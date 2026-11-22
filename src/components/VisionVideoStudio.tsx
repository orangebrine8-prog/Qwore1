import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Camera,
  Film,
  Image as ImageIcon,
  MessageSquare,
  Pause,
  Play,
  RotateCcw,
  Volume2,
} from 'lucide-react';
import { formatParamCount } from '../engine/architecture';
import {
  CameraMotionMode,
  MULTIMODAL_CONCEPTS,
  MultimodalCheckpoint,
  MultimodalHyperparams,
  MultimodalStepMetric,
  QoweVisionVideoEngine,
} from '../engine/multimodal';
import {
  generateMediaInPool,
  getBrowserWorkerCount,
  trainMultimodalPool,
} from '../engine/workerClient';
import type { NeuralAudioFrame } from '../utils/neuralAudioSynth';
import { CanvasMediaPlayer } from './CanvasMediaPlayer';

interface VisionVideoStudioProps {
  checkpoint: MultimodalCheckpoint | null;
  onSaveCheckpoint: (cp: MultimodalCheckpoint | null) => void;
  onOpenChat: () => void;
  onOpenMobileMenu: () => void;
}

const DEFAULT_MM_HYPERPARAMS: MultimodalHyperparams = {
  dCond: 24,
  hiddenDim: 80,
  numFreqs: 4,
  learningRate: 0.015,
  totalSteps: 160,
  seed: 2026,
  audioHiddenDim: 32,
};

const CAMERA_MODES: { value: CameraMotionMode; label: string }[] = [
  { value: 'auto', label: 'Auto (Prompt)' },
  { value: 'dolly-in', label: 'Dolly In' },
  { value: 'orbit', label: 'Orbit' },
  { value: 'pan-right', label: 'Pan Right' },
  { value: 'crane-up', label: 'Crane Up' },
  { value: 'handheld', label: 'Handheld' },
  { value: 'static', label: 'Static' },
];

interface ActivePreview {
  kind: 'image' | 'video';
  prompt: string;
  width: number;
  height: number;
  fps: number;
  frames: Uint8ClampedArray[];
  audioTrack?: NeuralAudioFrame[];
  cameraMotion?: CameraMotionMode;
}

export const VisionVideoStudio: React.FC<VisionVideoStudioProps> = ({
  checkpoint,
  onSaveCheckpoint,
  onOpenChat,
}) => {
  const [hp, setHp] = useState<MultimodalHyperparams>(
    checkpoint?.hyperparams ?? DEFAULT_MM_HYPERPARAMS,
  );
  const [isTraining, setIsTraining] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [metrics, setMetrics] = useState<MultimodalStepMetric[]>(checkpoint?.history ?? []);
  const [prompt, setPrompt] = useState<string>(MULTIMODAL_CONCEPTS[0].prompt);
  const [aspectRatio, setAspectRatio] = useState<'16:9' | '1:1'>('16:9');
  const [cameraMotion, setCameraMotion] = useState<CameraMotionMode>('auto');
  const [withAudio, setWithAudio] = useState<boolean>(true);
  const [preview, setPreview] = useState<ActivePreview | null>(null);
  const [workerError, setWorkerError] = useState<string | null>(null);

  const workerCount = useMemo(() => getBrowserWorkerCount(), []);
  const activeCancelRef = useRef<(() => void) | null>(null);
  const promptRef = useRef(prompt);
  promptRef.current = prompt;
  const aspectRef = useRef(aspectRatio);
  aspectRef.current = aspectRatio;
  const cameraRef = useRef(cameraMotion);
  cameraRef.current = cameraMotion;
  const audioRef = useRef(withAudio);
  audioRef.current = withAudio;

  useEffect(() => {
    return () => {
      activeCancelRef.current?.();
      activeCancelRef.current = null;
    };
  }, []);

  const paramCount = useMemo(() => {
    return new QoweVisionVideoEngine(undefined, hp).countParameters();
  }, [hp]);

  const startDistributedTraining = (resumeCheckpoint: MultimodalCheckpoint | null) => {
    activeCancelRef.current?.();
    setWorkerError(null);
    if (!resumeCheckpoint) {
      setMetrics([]);
    }
    setIsTraining(true);

    const handle = trainMultimodalPool({
      hyperparams: hp,
      previewPrompt: promptRef.current.trim() || MULTIMODAL_CONCEPTS[0].prompt,
      resumeFrom: resumeCheckpoint,
      onProgress: (ev) => {
        setMetrics(ev.history);
        if (ev.previewFrame && ev.previewWidth && ev.previewHeight) {
          setPreview({
            kind: 'image',
            prompt: promptRef.current,
            width: ev.previewWidth,
            height: ev.previewHeight,
            fps: 1,
            frames: [ev.previewFrame],
          });
        }
      },
      onDone: (cp, paused) => {
        activeCancelRef.current = null;
        setIsTraining(false);
        setMetrics(cp.history);
        onSaveCheckpoint(cp);
        if (!paused) {
          setIsGenerating(true);
          generateMediaInPool('video', promptRef.current, cp, 140, {
            aspectRatio: aspectRef.current,
            cameraMotion: cameraRef.current,
            withAudio: audioRef.current,
          })
            .then((media) => {
              setPreview({
                kind: 'video',
                prompt: promptRef.current,
                width: media.width,
                height: media.height,
                fps: media.fps,
                frames: media.frames,
                audioTrack: media.audioTrack,
                cameraMotion: media.cameraMotion,
              });
            })
            .catch((err) => {
              setWorkerError(err instanceof Error ? err.message : 'Unable to render 60 fps preview.');
            })
            .finally(() => setIsGenerating(false));
        }
      },
      onError: (msg) => {
        activeCancelRef.current = null;
        setIsTraining(false);
        setWorkerError(msg);
      },
    });

    activeCancelRef.current = handle.cancel;
  };

  const handleStartFreshTraining = () => {
    startDistributedTraining(null);
  };

  const handlePauseTraining = () => {
    activeCancelRef.current?.();
  };

  const handleResumeTraining = () => {
    startDistributedTraining(checkpoint);
  };

  const handleReset = () => {
    activeCancelRef.current?.();
    activeCancelRef.current = null;
    setIsTraining(false);
    setIsGenerating(false);
    setMetrics([]);
    setPreview(null);
    setWorkerError(null);
    onSaveCheckpoint(null);
  };

  const handleGenerate = async (kind: 'image' | 'video') => {
    if (!checkpoint || isTraining || isGenerating) return;
    setWorkerError(null);
    setIsGenerating(true);
    const targetPrompt = prompt.trim() || MULTIMODAL_CONCEPTS[0].prompt;
    try {
      const media = await generateMediaInPool(kind, targetPrompt, checkpoint, 140, {
        aspectRatio,
        cameraMotion,
        withAudio,
      });
      setPreview({
        kind,
        prompt: targetPrompt,
        width: media.width,
        height: media.height,
        fps: media.fps,
        frames: media.frames,
        audioTrack: media.audioTrack,
        cameraMotion: media.cameraMotion,
      });
    } catch (err) {
      setWorkerError(err instanceof Error ? err.message : 'Synthesis failed.');
    } finally {
      setIsGenerating(false);
    }
  };

  const latest = metrics[metrics.length - 1];
  const currentStep = latest?.step ?? checkpoint?.step ?? 0;
  const currentLoss = latest?.loss ?? checkpoint?.loss ?? null;
  const currentAudioLoss = latest?.audioLoss ?? null;
  const currentPsnr =
    latest?.psnr ?? (currentLoss !== null ? 10 * Math.log10(1 / Math.max(1e-6, currentLoss)) : null);
  const hasTrainedWeights = Boolean(checkpoint);

  const svgWidth = 600;
  const svgHeight = 160;
  const padX = 36;
  const padY = 16;
  const chartW = svgWidth - padX * 2;
  const chartH = svgHeight - padY * 2;

  const lossPoints = useMemo(() => {
    if (metrics.length === 0) return '';
    const maxLoss = Math.max(0.05, ...metrics.map((m) => m.loss));
    const minLoss = Math.min(0, ...metrics.map((m) => m.loss));
    const range = Math.max(0.01, maxLoss - minLoss);
    return metrics
      .map((m, idx) => {
        const x = padX + (metrics.length === 1 ? chartW / 2 : (idx / (metrics.length - 1)) * chartW);
        const y = padY + chartH - ((m.loss - minLoss) / range) * chartH;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  }, [metrics, chartW, chartH]);

  const audioLossPoints = useMemo(() => {
    const withAud = metrics.filter((m) => typeof m.audioLoss === 'number');
    if (withAud.length === 0) return '';
    const maxLoss = Math.max(0.05, ...metrics.map((m) => m.loss));
    const minLoss = Math.min(0, ...metrics.map((m) => m.loss));
    const range = Math.max(0.01, maxLoss - minLoss);
    return metrics
      .map((m, idx) => {
        const val = m.audioLoss ?? m.loss;
        const x = padX + (metrics.length === 1 ? chartW / 2 : (idx / (metrics.length - 1)) * chartW);
        const y = padY + chartH - (Math.min(maxLoss, Math.max(minLoss, val)) - minLoss) / range * chartH;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  }, [metrics, chartW, chartH]);

  const maxLossVal = metrics.length > 0 ? Math.max(...metrics.map((m) => m.loss)) : 0.25;
  const minLossVal = metrics.length > 0 ? Math.min(...metrics.map((m) => m.loss)) : 0.0;

  return (
    <div className="flex flex-col flex-1 h-full min-h-0 min-w-0 bg-[#f5f5f7] dark:bg-[#0b0b0d]">
      <header className="flex items-center justify-between gap-4 px-4 sm:px-6 py-2.5 sm:py-3 bg-white/95 dark:bg-[#161618]/95 border-b border-black/[0.06] dark:border-white/[0.08] shrink-0">
        <div className="flex items-center gap-3 min-w-0">
          <div className="min-w-0">
            <h1 className="text-sm sm:text-base font-semibold text-neutral-900 dark:text-white truncate">
              From-Scratch Audio &amp; 60 FPS Cinema Engine
            </h1>
            <p className="text-xs text-neutral-500 dark:text-neutral-400 truncate tabular-nums">
              Joint Audio-Visual Spatiotemporal Field
              <span aria-hidden="true"> · </span>
              {formatParamCount(paramCount)} trainable params
              <span aria-hidden="true"> · </span>
              {workerCount}-Core AllReduce (zero external video APIs)
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={onOpenChat}
          className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full border border-black/[0.08] dark:border-white/[0.1] text-xs font-medium text-neutral-700 dark:text-neutral-200 hover:bg-black/[0.04] dark:hover:bg-white/[0.08] transition-colors whitespace-nowrap shrink-0"
        >
          <MessageSquare className="w-3.5 h-3.5" />
          <span>Use in Chat</span>
        </button>
      </header>

      <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-6">
        <div className="max-w-6xl mx-auto space-y-6">
          {/* Metrics summary row */}
          <section
            aria-label="Multimodal training metrics"
            className="grid grid-cols-2 sm:grid-cols-4 gap-3.5"
          >
            <div className="p-4 rounded-[22px] bg-white/95 dark:bg-[#1c1c1e]/95 border border-black/[0.06] dark:border-white/[0.08] shadow-[0_1px_2px_rgba(0,0,0,0.03),0_8px_24px_rgba(0,0,0,0.03)]">
              <div className="text-xs text-neutral-500 dark:text-neutral-400">Training Step</div>
              <div className="mt-1 text-xl font-semibold text-neutral-900 dark:text-white tabular-nums">
                {currentStep} / {hp.totalSteps}
              </div>
            </div>
            <div className="p-4 rounded-[22px] bg-white/95 dark:bg-[#1c1c1e]/95 border border-black/[0.06] dark:border-white/[0.08] shadow-[0_1px_2px_rgba(0,0,0,0.03),0_8px_24px_rgba(0,0,0,0.03)]">
              <div className="text-xs text-neutral-500 dark:text-neutral-400">Visual / Audio MSE</div>
              <div className="mt-1 text-xl font-semibold text-neutral-900 dark:text-white tabular-nums">
                {currentLoss !== null ? currentLoss.toFixed(4) : '—'}
                {currentAudioLoss !== null && (
                  <span className="text-xs font-mono font-normal text-neutral-500 dark:text-neutral-400">
                    {' '}
                    / {currentAudioLoss.toFixed(4)}
                  </span>
                )}
              </div>
            </div>
            <div className="p-4 rounded-[22px] bg-white/95 dark:bg-[#1c1c1e]/95 border border-black/[0.06] dark:border-white/[0.08] shadow-[0_1px_2px_rgba(0,0,0,0.03),0_8px_24px_rgba(0,0,0,0.03)]">
              <div className="text-xs text-neutral-500 dark:text-neutral-400">Reconstruction PSNR</div>
              <div className="mt-1 text-xl font-semibold text-neutral-900 dark:text-white tabular-nums">
                {currentPsnr !== null ? `${currentPsnr.toFixed(1)} dB` : '—'}
              </div>
            </div>
            <div className="p-4 rounded-[22px] bg-white/95 dark:bg-[#1c1c1e]/95 border border-black/[0.06] dark:border-white/[0.08] shadow-[0_1px_2px_rgba(0,0,0,0.03),0_8px_24px_rgba(0,0,0,0.03)]">
              <div className="text-xs text-neutral-500 dark:text-neutral-400">Worker Pool Output</div>
              <div className="mt-1 text-xl font-semibold text-neutral-900 dark:text-white tabular-nums">
                {workerCount} cores · 60 fps + Sound
              </div>
            </div>
          </section>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left column: Visual Generator + Viewport + Loss Curve */}
            <div className="lg:col-span-7 space-y-6">
              <section className="p-5 rounded-[22px] bg-white/95 dark:bg-[#1c1c1e]/95 border border-black/[0.06] dark:border-white/[0.08] shadow-[0_1px_2px_rgba(0,0,0,0.03),0_8px_24px_rgba(0,0,0,0.03)] space-y-4">
                <div>
                  <h2 className="text-base font-semibold text-neutral-900 dark:text-white">
                    Synthesize 60 FPS Cinema Video + Synced Audio
                  </h2>
                  <p className="text-xs text-neutral-500 dark:text-neutral-400">
                    Trained end-to-end from scratch across {workerCount} CPU workers for camera motion, lighting bloom, and matching stereo sound — zero external video models.
                  </p>
                </div>

                {!hasTrainedWeights && !isTraining && (
                  <div className="p-4 rounded-2xl border border-amber-300/60 dark:border-amber-800/60 bg-amber-50/90 dark:bg-amber-950/30 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    <p className="text-xs text-amber-900 dark:text-amber-200">
                      Train the from-scratch audio-visual neural weights first so random layers learn scenes, camera motion, and acoustics.
                    </p>
                    <button
                      type="button"
                      onClick={handleStartFreshTraining}
                      className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-full bg-amber-900 hover:bg-amber-800 dark:bg-amber-600 dark:hover:bg-amber-500 text-white text-xs font-medium whitespace-nowrap shrink-0"
                    >
                      <Play className="w-3.5 h-3.5" />
                      <span>Initialize &amp; Train</span>
                    </button>
                  </div>
                )}

                {workerError && (
                  <div className="p-3.5 rounded-2xl bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900 text-xs text-red-700 dark:text-red-300">
                    {workerError}
                  </div>
                )}

                <div className="space-y-2.5">
                  <label htmlFor="vision-prompt-input" className="sr-only">
                    Visual generation prompt
                  </label>
                  <input
                    id="vision-prompt-input"
                    type="text"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Describe a scene + motion (e.g., dolly in neon cyber city rain reflections)"
                    className="w-full rounded-2xl border border-black/[0.08] dark:border-white/[0.1] bg-[#f5f5f7] dark:bg-[#121214] px-4 py-2.5 text-sm text-neutral-900 dark:text-neutral-100 focus:outline-none focus:border-[#007aff]"
                  />

                  <div className="flex sm:flex-wrap overflow-x-auto sm:overflow-visible gap-1.5 pb-1 sm:pb-0">
                    {[
                      ...MULTIMODAL_CONCEPTS.map((c) => c.prompt),
                      'sports car drifting through neon cyber city in heavy rain',
                      'rocket launching into crimson solar eclipse deep space',
                      'dragon flying over emerald aurora snowy mountains',
                      'golden dog running on sunny ocean beach waves',
                    ].map((preset) => {
                      const active = prompt === preset;
                      return (
                        <button
                          key={preset}
                          type="button"
                          onClick={() => setPrompt(preset)}
                          className={`px-3 py-1 rounded-full text-xs transition-colors whitespace-nowrap shrink-0 ${
                            active
                              ? 'bg-[#007aff] text-white dark:bg-[#0a84ff] font-medium'
                              : 'bg-black/[0.05] dark:bg-white/[0.08] text-neutral-600 dark:text-neutral-300 hover:bg-black/[0.08] dark:hover:bg-white/[0.12]'
                          }`}
                        >
                          {preset}
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Cinema Director Controls: Aspect Ratio, Camera Motion, Synced Audio */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 text-xs">
                  <div className="flex flex-col gap-1">
                    <span className="text-neutral-500 dark:text-neutral-400 font-medium">
                      Aspect Ratio
                    </span>
                    <div className="flex p-1 rounded-full bg-black/[0.05] dark:bg-white/[0.08]">
                      <button
                        type="button"
                        onClick={() => setAspectRatio('16:9')}
                        className={`flex-1 py-1 rounded-full text-xs font-medium transition-colors ${
                          aspectRatio === '16:9'
                            ? 'bg-white dark:bg-[#2c2c2e] text-neutral-900 dark:text-white shadow-xs'
                            : 'text-neutral-600 dark:text-neutral-400'
                        }`}
                      >
                        16:9 Cinema
                      </button>
                      <button
                        type="button"
                        onClick={() => setAspectRatio('1:1')}
                        className={`flex-1 py-1 rounded-full text-xs font-medium transition-colors ${
                          aspectRatio === '1:1'
                            ? 'bg-white dark:bg-[#2c2c2e] text-neutral-900 dark:text-white shadow-xs'
                            : 'text-neutral-600 dark:text-neutral-400'
                        }`}
                      >
                        1:1 Square
                      </button>
                    </div>
                  </div>

                  <label className="flex flex-col gap-1">
                    <span className="inline-flex items-center gap-1 text-neutral-500 dark:text-neutral-400 font-medium">
                      <Camera className="w-3 h-3" />
                      Camera Trajectory
                    </span>
                    <select
                      value={cameraMotion}
                      onChange={(e) => setCameraMotion(e.target.value as CameraMotionMode)}
                      className="rounded-full border border-black/[0.08] dark:border-white/[0.1] bg-[#f5f5f7] dark:bg-[#121214] px-3 py-1.5 text-xs text-neutral-900 dark:text-neutral-100"
                    >
                      {CAMERA_MODES.map((m) => (
                        <option key={m.value} value={m.value}>
                          {m.label}
                        </option>
                      ))}
                    </select>
                  </label>

                  <div className="flex flex-col gap-1">
                    <span className="inline-flex items-center gap-1 text-neutral-500 dark:text-neutral-400 font-medium">
                      <Volume2 className="w-3 h-3" />
                      Neural Audio Head
                    </span>
                    <button
                      type="button"
                      onClick={() => setWithAudio((a) => !a)}
                      className={`py-1.5 px-3 rounded-full border text-xs font-medium transition-colors ${
                        withAudio
                          ? 'border-[#007aff] text-[#007aff] bg-[#007aff]/10 dark:border-[#0a84ff] dark:text-[#0a84ff] dark:bg-[#0a84ff]/15'
                          : 'border-black/[0.08] dark:border-white/[0.1] text-neutral-600 dark:text-neutral-400'
                      }`}
                    >
                      {withAudio ? 'Synced Audio On' : 'Video Only'}
                    </button>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2">
                  <button
                    type="button"
                    disabled={!hasTrainedWeights || isTraining || isGenerating}
                    onClick={() => handleGenerate('image')}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[#007aff] hover:bg-[#0066d6] dark:bg-[#0a84ff] dark:hover:bg-[#1a8cff] disabled:bg-black/[0.08] dark:disabled:bg-white/[0.08] disabled:text-neutral-400 text-white text-xs font-medium transition-colors whitespace-nowrap"
                  >
                    <ImageIcon className="w-4 h-4" />
                    <span>Generate Image ({aspectRatio})</span>
                  </button>

                  <button
                    type="button"
                    disabled={!hasTrainedWeights || isTraining || isGenerating}
                    onClick={() => handleGenerate('video')}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-neutral-900 hover:bg-neutral-800 dark:bg-white dark:hover:bg-neutral-200 dark:text-neutral-900 disabled:bg-black/[0.08] dark:disabled:bg-white/[0.08] disabled:text-neutral-400 text-white text-xs font-medium transition-colors whitespace-nowrap"
                  >
                    <Film className="w-4 h-4" />
                    <span>
                      {isGenerating ? 'Synthesizing Audio + 60 FPS...' : 'Generate Video (60 fps + Audio)'}
                    </span>
                  </button>
                </div>

                {/* Zero-GC Canvas Viewport */}
                <div className="rounded-2xl bg-[#f5f5f7] dark:bg-[#121214] p-4 flex flex-col items-center justify-center min-h-[300px]">
                  {preview && preview.frames.length > 0 ? (
                    <CanvasMediaPlayer
                      frames={preview.frames}
                      width={preview.width}
                      height={preview.height}
                      fps={preview.fps}
                      label={preview.prompt}
                      audioTrack={preview.audioTrack}
                      cameraMotion={preview.cameraMotion}
                      showScrubber={preview.kind === 'video'}
                    />
                  ) : (
                    <div className="text-center space-y-1.5 px-4">
                      <p className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
                        {isTraining
                          ? `Training visual & audio heads across ${workerCount} workers...`
                          : '16:9 60 FPS Cinema & Sound Viewport Ready'}
                      </p>
                      <p className="text-xs text-neutral-500">
                        Train the neural weights on the right, then synthesize 60 fps camera-tracked video with synced stereo audio.
                      </p>
                    </div>
                  )}
                </div>
              </section>

              {/* Loss Curve */}
              <section className="p-5 rounded-[22px] bg-white/95 dark:bg-[#1c1c1e]/95 border border-black/[0.06] dark:border-white/[0.08] shadow-[0_1px_2px_rgba(0,0,0,0.03),0_8px_24px_rgba(0,0,0,0.03)] space-y-3">
                <div className="flex items-center justify-between gap-2">
                  <div>
                    <h2 className="text-base font-semibold text-neutral-900 dark:text-white">
                      Joint Visual &amp; Audio Loss (MSE)
                    </h2>
                    <p className="text-xs text-neutral-500 dark:text-neutral-400">
                      RGB reconstruction (blue) &amp; synced acoustic envelope loss (emerald) across {MULTIMODAL_CONCEPTS.length} scenes.
                    </p>
                  </div>
                  {metrics.length > 0 && (
                    <span className="text-xs font-mono text-neutral-600 dark:text-neutral-300 tabular-nums shrink-0">
                      Min {minLossVal.toFixed(4)} · Max {maxLossVal.toFixed(4)}
                    </span>
                  )}
                </div>

                <div className="rounded-2xl bg-[#f5f5f7] dark:bg-[#121214] p-3.5">
                  {metrics.length === 0 ? (
                    <div className="h-36 flex items-center justify-center text-xs text-neutral-500 dark:text-neutral-400">
                      Start training to track visual pixel MSE, audio synthesis loss, and PSNR.
                    </div>
                  ) : (
                    <svg
                      viewBox={`0 0 ${svgWidth} ${svgHeight}`}
                      className="w-full h-36"
                      role="img"
                      aria-label="Joint visual and audio MSE loss curve"
                    >
                      {[0.25, 0.5, 0.75, 1].map((frac) => {
                        const y = padY + chartH * (1 - frac);
                        return (
                          <line
                            key={frac}
                            x1={padX}
                            y1={y}
                            x2={svgWidth - padX}
                            y2={y}
                            stroke="currentColor"
                            className="text-black/[0.08] dark:text-white/[0.08]"
                            strokeWidth="1"
                          />
                        );
                      })}
                      {audioLossPoints && (
                        <polyline
                          fill="none"
                          stroke="#10b981"
                          strokeWidth="1.8"
                          strokeDasharray="4 3"
                          strokeLinejoin="round"
                          strokeLinecap="round"
                          points={audioLossPoints}
                        />
                      )}
                      <polyline
                        fill="none"
                        stroke="#007aff"
                        strokeWidth="2.4"
                        strokeLinejoin="round"
                        strokeLinecap="round"
                        points={lossPoints}
                      />
                      <text x={4} y={padY + 4} className="fill-neutral-400 text-[10px] font-mono">
                        {maxLossVal.toFixed(2)}
                      </text>
                      <text
                        x={4}
                        y={svgHeight - padY + 4}
                        className="fill-neutral-400 text-[10px] font-mono"
                      >
                        {minLossVal.toFixed(3)}
                      </text>
                    </svg>
                  )}
                </div>
              </section>
            </div>

            {/* Right column: Hyperparameters + Training Controls */}
            <div className="lg:col-span-5 space-y-6">
              <section className="p-5 rounded-[22px] bg-white/95 dark:bg-[#1c1c1e]/95 border border-black/[0.06] dark:border-white/[0.08] shadow-[0_1px_2px_rgba(0,0,0,0.03),0_8px_24px_rgba(0,0,0,0.03)] space-y-4">
                <div>
                  <h2 className="text-base font-semibold text-neutral-900 dark:text-white">
                    From-Scratch Audio-Video Trainer
                  </h2>
                  <p className="text-xs text-neutral-500 dark:text-neutral-400">
                    Fourier Features (x, y, r, t) · FiLM Conditioning · Joint Audio Head · {workerCount}-Core AllReduce AdamW
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <label className="block text-xs">
                    <span className="text-neutral-600 dark:text-neutral-400 font-medium">
                      Hidden Dim
                    </span>
                    <select
                      value={hp.hiddenDim}
                      disabled={isTraining}
                      onChange={(e) => setHp({ ...hp, hiddenDim: Number(e.target.value) })}
                      className="mt-1 w-full rounded-2xl border border-black/[0.08] dark:border-white/[0.1] bg-[#f5f5f7] dark:bg-[#121214] px-3 py-2 text-xs font-mono text-neutral-900 dark:text-neutral-100"
                    >
                      <option value={64}>64</option>
                      <option value={80}>80</option>
                      <option value={96}>96</option>
                    </select>
                  </label>

                  <label className="block text-xs">
                    <span className="text-neutral-600 dark:text-neutral-400 font-medium">
                      Fourier Bands
                    </span>
                    <select
                      value={hp.numFreqs}
                      disabled={isTraining}
                      onChange={(e) => setHp({ ...hp, numFreqs: Number(e.target.value) })}
                      className="mt-1 w-full rounded-2xl border border-black/[0.08] dark:border-white/[0.1] bg-[#f5f5f7] dark:bg-[#121214] px-3 py-2 text-xs font-mono text-neutral-900 dark:text-neutral-100"
                    >
                      <option value={3}>3 bands</option>
                      <option value={4}>4 bands</option>
                      <option value={5}>5 bands</option>
                    </select>
                  </label>

                  <label className="block text-xs">
                    <span className="text-neutral-600 dark:text-neutral-400 font-medium">
                      Learning Rate
                    </span>
                    <input
                      type="number"
                      step="0.002"
                      min="0.002"
                      max="0.04"
                      disabled={isTraining}
                      value={hp.learningRate}
                      onChange={(e) =>
                        setHp({ ...hp, learningRate: Math.max(0.001, Number(e.target.value)) })
                      }
                      className="mt-1 w-full rounded-2xl border border-black/[0.08] dark:border-white/[0.1] bg-[#f5f5f7] dark:bg-[#121214] px-3 py-2 text-xs font-mono tabular-nums text-neutral-900 dark:text-neutral-100"
                    />
                  </label>

                  <label className="block text-xs">
                    <span className="text-neutral-600 dark:text-neutral-400 font-medium">
                      Total Steps
                    </span>
                    <input
                      type="number"
                      step="20"
                      min="40"
                      max="400"
                      disabled={isTraining}
                      value={hp.totalSteps}
                      onChange={(e) =>
                        setHp({ ...hp, totalSteps: Math.max(20, Number(e.target.value)) })
                      }
                      className="mt-1 w-full rounded-2xl border border-black/[0.08] dark:border-white/[0.1] bg-[#f5f5f7] dark:bg-[#121214] px-3 py-2 text-xs font-mono tabular-nums text-neutral-900 dark:text-neutral-100"
                    />
                  </label>
                </div>

                <div className="pt-2 flex flex-wrap gap-2">
                  {!isTraining ? (
                    <button
                      type="button"
                      onClick={handleStartFreshTraining}
                      className="flex-1 inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-full bg-[#007aff] hover:bg-[#0066d6] dark:bg-[#0a84ff] dark:hover:bg-[#1a8cff] text-white text-sm font-medium transition-colors whitespace-nowrap"
                    >
                      <Play className="w-4 h-4" />
                      <span>Initialize &amp; Train</span>
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={handlePauseTraining}
                      className="flex-1 inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-full bg-amber-600 hover:bg-amber-700 text-white text-sm font-medium transition-colors whitespace-nowrap"
                    >
                      <Pause className="w-4 h-4" />
                      <span>Pause Training</span>
                    </button>
                  )}

                  {!isTraining && checkpoint && currentStep > 0 && currentStep < hp.totalSteps && (
                    <button
                      type="button"
                      onClick={handleResumeTraining}
                      className="px-4 py-2.5 rounded-full border border-black/[0.08] dark:border-white/[0.1] text-xs font-medium text-neutral-700 dark:text-neutral-200 hover:bg-black/[0.04] dark:hover:bg-white/[0.06] whitespace-nowrap"
                    >
                      Resume
                    </button>
                  )}

                  <button
                    type="button"
                    onClick={handleReset}
                    aria-label="Reset multimodal weights"
                    className="p-2.5 rounded-full border border-black/[0.08] dark:border-white/[0.1] text-neutral-600 dark:text-neutral-300 hover:bg-black/[0.04] dark:hover:bg-white/[0.06]"
                  >
                    <RotateCcw className="w-4 h-4" />
                  </button>
                </div>
              </section>

              <section className="p-5 rounded-[22px] bg-white/95 dark:bg-[#1c1c1e]/95 border border-black/[0.06] dark:border-white/[0.08] shadow-[0_1px_2px_rgba(0,0,0,0.03),0_8px_24px_rgba(0,0,0,0.03)] space-y-3">
                <h2 className="text-base font-semibold text-neutral-900 dark:text-white">
                  Compositional Neural Cinema Architecture
                </h2>
                <ul className="space-y-2 text-xs text-neutral-600 dark:text-neutral-400 leading-relaxed">
                  <li>
                    <strong className="text-neutral-900 dark:text-neutral-200">1. {workerCount}-Worker Tri-Head AllReduce:</strong> Jointly trains visual pixels, a 14-latent compositional scene predictor, and an 8-channel neural audio head from scratch across all CPU workers.
                  </li>
                  <li>
                    <strong className="text-neutral-900 dark:text-neutral-200">2. Sharp 320×180 2.5D Volumetric Cinema:</strong> Fuses the trained neural lighting field with multi-plane reflections, articulated dynamic subjects (cars, rockets, dragons, dogs, boats), perspective rain/embers, and camera trajectories.
                  </li>
                  <li>
                    <strong className="text-neutral-900 dark:text-neutral-200">3. Synced Stereo Audio &amp; WebM Muxing:</strong> Synthesizes resonant surf, rain, turbine roar, chimes, and sub-bass synced to the 60 fps loop and exports muxed <span className="font-mono">.webm</span> files.
                  </li>
                </ul>
              </section>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
