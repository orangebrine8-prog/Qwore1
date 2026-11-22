import React, { useDeferredValue, useEffect, useMemo, useRef, useState } from 'react';
import {
  Download,
  MessageSquare,
  Pause,
  Play,
  RotateCcw,
  Upload,
} from 'lucide-react';
import { TRAINING_DATASETS } from '../engine/datasets';
import { formatParamCount } from '../engine/architecture';
import { tokenizeWithPieces, trainTokenizer } from '../engine/tokenizer';
import {
  countKernelParameters,
  QoweCheckpoint,
  QoweTrainerEngine,
  QoweTrainingHyperparams,
  TrainingStepMetric,
} from '../engine/trainer';
import { postWorkerCommand, subscribeWorkerEvents } from '../engine/workerClient';
import type { WorkspaceTab } from '../types/qowe';

interface TrainingStudioProps {
  corpus: string;
  onChangeCorpus: (next: string) => void;
  checkpoint: QoweCheckpoint | null;
  onSaveCheckpoint: (cp: QoweCheckpoint | null) => void;
  onOpenChatWithWeights: () => void;
  onNavigateTab: (tab: WorkspaceTab) => void;
  onOpenMobileMenu: () => void;
  targetParamsFormatted: string;
}

const DEFAULT_HYPERPARAMS: QoweTrainingHyperparams = {
  dModel: 48,
  nLayers: 1,
  nHeads: 4,
  dFF: 96,
  contextLength: 48,
  learningRate: 0.015,
  totalSteps: 120,
  seed: 1337,
};

export const TrainingStudio: React.FC<TrainingStudioProps> = ({
  corpus,
  onChangeCorpus,
  checkpoint,
  onSaveCheckpoint,
  onOpenChatWithWeights,
  onOpenMobileMenu,
  targetParamsFormatted,
}) => {
  const [hp, setHp] = useState<QoweTrainingHyperparams>(
    checkpoint?.hyperparams ?? DEFAULT_HYPERPARAMS,
  );
  const [isTraining, setIsTraining] = useState(false);
  const [metrics, setMetrics] = useState<TrainingStepMetric[]>(checkpoint?.history ?? []);
  const [samplePrompt, setSamplePrompt] = useState('<|user|> Who are you?\n<|assistant|>');
  const [sampleOutput, setSampleOutput] = useState<string>('');
  const [sampleTopTokens, setSampleTopTokens] = useState<
    { id: number; token: string; probability: number }[]
  >([]);
  const [showTokenizerDetails, setShowTokenizerDetails] = useState(true);

  const engineRef = useRef<QoweTrainerEngine | null>(null);
  const activeJobIdRef = useRef<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const corpusRef = useRef(corpus);
  const onSaveCheckpointRef = useRef(onSaveCheckpoint);
  const samplePromptRef = useRef(samplePrompt);
  corpusRef.current = corpus;
  onSaveCheckpointRef.current = onSaveCheckpoint;
  samplePromptRef.current = samplePrompt;

  const deferredCorpus = useDeferredValue(corpus);

  // Live tokenizer preview from deferred corpus so typing never lags
  const tokenizer = useMemo(() => trainTokenizer(deferredCorpus || ' ', 360), [deferredCorpus]);
  const tokenPieces = useMemo(
    () => tokenizeWithPieces(tokenizer, deferredCorpus.slice(0, 420)),
    [tokenizer, deferredCorpus],
  );
  const activeKernelParams = useMemo(
    () => countKernelParameters(tokenizer.vocab.length, hp),
    [tokenizer.vocab.length, hp],
  );

  useEffect(() => {
    if (checkpoint && !engineRef.current) {
      try {
        engineRef.current = QoweTrainerEngine.fromCheckpoint(checkpoint, corpusRef.current);
      } catch {
        engineRef.current = null;
      }
    }
  }, [checkpoint]);

  useEffect(() => {
    const unsub = subscribeWorkerEvents((ev) => {
      if (ev.jobId !== activeJobIdRef.current) return;
      if (ev.type === 'text-progress') {
        setMetrics(ev.history);
      } else if (ev.type === 'text-done') {
        activeJobIdRef.current = null;
        setIsTraining(false);
        setMetrics(ev.checkpoint.history);
        onSaveCheckpointRef.current(ev.checkpoint);
        try {
          const restored = QoweTrainerEngine.fromCheckpoint(ev.checkpoint, corpusRef.current);
          engineRef.current = restored;
          const gen = restored.generate(samplePromptRef.current, {
            maxNewTokens: 36,
            temperature: 0.25,
            topK: 3,
          });
          setSampleOutput(gen.text);
          setSampleTopTokens(gen.topCandidatesAtLastStep);
        } catch {
          // Ignore sample error
        }
      } else if (ev.type === 'error') {
        activeJobIdRef.current = null;
        setIsTraining(false);
      }
    });
    return () => {
      unsub();
      if (activeJobIdRef.current) {
        postWorkerCommand({ type: 'pause', jobId: activeJobIdRef.current });
      }
    };
  }, []);

  const handleStartFreshTraining = () => {
    if (activeJobIdRef.current) {
      postWorkerCommand({ type: 'pause', jobId: activeJobIdRef.current });
    }
    const jobId = `txt-${Date.now()}`;
    activeJobIdRef.current = jobId;
    engineRef.current = null;
    setMetrics([]);
    setSampleOutput('');
    setSampleTopTokens([]);
    setIsTraining(true);
    postWorkerCommand({
      type: 'train-text',
      jobId,
      corpus: corpus || 'Hello Qowe 1.0',
      hyperparams: hp,
      resumeFrom: null,
    });
  };

  const handleResumeTraining = () => {
    if (!checkpoint) {
      handleStartFreshTraining();
      return;
    }
    const jobId = `txt-${Date.now()}`;
    activeJobIdRef.current = jobId;
    setIsTraining(true);
    postWorkerCommand({
      type: 'train-text',
      jobId,
      corpus: corpus || 'Hello Qowe 1.0',
      hyperparams: hp,
      resumeFrom: checkpoint,
    });
  };

  const handlePauseTraining = () => {
    if (activeJobIdRef.current) {
      postWorkerCommand({ type: 'pause', jobId: activeJobIdRef.current });
    }
  };

  const handleReset = () => {
    if (activeJobIdRef.current) {
      postWorkerCommand({ type: 'pause', jobId: activeJobIdRef.current });
      activeJobIdRef.current = null;
    }
    setIsTraining(false);
    engineRef.current = null;
    setMetrics([]);
    setSampleOutput('');
    setSampleTopTokens([]);
    onSaveCheckpoint(null);
  };

  const handleGenerateSample = () => {
    const engine =
      engineRef.current ??
      (checkpoint ? QoweTrainerEngine.fromCheckpoint(checkpoint, corpus) : null);
    if (!engine) return;
    engineRef.current = engine;
    const gen = engine.generate(samplePrompt, { maxNewTokens: 36, temperature: 0.25, topK: 3 });
    setSampleOutput(gen.text);
    setSampleTopTokens(gen.topCandidatesAtLastStep);
  };

  const handleDownloadCheckpoint = () => {
    const cp = engineRef.current ? engineRef.current.exportCheckpoint() : checkpoint;
    if (!cp) return;
    const blob = new Blob([JSON.stringify(cp, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `qowe-1.0-step-${cp.step}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleUploadCorpus = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string' && reader.result.trim()) {
        onChangeCorpus(reader.result);
      }
    };
    reader.readAsText(file);
  };

  const latest = metrics[metrics.length - 1];
  const currentStep = latest?.step ?? checkpoint?.step ?? 0;
  const currentLoss = latest?.loss ?? checkpoint?.loss ?? null;
  const currentPpl = latest?.perplexity ?? checkpoint?.perplexity ?? null;

  // Build SVG path for loss curve
  const svgWidth = 600;
  const svgHeight = 190;
  const padX = 36;
  const padY = 18;
  const chartW = svgWidth - padX * 2;
  const chartH = svgHeight - padY * 2;

  const lossPoints = useMemo(() => {
    if (metrics.length === 0) return '';
    const maxLoss = Math.max(0.5, ...metrics.map((m) => m.loss));
    const minLoss = Math.min(0, ...metrics.map((m) => m.loss));
    const range = Math.max(0.1, maxLoss - minLoss);
    return metrics
      .map((m, idx) => {
        const x = padX + (metrics.length === 1 ? chartW / 2 : (idx / (metrics.length - 1)) * chartW);
        const y = padY + chartH - ((m.loss - minLoss) / range) * chartH;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  }, [metrics, chartW, chartH]);

  const maxLossVal = metrics.length > 0 ? Math.max(...metrics.map((m) => m.loss)) : 5.0;
  const minLossVal = metrics.length > 0 ? Math.min(...metrics.map((m) => m.loss)) : 0.0;

  return (
    <div className="flex flex-col flex-1 h-full min-h-0 min-w-0 bg-[#f5f5f7] dark:bg-[#0b0b0d]">
      <header className="flex items-center justify-between gap-4 px-4 sm:px-6 py-2.5 sm:py-3 bg-white/95 dark:bg-[#161618]/95 border-b border-black/[0.06] dark:border-white/[0.08] shrink-0">
        <div className="flex items-center gap-3 min-w-0">
          <div className="min-w-0">
            <h1 className="text-sm sm:text-base font-semibold text-neutral-900 dark:text-white truncate">
              Train Qowe 1.0 from Scratch
            </h1>
            <p className="text-xs text-neutral-500 dark:text-neutral-400 truncate tabular-nums">
              Target Blueprint: {targetParamsFormatted} params
              <span aria-hidden="true"> · </span>
              Live Browser Kernel: {formatParamCount(activeKernelParams)} trainable params
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {(checkpoint || metrics.length > 0) && (
            <button
              type="button"
              onClick={onOpenChatWithWeights}
              className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-full bg-[#007aff] hover:bg-[#0066d6] dark:bg-[#0a84ff] dark:hover:bg-[#1a8cff] text-white text-xs font-medium transition-colors whitespace-nowrap shrink-0"
            >
              <MessageSquare className="w-3.5 h-3.5" />
              <span>Chat with Trained Weights</span>
            </button>
          )}
        </div>
      </header>

      <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-6">
        <div className="max-w-6xl mx-auto space-y-6">
          {/* Metrics summary row */}
          <section
            aria-label="Training metrics summary"
            className="grid grid-cols-2 sm:grid-cols-5 gap-3.5"
          >
            <div className="p-4 rounded-[22px] bg-white/95 dark:bg-[#1c1c1e]/95 border border-black/[0.06] dark:border-white/[0.08] shadow-[0_1px_2px_rgba(0,0,0,0.03),0_8px_24px_rgba(0,0,0,0.03)]">
              <div className="text-xs text-neutral-500 dark:text-neutral-400">Training Step</div>
              <div className="mt-1 text-xl font-semibold text-neutral-900 dark:text-white tabular-nums">
                {currentStep} / {hp.totalSteps}
              </div>
            </div>
            <div className="p-4 rounded-[22px] bg-white/95 dark:bg-[#1c1c1e]/95 border border-black/[0.06] dark:border-white/[0.08] shadow-[0_1px_2px_rgba(0,0,0,0.03),0_8px_24px_rgba(0,0,0,0.03)]">
              <div className="text-xs text-neutral-500 dark:text-neutral-400">Cross-Entropy Loss</div>
              <div className="mt-1 text-xl font-semibold text-neutral-900 dark:text-white tabular-nums">
                {currentLoss !== null ? currentLoss.toFixed(4) : '—'}
              </div>
            </div>
            <div className="p-4 rounded-[22px] bg-white/95 dark:bg-[#1c1c1e]/95 border border-black/[0.06] dark:border-white/[0.08] shadow-[0_1px_2px_rgba(0,0,0,0.03),0_8px_24px_rgba(0,0,0,0.03)]">
              <div className="text-xs text-neutral-500 dark:text-neutral-400">Perplexity</div>
              <div className="mt-1 text-xl font-semibold text-neutral-900 dark:text-white tabular-nums">
                {currentPpl !== null ? currentPpl.toFixed(2) : '—'}
              </div>
            </div>
            <div className="p-4 rounded-[22px] bg-white/95 dark:bg-[#1c1c1e]/95 border border-black/[0.06] dark:border-white/[0.08] shadow-[0_1px_2px_rgba(0,0,0,0.03),0_8px_24px_rgba(0,0,0,0.03)]">
              <div className="text-xs text-neutral-500 dark:text-neutral-400">Gradient Norm</div>
              <div className="mt-1 text-xl font-semibold text-neutral-900 dark:text-white tabular-nums">
                {latest ? latest.gradNorm.toFixed(3) : '—'}
              </div>
            </div>
            <div className="p-4 rounded-[22px] bg-white/95 dark:bg-[#1c1c1e]/95 border border-black/[0.06] dark:border-white/[0.08] shadow-[0_1px_2px_rgba(0,0,0,0.03),0_8px_24px_rgba(0,0,0,0.03)] col-span-2 sm:col-span-1">
              <div className="text-xs text-neutral-500 dark:text-neutral-400">Throughput</div>
              <div className="mt-1 text-xl font-semibold text-neutral-900 dark:text-white tabular-nums">
                {latest ? `${latest.tokensPerSec} tok/s` : '—'}
              </div>
            </div>
          </section>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left column: Dataset + Tokenizer */}
            <div className="lg:col-span-7 space-y-6">
              <section className="p-5 rounded-[22px] bg-white/95 dark:bg-[#1c1c1e]/95 border border-black/[0.06] dark:border-white/[0.08] shadow-[0_1px_2px_rgba(0,0,0,0.03),0_8px_24px_rgba(0,0,0,0.03)] space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h2 className="text-base font-semibold text-neutral-900 dark:text-white">
                      Training Corpus
                    </h2>
                    <p className="text-xs text-neutral-500 dark:text-neutral-400">
                      Select a starter dataset, upload a text file, or write custom dialogues.
                    </p>
                  </div>
                  <div>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".txt,.md,.jsonl"
                      onChange={handleUploadCorpus}
                      className="hidden"
                    />
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full border border-black/[0.08] dark:border-white/[0.1] text-xs font-medium text-neutral-700 dark:text-neutral-300 hover:bg-black/[0.04] dark:hover:bg-white/[0.06] whitespace-nowrap shrink-0"
                    >
                      <Upload className="w-3.5 h-3.5" />
                      <span>Upload .txt</span>
                    </button>
                  </div>
                </div>

                <div className="flex flex-wrap gap-1.5 p-1 rounded-full bg-black/[0.05] dark:bg-white/[0.08]">
                  {TRAINING_DATASETS.map((ds) => {
                    const active = corpus === ds.corpus;
                    return (
                      <button
                        key={ds.id}
                        type="button"
                        onClick={() => onChangeCorpus(ds.corpus)}
                        className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                          active
                            ? 'bg-white dark:bg-[#2c2c2e] text-neutral-900 dark:text-white shadow-xs'
                            : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white'
                        }`}
                      >
                        {ds.name}
                      </button>
                    );
                  })}
                </div>

                <div>
                  <label htmlFor="training-corpus-editor" className="sr-only">
                    Training corpus text
                  </label>
                  <textarea
                    id="training-corpus-editor"
                    rows={8}
                    value={corpus}
                    onChange={(e) => onChangeCorpus(e.target.value)}
                    className="w-full rounded-2xl border border-black/[0.08] dark:border-white/[0.1] bg-[#f5f5f7] dark:bg-[#121214] p-3.5 font-mono text-xs leading-relaxed text-neutral-800 dark:text-neutral-200 focus:outline-none focus:border-[#007aff]"
                  />
                </div>

                <div className="flex items-center justify-between text-xs text-neutral-500 dark:text-neutral-400 tabular-nums">
                  <span>
                    {corpus.length.toLocaleString()} characters
                    <span aria-hidden="true"> · </span>
                    Vocab: {tokenizer.vocab.length} tokens
                    <span aria-hidden="true"> · </span>
                    {tokenPieces.length} preview tokens
                  </span>
                  <button
                    type="button"
                    onClick={() => setShowTokenizerDetails((s) => !s)}
                    className="text-[#007aff] dark:text-[#0a84ff] hover:underline font-medium"
                  >
                    {showTokenizerDetails ? 'Hide Tokens' : 'Inspect Tokens'}
                  </button>
                </div>

                {showTokenizerDetails && (
                  <div className="p-3.5 rounded-2xl bg-[#f5f5f7] dark:bg-[#121214] max-h-36 overflow-y-auto">
                    <div className="flex flex-wrap gap-1 font-mono text-xs">
                      {tokenPieces.slice(0, 90).map((piece, idx) => (
                        <span
                          key={`${piece.id}-${idx}`}
                          title={`Token ID: ${piece.id}`}
                          className={`px-1.5 py-0.5 rounded-md ${
                            idx % 3 === 0
                              ? 'bg-[#007aff]/10 dark:bg-[#0a84ff]/20 text-[#0055b3] dark:text-[#66b2ff]'
                              : idx % 3 === 1
                                ? 'bg-black/[0.06] dark:bg-white/[0.08] text-neutral-800 dark:text-neutral-200'
                                : 'bg-emerald-500/12 dark:bg-emerald-500/20 text-emerald-800 dark:text-emerald-200'
                          }`}
                        >
                          {piece.text.replace(/\n/g, '↵').replace(/ /g, '·')}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </section>

              {/* Loss Curve */}
              <section className="p-5 rounded-[22px] bg-white/95 dark:bg-[#1c1c1e]/95 border border-black/[0.06] dark:border-white/[0.08] shadow-[0_1px_2px_rgba(0,0,0,0.03),0_8px_24px_rgba(0,0,0,0.03)] space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-base font-semibold text-neutral-900 dark:text-white">
                      Cross-Entropy Loss Curve
                    </h2>
                    <p className="text-xs text-neutral-500 dark:text-neutral-400">
                      Real-time causal next-token prediction loss across AdamW optimizer steps.
                    </p>
                  </div>
                  {metrics.length > 0 && (
                    <span className="text-xs font-mono text-neutral-600 dark:text-neutral-300 tabular-nums">
                      Min {minLossVal.toFixed(3)} · Max {maxLossVal.toFixed(3)}
                    </span>
                  )}
                </div>

                <div className="rounded-2xl bg-[#f5f5f7] dark:bg-[#121214] p-3.5">
                  {metrics.length === 0 ? (
                    <div className="h-44 flex items-center justify-center text-xs text-neutral-500 dark:text-neutral-400">
                      Click "Initialize &amp; Train" to start gradient descent from random Gaussian weights.
                    </div>
                  ) : (
                    <svg
                      viewBox={`0 0 ${svgWidth} ${svgHeight}`}
                      className="w-full h-44"
                      role="img"
                      aria-label="Training loss curve over steps"
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
                      <polyline
                        fill="none"
                        stroke="#007aff"
                        strokeWidth="2.4"
                        strokeLinejoin="round"
                        strokeLinecap="round"
                        points={lossPoints}
                      />
                      <text
                        x={4}
                        y={padY + 4}
                        className="fill-neutral-400 text-[10px] font-mono"
                      >
                        {maxLossVal.toFixed(1)}
                      </text>
                      <text
                        x={4}
                        y={svgHeight - padY + 4}
                        className="fill-neutral-400 text-[10px] font-mono"
                      >
                        {minLossVal.toFixed(1)}
                      </text>
                    </svg>
                  )}
                </div>
              </section>
            </div>

            {/* Right column: Hyperparameters + Controls + Sample Inference */}
            <div className="lg:col-span-5 space-y-6">
              <section className="p-5 rounded-[22px] bg-white/95 dark:bg-[#1c1c1e]/95 border border-black/[0.06] dark:border-white/[0.08] shadow-[0_1px_2px_rgba(0,0,0,0.03),0_8px_24px_rgba(0,0,0,0.03)] space-y-4">
                <div>
                  <h2 className="text-base font-semibold text-neutral-900 dark:text-white">
                    Training Configuration
                  </h2>
                  <p className="text-xs text-neutral-500 dark:text-neutral-400">
                    Random initialization N(0, 0.08) · AdamW · Cosine decay · Grad clip 1.0
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <label className="block text-xs">
                    <span className="text-neutral-600 dark:text-neutral-400 font-medium">
                      Hidden Dim (d_model)
                    </span>
                    <select
                      value={hp.dModel}
                      disabled={isTraining}
                      onChange={(e) => {
                        const d = Number(e.target.value);
                        setHp({ ...hp, dModel: d, dFF: d * 2 });
                      }}
                      className="mt-1 w-full rounded-2xl border border-black/[0.08] dark:border-white/[0.1] bg-[#f5f5f7] dark:bg-[#121214] px-3 py-2 text-xs font-mono text-neutral-900 dark:text-neutral-100"
                    >
                      <option value={24}>24</option>
                      <option value={32}>32</option>
                      <option value={48}>48</option>
                    </select>
                  </label>

                  <label className="block text-xs">
                    <span className="text-neutral-600 dark:text-neutral-400 font-medium">
                      Transformer Layers
                    </span>
                    <select
                      value={hp.nLayers}
                      disabled={isTraining}
                      onChange={(e) => setHp({ ...hp, nLayers: Number(e.target.value) })}
                      className="mt-1 w-full rounded-2xl border border-black/[0.08] dark:border-white/[0.1] bg-[#f5f5f7] dark:bg-[#121214] px-3 py-2 text-xs font-mono text-neutral-900 dark:text-neutral-100"
                    >
                      <option value={1}>1 Layer</option>
                      <option value={2}>2 Layers</option>
                      <option value={3}>3 Layers</option>
                    </select>
                  </label>

                  <label className="block text-xs">
                    <span className="text-neutral-600 dark:text-neutral-400 font-medium">
                      Context Window
                    </span>
                    <select
                      value={hp.contextLength}
                      disabled={isTraining}
                      onChange={(e) => setHp({ ...hp, contextLength: Number(e.target.value) })}
                      className="mt-1 w-full rounded-2xl border border-black/[0.08] dark:border-white/[0.1] bg-[#f5f5f7] dark:bg-[#121214] px-3 py-2 text-xs font-mono text-neutral-900 dark:text-neutral-100"
                    >
                      <option value={24}>24 tokens</option>
                      <option value={32}>32 tokens</option>
                      <option value={48}>48 tokens</option>
                    </select>
                  </label>

                  <label className="block text-xs">
                    <span className="text-neutral-600 dark:text-neutral-400 font-medium">
                      Learning Rate
                    </span>
                    <input
                      type="number"
                      step="0.001"
                      min="0.0005"
                      max="0.05"
                      disabled={isTraining}
                      value={hp.learningRate}
                      onChange={(e) =>
                        setHp({ ...hp, learningRate: Math.max(0.0005, Number(e.target.value)) })
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
                      min="20"
                      max="600"
                      disabled={isTraining}
                      value={hp.totalSteps}
                      onChange={(e) =>
                        setHp({ ...hp, totalSteps: Math.max(10, Number(e.target.value)) })
                      }
                      className="mt-1 w-full rounded-2xl border border-black/[0.08] dark:border-white/[0.1] bg-[#f5f5f7] dark:bg-[#121214] px-3 py-2 text-xs font-mono tabular-nums text-neutral-900 dark:text-neutral-100"
                    />
                  </label>

                  <label className="block text-xs">
                    <span className="text-neutral-600 dark:text-neutral-400 font-medium">
                      Random Seed
                    </span>
                    <input
                      type="number"
                      disabled={isTraining}
                      value={hp.seed}
                      onChange={(e) => setHp({ ...hp, seed: Number(e.target.value) || 1337 })}
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

                  {!isTraining && engineRef.current && currentStep > 0 && currentStep < hp.totalSteps && (
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
                    aria-label="Reset trained weights"
                    className="p-2.5 rounded-full border border-black/[0.08] dark:border-white/[0.1] text-neutral-600 dark:text-neutral-300 hover:bg-black/[0.04] dark:hover:bg-white/[0.06]"
                  >
                    <RotateCcw className="w-4 h-4" />
                  </button>

                  {(checkpoint || currentStep > 0) && (
                    <button
                      type="button"
                      onClick={handleDownloadCheckpoint}
                      aria-label="Export trained checkpoint JSON"
                      className="p-2.5 rounded-full border border-black/[0.08] dark:border-white/[0.1] text-neutral-600 dark:text-neutral-300 hover:bg-black/[0.04] dark:hover:bg-white/[0.06]"
                    >
                      <Download className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </section>

              {/* Quick checkpoint sampler */}
              <section className="p-5 rounded-[22px] bg-white/95 dark:bg-[#1c1c1e]/95 border border-black/[0.06] dark:border-white/[0.08] shadow-[0_1px_2px_rgba(0,0,0,0.03),0_8px_24px_rgba(0,0,0,0.03)] space-y-3">
                <h2 className="text-base font-semibold text-neutral-900 dark:text-white">
                  Checkpoint Sampler
                </h2>
                <p className="text-xs text-neutral-500 dark:text-neutral-400">
                  Test raw next-token completion directly from your trained weights.
                </p>

                <label htmlFor="sample-prompt-input" className="sr-only">
                  Sample prompt
                </label>
                <input
                  id="sample-prompt-input"
                  type="text"
                  value={samplePrompt}
                  onChange={(e) => setSamplePrompt(e.target.value)}
                  className="w-full rounded-2xl border border-black/[0.08] dark:border-white/[0.1] bg-[#f5f5f7] dark:bg-[#121214] px-3.5 py-2 font-mono text-xs text-neutral-900 dark:text-neutral-100 focus:outline-none focus:border-[#007aff]"
                />

                <button
                  type="button"
                  disabled={!checkpoint && !engineRef.current}
                  onClick={handleGenerateSample}
                  className="w-full px-4 py-2.5 rounded-full border border-black/[0.08] dark:border-white/[0.1] bg-[#f5f5f7] dark:bg-[#2c2c2e] hover:bg-black/[0.05] dark:hover:bg-[#3a3a3c] disabled:opacity-50 text-xs font-medium text-neutral-900 dark:text-white transition-colors"
                >
                  Generate Completion
                </button>

                {sampleOutput && (
                  <div className="p-3.5 rounded-2xl bg-[#f5f5f7] dark:bg-[#121214] space-y-2">
                    <div className="font-mono text-xs text-neutral-800 dark:text-neutral-200 whitespace-pre-wrap">
                      {sampleOutput}
                    </div>
                    {sampleTopTokens.length > 0 && (
                      <div className="pt-2 border-t border-black/[0.06] dark:border-white/[0.08] space-y-1">
                        <div className="text-[11px] text-neutral-500 dark:text-neutral-400">
                          Top-5 Next-Token Distribution (Final Step)
                        </div>
                        {sampleTopTokens.map((t) => (
                          <div key={`${t.id}-${t.token}`} className="flex items-center gap-2 text-xs font-mono tabular-nums">
                            <span className="w-20 truncate text-neutral-700 dark:text-neutral-300">
                              "{t.token.replace(/\n/g, '\\n')}"
                            </span>
                            <div className="flex-1 h-1.5 rounded-full bg-black/[0.08] dark:bg-white/[0.1] overflow-hidden">
                              <div
                                className="h-full rounded-full bg-[#007aff] dark:bg-[#0a84ff]"
                                style={{ width: `${Math.round(t.probability * 100)}%` }}
                              />
                            </div>
                            <span className="w-12 text-right text-neutral-500 dark:text-neutral-400">
                              {(t.probability * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </section>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
