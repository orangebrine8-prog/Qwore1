import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Download, Pause, Play, Volume2, VolumeX } from 'lucide-react';
import type { LtxCinemaPayload } from '../engine/ltx25DiT';
import {
  exportLtxCinemaToWebm,
  LtxWebglRenderer,
  renderLtxStillToDataUrl,
} from '../engine/ltxCinemaWebgl';
import type { CameraMotionMode } from '../engine/multimodal';
import { isWebgl2Available } from '../engine/workerClient';
import {
  createAudioBufferFromPcm,
  type NeuralAudioFrame,
  renderNeuralAudioPcm,
} from '../utils/neuralAudioSynth';
import {
  recordRgbaFramesToWebm,
  rgbaFrameToDataUrl,
  triggerBlobDownload,
  triggerDataUrlDownload,
} from '../utils/mediaExport';

interface CanvasMediaPlayerProps {
  frames: Uint8ClampedArray[];
  width: number;
  height: number;
  fps: number;
  label: string;
  audioTrack?: NeuralAudioFrame[];
  cameraMotion?: CameraMotionMode;
  cinemaPayload?: LtxCinemaPayload;
  showScrubber?: boolean;
  displaySizeClass?: string;
}

export const CanvasMediaPlayer: React.FC<CanvasMediaPlayerProps> = React.memo(({
  frames,
  width,
  height,
  fps,
  label,
  audioTrack,
  cameraMotion,
  cinemaPayload,
  showScrubber = false,
  displaySizeClass,
}) => {
  const safeFrames = useMemo(
    () =>
      Array.isArray(frames)
        ? frames.filter(
            (f): f is Uint8ClampedArray => Boolean(f && typeof f.length === 'number' && f.length > 0),
          )
        : [],
    [frames],
  );

  const [webglError, setWebglError] = useState<string | null>(null);
  const useWebgl = Boolean(cinemaPayload && !webglError && isWebgl2Available());

  const isWide = width > height;
  const resolvedCanvasW = useWebgl ? (isWide ? 960 : 640) : isWide ? 512 : 288;
  const resolvedCanvasH = useWebgl ? (isWide ? 540 : 640) : isWide ? 288 : 288;
  const resolvedClass =
    displaySizeClass ??
    (isWide
      ? 'w-full max-w-[640px] aspect-video'
      : 'w-full max-w-[320px] sm:max-w-[360px] aspect-square');

  const isVideo = useWebgl ? fps > 1 : safeFrames.length > 1;
  const totalFrames = useWebgl
    ? isVideo
      ? Math.max(2, Math.round((cinemaPayload?.durationSec || 4) * Math.max(1, fps)))
      : 1
    : safeFrames.length;

  const hasAudio = Array.isArray(audioTrack) && audioTrack.length > 0 && isVideo;
  const [isPlaying, setIsPlaying] = useState(true);
  const [isMuted, setIsMuted] = useState(true);
  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [isInViewport, setIsInViewport] = useState(true);
  const [isTabVisible, setIsTabVisible] = useState(
    () => typeof document === 'undefined' || !document.hidden,
  );

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const webglRendererRef = useRef<LtxWebglRenderer | null>(null);
  const scratchCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const sCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const dCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const imgDataRef = useRef<ImageData | null>(null);
  const scrubberRef = useRef<HTMLInputElement | null>(null);
  const frameLabelRef = useRef<HTMLSpanElement | null>(null);
  const currentIdxRef = useRef(0);

  const audioCtxRef = useRef<AudioContext | null>(null);
  const audioBufferRef = useRef<AudioBuffer | null>(null);
  const audioSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const audioGainRef = useRef<GainNode | null>(null);

  const stopAudioPlayback = useCallback(() => {
    if (audioSourceRef.current) {
      try {
        audioSourceRef.current.stop();
        audioSourceRef.current.disconnect();
      } catch {
        // Ignore already stopped
      }
      audioSourceRef.current = null;
    }
  }, []);

  // Pre-render PCM buffer cleanly when audioTrack updates
  useEffect(() => {
    audioBufferRef.current = null;
    stopAudioPlayback();
  }, [audioTrack, totalFrames, fps, stopAudioPlayback]);

  useEffect(() => {
    return () => {
      stopAudioPlayback();
      if (audioCtxRef.current) {
        void audioCtxRef.current.close().catch(() => {});
        audioCtxRef.current = null;
      }
    };
  }, [stopAudioPlayback]);

  // Initialize / dispose WebGL2 3D PBR raymarching renderer when cinemaPayload is active
  useEffect(() => {
    if (!useWebgl) {
      webglRendererRef.current?.dispose();
      webglRendererRef.current = null;
      return;
    }
    const canvas = canvasRef.current;
    if (!canvas) return;
    try {
      const renderer = new LtxWebglRenderer(canvas);
      webglRendererRef.current = renderer;
      const tNorm = totalFrames > 1 ? (currentIdxRef.current % totalFrames) / totalFrames : 0.35;
      if (cinemaPayload) {
        renderer.render(cinemaPayload, tNorm, canvas.width, canvas.height);
      }
      return () => {
        renderer.dispose();
        if (webglRendererRef.current === renderer) {
          webglRendererRef.current = null;
        }
      };
    } catch (err) {
      setWebglError(err instanceof Error ? err.message : 'WebGL2 shader initialization failed');
    }
  }, [useWebgl, cinemaPayload, resolvedCanvasW, resolvedCanvasH, totalFrames]);

  const startSyncedAudioAt = useCallback(
    async (frameIndex: number) => {
      if (!hasAudio || !audioTrack || typeof AudioContext === 'undefined') return;
      stopAudioPlayback();
      if (!audioCtxRef.current) {
        audioCtxRef.current = new AudioContext();
      }
      const ctx = audioCtxRef.current;
      if (ctx.state === 'suspended') {
        await ctx.resume().catch(() => {});
      }
      if (!audioBufferRef.current) {
        const cycleSec = useWebgl
          ? Math.max(1, cinemaPayload?.durationSec || 4)
          : totalFrames / Math.max(1, fps);
        const pcm = renderNeuralAudioPcm(audioTrack, cycleSec, ctx.sampleRate);
        audioBufferRef.current = createAudioBufferFromPcm(ctx, pcm);
      }
      if (!audioGainRef.current) {
        audioGainRef.current = ctx.createGain();
        audioGainRef.current.gain.value = 0.72;
        audioGainRef.current.connect(ctx.destination);
      }
      const src = ctx.createBufferSource();
      src.buffer = audioBufferRef.current;
      src.loop = true;
      src.connect(audioGainRef.current);
      const offsetSec = (frameIndex / Math.max(1, fps)) % audioBufferRef.current.duration;
      src.start(0, offsetSec);
      audioSourceRef.current = src;
    },
    [hasAudio, audioTrack, useWebgl, cinemaPayload?.durationSec, totalFrames, fps, stopAudioPlayback],
  );

  // Sync Web Audio start/stop with playback, mute, viewport, and tab visibility
  useEffect(() => {
    if (!hasAudio || isMuted || !isPlaying || !isInViewport || !isTabVisible) {
      stopAudioPlayback();
      return;
    }
    void startSyncedAudioAt(currentIdxRef.current);
    return () => {
      stopAudioPlayback();
    };
  }, [hasAudio, isMuted, isPlaying, isInViewport, isTabVisible, startSyncedAudioAt, stopAudioPlayback]);

  const drawFrameAt = useCallback(
    (idx: number) => {
      const canvas = canvasRef.current;
      if (!canvas || totalFrames === 0) return;
      const safeIdx = Math.max(0, Math.min(totalFrames - 1, idx));
      currentIdxRef.current = safeIdx;

      if (useWebgl && cinemaPayload && webglRendererRef.current) {
        const tNorm = totalFrames > 1 ? safeIdx / totalFrames : 0.35;
        webglRendererRef.current.render(cinemaPayload, tNorm, canvas.width, canvas.height);
      } else {
        if (safeFrames.length === 0) return;
        const frameBuf = safeFrames[safeIdx];
        if (!frameBuf) return;

        if (!scratchCanvasRef.current) {
          scratchCanvasRef.current = document.createElement('canvas');
        }
        const scratch = scratchCanvasRef.current;
        if (scratch.width !== width || scratch.height !== height) {
          scratch.width = width;
          scratch.height = height;
          sCtxRef.current = scratch.getContext('2d');
          imgDataRef.current = null;
        }

        if (!sCtxRef.current) {
          sCtxRef.current = scratch.getContext('2d');
        }
        if (!dCtxRef.current) {
          dCtxRef.current = canvas.getContext('2d');
        }
        const sCtx = sCtxRef.current;
        const dCtx = dCtxRef.current;
        if (!sCtx || !dCtx) return;

        if (
          !imgDataRef.current ||
          imgDataRef.current.width !== width ||
          imgDataRef.current.height !== height
        ) {
          imgDataRef.current = sCtx.createImageData(width, height);
        }
        const imgData = imgDataRef.current;
        imgData.data.set(frameBuf);
        sCtx.putImageData(imgData, 0, 0);

        dCtx.imageSmoothingEnabled = true;
        dCtx.imageSmoothingQuality = 'high';
        dCtx.drawImage(scratch, 0, 0, canvas.width, canvas.height);
      }

      if (frameLabelRef.current) {
        frameLabelRef.current.textContent = `Frame ${safeIdx + 1}/${totalFrames} · ${fps} fps`;
      }
      if (scrubberRef.current && document.activeElement !== scrubberRef.current) {
        scrubberRef.current.value = String(safeIdx);
      }
    },
    [useWebgl, cinemaPayload, totalFrames, safeFrames, width, height, fps],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || typeof IntersectionObserver === 'undefined') return;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.some((entry) => entry.isIntersecting);
        setIsInViewport(visible);
      },
      { threshold: 0.05 },
    );
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [useWebgl]);

  useEffect(() => {
    const onVisibility = () => setIsTabVisible(!document.hidden);
    document.addEventListener('visibilitychange', onVisibility);
    return () => document.removeEventListener('visibilitychange', onVisibility);
  }, []);

  useEffect(() => {
    if (totalFrames === 0) return;
    drawFrameAt(currentIdxRef.current % totalFrames);
    if (!isVideo || !isPlaying || !isInViewport || !isTabVisible) return;

    let rafId = 0;
    const start = performance.now() - (currentIdxRef.current / Math.max(1, fps)) * 1000;
    const tick = (now: number) => {
      const elapsedSec = (now - start) / 1000;
      const nextIdx = Math.floor(elapsedSec * fps) % totalFrames;
      if (nextIdx !== currentIdxRef.current) {
        drawFrameAt(nextIdx);
      }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [totalFrames, fps, isVideo, isPlaying, isInViewport, isTabVisible, drawFrameAt]);

  const handleDownloadPng = () => {
    if (useWebgl && cinemaPayload) {
      const tNorm = totalFrames > 1 ? currentIdxRef.current / totalFrames : 0.35;
      const url =
        renderLtxStillToDataUrl(cinemaPayload, tNorm, resolvedCanvasW, resolvedCanvasH) ??
        canvasRef.current?.toDataURL('image/png');
      if (url) {
        triggerDataUrlDownload(url, 'qowe-ltx25-frame.png');
      }
      return;
    }
    if (safeFrames.length === 0) return;
    const frameBuf = safeFrames[currentIdxRef.current];
    if (!frameBuf) return;
    const url = rgbaFrameToDataUrl(frameBuf, width, height, 4);
    triggerDataUrlDownload(url, 'qowe-1.0-frame.png');
  };

  const handleDownloadWebm = async () => {
    if (!isVideo || isExporting) return;
    if (!useWebgl && safeFrames.length === 0) return;
    setExportError(null);
    setIsExporting(true);
    try {
      if (useWebgl && cinemaPayload) {
        const blob = await exportLtxCinemaToWebm(
          cinemaPayload,
          audioTrack,
          resolvedCanvasW,
          resolvedCanvasH,
        );
        triggerBlobDownload(blob, 'qowe-ltx25-cinema.webm');
      } else {
        const outW = isWide ? 512 : 384;
        const outH = isWide ? 288 : 384;
        const blob = await recordRgbaFramesToWebm(
          safeFrames,
          width,
          height,
          outW,
          fps,
          outH,
          audioTrack,
        );
        triggerBlobDownload(blob, 'qowe-1.0-60fps-audio-video.webm');
      }
    } catch (err) {
      setExportError(err instanceof Error ? err.message : 'Unable to encode WebM video.');
    } finally {
      setIsExporting(false);
    }
  };

  const handleToggleAudio = () => {
    if (!hasAudio) return;
    setIsMuted((prev) => !prev);
  };

  if (!useWebgl && safeFrames.length === 0) {
    return (
      <div className="p-3.5 rounded-2xl border border-black/[0.08] dark:border-white/[0.1] bg-[#f5f5f7] dark:bg-[#121214] text-xs text-neutral-500 dark:text-neutral-400">
        {webglError
          ? `Renderer notice: ${webglError}. Re-run the prompt to synthesize frames.`
          : 'Video buffer unavailable. Re-run the prompt to synthesize fresh 60 fps frames.'}
      </div>
    );
  }

  return (
    <div className="w-full space-y-2.5 flex flex-col items-start">
      <div className="rounded-2xl overflow-hidden border border-black/[0.08] dark:border-white/[0.1] bg-neutral-950 max-w-full">
        <canvas
          key={useWebgl ? 'webgl2' : 'canvas2d'}
          ref={canvasRef}
          width={resolvedCanvasW}
          height={resolvedCanvasH}
          aria-label={label}
          className={`${resolvedClass} block`}
        />
      </div>

      {(isVideo || useWebgl) && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-neutral-500 dark:text-neutral-400">
          {useWebgl && cinemaPayload && (
            <span className="font-mono">
              LTX-2.5 Dual-Stream · {resolvedCanvasW}×{resolvedCanvasH} · Flow{' '}
              {(cinemaPayload.flowConsistencyScore * 100).toFixed(1)}%
            </span>
          )}
          {isVideo && cameraMotion && (
            <span className="font-mono">Camera: {cameraMotion}</span>
          )}
          {hasAudio && (
            <span className="font-mono">
              {isMuted ? 'Synced Neural Audio (muted)' : 'Synced Neural Audio playing'}
            </span>
          )}
        </div>
      )}

      {isVideo && showScrubber && (
        <div className="w-full max-w-md flex items-center gap-2 text-xs">
          <input
            ref={scrubberRef}
            type="range"
            min={0}
            max={Math.max(0, totalFrames - 1)}
            defaultValue={0}
            aria-label="Scrub video frames"
            onChange={(e) => {
              setIsPlaying(false);
              drawFrameAt(Number(e.target.value));
            }}
            className="flex-1 accent-[#007aff]"
          />
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2 text-xs">
        {isVideo && (
          <>
            <button
              type="button"
              onClick={() => setIsPlaying((p) => !p)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-black/[0.08] dark:border-white/[0.1] text-neutral-700 dark:text-neutral-200 hover:bg-black/[0.04] dark:hover:bg-white/[0.08] whitespace-nowrap"
            >
              {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
              <span>{isPlaying ? 'Pause' : 'Play'}</span>
            </button>

            {hasAudio && (
              <button
                type="button"
                onClick={handleToggleAudio}
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border text-xs transition-colors whitespace-nowrap ${
                  !isMuted
                    ? 'border-[#007aff] text-[#007aff] bg-[#007aff]/10 dark:border-[#0a84ff] dark:text-[#0a84ff] dark:bg-[#0a84ff]/15 font-medium'
                    : 'border-black/[0.08] dark:border-white/[0.1] text-neutral-700 dark:text-neutral-200 hover:bg-black/[0.04] dark:hover:bg-white/[0.08]'
                }`}
              >
                {isMuted ? <VolumeX className="w-3.5 h-3.5" /> : <Volume2 className="w-3.5 h-3.5" />}
                <span>{isMuted ? 'Unmute Audio' : 'Sound On'}</span>
              </button>
            )}

            <span
              ref={frameLabelRef}
              className="font-mono text-neutral-500 dark:text-neutral-400 tabular-nums"
            >
              Frame 1/{totalFrames} · {fps} fps
            </span>
          </>
        )}

        <button
          type="button"
          onClick={handleDownloadPng}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-black/[0.08] dark:border-white/[0.1] text-neutral-700 dark:text-neutral-200 hover:bg-black/[0.04] dark:hover:bg-white/[0.08] whitespace-nowrap"
        >
          <Download className="w-3.5 h-3.5" />
          <span>Download PNG</span>
        </button>

        {isVideo && (
          <button
            type="button"
            disabled={isExporting}
            onClick={handleDownloadWebm}
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-[#007aff] hover:bg-[#0066d6] dark:bg-[#0a84ff] dark:hover:bg-[#1a8cff] disabled:opacity-60 text-white font-medium whitespace-nowrap"
          >
            <Download className="w-3.5 h-3.5" />
            <span>
              {isExporting
                ? 'Encoding...'
                : hasAudio
                  ? `Download .webm + Audio (${fps} fps)`
                  : `Download .webm (${fps} fps)`}
            </span>
          </button>
        )}
      </div>

      {exportError && <p className="text-xs text-red-600 dark:text-red-400">{exportError}</p>}
    </div>
  );
});
