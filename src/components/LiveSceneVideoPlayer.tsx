import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Download, Pause, Play, RotateCcw } from 'lucide-react';
import type { VideoSceneSpec } from '../engine/intentAndMedia';
import { recordVideoSceneToWebm, renderVideoSceneFrame } from '../engine/sceneVideoRenderer';
import { triggerBlobDownload, triggerDataUrlDownload } from '../utils/mediaExport';

interface LiveSceneVideoPlayerProps {
  spec: VideoSceneSpec;
  showScrubber?: boolean;
}

export const LiveSceneVideoPlayer: React.FC<LiveSceneVideoPlayerProps> = React.memo(({
  spec,
  showScrubber = true,
}) => {
  const fps = 60;
  const width = spec.aspectRatio === '1:1' ? 512 : 640;
  const height = spec.aspectRatio === '1:1' ? 512 : 360;
  const totalFrames = Math.max(60, Math.round(spec.durationSec * fps));

  const [isPlaying, setIsPlaying] = useState(true);
  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [isInViewport, setIsInViewport] = useState(true);
  const [isTabVisible, setIsTabVisible] = useState(
    () => typeof document === 'undefined' || !document.hidden,
  );

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const scrubberRef = useRef<HTMLInputElement | null>(null);
  const frameLabelRef = useRef<HTMLSpanElement | null>(null);
  const frameIndexRef = useRef(0);

  const drawAtFrame = useCallback(
    (frameIdx: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      if (!ctxRef.current) {
        ctxRef.current = canvas.getContext('2d');
      }
      const ctx = ctxRef.current;
      if (!ctx) return;

      const safeIdx = ((frameIdx % totalFrames) + totalFrames) % totalFrames;
      frameIndexRef.current = safeIdx;
      renderVideoSceneFrame(ctx, spec, width, height, safeIdx / fps);

      if (frameLabelRef.current) {
        frameLabelRef.current.textContent = `Frame ${safeIdx + 1}/${totalFrames} · ${fps} fps`;
      }
      if (scrubberRef.current && document.activeElement !== scrubberRef.current) {
        scrubberRef.current.value = String(safeIdx);
      }
    },
    [spec, width, height, totalFrames],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || typeof IntersectionObserver === 'undefined') return;
    const observer = new IntersectionObserver(
      (entries) => {
        setIsInViewport(entries.some((entry) => entry.isIntersecting));
      },
      { threshold: 0.05 },
    );
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const onVisibility = () => setIsTabVisible(!document.hidden);
    document.addEventListener('visibilitychange', onVisibility);
    return () => document.removeEventListener('visibilitychange', onVisibility);
  }, []);

  useEffect(() => {
    drawAtFrame(frameIndexRef.current);
    if (!isPlaying || !isInViewport || !isTabVisible) return;

    let rafId = 0;
    const startMs = performance.now() - (frameIndexRef.current / fps) * 1000;

    const tick = (now: number) => {
      const elapsedSec = (now - startMs) / 1000;
      const nextFrame = Math.floor(elapsedSec * fps) % totalFrames;
      if (nextFrame !== frameIndexRef.current) {
        drawAtFrame(nextFrame);
      }
      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [isPlaying, isInViewport, isTabVisible, totalFrames, drawAtFrame]);

  const handleDownloadPng = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    triggerDataUrlDownload(canvas.toDataURL('image/png'), 'qowe-1.0-video-frame.png');
  };

  const handleDownloadWebm = async () => {
    if (isExporting) return;
    setExportError(null);
    setIsExporting(true);
    try {
      const blob = await recordVideoSceneToWebm(spec, width, height, fps);
      triggerBlobDownload(blob, 'qowe-1.0-60fps-video.webm');
    } catch (err) {
      setExportError(err instanceof Error ? err.message : 'Unable to encode WebM video.');
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="space-y-2.5 flex flex-col items-start w-full">
      <div className="w-full max-w-lg rounded-2xl overflow-hidden border border-black/[0.08] dark:border-white/[0.1] bg-neutral-950">
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          aria-label={spec.title || 'Generated 60 fps video'}
          className="w-full h-auto block"
        />
      </div>

      {showScrubber && (
        <div className="w-full max-w-lg flex items-center gap-2 text-xs">
          <input
            ref={scrubberRef}
            type="range"
            min={0}
            max={totalFrames - 1}
            defaultValue={0}
            aria-label="Scrub 60 fps video timeline"
            onChange={(e) => {
              setIsPlaying(false);
              drawAtFrame(Number(e.target.value));
            }}
            className="flex-1 accent-[#007aff]"
          />
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2 text-xs">
        <button
          type="button"
          onClick={() => setIsPlaying((p) => !p)}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-black/[0.08] dark:border-white/[0.1] text-neutral-700 dark:text-neutral-200 hover:bg-black/[0.04] dark:hover:bg-white/[0.08] whitespace-nowrap"
        >
          {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
          <span>{isPlaying ? 'Pause' : 'Play'}</span>
        </button>

        <button
          type="button"
          onClick={() => {
            drawAtFrame(0);
            setIsPlaying(true);
          }}
          aria-label="Restart video"
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-black/[0.08] dark:border-white/[0.1] text-neutral-700 dark:text-neutral-200 hover:bg-black/[0.04] dark:hover:bg-white/[0.08] whitespace-nowrap"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span>Replay</span>
        </button>

        <span
          ref={frameLabelRef}
          className="font-mono text-neutral-500 dark:text-neutral-400 tabular-nums"
        >
          Frame 1/{totalFrames} · {fps} fps
        </span>

        <button
          type="button"
          onClick={handleDownloadPng}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-black/[0.08] dark:border-white/[0.1] text-neutral-700 dark:text-neutral-200 hover:bg-black/[0.04] dark:hover:bg-white/[0.08] whitespace-nowrap"
        >
          <Download className="w-3.5 h-3.5" />
          <span>Snapshot PNG</span>
        </button>

        <button
          type="button"
          disabled={isExporting}
          onClick={handleDownloadWebm}
          className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-[#007aff] hover:bg-[#0066d6] dark:bg-[#0a84ff] dark:hover:bg-[#1a8cff] disabled:opacity-60 text-white font-medium whitespace-nowrap"
        >
          <Download className="w-3.5 h-3.5" />
          <span>{isExporting ? 'Recording 60 fps...' : 'Download .webm (60 fps)'}</span>
        </button>
      </div>

      {exportError && <p className="text-xs text-red-600 dark:text-red-400">{exportError}</p>}
    </div>
  );
});
