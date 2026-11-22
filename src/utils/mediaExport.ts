import type { GeneratedMediaFrames } from '../engine/multimodal';
import {
  createAudioBufferFromPcm,
  type NeuralAudioFrame,
  renderNeuralAudioPcm,
} from './neuralAudioSynth';

export function rgbaFrameToDataUrl(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  scale = 4,
): string {
  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = width;
  srcCanvas.height = height;
  const srcCtx = srcCanvas.getContext('2d');
  if (!srcCtx) return '';

  const imgData = srcCtx.createImageData(width, height);
  imgData.data.set(rgba);
  srcCtx.putImageData(imgData, 0, 0);

  if (scale <= 1) {
    return srcCanvas.toDataURL('image/png');
  }

  const outCanvas = document.createElement('canvas');
  outCanvas.width = width * scale;
  outCanvas.height = height * scale;
  const outCtx = outCanvas.getContext('2d');
  if (!outCtx) return srcCanvas.toDataURL('image/png');

  outCtx.imageSmoothingEnabled = true;
  outCtx.imageSmoothingQuality = 'high';
  outCtx.drawImage(srcCanvas, 0, 0, outCanvas.width, outCanvas.height);
  return outCanvas.toDataURL('image/png');
}

export function mediaToFrameDataUrls(media: GeneratedMediaFrames, scale = 4): string[] {
  return media.frames.map((frame) => rgbaFrameToDataUrl(frame, media.width, media.height, scale));
}

export function triggerBlobDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function triggerDataUrlDownload(dataUrl: string, filename: string): void {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

export async function recordRgbaFramesToWebm(
  frames: Uint8ClampedArray[],
  srcWidth: number,
  srcHeight: number,
  outWidth: number,
  fps: number,
  outHeight?: number,
  audioTrack?: NeuralAudioFrame[],
): Promise<Blob> {
  if (frames.length === 0) {
    throw new Error('No video frames available to export.');
  }
  if (typeof MediaRecorder === 'undefined') {
    throw new Error('Video recording is not supported in this browser.');
  }

  const resolvedOutHeight =
    outHeight ?? Math.max(16, Math.round((outWidth * srcHeight) / Math.max(1, srcWidth)));

  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = srcWidth;
  srcCanvas.height = srcHeight;
  const srcCtx = srcCanvas.getContext('2d');

  const outCanvas = document.createElement('canvas');
  outCanvas.width = outWidth;
  outCanvas.height = resolvedOutHeight;
  const outCtx = outCanvas.getContext('2d');

  if (!srcCtx || !outCtx || typeof outCanvas.captureStream !== 'function') {
    throw new Error('Canvas stream recording is unavailable in this browser.');
  }
  outCtx.imageSmoothingEnabled = true;
  outCtx.imageSmoothingQuality = 'high';

  const canvasStream = outCanvas.captureStream(fps);
  const tracks: MediaStreamTrack[] = [...canvasStream.getVideoTracks()];

  let audioCtx: AudioContext | null = null;
  let audioSource: AudioBufferSourceNode | null = null;
  const hasAudio = Array.isArray(audioTrack) && audioTrack.length > 0;
  const cycleSec = frames.length / Math.max(1, fps);
  const cycles = cycleSec < 1.5 ? 2 : 1;
  const totalDurationMs = Math.max(1200, cycleSec * cycles * 1000);

  if (hasAudio && typeof AudioContext !== 'undefined') {
    try {
      audioCtx = new AudioContext();
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume().catch(() => {});
      }
      const pcm = renderNeuralAudioPcm(audioTrack, cycleSec, audioCtx.sampleRate);
      const buffer = createAudioBufferFromPcm(audioCtx, pcm);
      const dest = audioCtx.createMediaStreamDestination();
      const src = audioCtx.createBufferSource();
      src.buffer = buffer;
      src.loop = true;
      src.connect(dest);
      audioSource = src;
      tracks.push(...dest.stream.getAudioTracks());
    } catch {
      if (audioCtx) {
        await audioCtx.close().catch(() => {});
        audioCtx = null;
      }
      audioSource = null;
    }
  }

  const combinedStream = new MediaStream(tracks);
  const mimeCandidates = audioSource
    ? ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm']
    : ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  const mimeType = mimeCandidates.find((m) => MediaRecorder.isTypeSupported(m)) ?? '';

  const recorder = mimeType
    ? new MediaRecorder(combinedStream, { mimeType, videoBitsPerSecond: 3_500_000 })
    : new MediaRecorder(combinedStream);

  const chunks: BlobPart[] = [];
  recorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) chunks.push(e.data);
  };

  const stopped = new Promise<Blob>((resolve, reject) => {
    recorder.onstop = () => {
      resolve(new Blob(chunks, { type: mimeType || 'video/webm' }));
    };
    recorder.onerror = () => reject(new Error('MediaRecorder failed while encoding video.'));
  });

  const imgData = srcCtx.createImageData(srcWidth, srcHeight);
  const drawFrame = (rgba: Uint8ClampedArray) => {
    imgData.data.set(rgba);
    srcCtx.putImageData(imgData, 0, 0);
    outCtx.drawImage(srcCanvas, 0, 0, outWidth, resolvedOutHeight);
  };

  try {
    drawFrame(frames[0]);
    recorder.start();
    if (audioSource) {
      audioSource.start(0);
    }

    await new Promise<void>((resolve) => {
      const start = performance.now();
      const step = (now: number) => {
        const elapsed = now - start;
        if (elapsed >= totalDurationMs) {
          resolve();
          return;
        }
        const idx = Math.floor((elapsed / 1000) * fps) % frames.length;
        drawFrame(frames[idx]);
        requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });

    if (audioSource) {
      try {
        audioSource.stop();
      } catch {
        // Ignore already stopped
      }
    }
    recorder.stop();
    return await stopped;
  } finally {
    for (const track of combinedStream.getTracks()) {
      track.stop();
    }
    if (audioCtx) {
      await audioCtx.close().catch(() => {});
    }
  }
}

export async function svgToPngDataUrl(
  svgMarkup: string,
  width = 1024,
  height = 1024,
): Promise<string> {
  const svgDataUrl = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgMarkup)}`;
  return new Promise<string>((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        resolve(svgDataUrl);
        return;
      }
      ctx.drawImage(img, 0, 0, width, height);
      try {
        resolve(canvas.toDataURL('image/png'));
      } catch {
        resolve(svgDataUrl);
      }
    };
    img.onerror = () => resolve(svgDataUrl);
    img.src = svgDataUrl;
  });
}
