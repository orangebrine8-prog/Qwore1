import type { WorkerCommand, WorkerEvent } from './qowe.worker';
import {
  CameraMotionMode,
  MultimodalCheckpoint,
  MultimodalHyperparams,
  MultimodalStepMetric,
  QoweVisionVideoEngine,
  resolveCameraMotion,
} from './multimodal';
import type { NeuralAudioFrame } from '../utils/neuralAudioSynth';

let sharedWorker: Worker | null = null;
const listeners = new Set<(event: WorkerEvent) => void>();

// Multi-core browser worker pool for distributed video/image training & 60 fps rendering
const poolWorkers: Worker[] = [];

export function getBrowserWorkerCount(): number {
  if (typeof navigator === 'undefined') return 2;
  return Math.min(8, Math.max(2, navigator.hardwareConcurrency || 4));
}

function getWorker(): Worker {
  if (!sharedWorker) {
    sharedWorker = new Worker(new URL('./qowe.worker.ts', import.meta.url), { type: 'module' });
    sharedWorker.onmessage = (ev: MessageEvent<WorkerEvent>) => {
      for (const listener of listeners) {
        listener(ev.data);
      }
    };
  }
  return sharedWorker;
}

function getWorkerPool(): Worker[] {
  const targetCount = getBrowserWorkerCount();
  while (poolWorkers.length < targetCount) {
    const w = new Worker(new URL('./qowe.worker.ts', import.meta.url), { type: 'module' });
    poolWorkers.push(w);
  }
  return poolWorkers;
}

export function postWorkerCommand(cmd: WorkerCommand): void {
  getWorker().postMessage(cmd);
}

export function subscribeWorkerEvents(handler: (ev: WorkerEvent) => void): () => void {
  getWorker();
  listeners.add(handler);
  return () => {
    listeners.delete(handler);
  };
}

function dispatchToPoolWorker<T extends WorkerEvent>(
  worker: Worker,
  cmd: WorkerCommand,
  expectedType: T['type'],
): Promise<T> {
  return new Promise((resolve, reject) => {
    const onMessage = (ev: MessageEvent<WorkerEvent>) => {
      const data = ev.data;
      if ('jobId' in cmd && data.jobId !== cmd.jobId) return;
      if (data.type === expectedType) {
        cleanup();
        resolve(data as T);
      } else if (data.type === 'error') {
        cleanup();
        reject(new Error(data.message));
      }
    };
    const onError = (err: ErrorEvent) => {
      cleanup();
      reject(new Error(err.message || 'Worker execution failed'));
    };
    const cleanup = () => {
      worker.removeEventListener('message', onMessage);
      worker.removeEventListener('error', onError);
    };
    worker.addEventListener('message', onMessage);
    worker.addEventListener('error', onError);
    worker.postMessage(cmd);
  });
}

export interface MultimodalPoolTrainOptions {
  hyperparams: MultimodalHyperparams;
  previewPrompt: string;
  resumeFrom?: MultimodalCheckpoint | null;
  onProgress: (payload: {
    history: MultimodalStepMetric[];
    previewFrame?: Uint8ClampedArray;
    previewWidth?: number;
    previewHeight?: number;
  }) => void;
  onDone: (checkpoint: MultimodalCheckpoint, paused: boolean) => void;
  onError: (message: string) => void;
}

export function trainMultimodalPool(options: MultimodalPoolTrainOptions): { cancel: () => void } {
  let cancelled = false;
  let stopped = false;

  const run = async () => {
    try {
      const pool = getWorkerPool();
      const numWorkers = pool.length;
      const engine = options.resumeFrom
        ? QoweVisionVideoEngine.fromCheckpoint(options.resumeFrom)
        : new QoweVisionVideoEngine(undefined, options.hyperparams);

      // Distribute sample batch across all browser CPU workers
      const samplesPerWorker = Math.max(8, Math.ceil(64 / numWorkers));
      let lastPost = performance.now();
      let lastYield = lastPost;
      let lastPreviewStep = engine.step;
      let previewInFlight = false;

      while (engine.step < options.hyperparams.totalSteps && !cancelled) {
        const currentWeights = engine.getWeightArrays();
        const stepBaseSeed = (options.hyperparams.seed * 31 + engine.step * 997) >>> 0;

        // Parallel data-parallel forward + backward across ALL browser worker threads
        const workerPackets = await Promise.all(
          pool.map((w, idx) =>
            dispatchToPoolWorker<Extract<WorkerEvent, { type: 'mm-grads-done' }>>(
              w,
              {
                type: 'compute-mm-grads',
                jobId: `mm-grad-${engine.step}-${idx}-${Math.random().toString(36).slice(2, 6)}`,
                hyperparams: engine.hyperparams,
                tokenizer: engine.tokenizer,
                weights: currentWeights,
                samplesPerConcept: samplesPerWorker,
                stepSeed: (stepBaseSeed + idx * 7919) >>> 0,
              },
              'mm-grads-done',
            ),
          ),
        );

        if (cancelled) break;

        // AllReduce gradient average + joint Audio-Visual MSE reduction
        const averagedGrads = workerPackets[0].grads.map((g) => new Float32Array(g.length));
        let totalSqErr = 0;
        let totalSamples = 0;
        let totalVisualSq = 0;
        let totalVisualSamples = 0;
        let totalAudioSq = 0;
        let totalAudioSamples = 0;
        const invW = 1 / workerPackets.length;

        for (const pkt of workerPackets) {
          totalSqErr += pkt.sqErrSum;
          totalSamples += pkt.sampleCount;
          if (pkt.visualSqErrSum !== undefined && pkt.visualSampleCount !== undefined) {
            totalVisualSq += pkt.visualSqErrSum;
            totalVisualSamples += pkt.visualSampleCount;
          }
          if (pkt.audioSqErrSum !== undefined && pkt.audioSampleCount !== undefined) {
            totalAudioSq += pkt.audioSqErrSum;
            totalAudioSamples += pkt.audioSampleCount;
          }
          for (let p = 0; p < averagedGrads.length; p++) {
            const dst = averagedGrads[p];
            const src = pkt.grads[p];
            for (let i = 0; i < dst.length; i++) {
              dst[i] += src[i] * invW;
            }
          }
        }

        engine.applyExternalGradientsAndStep(
          averagedGrads,
          totalSqErr,
          totalSamples,
          totalVisualSamples > 0 ? totalVisualSq : undefined,
          totalVisualSamples > 0 ? totalVisualSamples : undefined,
          totalAudioSamples > 0 ? totalAudioSq : undefined,
          totalAudioSamples > 0 ? totalAudioSamples : undefined,
        );

        const now = performance.now();
        const shouldPreview =
          !previewInFlight &&
          (engine.step - lastPreviewStep >= 14 || engine.step >= options.hyperparams.totalSteps);

        if (shouldPreview) {
          previewInFlight = true;
          lastPreviewStep = engine.step;
          const previewWeights = engine.getWeightArrays();
          void dispatchToPoolWorker<Extract<WorkerEvent, { type: 'mm-frames-slice-done' }>>(
            pool[0],
            {
              type: 'render-mm-frames-slice',
              jobId: `mm-prev-${engine.step}-${Math.random().toString(36).slice(2, 6)}`,
              hyperparams: engine.hyperparams,
              tokenizer: engine.tokenizer,
              weights: previewWeights,
              prompt: options.previewPrompt,
              width: 96,
              height: 54,
              times: [0.35],
              cameraMotion: 'static',
            },
            'mm-frames-slice-done',
          )
            .then((res) => {
              if (!cancelled && res.frames[0]) {
                options.onProgress({
                  history: [...engine.history],
                  previewFrame: res.frames[0],
                  previewWidth: 96,
                  previewHeight: 54,
                });
              }
            })
            .catch(() => {
              // Non-fatal preview error
            })
            .finally(() => {
              previewInFlight = false;
            });
        }

        if (now - lastPost >= 55 || engine.step >= options.hyperparams.totalSteps) {
          options.onProgress({
            history: [...engine.history],
          });
          lastPost = now;
        }

        if (now - lastYield >= 12) {
          lastYield = now;
          await new Promise((resolve) => setTimeout(resolve, 0));
        }
      }

      if (!stopped) {
        stopped = true;
        options.onDone(engine.exportCheckpoint(), cancelled);
      }
    } catch (err) {
      if (!stopped) {
        stopped = true;
        options.onError(err instanceof Error ? err.message : 'Distributed training failed');
      }
    }
  };

  void run();

  return {
    cancel: () => {
      cancelled = true;
    },
  };
}

export interface GenerateMediaPoolOptions {
  aspectRatio?: '16:9' | '1:1';
  cameraMotion?: CameraMotionMode;
  withAudio?: boolean;
  numFrames?: number;
}

export async function generateMediaInPool(
  kind: 'image' | 'video',
  prompt: string,
  checkpoint: MultimodalCheckpoint | null,
  autoTrainSteps = 140,
  options: GenerateMediaPoolOptions = {},
): Promise<{
  width: number;
  height: number;
  fps: number;
  frames: Uint8ClampedArray[];
  audioTrack?: NeuralAudioFrame[];
  cameraMotion?: Exclude<CameraMotionMode, 'auto'>;
  checkpoint: MultimodalCheckpoint;
}> {
  let activeCheckpoint = checkpoint;

  if (!activeCheckpoint) {
    activeCheckpoint = await new Promise<MultimodalCheckpoint>((resolve, reject) => {
      trainMultimodalPool({
        hyperparams: {
          dCond: 24,
          hiddenDim: 80,
          numFreqs: 4,
          learningRate: 0.015,
          totalSteps: autoTrainSteps,
          seed: 2026,
          audioHiddenDim: 32,
        },
        previewPrompt: prompt,
        onProgress: () => {},
        onDone: (cp) => resolve(cp),
        onError: (msg) => reject(new Error(msg)),
      });
    });
  }

  const pool = getWorkerPool();
  const engine = QoweVisionVideoEngine.fromCheckpoint(activeCheckpoint);
  const weights = engine.getWeightArrays();
  const aspectRatio = options.aspectRatio ?? '16:9';

  if (kind === 'image') {
    const width = aspectRatio === '16:9' ? 384 : 256;
    const height = aspectRatio === '16:9' ? 216 : 256;
    const res = await dispatchToPoolWorker<Extract<WorkerEvent, { type: 'mm-frames-slice-done' }>>(
      pool[0],
      {
        type: 'render-mm-frames-slice',
        jobId: `render-img-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        hyperparams: engine.hyperparams,
        tokenizer: engine.tokenizer,
        weights,
        prompt,
        width,
        height,
        times: [0.35],
        cameraMotion: 'static',
      },
      'mm-frames-slice-done',
    );
    return {
      width,
      height,
      fps: 1,
      frames: res.frames,
      checkpoint: activeCheckpoint,
    };
  }

  // Parallel 60 fps Veo-style audio-video synthesis across all browser worker threads
  const numFrames = options.numFrames ?? 120;
  const width = aspectRatio === '16:9' ? 320 : 240;
  const height = aspectRatio === '16:9' ? 180 : 240;
  const fps = 60;
  const resolvedCam = resolveCameraMotion(prompt, options.cameraMotion ?? 'auto');
  const times = Array.from({ length: numFrames }, (_, idx) => idx / numFrames);
  const chunkSize = Math.ceil(numFrames / pool.length);

  const audioTrack =
    options.withAudio === false ? undefined : engine.predictAudioTrack(prompt, times, resolvedCam);

  const slices = await Promise.all(
    pool.map((w, idx) => {
      const sliceTimes = times.slice(idx * chunkSize, (idx + 1) * chunkSize);
      if (sliceTimes.length === 0) {
        return Promise.resolve({ frames: [] as Uint8ClampedArray[] });
      }
      return dispatchToPoolWorker<Extract<WorkerEvent, { type: 'mm-frames-slice-done' }>>(
        w,
        {
          type: 'render-mm-frames-slice',
          jobId: `render-vid-${idx}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
          hyperparams: engine.hyperparams,
          tokenizer: engine.tokenizer,
          weights,
          prompt,
          width,
          height,
          times: sliceTimes,
          cameraMotion: resolvedCam,
        },
        'mm-frames-slice-done',
      );
    }),
  );

  const frames = slices.flatMap((s) => s.frames);
  return {
    width,
    height,
    fps,
    frames,
    audioTrack,
    cameraMotion: resolvedCam,
    checkpoint: activeCheckpoint,
  };
}

// Keep backward-compatible wrapper delegating to multi-core pool
export function generateMediaInWorker(
  kind: 'image' | 'video',
  prompt: string,
  checkpoint: MultimodalCheckpoint | null,
  autoTrainSteps = 140,
) {
  return generateMediaInPool(kind, prompt, checkpoint, autoTrainSteps);
}
