import {
  CameraMotionMode,
  MultimodalCheckpoint,
  MultimodalHyperparams,
  MultimodalStepMetric,
  QoweVisionVideoEngine,
} from './multimodal';
import type { NeuralAudioFrame } from '../utils/neuralAudioSynth';
import { QoweTokenizer, trainTokenizer } from './tokenizer';
import {
  QoweCheckpoint,
  QoweTrainerEngine,
  QoweTrainingHyperparams,
  TrainingStepMetric,
} from './trainer';

export type WorkerCommand =
  | {
      type: 'train-text';
      jobId: string;
      corpus: string;
      hyperparams: QoweTrainingHyperparams;
      resumeFrom?: QoweCheckpoint | null;
    }
  | {
      type: 'train-multimodal';
      jobId: string;
      hyperparams: MultimodalHyperparams;
      previewPrompt: string;
      resumeFrom?: MultimodalCheckpoint | null;
    }
  | {
      type: 'compute-mm-grads';
      jobId: string;
      hyperparams: MultimodalHyperparams;
      tokenizer: QoweTokenizer;
      weights: Float32Array[];
      samplesPerConcept: number;
      stepSeed: number;
    }
  | {
      type: 'render-mm-frames-slice';
      jobId: string;
      hyperparams: MultimodalHyperparams;
      tokenizer: QoweTokenizer;
      weights: Float32Array[];
      prompt: string;
      width: number;
      height: number;
      times: number[];
      cameraMotion?: CameraMotionMode;
    }
  | {
      type: 'pause';
      jobId: string;
    }
  | {
      type: 'generate-media';
      jobId: string;
      kind: 'image' | 'video';
      prompt: string;
      checkpoint: MultimodalCheckpoint | null;
      autoTrainSteps?: number;
      cameraMotion?: CameraMotionMode;
    };

export type WorkerEvent =
  | {
      type: 'text-progress';
      jobId: string;
      history: TrainingStepMetric[];
    }
  | {
      type: 'text-done';
      jobId: string;
      checkpoint: QoweCheckpoint;
      paused: boolean;
    }
  | {
      type: 'mm-progress';
      jobId: string;
      history: MultimodalStepMetric[];
      previewFrame?: Uint8ClampedArray;
      previewWidth?: number;
      previewHeight?: number;
    }
  | {
      type: 'mm-done';
      jobId: string;
      checkpoint: MultimodalCheckpoint;
      paused: boolean;
    }
  | {
      type: 'mm-grads-done';
      jobId: string;
      grads: Float32Array[];
      sqErrSum: number;
      sampleCount: number;
      visualSqErrSum?: number;
      visualSampleCount?: number;
      audioSqErrSum?: number;
      audioSampleCount?: number;
    }
  | {
      type: 'mm-frames-slice-done';
      jobId: string;
      frames: Uint8ClampedArray[];
    }
  | {
      type: 'media-done';
      jobId: string;
      kind: 'image' | 'video';
      width: number;
      height: number;
      fps: number;
      frames: Uint8ClampedArray[];
      audioTrack?: NeuralAudioFrame[];
      cameraMotion?: CameraMotionMode;
      checkpoint: MultimodalCheckpoint;
    }
  | {
      type: 'error';
      jobId: string;
      message: string;
    };

let activeJobId: string | null = null;
let pauseRequested = false;

// Reusable worker-local multimodal engine avoids reallocation on every AllReduce step
let cachedMmEngine: QoweVisionVideoEngine | null = null;
let cachedMmKey = '';

function getOrSyncMmEngine(
  tokenizer: QoweTokenizer,
  hyperparams: MultimodalHyperparams,
  weights: Float32Array[],
): QoweVisionVideoEngine {
  const key = `${tokenizer.vocab.length}:${hyperparams.dCond}:${hyperparams.hiddenDim}:${hyperparams.numFreqs}:${hyperparams.audioHiddenDim ?? 32}`;
  if (!cachedMmEngine || cachedMmKey !== key) {
    cachedMmEngine = new QoweVisionVideoEngine(tokenizer, hyperparams);
    cachedMmKey = key;
  }
  cachedMmEngine.setWeightArrays(weights);
  return cachedMmEngine;
}

function yieldSlice(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

self.onmessage = async (event: MessageEvent<WorkerCommand>) => {
  const msg = event.data;

  if (msg.type === 'pause') {
    if (activeJobId === msg.jobId) {
      pauseRequested = true;
    }
    return;
  }

  if (msg.type === 'compute-mm-grads') {
    try {
      const engine = getOrSyncMmEngine(msg.tokenizer, msg.hyperparams, msg.weights);
      const pkt = engine.computeBatchGradients(msg.samplesPerConcept, msg.stepSeed);
      const transferables = pkt.grads.map((g) => g.buffer);
      (self as unknown as Worker).postMessage(
        {
          type: 'mm-grads-done',
          jobId: msg.jobId,
          grads: pkt.grads,
          sqErrSum: pkt.sqErrSum,
          sampleCount: pkt.sampleCount,
          visualSqErrSum: pkt.visualSqErrSum,
          visualSampleCount: pkt.visualSampleCount,
          audioSqErrSum: pkt.audioSqErrSum,
          audioSampleCount: pkt.audioSampleCount,
        } satisfies WorkerEvent,
        transferables,
      );
    } catch (err) {
      (self as unknown as Worker).postMessage({
        type: 'error',
        jobId: msg.jobId,
        message: err instanceof Error ? err.message : 'Distributed gradient worker error',
      } satisfies WorkerEvent);
    }
    return;
  }

  if (msg.type === 'render-mm-frames-slice') {
    try {
      const engine = getOrSyncMmEngine(msg.tokenizer, msg.hyperparams, msg.weights);
      const frames = engine.renderFramesBatch(
        msg.prompt,
        msg.width,
        msg.height,
        msg.times,
        msg.cameraMotion ?? 'auto',
      );
      const transferables = frames.map((f) => f.buffer);
      (self as unknown as Worker).postMessage(
        {
          type: 'mm-frames-slice-done',
          jobId: msg.jobId,
          frames,
        } satisfies WorkerEvent,
        transferables,
      );
    } catch (err) {
      (self as unknown as Worker).postMessage({
        type: 'error',
        jobId: msg.jobId,
        message: err instanceof Error ? err.message : 'Distributed frame render error',
      } satisfies WorkerEvent);
    }
    return;
  }

  if (msg.type === 'train-text') {
    activeJobId = msg.jobId;
    pauseRequested = false;
    try {
      const engine = msg.resumeFrom
        ? QoweTrainerEngine.fromCheckpoint(msg.resumeFrom, msg.corpus)
        : new QoweTrainerEngine(
            trainTokenizer(msg.corpus || 'Hello Qowe 1.0', 360),
            msg.corpus || 'Hello Qowe 1.0',
            msg.hyperparams,
          );

      let lastPost = performance.now();
      while (engine.step < msg.hyperparams.totalSteps && !pauseRequested && activeJobId === msg.jobId) {
        engine.trainStep();
        const now = performance.now();
        if (now - lastPost >= 50 || engine.step >= msg.hyperparams.totalSteps) {
          (self as unknown as Worker).postMessage({
            type: 'text-progress',
            jobId: msg.jobId,
            history: [...engine.history],
          } satisfies WorkerEvent);
          lastPost = now;
          await yieldSlice();
        }
      }

      const cp = engine.exportCheckpoint();
      (self as unknown as Worker).postMessage({
        type: 'text-done',
        jobId: msg.jobId,
        checkpoint: cp,
        paused: pauseRequested,
      } satisfies WorkerEvent);
    } catch (err) {
      (self as unknown as Worker).postMessage({
        type: 'error',
        jobId: msg.jobId,
        message: err instanceof Error ? err.message : 'Text training worker error',
      } satisfies WorkerEvent);
    } finally {
      if (activeJobId === msg.jobId) activeJobId = null;
    }
    return;
  }

  if (msg.type === 'train-multimodal') {
    activeJobId = msg.jobId;
    pauseRequested = false;
    try {
      const engine = msg.resumeFrom
        ? QoweVisionVideoEngine.fromCheckpoint(msg.resumeFrom)
        : new QoweVisionVideoEngine(undefined, msg.hyperparams);

      let lastPost = performance.now();
      let lastPreviewStep = 0;

      while (engine.step < msg.hyperparams.totalSteps && !pauseRequested && activeJobId === msg.jobId) {
        engine.trainStep();
        const now = performance.now();
        const shouldPreview = engine.step - lastPreviewStep >= 16 || engine.step >= msg.hyperparams.totalSteps;

        if (now - lastPost >= 50 || shouldPreview) {
          let previewFrame: Uint8ClampedArray | undefined;
          let previewWidth: number | undefined;
          let previewHeight: number | undefined;
          if (shouldPreview) {
            previewWidth = 64;
            previewHeight = 64;
            previewFrame = engine.renderFrame(msg.previewPrompt, 64, 64, 0.35);
            lastPreviewStep = engine.step;
          }

          const payload: WorkerEvent = {
            type: 'mm-progress',
            jobId: msg.jobId,
            history: [...engine.history],
            previewFrame,
            previewWidth,
            previewHeight,
          };
          if (previewFrame) {
            (self as unknown as Worker).postMessage(payload, [previewFrame.buffer]);
          } else {
            (self as unknown as Worker).postMessage(payload);
          }
          lastPost = now;
          await yieldSlice();
        }
      }

      const cp = engine.exportCheckpoint();
      (self as unknown as Worker).postMessage({
        type: 'mm-done',
        jobId: msg.jobId,
        checkpoint: cp,
        paused: pauseRequested,
      } satisfies WorkerEvent);
    } catch (err) {
      (self as unknown as Worker).postMessage({
        type: 'error',
        jobId: msg.jobId,
        message: err instanceof Error ? err.message : 'Multimodal training worker error',
      } satisfies WorkerEvent);
    } finally {
      if (activeJobId === msg.jobId) activeJobId = null;
    }
    return;
  }

  if (msg.type === 'generate-media') {
    try {
      const engine = msg.checkpoint
        ? QoweVisionVideoEngine.fromCheckpoint(msg.checkpoint)
        : new QoweVisionVideoEngine();

      if (!msg.checkpoint) {
        const steps = msg.autoTrainSteps ?? 140;
        for (let i = 0; i < steps; i++) {
          engine.trainStep();
          if (i % 20 === 19) await yieldSlice();
        }
      }

      const media =
        msg.kind === 'image'
          ? engine.generateImage(msg.prompt, 128)
          : engine.generateCinematicVideo(msg.prompt, {
              cameraMotion: msg.cameraMotion ?? 'auto',
            });

      const cp = engine.exportCheckpoint();
      const transferables = media.frames.map((f) => f.buffer);
      (self as unknown as Worker).postMessage(
        {
          type: 'media-done',
          jobId: msg.jobId,
          kind: msg.kind,
          width: media.width,
          height: media.height,
          fps: media.fps,
          frames: media.frames,
          audioTrack: media.audioTrack,
          cameraMotion: media.cameraMotion,
          checkpoint: cp,
        } satisfies WorkerEvent,
        transferables,
      );
    } catch (err) {
      (self as unknown as Worker).postMessage({
        type: 'error',
        jobId: msg.jobId,
        message: err instanceof Error ? err.message : 'Media synthesis error',
      } satisfies WorkerEvent);
    }
  }
};
