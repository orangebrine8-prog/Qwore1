import test from 'node:test';
import assert from 'node:assert/strict';
import { calculateArchitectureSpecs, QOWE_PRESETS } from './architecture';
import { TRAINING_DATASETS } from './datasets';
import {
  buildFullModelBundle,
  buildPytorchScript,
  buildStandaloneChatHtml,
  ensureTrainedTextCheckpoint,
  isTextCheckpointReadyForChat,
} from './exporter';
import { QoweVisionVideoEngine } from './multimodal';
import { decode, encode, trainTokenizer } from './tokenizer';
import { QoweTrainerEngine } from './trainer';
import { renderNeuralAudioPcm } from '../utils/neuralAudioSynth';

test('Qowe 1.0 Ultra 800B preset exceeds 800B parameters with exact component sum', () => {
  const ultra = QOWE_PRESETS[0];
  const specs = calculateArchitectureSpecs(ultra);

  assert.equal(ultra.name, 'Qowe 1.0 Ultra 800B');
  assert.ok(specs.totalParams > 800_000_000_000, `Expected > 800B, got ${specs.totalParams}`);
  assert.equal(
    specs.tokenEmbeddingParams +
      specs.positionEmbeddingParams +
      specs.totalBlockParams +
      specs.finalLayerNormParams +
      specs.lmHeadParams,
    specs.totalParams,
  );
});

test('Qowe 1.0 Base preset exceeds 100M parameters with exact component sum', () => {
  const base = QOWE_PRESETS[1];
  const specs = calculateArchitectureSpecs(base);

  assert.equal(specs.isOver100M, true);
  assert.ok(specs.totalParams > 100_000_000);
  assert.equal(specs.totalParams, 124_439_808);
  assert.equal(
    specs.tokenEmbeddingParams +
      specs.positionEmbeddingParams +
      specs.totalBlockParams +
      specs.finalLayerNormParams +
      specs.lmHeadParams,
    specs.totalParams,
  );
});

test('Word-preserving tokenizer round-trips chat & code text cleanly', () => {
  const sample =
    '<|user|> Who are you?\n<|assistant|> I am Qowe 1.0, an 801B parameter transformer.<|endoftext|>';
  const tokenizer = trainTokenizer(sample + ' ' + sample, 320);
  const encoded = encode(tokenizer, sample);
  const decoded = decode(tokenizer, encoded);

  assert.ok(encoded.length > 0);
  assert.equal(decoded, sample);
  assert.ok(tokenizer.vocab.includes(' Qowe') || tokenizer.vocab.includes('Qowe'));
});

test('From-scratch transformer training converges to coherent English words on dialogue corpus', () => {
  const corpus = TRAINING_DATASETS[0].corpus;
  const tokenizer = trainTokenizer(corpus, 360);
  const engine = new QoweTrainerEngine(tokenizer, corpus, {
    dModel: 48,
    nLayers: 1,
    nHeads: 4,
    dFF: 96,
    contextLength: 48,
    learningRate: 0.015,
    totalSteps: 100,
    seed: 1337,
  });

  const first = engine.trainStep();
  let last = first;
  for (let i = 1; i < 100; i++) {
    last = engine.trainStep();
  }

  assert.ok(Number.isFinite(first.loss));
  assert.ok(Number.isFinite(last.loss));
  assert.ok(last.loss < 1.2, `Expected converged loss < 1.2, got ${last.loss.toFixed(4)}`);

  const output = engine.generate('<|user|> Who are you?\n<|assistant|>', {
    maxNewTokens: 28,
    temperature: 0.25,
    topK: 3,
  });

  assert.ok(output.generatedTokens > 3);
  assert.ok(
    output.text.includes('Qowe') || output.text.includes('transformer') || output.text.includes('model'),
    `Expected coherent English output from corpus, got: ${JSON.stringify(output.text)}`,
  );
});

test('Multi-worker AllReduce training reduces visual+audio loss and renders 16:9 60fps video with synced stereo audio', () => {
  const coordinator = new QoweVisionVideoEngine();
  const workerA = new QoweVisionVideoEngine(coordinator.tokenizer, coordinator.hyperparams);
  const workerB = new QoweVisionVideoEngine(coordinator.tokenizer, coordinator.hyperparams);

  let firstLoss = 0;
  let lastLoss = 0;
  let firstAudioLoss = 0;
  let lastAudioLoss = 0;

  for (let step = 0; step < 30; step++) {
    const weights = coordinator.getWeightArrays();
    workerA.setWeightArrays(weights);
    workerB.setWeightArrays(weights);

    const pktA = workerA.computeBatchGradients(16, 1000 + step * 2);
    const pktB = workerB.computeBatchGradients(16, 1001 + step * 2);

    const avgGrads = pktA.grads.map((g, idx) => {
      const out = new Float32Array(g.length);
      for (let i = 0; i < g.length; i++) {
        out[i] = 0.5 * (g[i] + pktB.grads[idx][i]);
      }
      return out;
    });

    const metric = coordinator.applyExternalGradientsAndStep(
      avgGrads,
      pktA.sqErrSum + pktB.sqErrSum,
      pktA.sampleCount + pktB.sampleCount,
      (pktA.visualSqErrSum ?? 0) + (pktB.visualSqErrSum ?? 0),
      (pktA.visualSampleCount ?? 0) + (pktB.visualSampleCount ?? 0),
      (pktA.audioSqErrSum ?? 0) + (pktB.audioSqErrSum ?? 0),
      (pktA.audioSampleCount ?? 0) + (pktB.audioSampleCount ?? 0),
    );
    if (step === 0) {
      firstLoss = metric.loss;
      firstAudioLoss = metric.audioLoss ?? 0;
    }
    lastLoss = metric.loss;
    lastAudioLoss = metric.audioLoss ?? 0;
  }

  assert.ok(Number.isFinite(firstLoss));
  assert.ok(Number.isFinite(lastLoss));
  assert.ok(lastLoss < firstLoss, `Expected AllReduce visual MSE drop: ${firstLoss} -> ${lastLoss}`);
  assert.ok(
    lastAudioLoss < firstAudioLoss,
    `Expected AllReduce audio MSE drop: ${firstAudioLoss} -> ${lastAudioLoss}`,
  );

  const img = coordinator.generateImage('studio banana gold product shot on dark slate', 128);
  assert.equal(img.frames.length, 1);
  assert.equal(img.frames[0].length, 128 * 128 * 4);

  const vid = coordinator.generateVideo('neon cyber city rain reflections at night', 48, 60, 60);
  assert.equal(vid.fps, 60);
  assert.equal(vid.frames.length, 60);
  let diffSum = 0;
  for (let i = 0; i < vid.frames[0].length; i += 4) {
    diffSum += Math.abs(vid.frames[0][i] - vid.frames[15][i]);
  }
  assert.ok(diffSum > 0, 'Expected visible motion across 60fps generated video frames');

  const cinema = coordinator.generateCinematicVideo(
    'orbit golden sunrise over calm ocean waves',
    { width: 64, height: 36, numFrames: 30, fps: 60 },
  );
  assert.equal(cinema.width, 64);
  assert.equal(cinema.height, 36);
  assert.equal(cinema.cameraMotion, 'orbit');
  assert.equal(cinema.audioTrack?.length, 30);

  const openPrompt = coordinator.generateCinematicVideo(
    'sports car drifting through neon cyber city in heavy rain',
    { width: 160, height: 90, numFrames: 10, fps: 60 },
  );
  assert.equal(openPrompt.width, 160);
  assert.equal(openPrompt.height, 90);
  assert.equal(openPrompt.frames[0].length, 160 * 90 * 4);
  assert.equal(openPrompt.audioTrack?.length, 10);

  const pcm = renderNeuralAudioPcm(cinema.audioTrack!, 1.0, 22050);
  assert.equal(pcm.left.length, 22050);
  let energy = 0;
  for (let i = 0; i < pcm.left.length; i += 10) {
    energy += Math.abs(pcm.left[i]);
  }
  assert.ok(energy > 1, 'Expected audible neural waveform synthesis');

  const ltxResult = coordinator.predictCinemaPayload(
    'stormy ocean waves crashing against jagged volcanic cliffs under emerald aurora',
    { aspectRatio: '16:9', durationSec: 4, fps: 60, keyframeCount: 8, denoisingSteps: 8 },
  );
  assert.equal(ltxResult.payload.aspectRatio, '16:9');
  assert.equal(ltxResult.payload.denoisingSteps, 8);
  assert.equal(ltxResult.payload.neuralKeyframes.length, 8);
  assert.equal(ltxResult.payload.sceneLatents.length, 14);
  assert.ok(
    ltxResult.payload.flowConsistencyScore > 0.5 && ltxResult.payload.flowConsistencyScore <= 1,
    `Expected flowConsistencyScore in (0.5, 1], got ${ltxResult.payload.flowConsistencyScore}`,
  );
  assert.equal(ltxResult.audioTrack.length, 240);
});

test('Standalone PyTorch exporter reflects exact 800B, Base, and cloned LTX-2.5 Dual-Stream DiT configs', () => {
  const scriptUltra = buildPytorchScript(QOWE_PRESETS[0]);
  assert.ok(scriptUltra.includes('801.32B'));
  assert.ok(scriptUltra.includes('d_model: int = 20480'));
  assert.ok(scriptUltra.includes('n_layers: int = 158'));
  assert.ok(scriptUltra.includes('class Qowe1Model'));
  assert.ok(scriptUltra.includes('class QoweLTX25DualStreamDiT'));
  assert.ok(scriptUltra.includes('class AsymmetricDualStreamDiTBlock'));
  assert.ok(scriptUltra.includes('class RectifiedFlowScheduler'));

  const scriptBase = buildPytorchScript(QOWE_PRESETS[1]);
  assert.ok(scriptBase.includes('124,439,808'));
  assert.ok(scriptBase.includes('d_model: int = 768'));
  assert.ok(scriptBase.includes('n_layers: int = 12'));
});

test('Exported JSON bundle auto-trains assistant weights and referenceInferenceJs generates coherent assistant replies', () => {
  const trainedCp = ensureTrainedTextCheckpoint(null, TRAINING_DATASETS[0].corpus, 140);
  assert.equal(isTextCheckpointReadyForChat(trainedCp), true);
  assert.ok(trainedCp.loss < 1.1, `Expected converged assistant loss < 1.1, got ${trainedCp.loss}`);

  const rawBundle = buildFullModelBundle(QOWE_PRESETS[1], trainedCp, null);
  const roundTripped = JSON.parse(JSON.stringify(rawBundle)) as typeof rawBundle;

  assert.equal(roundTripped.format, 'qowe-1.0-bundle');
  assert.equal(roundTripped.inferenceSpec.readyForChat, true);
  assert.deepEqual(roundTripped.inferenceSpec.stopTokenIds, [1, 2]);
  assert.ok(roundTripped.inferenceSpec.tensorLayouts.lmHead.includes('[vocabSize, dModel]'));

  const qoweGenerate = new Function(
    `${roundTripped.inferenceSpec.referenceInferenceJs}\nreturn qoweGenerate;`,
  )() as (
    bundle: typeof roundTripped,
    prompt: string,
    opts?: { temperature?: number; topK?: number; maxNewTokens?: number; greedy?: boolean },
  ) => { text: string; generatedTokens: number };

  const replyIdentity = qoweGenerate(roundTripped, 'Who are you?', {
    temperature: 0.2,
    topK: 1,
    maxNewTokens: 32,
    greedy: true,
  });
  assert.ok(replyIdentity.generatedTokens > 3);
  assert.ok(
    replyIdentity.text.includes('Qowe') || replyIdentity.text.includes('transformer'),
    `Expected coherent assistant identity reply, got: ${JSON.stringify(replyIdentity.text)}`,
  );

  const replyGreeting = qoweGenerate(roundTripped, 'Hi', {
    temperature: 0.2,
    topK: 1,
    maxNewTokens: 32,
    greedy: true,
  });
  assert.ok(replyGreeting.generatedTokens > 3);
  assert.ok(
    replyGreeting.text.includes('Qowe') ||
      replyGreeting.text.includes('Hello') ||
      replyGreeting.text.includes('assistant'),
    `Expected coherent assistant greeting reply, got: ${JSON.stringify(replyGreeting.text)}`,
  );

  const html = buildStandaloneChatHtml(roundTripped);
  assert.ok(html.includes('<!DOCTYPE html>'));
  assert.ok(html.includes('function qoweGenerate'));
  assert.ok(html.includes('Qowe 1.0 Standalone Local Assistant'));
  // Embedded JSON must escape '<' so '</script' cannot break HTML parsing
  assert.equal(html.match(/<\/script>/gi)?.length, 1);
});
