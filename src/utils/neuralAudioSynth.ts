export type NeuralAudioFrame = [
  number, // 0: bassNorm (sub/bass pitch 0..1 -> 38..110 Hz)
  number, // 1: harmonicMix (pad/chime harmonic shimmer 0..1)
  number, // 2: noiseLevel (wind/rain/surf/roar 0..1)
  number, // 3: filterNorm (lowpass cutoff brightness 0..1)
  number, // 4: pulseRate (rhythmic swell/modulation speed 0..1)
  number, // 5: transientAmp (droplets/crackles/chimes/thunder 0..1)
  number, // 6: stereoWidth (spatial motion pan 0..1)
  number, // 7: subRumble (low cinema impact/engine/rumble 0..1)
];

export const AUDIO_PARAM_DIM = 8;

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function createAudioRng(seed: number): () => number {
  let s = (seed >>> 0) || 1987;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface StereoPcmBuffer {
  sampleRate: number;
  left: Float32Array;
  right: Float32Array;
  durationSec: number;
}

/**
 * Synthesizes 44.1kHz stereo cinematic audio driven by Qowe's trained-from-scratch
 * 8-channel neural audio head: warm harmonic pads, dual-stage resonant surf/rain/wind
 * noise, thruster/engine rumble harmonics, crisp foley transients, and stereo pan sweeps.
 */
export function renderNeuralAudioPcm(
  audioTrack: NeuralAudioFrame[],
  durationSec: number,
  sampleRate = 44100,
): StereoPcmBuffer {
  const safeDuration = Math.max(0.5, Math.min(8, durationSec));
  const totalSamples = Math.max(1, Math.floor(safeDuration * sampleRate));
  const left = new Float32Array(totalSamples);
  const right = new Float32Array(totalSamples);

  if (audioTrack.length === 0) {
    return { sampleRate, left, right, durationSec: safeDuration };
  }

  const rand = createAudioRng(audioTrack.length * 919 + 42);
  const frameCount = audioTrack.length;
  const twoPi = Math.PI * 2;
  const invSr = 1 / sampleRate;

  let phaseSub = 0;
  let phaseFifth = 0;
  let phaseNinth = 0;
  let phaseEngine = 0;
  let phaseMod = 0;
  let lp1L = 0;
  let lp2L = 0;
  let lp1R = 0;
  let lp2R = 0;
  let transientEnv = 0;
  let transientPhase = 0;
  let transientFreq = 680;
  let transientPan = 0.5;
  let rumbleEnv = 0;

  for (let i = 0; i < totalSamples; i++) {
    const u = i / totalSamples;
    const fPos = u * frameCount;
    const idx0 = Math.floor(fPos) % frameCount;
    const idx1 = (idx0 + 1) % frameCount;
    const frac = fPos - Math.floor(fPos);

    const a0 = audioTrack[idx0];
    const a1 = audioTrack[idx1];

    const bassNorm = clamp01(a0[0] + (a1[0] - a0[0]) * frac);
    const harmonicMix = clamp01(a0[1] + (a1[1] - a0[1]) * frac);
    const noiseLevel = clamp01(a0[2] + (a1[2] - a0[2]) * frac);
    const filterNorm = clamp01(a0[3] + (a1[3] - a0[3]) * frac);
    const pulseRate = clamp01(a0[4] + (a1[4] - a0[4]) * frac);
    const transientAmp = clamp01(a0[5] + (a1[5] - a0[5]) * frac);
    const stereoWidth = clamp01(a0[6] + (a1[6] - a0[6]) * frac);
    const subRumble = clamp01(a0[7] + (a1[7] - a0[7]) * frac);

    const baseHz = 40 + bassNorm * 70;
    const lfoHz = 0.55 + pulseRate * 5.6;

    phaseSub = (phaseSub + twoPi * baseHz * invSr) % twoPi;
    phaseFifth = (phaseFifth + twoPi * (baseHz * 1.498) * invSr) % twoPi;
    phaseNinth = (phaseNinth + twoPi * (baseHz * 2.245) * invSr) % twoPi;
    phaseEngine = (phaseEngine + twoPi * (baseHz * 0.75 + pulseRate * 48) * invSr) % twoPi;
    phaseMod = (phaseMod + twoPi * lfoHz * invSr) % twoPi;

    const pulseEnv = 0.65 + 0.35 * Math.sin(phaseMod);

    // Rich warm sub + mechanical/cinema turbine texture
    const sub = Math.sin(phaseSub) * (0.14 + subRumble * 0.28);
    const engineGrowl =
      Math.tanh(Math.sin(phaseEngine) * 2.2 + Math.sin(phaseEngine * 2.02) * 0.9) *
      subRumble *
      pulseRate *
      0.16;

    const harm =
      (Math.sin(phaseFifth) * 0.52 +
        Math.sin(phaseNinth + 0.5) * 0.35 +
        Math.sin(phaseSub * 2.005) * 0.25) *
      harmonicMix *
      0.24 *
      pulseEnv;

    // 2-pole warm resonant noise shaping (surf / rain / rocket roar / wind)
    const whiteL = rand() * 2 - 1;
    const whiteR = rand() * 2 - 1;
    const cutoffCoeff = 0.014 + filterNorm * filterNorm * 0.34;
    lp1L += cutoffCoeff * (whiteL - lp1L);
    lp2L += cutoffCoeff * (lp1L - lp2L);
    lp1R += cutoffCoeff * (whiteR - lp1R);
    lp2R += cutoffCoeff * (lp1R - lp2R);

    const midNoise = (lp2L + lp2R) * 0.5;
    const sideNoise = (lp2L - lp2R) * stereoWidth;
    const nGain = noiseLevel * (0.3 + 0.18 * pulseEnv);
    const texL = (midNoise + sideNoise) * nGain;
    const texR = (midNoise - sideNoise) * nGain;

    // Crisp transient layer (raindrops, fire crackles, chimes, distant thunder impacts)
    if (
      transientAmp > 0.06 &&
      transientEnv < 0.015 &&
      rand() < 0.00022 + transientAmp * 0.0018
    ) {
      transientEnv = 0.26 + transientAmp * 0.56;
      transientFreq = 260 + rand() * 1650 * (0.35 + filterNorm);
      transientPan = 0.5 + (rand() - 0.5) * stereoWidth;
      if (subRumble > 0.55 && rand() < 0.18) {
        rumbleEnv = Math.max(rumbleEnv, 0.35);
      }
    }

    let pingL = 0;
    let pingR = 0;
    if (transientEnv > 0.001) {
      transientPhase = (transientPhase + twoPi * transientFreq * invSr) % twoPi;
      const p = Math.sin(transientPhase) * transientEnv * 0.2;
      pingL = p * (1 - transientPan * 0.6);
      pingR = p * (0.4 + transientPan * 0.6);
      transientEnv *= 0.9991;
    }

    let impact = 0;
    if (rumbleEnv > 0.001) {
      impact = (midNoise * 0.7 + Math.sin(phaseSub * 0.5) * 0.3) * rumbleEnv * 0.45;
      rumbleEnv *= 0.9996;
    }

    const panWave = Math.sin(u * twoPi) * 0.28 * stereoWidth;
    const baseSignal = sub + engineGrowl + harm + impact;
    const outL = Math.tanh((baseSignal * (1 - panWave) + texL + pingL) * 0.95);
    const outR = Math.tanh((baseSignal * (1 + panWave) + texR + pingR) * 0.95);

    const edgeSamples = Math.min(900, Math.floor(totalSamples * 0.04));
    let env = 1;
    if (i < edgeSamples) env = i / edgeSamples;
    else if (i > totalSamples - edgeSamples) env = (totalSamples - i) / edgeSamples;

    left[i] = outL * env;
    right[i] = outR * env;
  }

  return { sampleRate, left, right, durationSec: safeDuration };
}

export function createAudioBufferFromPcm(
  ctx: BaseAudioContext,
  pcm: StereoPcmBuffer,
): AudioBuffer {
  const buf = ctx.createBuffer(2, pcm.left.length, pcm.sampleRate);
  buf.getChannelData(0).set(pcm.left);
  buf.getChannelData(1).set(pcm.right);
  return buf;
}

export function encodeStereoPcmToWavBlob(pcm: StereoPcmBuffer): Blob {
  const numChannels = 2;
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = pcm.sampleRate * blockAlign;
  const dataBytes = pcm.left.length * blockAlign;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);

  const writeStr = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };

  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + dataBytes, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, pcm.sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, dataBytes, true);

  let off = 44;
  for (let i = 0; i < pcm.left.length; i++) {
    const l = Math.max(-1, Math.min(1, pcm.left[i]));
    const r = Math.max(-1, Math.min(1, pcm.right[i]));
    view.setInt16(off, l < 0 ? l * 0x8000 : l * 0x7fff, true);
    view.setInt16(off + 2, r < 0 ? r * 0x8000 : r * 0x7fff, true);
    off += 4;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}
