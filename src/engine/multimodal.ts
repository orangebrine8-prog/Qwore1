import { encode, QoweTokenizer, trainTokenizer } from './tokenizer';
import { AUDIO_PARAM_DIM, type NeuralAudioFrame } from '../utils/neuralAudioSynth';
import {
  type CameraMotionMode,
  composeCinematicFrame,
  parseCinemaPrompt,
  SCENE_LATENT_DIM,
} from './cinemaCompositor';
import { type LtxCinemaPayload, runLtx25DualStreamFlow } from './ltx25DiT';

export type { CameraMotionMode, LtxCinemaPayload };

export interface MultimodalHyperparams {
  dCond: number;
  hiddenDim: number;
  numFreqs: number;
  learningRate: number;
  totalSteps: number;
  seed: number;
  audioHiddenDim?: number;
}

export interface MultimodalStepMetric {
  step: number;
  loss: number;
  psnr: number;
  visualLoss?: number;
  audioLoss?: number;
}

export interface MultimodalCheckpoint {
  version: string;
  step: number;
  loss: number;
  hyperparams: MultimodalHyperparams;
  tokenizer: QoweTokenizer;
  weights: {
    textEmb: number[];
    w1: number[];
    b1: number[];
    w2: number[];
    b2: number[];
    w3: number[];
    b3: number[];
    wOut: number[];
    bOut: number[];
    filmGamma: number[];
    filmBeta: number[];
    wAud1?: number[];
    bAud1?: number[];
    wAudOut?: number[];
    bAudOut?: number[];
    wScene1?: number[];
    bScene1?: number[];
    wSceneOut?: number[];
    bSceneOut?: number[];
  };
  history: MultimodalStepMetric[];
}

export interface GeneratedMediaFrames {
  width: number;
  height: number;
  fps: number;
  frames: Uint8ClampedArray[];
  audioTrack?: NeuralAudioFrame[];
  cameraMotion?: CameraMotionMode;
}

interface ConceptTarget {
  prompt: string;
  aliases: string[];
  defaultCamera: Exclude<CameraMotionMode, 'auto'>;
  // [warmth, neon, space, water, mountains, city, forest, volcano, rain, fireflies, embers, kinetic, godRays, studioProduct]
  sceneLatents: number[];
  evalPixel: (x: number, y: number, t: number) => [number, number, number];
  evalAudio: (t: number) => NeuralAudioFrame;
}

export const MULTIMODAL_CONCEPTS: ConceptTarget[] = [
  {
    prompt: 'golden sunrise over calm ocean waves',
    aliases: [
      'golden sunrise over calm ocean waves',
      'ocean sunset golden sun water horizon waves boat whale',
      'sunny beach sunrise sea reflection golden dog running',
    ],
    defaultCamera: 'dolly-in',
    sceneLatents: [0.95, 0.05, 0.05, 0.96, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.85, 0.0],
    evalPixel: (x, y, t) => {
      const horizon = 0.06;
      const tau = t * Math.PI * 2;
      const sunY = -0.26 - 0.05 * Math.sin(tau);
      const distSun = Math.hypot(x, y - sunY);
      if (y < horizon) {
        const skyT = Math.max(0, Math.min(1, (y + 1) / (horizon + 1)));
        let r = 0.1 + 0.85 * Math.pow(skyT, 1.3);
        let g = 0.16 + 0.54 * Math.pow(skyT, 1.5);
        let b = 0.48 - 0.22 * skyT;
        const sunCore = distSun < 0.17 ? 1 : Math.exp(-Math.pow(distSun - 0.15, 2) * 38);
        const sunHalo = Math.exp(-distSun * distSun * 6.5);
        const cloud = 0.05 * Math.sin(x * 7 + t * 1.8) * Math.exp(-Math.pow((y + 0.35) * 4, 2));
        r = Math.min(1, r + 0.85 * sunCore + 0.35 * sunHalo + Math.max(0, cloud));
        g = Math.min(1, g + 0.72 * sunCore + 0.25 * sunHalo + Math.max(0, cloud * 0.7));
        b = Math.min(1, b + 0.28 * sunCore + 0.08 * sunHalo);
        return [r, g, b];
      }
      const depth = Math.max(0.05, y - horizon);
      const ripple = Math.sin(x * 12 + tau + y * 14) * 0.075 + Math.cos(x * 21 - tau * 1.5) * 0.03;
      const reflect = Math.exp(-(x * x) / (0.015 + depth * 0.095)) * (1 - depth * 0.52);
      const r = Math.min(1, Math.max(0, 0.04 + 0.92 * reflect + ripple));
      const g = Math.min(1, Math.max(0, 0.18 + 0.62 * reflect + ripple * 0.75));
      const b = Math.min(1, Math.max(0, 0.44 + 0.22 * reflect + ripple * 0.45));
      return [r, g, b];
    },
    evalAudio: (t) => {
      const swell = 0.5 + 0.5 * Math.sin(t * Math.PI * 2 - 0.6);
      return [
        0.24 + 0.06 * swell,
        0.58 + 0.14 * swell,
        0.48 + 0.36 * swell,
        0.28 + 0.34 * swell,
        0.18,
        0.08,
        0.65,
        0.22 + 0.15 * swell,
      ];
    },
  },
  {
    prompt: 'crimson solar eclipse in deep space starfield',
    aliases: [
      'crimson solar eclipse in deep space starfield',
      'deep space galaxy stars red eclipse moon rocket astronaut',
      'cosmic black hole solar flare cosmos starship',
    ],
    defaultCamera: 'orbit',
    sceneLatents: [0.4, 0.15, 0.98, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.5, 0.2, 0.0],
    evalPixel: (x, y, t) => {
      const tau = t * Math.PI * 2;
      const cx = 0.08 * Math.cos(tau);
      const cy = 0.06 * Math.sin(tau);
      const dx = x - cx;
      const dy = y - cy;
      const d = Math.hypot(dx, dy);
      const ang = Math.atan2(dy, dx);
      const flare = 1 + 0.25 * Math.sin(ang * 5 - tau) * Math.cos(ang * 3 + tau);
      const corona = Math.exp(-Math.pow((d - 0.36) * 8.5, 2)) * flare;
      const outerGlow = Math.exp(-Math.pow(Math.max(0, d - 0.34) * 3.2, 2));
      const eclipseMask = d < 0.32 ? 0.03 : 1;
      const starNoise = Math.pow(
        Math.max(0, Math.sin(x * 29 + y * 43) * Math.cos(x * 23 - y * 37)),
        12,
      );
      const r = Math.min(1, (0.03 + 0.96 * corona + 0.35 * outerGlow) * eclipseMask + starNoise * 0.75);
      const g = Math.min(1, (0.03 + 0.38 * corona + 0.1 * outerGlow) * eclipseMask + starNoise * 0.75);
      const b = Math.min(1, (0.09 + 0.2 * corona + 0.14 * outerGlow) * eclipseMask + starNoise * 0.95);
      return [r, g, b];
    },
    evalAudio: (t) => {
      const totality = Math.exp(-Math.pow((t - 0.5) * 3.1, 2));
      return [
        0.1 + 0.12 * totality,
        0.65 + 0.22 * totality,
        0.18 + 0.1 * totality,
        0.22 + 0.28 * totality,
        0.12,
        0.04,
        0.82,
        0.72 + 0.22 * totality,
      ];
    },
  },
  {
    prompt: 'emerald aurora lights over mountain peaks',
    aliases: [
      'emerald aurora lights over mountain peaks',
      'northern lights green aurora snowy mountain night dragon bird',
      'alpine sky glowing emerald ribbons arctic wind',
    ],
    defaultCamera: 'pan-right',
    sceneLatents: [0.05, 0.35, 0.45, 0.55, 0.95, 0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.3, 0.25, 0.0],
    evalPixel: (x, y, t) => {
      const tau = t * Math.PI * 2;
      const ridge = 0.28 + 0.22 * Math.sin(x * 3.2 + 0.5) + 0.11 * Math.cos(x * 7.2);
      if (y > ridge) {
        const snow = Math.exp(-Math.pow((y - ridge) * 18, 2)) * 0.18;
        return [0.05 + snow, 0.07 + snow * 1.2, 0.13 + snow * 1.5];
      }
      const waveY = -0.32 + 0.18 * Math.sin(x * 2.8 + tau);
      const band = Math.exp(-Math.pow((y - waveY) * 4.5, 2));
      const curtain = 0.45 + 0.55 * Math.sin(x * 8 - tau);
      const violetTop = Math.exp(-Math.pow((y + 0.65) * 3.5, 2)) * curtain;
      const r = Math.min(1, 0.04 + 0.18 * band * curtain + 0.45 * violetTop);
      const g = Math.min(1, 0.09 + 0.88 * band * (0.45 + 0.55 * curtain));
      const b = Math.min(1, 0.18 + 0.56 * band + 0.62 * violetTop);
      return [r, g, b];
    },
    evalAudio: (t) => {
      const shimmer = 0.5 + 0.5 * Math.sin(t * Math.PI * 4);
      return [
        0.38,
        0.82 + 0.12 * shimmer,
        0.34 + 0.18 * (1 - shimmer),
        0.48 + 0.22 * shimmer,
        0.24,
        0.18,
        0.88,
        0.16,
      ];
    },
  },
  {
    prompt: 'cobalt plasma vortex energy pulse',
    aliases: [
      'cobalt plasma vortex energy pulse',
      'electric blue energy spiral vortex swirl crystal robot',
      'quantum portal glowing blue orb hum',
    ],
    defaultCamera: 'dolly-in',
    sceneLatents: [0.1, 0.92, 0.72, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.85, 0.5, 0.0],
    evalPixel: (x, y, t) => {
      const tau = t * Math.PI * 2;
      const rDist = Math.hypot(x, y);
      const angle = Math.atan2(y, x);
      const spiral = 0.5 + 0.5 * Math.sin(rDist * 13 - angle * 3 - tau);
      const core = Math.exp(-rDist * rDist * 18);
      const fade = Math.exp(-rDist * 1.15);
      const r = Math.min(1, 0.06 + 0.48 * spiral * fade + 0.85 * core);
      const g = Math.min(1, 0.14 + 0.72 * spiral * fade + 0.9 * core);
      const b = Math.min(1, 0.28 + 0.72 * fade + 0.95 * core);
      return [r, g, b];
    },
    evalAudio: (t) => {
      const surge = 0.5 + 0.5 * Math.sin(t * Math.PI * 6);
      return [
        0.45 + 0.22 * surge,
        0.74,
        0.26 + 0.18 * surge,
        0.58 + 0.3 * surge,
        0.78,
        0.42,
        0.76,
        0.52,
      ];
    },
  },
  {
    prompt: 'neon cyber city rain reflections at night',
    aliases: [
      'neon cyber city rain reflections at night',
      'cyberpunk skyline glowing magenta cyan buildings sports car drift',
      'futuristic city street lights night heavy rain cat robot',
    ],
    defaultCamera: 'handheld',
    sceneLatents: [0.15, 0.98, 0.15, 0.78, 0.0, 0.98, 0.0, 0.0, 0.95, 0.0, 0.0, 0.78, 0.25, 0.0],
    evalPixel: (x, y, t) => {
      const tau = t * Math.PI * 2;
      const horizon = 0.12;
      const col = Math.floor((x + 1) * 5);
      const towerH = -0.1 - 0.45 * Math.abs(Math.sin(col * 1.9 + 0.4));
      const isMagenta = col % 2 === 0;
      if (y < horizon) {
        const inBuilding = y > towerH;
        if (!inBuilding) return [0.05, 0.04, 0.14];
        const neonWindow = Math.sin(y * 32) > 0.35 && Math.sin(x * 26) > 0.2 ? 0.85 : 0.08;
        return isMagenta
          ? [0.12 + neonWindow * 0.88, 0.06 + neonWindow * 0.18, 0.2 + neonWindow * 0.65]
          : [0.06 + neonWindow * 0.14, 0.12 + neonWindow * 0.82, 0.22 + neonWindow * 0.75];
      }
      const ripple = 0.7 + 0.3 * Math.sin(y * 28 - tau + x * 6);
      const shimmer = Math.exp(-Math.pow(x % 0.32, 2) * 18) * ripple;
      return isMagenta
        ? [0.12 + 0.68 * shimmer, 0.06 + 0.15 * shimmer, 0.22 + 0.55 * shimmer]
        : [0.06 + 0.12 * shimmer, 0.14 + 0.65 * shimmer, 0.24 + 0.68 * shimmer];
    },
    evalAudio: (t) => {
      const passBy = 0.5 + 0.5 * Math.sin(t * Math.PI * 2);
      return [
        0.34,
        0.48 + 0.14 * passBy,
        0.64,
        0.55 + 0.22 * passBy,
        0.44,
        0.72,
        0.72,
        0.36,
      ];
    },
  },
  {
    prompt: 'studio banana gold product shot on dark slate',
    aliases: [
      'studio banana gold product shot on dark slate',
      'yellow banana fruit gold studio lighting',
      'clean product render golden crescent turntable',
    ],
    defaultCamera: 'orbit',
    sceneLatents: [0.75, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.3, 0.96],
    evalPixel: (x, y, t) => {
      const tau = t * Math.PI * 2;
      const rotX = x * Math.cos(0.15 * Math.sin(tau)) - y * Math.sin(0.15 * Math.sin(tau));
      const rotY = x * Math.sin(0.15 * Math.sin(tau)) + y * Math.cos(0.15 * Math.sin(tau));
      const curveY = 0.35 * rotX * rotX - 0.15;
      const distCurve = Math.abs(rotY - curveY);
      const bodyMask = Math.abs(rotX) < 0.65 && distCurve < 0.22 * (1 - 0.45 * rotX * rotX);
      if (bodyMask) {
        const highlight = Math.exp(-Math.pow((rotY - (curveY - 0.08)) * 12, 2));
        const r = Math.min(1, 0.86 + 0.14 * highlight);
        const g = Math.min(1, 0.7 + 0.26 * highlight);
        const b = Math.min(1, 0.1 + 0.35 * highlight);
        return [r, g, b];
      }
      const floorGlow = Math.exp(-Math.hypot(x, (y - 0.32) * 2.2) * 3.2) * 0.22;
      const rim = Math.exp(-Math.hypot(x, y) * 1.4) * 0.18;
      return [0.08 + floorGlow + rim, 0.09 + floorGlow * 0.9 + rim * 0.85, 0.13 + rim];
    },
    evalAudio: (t) => {
      const shimmer = 0.5 + 0.5 * Math.cos(t * Math.PI * 2);
      return [0.52, 0.72 + 0.12 * shimmer, 0.04, 0.42 + 0.18 * shimmer, 0.28, 0.06, 0.45, 0.12];
    },
  },
  {
    prompt: 'synthwave retro laser grid sunset',
    aliases: [
      'synthwave retro laser grid sunset',
      'vaporwave neon purple pink horizon grid sun supercar',
      '80s retro outrun sun highway sports car speed',
    ],
    defaultCamera: 'dolly-in',
    sceneLatents: [0.65, 0.98, 0.3, 0.0, 0.35, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95, 0.45, 0.0],
    evalPixel: (x, y, t) => {
      const tau = t * Math.PI * 2;
      const horizon = 0.08;
      if (y < horizon) {
        const sunD = Math.hypot(x, y + 0.22);
        const sky = Math.max(0, Math.min(1, (y + 1) / (horizon + 1)));
        let r = 0.18 + 0.55 * sky;
        let g = 0.05 + 0.15 * sky;
        let b = 0.36 + 0.32 * (1 - sky);
        if (sunD < 0.42) {
          const stripe = y > -0.15 && Math.sin(y * 42 + tau * 0.5) < -0.25 ? 0.25 : 1;
          r = Math.min(1, (0.98 - (y + 0.5) * 0.3) * stripe);
          g = Math.min(1, (0.55 - (y + 0.5) * 0.65) * stripe);
          b = Math.min(1, 0.25 * stripe);
        }
        return [Math.max(0, r), Math.max(0, g), Math.max(0, b)];
      }
      const z = Math.max(0.06, y - horizon);
      const px = x / z;
      const py = 1 / z - ((t * 2) % 1);
      const gridX = Math.exp(-Math.pow(Math.sin(px * Math.PI * 1.2), 2) * 45);
      const gridY = Math.exp(-Math.pow(Math.sin(py * Math.PI * 1.2), 2) * 45);
      const glow = Math.min(1, gridX + gridY);
      return [0.08 + 0.85 * glow, 0.04 + 0.28 * glow, 0.18 + 0.82 * glow];
    },
    evalAudio: (t) => {
      const beat = 0.5 + 0.5 * Math.sin(t * Math.PI * 8);
      return [0.68, 0.78 + 0.14 * beat, 0.1, 0.48 + 0.34 * beat, 0.86, 0.16, 0.6, 0.42];
    },
  },
  {
    prompt: 'bioluminescent forest glowing fireflies',
    aliases: [
      'bioluminescent forest glowing fireflies',
      'magical emerald woods glowing fireflies night nature puppy',
      'enchanted forest trees glowing orbs crickets',
    ],
    defaultCamera: 'crane-up',
    sceneLatents: [0.15, 0.3, 0.25, 0.35, 0.2, 0.0, 0.98, 0.0, 0.0, 0.96, 0.0, 0.3, 0.55, 0.0],
    evalPixel: (x, y, t) => {
      const tau = t * Math.PI * 2;
      const trunk = Math.sin(x * 11 + 0.8) > 0.72 ? 0.04 : 0.12;
      let r = 0.04 + trunk * 0.3;
      let g = 0.11 + trunk * 0.7 + (1 - (y + 1) * 0.4) * 0.08;
      let b = 0.09 + trunk * 0.5;
      for (let i = 0; i < 4; i++) {
        const fx = 0.55 * Math.sin(tau + i * 1.7) + 0.2 * Math.cos(i * 2.9);
        const fy = 0.45 * Math.cos(tau + i * 2.3) - 0.1 * Math.sin(i * 1.4);
        const d = Math.hypot(x - fx, y - fy);
        const orb = Math.exp(-d * d * 42);
        r += 0.65 * orb;
        g += 0.95 * orb;
        b += 0.35 * orb;
      }
      return [Math.min(1, r), Math.min(1, g), Math.min(1, b)];
    },
    evalAudio: (t) => {
      const glow = 0.5 + 0.5 * Math.sin(t * Math.PI * 2);
      return [0.32, 0.74, 0.24, 0.54 + 0.18 * glow, 0.34, 0.68, 0.84, 0.1];
    },
  },
  {
    prompt: 'volcanic lava eruption glowing embers at dusk',
    aliases: [
      'volcanic lava eruption glowing embers at dusk',
      'volcano molten magma crater sparks fire roar dragon',
      'crimson lava flow dark obsidian smoke',
    ],
    defaultCamera: 'crane-up',
    sceneLatents: [0.9, 0.15, 0.1, 0.0, 0.85, 0.0, 0.0, 0.98, 0.0, 0.0, 0.96, 0.75, 0.3, 0.0],
    evalPixel: (x, y, t) => {
      const tau = t * Math.PI * 2;
      const slope = 0.25 + 0.55 * Math.abs(x);
      if (y < slope - 0.15) {
        const plume = Math.exp(-x * x * 9) * (0.5 + 0.5 * Math.sin(y * 8 + tau * 2));
        const ember = Math.pow(
          Math.max(0, Math.sin(x * 24 + tau) * Math.cos(y * 22 + tau * 2)),
          10,
        );
        return [
          Math.min(1, 0.16 + 0.72 * plume + ember),
          Math.min(1, 0.08 + 0.34 * plume + ember * 0.55),
          Math.min(1, 0.14 + 0.08 * plume),
        ];
      }
      const vein = Math.exp(-Math.pow(Math.sin(x * 7 + y * 3 - tau), 2) * 20);
      return [
        Math.min(1, 0.12 + 0.88 * vein),
        Math.min(1, 0.05 + 0.42 * vein),
        Math.min(1, 0.05 + 0.08 * vein),
      ];
    },
    evalAudio: (t) => {
      const surge = 0.5 + 0.5 * Math.sin(t * Math.PI * 2 + 0.8);
      return [0.08, 0.18, 0.72 + 0.2 * surge, 0.24 + 0.26 * surge, 0.36, 0.78, 0.64, 0.94];
    },
  },
  {
    prompt: 'aerial rocky waterfall golden mist sunlight',
    aliases: [
      'aerial rocky waterfall golden mist sunlight',
      'nature canyon river cascading falls mist eagle',
      'drone flyover emerald cliff waterfall',
    ],
    defaultCamera: 'orbit',
    sceneLatents: [0.65, 0.05, 0.0, 0.92, 0.85, 0.0, 0.75, 0.0, 0.15, 0.25, 0.0, 0.6, 0.85, 0.0],
    evalPixel: (x, y, t) => {
      const tau = t * Math.PI * 2;
      const inStream = Math.abs(x + 0.06 * Math.sin(y * 5)) < 0.22;
      if (!inStream) {
        const moss = 0.5 + 0.5 * Math.sin(x * 11 + y * 7);
        return [0.11 + 0.08 * moss, 0.22 + 0.18 * moss, 0.14 + 0.06 * moss];
      }
      const flow = 0.5 + 0.5 * Math.sin(y * 22 - tau * 2 + x * 10);
      const mist = Math.exp(-Math.pow((y - 0.65) * 4, 2));
      return [
        Math.min(1, 0.38 + 0.42 * flow + 0.35 * mist),
        Math.min(1, 0.66 + 0.28 * flow + 0.25 * mist),
        Math.min(1, 0.82 + 0.16 * flow + 0.14 * mist),
      ];
    },
    evalAudio: (t) => {
      const rush = 0.5 + 0.5 * Math.sin(t * Math.PI * 2);
      return [0.2, 0.44, 0.92, 0.62 + 0.16 * rush, 0.2, 0.48, 0.78, 0.6];
    },
  },
];

export const DEFAULT_MULTIMODAL_CORPUS =
  MULTIMODAL_CONCEPTS.flatMap((c) => c.aliases).join('\n') +
  '\nrocket launch orbit space shuttle ocean car sports car drift highway dragon fire dog retriever cat bird eagle astronaut robot boat whale sunset neon rain thunderstorm storm blizzard embers dusk dawn waterfall';

export function resolveCameraMotion(
  prompt: string,
  requested: CameraMotionMode = 'auto',
): Exclude<CameraMotionMode, 'auto'> {
  if (requested !== 'auto') return requested;
  const lower = prompt.toLowerCase();
  if (/dolly|zoom|push|forward|tunnel|warp|vortex|highway|grid/.test(lower)) return 'dolly-in';
  if (/orbit|turntable|rotate|spin|eclipse|banana|planet|waterfall/.test(lower)) return 'orbit';
  if (/pan|horizon|skyline|aurora|mountain|track|dog|cat|car/.test(lower)) return 'pan-right';
  if (/crane|rise|launch|rocket|volcano|forest|tower|lift/.test(lower)) return 'crane-up';
  if (/handheld|rain|street|city|chase|cyber/.test(lower)) return 'handheld';
  return 'dolly-in';
}

function createRng(seed: number): () => number {
  let s = (seed >>> 0) || 4242;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function fillGaussian(arr: Float32Array, std: number, rand: () => number): void {
  for (let i = 0; i < arr.length; i += 2) {
    const u1 = Math.max(1e-7, rand());
    const u2 = rand();
    const mag = std * Math.sqrt(-2.0 * Math.log(u1));
    arr[i] = mag * Math.cos(2 * Math.PI * u2);
    if (i + 1 < arr.length) arr[i + 1] = mag * Math.sin(2 * Math.PI * u2);
  }
}

function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

export interface MultimodalGradientPacket {
  grads: Float32Array[];
  sqErrSum: number;
  sampleCount: number;
  visualSqErrSum?: number;
  visualSampleCount?: number;
  audioSqErrSum?: number;
  audioSampleCount?: number;
}

export class QoweVisionVideoEngine {
  public readonly tokenizer: QoweTokenizer;
  public readonly hyperparams: MultimodalHyperparams;
  public readonly inDim: number;
  public readonly audioInDim: number;
  public readonly audioHiddenDim: number;
  public readonly sceneHiddenDim = 28;
  public step = 0;
  public lastLoss = 0.25;
  public history: MultimodalStepMetric[] = [];

  private textEmb: Float32Array;
  private w1: Float32Array;
  private b1: Float32Array;
  private w2: Float32Array;
  private b2: Float32Array;
  private w3: Float32Array;
  private b3: Float32Array;
  private wOut: Float32Array;
  private bOut: Float32Array;
  private filmGamma: Float32Array;
  private filmBeta: Float32Array;
  private wAud1: Float32Array;
  private bAud1: Float32Array;
  private wAudOut: Float32Array;
  private bAudOut: Float32Array;
  private wScene1: Float32Array;
  private bScene1: Float32Array;
  private wSceneOut: Float32Array;
  private bSceneOut: Float32Array;

  private grads: Record<string, Float32Array> = {};
  private adamM: Record<string, Float32Array> = {};
  private adamV: Record<string, Float32Array> = {};
  private rng: () => number;

  private readonly freqs: Float32Array;
  private readonly promptTokenCache = new Map<string, number[]>();
  private readonly scratchCond: Float32Array;
  private readonly scratchGamma: Float32Array;
  private readonly scratchBeta: Float32Array;
  private readonly scratchDGamma: Float32Array;
  private readonly scratchDBeta: Float32Array;
  private readonly scratchDOutPre = new Float32Array(3);
  private readonly scratchDA3: Float32Array;
  private readonly scratchDZ3: Float32Array;
  private readonly scratchDA2: Float32Array;
  private readonly scratchDA1: Float32Array;
  private readonly scratchDZ2Pre: Float32Array;
  private readonly scratchDCondVec: Float32Array;
  private readonly scratchAudIn: Float32Array;
  private readonly scratchAudH: Float32Array;
  private readonly scratchAudZ: Float32Array;
  private readonly scratchAudOut = new Float32Array(AUDIO_PARAM_DIM);
  private readonly scratchDAudOutPre = new Float32Array(AUDIO_PARAM_DIM);
  private readonly scratchDAudH: Float32Array;
  private readonly scratchSceneZ: Float32Array;
  private readonly scratchSceneH: Float32Array;
  private readonly scratchSceneOut = new Float32Array(SCENE_LATENT_DIM);
  private readonly scratchDSceneOutPre = new Float32Array(SCENE_LATENT_DIM);
  private readonly scratchDSceneH: Float32Array;

  private fwdBatchCap = 0;
  private fwdCoords = new Float32Array(0);
  private fwdTargets = new Float32Array(0);
  private fwdZ1 = new Float32Array(0);
  private fwdA1 = new Float32Array(0);
  private fwdZ2Pre = new Float32Array(0);
  private fwdZ2 = new Float32Array(0);
  private fwdA2 = new Float32Array(0);
  private fwdZ3 = new Float32Array(0);
  private fwdA3 = new Float32Array(0);
  private fwdRgb = new Float32Array(0);

  constructor(
    tokenizer?: QoweTokenizer,
    hyperparams: MultimodalHyperparams = {
      dCond: 24,
      hiddenDim: 80,
      numFreqs: 4,
      learningRate: 0.015,
      totalSteps: 160,
      seed: 2026,
      audioHiddenDim: 32,
    },
  ) {
    this.tokenizer = tokenizer ?? trainTokenizer(DEFAULT_MULTIMODAL_CORPUS, 460);
    this.hyperparams = {
      ...hyperparams,
      audioHiddenDim: hyperparams.audioHiddenDim ?? 32,
    };
    this.inDim = 4 + 6 * this.hyperparams.numFreqs;
    this.audioHiddenDim = this.hyperparams.audioHiddenDim ?? 32;
    this.audioInDim = this.hyperparams.dCond + 5;
    this.rng = createRng(this.hyperparams.seed);

    const V = this.tokenizer.vocab.length;
    const { dCond, hiddenDim: H, numFreqs } = this.hyperparams;
    const A = this.audioHiddenDim;
    const S = this.sceneHiddenDim;

    this.freqs = new Float32Array(numFreqs);
    for (let f = 0; f < numFreqs; f++) {
      this.freqs[f] = Math.pow(1.7, f) * Math.PI;
    }

    this.scratchCond = new Float32Array(dCond);
    this.scratchGamma = new Float32Array(H);
    this.scratchBeta = new Float32Array(H);
    this.scratchDGamma = new Float32Array(H);
    this.scratchDBeta = new Float32Array(H);
    this.scratchDA3 = new Float32Array(H);
    this.scratchDZ3 = new Float32Array(H);
    this.scratchDA2 = new Float32Array(H);
    this.scratchDA1 = new Float32Array(H);
    this.scratchDZ2Pre = new Float32Array(H);
    this.scratchDCondVec = new Float32Array(dCond);
    this.scratchAudIn = new Float32Array(this.audioInDim);
    this.scratchAudH = new Float32Array(A);
    this.scratchAudZ = new Float32Array(A);
    this.scratchDAudH = new Float32Array(A);
    this.scratchSceneZ = new Float32Array(S);
    this.scratchSceneH = new Float32Array(S);
    this.scratchDSceneH = new Float32Array(S);

    this.textEmb = new Float32Array(V * dCond);
    this.w1 = new Float32Array(this.inDim * H);
    this.b1 = new Float32Array(H);
    this.w2 = new Float32Array(H * H);
    this.b2 = new Float32Array(H);
    this.w3 = new Float32Array(H * H);
    this.b3 = new Float32Array(H);
    this.wOut = new Float32Array(H * 3);
    this.bOut = new Float32Array(3);
    this.filmGamma = new Float32Array(dCond * H);
    this.filmBeta = new Float32Array(dCond * H);

    this.wAud1 = new Float32Array(this.audioInDim * A);
    this.bAud1 = new Float32Array(A);
    this.wAudOut = new Float32Array(A * AUDIO_PARAM_DIM);
    this.bAudOut = new Float32Array(AUDIO_PARAM_DIM);

    this.wScene1 = new Float32Array(dCond * S);
    this.bScene1 = new Float32Array(S);
    this.wSceneOut = new Float32Array(S * SCENE_LATENT_DIM);
    this.bSceneOut = new Float32Array(SCENE_LATENT_DIM);

    fillGaussian(this.textEmb, 0.25, this.rng);
    fillGaussian(this.w1, Math.sqrt(2 / this.inDim), this.rng);
    fillGaussian(this.w2, Math.sqrt(2 / H), this.rng);
    fillGaussian(this.w3, Math.sqrt(2 / H), this.rng);
    fillGaussian(this.wOut, Math.sqrt(1 / H), this.rng);
    fillGaussian(this.filmGamma, 0.15, this.rng);
    fillGaussian(this.filmBeta, 0.1, this.rng);
    fillGaussian(this.wAud1, Math.sqrt(2 / this.audioInDim), this.rng);
    fillGaussian(this.wAudOut, Math.sqrt(1 / A), this.rng);
    fillGaussian(this.wScene1, Math.sqrt(2 / dCond), this.rng);
    fillGaussian(this.wSceneOut, Math.sqrt(1 / S), this.rng);

    for (const [name, arr] of this.paramEntries()) {
      this.grads[name] = new Float32Array(arr.length);
      this.adamM[name] = new Float32Array(arr.length);
      this.adamV[name] = new Float32Array(arr.length);
    }
  }

  private ensureBatchBuffers(B: number): void {
    if (this.fwdBatchCap >= B) return;
    const H = this.hyperparams.hiddenDim;
    const D = this.inDim;
    this.fwdBatchCap = B;
    this.fwdCoords = new Float32Array(B * D);
    this.fwdTargets = new Float32Array(B * 3);
    this.fwdZ1 = new Float32Array(B * H);
    this.fwdA1 = new Float32Array(B * H);
    this.fwdZ2Pre = new Float32Array(B * H);
    this.fwdZ2 = new Float32Array(B * H);
    this.fwdA2 = new Float32Array(B * H);
    this.fwdZ3 = new Float32Array(B * H);
    this.fwdA3 = new Float32Array(B * H);
    this.fwdRgb = new Float32Array(B * 3);
  }

  public paramEntries(): [string, Float32Array][] {
    return [
      ['textEmb', this.textEmb],
      ['w1', this.w1],
      ['b1', this.b1],
      ['w2', this.w2],
      ['b2', this.b2],
      ['w3', this.w3],
      ['b3', this.b3],
      ['wOut', this.wOut],
      ['bOut', this.bOut],
      ['filmGamma', this.filmGamma],
      ['filmBeta', this.filmBeta],
      ['wAud1', this.wAud1],
      ['bAud1', this.bAud1],
      ['wAudOut', this.wAudOut],
      ['bAudOut', this.bAudOut],
      ['wScene1', this.wScene1],
      ['bScene1', this.bScene1],
      ['wSceneOut', this.wSceneOut],
      ['bSceneOut', this.bSceneOut],
    ];
  }

  public getWeightArrays(): Float32Array[] {
    return this.paramEntries().map(([, arr]) => arr);
  }

  public setWeightArrays(arrays: Float32Array[]): void {
    const entries = this.paramEntries();
    for (let i = 0; i < entries.length; i++) {
      if (arrays[i] && arrays[i].length === entries[i][1].length) {
        entries[i][1].set(arrays[i]);
      }
    }
  }

  public countParameters(): number {
    return this.paramEntries().reduce((sum, [, arr]) => sum + arr.length, 0);
  }

  private encodeFeaturesInto(x: number, y: number, t: number, out: Float32Array, off = 0): void {
    const tNorm = t * 2 - 1;
    out[off] = x;
    out[off + 1] = y;
    out[off + 2] = Math.hypot(x, y);
    out[off + 3] = tNorm;
    let idx = off + 4;
    const freqs = this.freqs;
    for (let f = 0; f < freqs.length; f++) {
      const freq = freqs[f];
      out[idx++] = Math.sin(freq * x);
      out[idx++] = Math.cos(freq * x);
      out[idx++] = Math.sin(freq * y);
      out[idx++] = Math.cos(freq * y);
      out[idx++] = Math.sin(freq * tNorm);
      out[idx++] = Math.cos(freq * tNorm);
    }
  }

  private poolPrompt(prompt: string): { tokenIds: number[]; cond: Float32Array } {
    const key = prompt.toLowerCase();
    let valid = this.promptTokenCache.get(key);
    if (!valid) {
      const ids = encode(this.tokenizer, key);
      valid = ids.length > 0 ? ids : [0];
      if (this.promptTokenCache.size < 256) {
        this.promptTokenCache.set(key, valid);
      }
    }
    const dCond = this.hyperparams.dCond;
    const cond = this.scratchCond;
    cond.fill(0);
    const invN = 1 / valid.length;
    for (let k = 0; k < valid.length; k++) {
      const off = valid[k] * dCond;
      if (off + dCond <= this.textEmb.length) {
        for (let d = 0; d < dCond; d++) {
          cond[d] += this.textEmb[off + d] * invN;
        }
      }
    }
    return { tokenIds: valid, cond };
  }

  private computeFilm(cond: Float32Array): { gamma: Float32Array; beta: Float32Array } {
    const { dCond, hiddenDim: H } = this.hyperparams;
    const gamma = this.scratchGamma;
    const beta = this.scratchBeta;
    gamma.fill(1);
    beta.fill(0);
    for (let d = 0; d < dCond; d++) {
      const c = cond[d];
      const off = d * H;
      for (let h = 0; h < H; h++) {
        gamma[h] += c * this.filmGamma[off + h];
        beta[h] += c * this.filmBeta[off + h];
      }
    }
    return { gamma, beta };
  }

  private forwardSceneLatents(cond: Float32Array): Float32Array {
    const dCond = this.hyperparams.dCond;
    const S = this.sceneHiddenDim;
    const z = this.scratchSceneZ;
    const h = this.scratchSceneH;
    for (let j = 0; j < S; j++) {
      let sum = this.bScene1[j];
      for (let d = 0; d < dCond; d++) {
        sum += cond[d] * this.wScene1[d * S + j];
      }
      z[j] = sum;
      h[j] = sum > 0 ? sum : 0.08 * sum;
    }

    const out = this.scratchSceneOut;
    for (let k = 0; k < SCENE_LATENT_DIM; k++) {
      let sum = this.bSceneOut[k];
      for (let j = 0; j < S; j++) {
        sum += h[j] * this.wSceneOut[j * SCENE_LATENT_DIM + k];
      }
      out[k] = sigmoid(sum);
    }
    return out;
  }

  public predictSceneLatents(prompt: string): Float32Array {
    const { cond } = this.poolPrompt(prompt);
    const pred = new Float32Array(this.forwardSceneLatents(cond));

    // Gently refine out-of-distribution prompt attributes so arbitrary prompts compose cleanly
    const lower = prompt.toLowerCase();
    const boost = (idx: number, val: number) => {
      pred[idx] = Math.max(pred[idx], val);
    };
    const damp = (idx: number, val: number) => {
      pred[idx] = Math.min(pred[idx], val);
    };

    if (/\b(sunset|sunrise|golden|warm|fire|lava|desert)\b/.test(lower)) boost(0, 0.9);
    if (/\b(neon|cyber|cyberpunk|synthwave|laser|night city|magenta|cyan)\b/.test(lower)) boost(1, 0.92);
    if (/\b(space|galaxy|cosmos|nebula|planet|moon|star|eclipse|rocket|astronaut)\b/.test(lower)) {
      boost(2, 0.9);
      damp(3, 0.1);
    }
    if (/\b(ocean|sea|water|beach|wave|river|waterfall|lake|boat|whale)\b/.test(lower)) boost(3, 0.92);
    if (/\b(mountain|alpine|peak|snow|cliff|ridge|aurora)\b/.test(lower)) boost(4, 0.88);
    if (/\b(city|skyline|street|tower|downtown|urban|cyberpunk)\b/.test(lower)) boost(5, 0.9);
    if (/\b(forest|woods|trees|jungle|nature|fireflies)\b/.test(lower)) boost(6, 0.9);
    if (/\b(volcano|lava|magma|eruption)\b/.test(lower)) {
      boost(7, 0.95);
      boost(10, 0.9);
    }
    if (/\b(rain|storm|drizzle|monsoon|wet)\b/.test(lower)) {
      boost(8, 0.92);
      boost(3, 0.65);
    }
    if (/\b(fireflies|glowing orbs|magic|fairy|snow)\b/.test(lower)) boost(9, 0.88);
    if (/\b(ember|spark|rocket|dragon|fire|explosion)\b/.test(lower)) boost(10, 0.82);
    if (/\b(drift|speed|race|chase|launch|fly|run|gallop|fast)\b/.test(lower)) boost(11, 0.88);

    return pred;
  }

  private forwardAudioAt(cond: Float32Array, t: number): Float32Array {
    const dCond = this.hyperparams.dCond;
    const A = this.audioHiddenDim;
    const audIn = this.scratchAudIn;
    for (let d = 0; d < dCond; d++) audIn[d] = cond[d];
    const tau = t * Math.PI * 2;
    audIn[dCond] = Math.sin(tau);
    audIn[dCond + 1] = Math.cos(tau);
    audIn[dCond + 2] = Math.sin(2 * tau);
    audIn[dCond + 3] = Math.cos(2 * tau);
    audIn[dCond + 4] = t * 2 - 1;

    const z = this.scratchAudZ;
    const h = this.scratchAudH;
    for (let j = 0; j < A; j++) {
      let sum = this.bAud1[j];
      for (let i = 0; i < this.audioInDim; i++) {
        sum += audIn[i] * this.wAud1[i * A + j];
      }
      z[j] = sum;
      h[j] = sum > 0 ? sum : 0.08 * sum;
    }

    const out = this.scratchAudOut;
    for (let k = 0; k < AUDIO_PARAM_DIM; k++) {
      let sum = this.bAudOut[k];
      for (let j = 0; j < A; j++) {
        sum += h[j] * this.wAudOut[j * AUDIO_PARAM_DIM + k];
      }
      out[k] = sigmoid(sum);
    }
    return out;
  }

  private forwardBatch(B: number, gamma: Float32Array, beta: Float32Array): void {
    const H = this.hyperparams.hiddenDim;
    const D = this.inDim;

    const coords = this.fwdCoords;
    const z1 = this.fwdZ1;
    const a1 = this.fwdA1;
    const z2Pre = this.fwdZ2Pre;
    const z2 = this.fwdZ2;
    const a2 = this.fwdA2;
    const z3 = this.fwdZ3;
    const a3 = this.fwdA3;
    const rgb = this.fwdRgb;

    const w1 = this.w1;
    const b1 = this.b1;
    const w2 = this.w2;
    const b2 = this.b2;
    const w3 = this.w3;
    const b3 = this.b3;
    const wOut = this.wOut;
    const bOut = this.bOut;

    for (let b = 0; b < B; b++) {
      const fOff = b * D;
      const hOff = b * H;

      for (let j = 0; j < H; j++) {
        let s = b1[j];
        for (let i = 0; i < D; i++) s += coords[fOff + i] * w1[i * H + j];
        z1[hOff + j] = s;
        a1[hOff + j] = s > 0 ? s : 0.05 * s;
      }

      for (let j = 0; j < H; j++) {
        let s = b2[j];
        for (let i = 0; i < H; i++) s += a1[hOff + i] * w2[i * H + j];
        z2Pre[hOff + j] = s;
        const mod = s * gamma[j] + beta[j];
        z2[hOff + j] = mod;
        a2[hOff + j] = mod > 0 ? mod : 0.05 * mod;
      }

      for (let j = 0; j < H; j++) {
        let s = b3[j];
        for (let i = 0; i < H; i++) s += a2[hOff + i] * w3[i * H + j];
        const res = s + a1[hOff + j];
        z3[hOff + j] = res;
        a3[hOff + j] = res > 0 ? res : 0.05 * res;
      }

      const oOff = b * 3;
      for (let c = 0; c < 3; c++) {
        let s = bOut[c];
        for (let i = 0; i < H; i++) s += a3[hOff + i] * wOut[i * 3 + c];
        rgb[oOff + c] = sigmoid(s);
      }
    }
  }

  public computeBatchGradients(samplesPerConcept = 24, stepSeed?: number): MultimodalGradientPacket {
    const entries = this.paramEntries();
    for (let i = 0; i < entries.length; i++) {
      this.grads[entries[i][0]].fill(0);
    }

    this.ensureBatchBuffers(samplesPerConcept);

    const rand = stepSeed !== undefined ? createRng(stepSeed) : this.rng;
    const H = this.hyperparams.hiddenDim;
    const D = this.inDim;
    const dCond = this.hyperparams.dCond;
    const A = this.audioHiddenDim;
    const S = this.sceneHiddenDim;
    const totalSamples = MULTIMODAL_CONCEPTS.length * samplesPerConcept;
    const invTotalChannels = 2 / (totalSamples * 3);
    const audioSamplesPerConcept = 6;
    const invTotalAudio = 1.5 / (MULTIMODAL_CONCEPTS.length * audioSamplesPerConcept * AUDIO_PARAM_DIM);
    const invTotalScene = 1.4 / (MULTIMODAL_CONCEPTS.length * SCENE_LATENT_DIM);

    let visualSqErr = 0;
    let audioSqErr = 0;

    const coords = this.fwdCoords;
    const targets = this.fwdTargets;
    const dGamma = this.scratchDGamma;
    const dBeta = this.scratchDBeta;
    const dOutPre = this.scratchDOutPre;
    const dA3 = this.scratchDA3;
    const dZ3 = this.scratchDZ3;
    const dA2 = this.scratchDA2;
    const dA1 = this.scratchDA1;
    const dZ2Pre = this.scratchDZ2Pre;
    const dCondVec = this.scratchDCondVec;
    const dAudOutPre = this.scratchDAudOutPre;
    const dAudH = this.scratchDAudH;
    const dSceneOutPre = this.scratchDSceneOutPre;
    const dSceneH = this.scratchDSceneH;

    const z1 = this.fwdZ1;
    const a1 = this.fwdA1;
    const z2Pre = this.fwdZ2Pre;
    const z2 = this.fwdZ2;
    const a2 = this.fwdA2;
    const z3 = this.fwdZ3;
    const a3 = this.fwdA3;
    const rgb = this.fwdRgb;

    for (let cIdx = 0; cIdx < MULTIMODAL_CONCEPTS.length; cIdx++) {
      const concept = MULTIMODAL_CONCEPTS[cIdx];
      const aliasIdx = Math.floor(rand() * concept.aliases.length);
      const phrase = concept.aliases[aliasIdx] ?? concept.prompt;
      const { tokenIds, cond } = this.poolPrompt(phrase);
      const { gamma, beta } = this.computeFilm(cond);

      for (let s = 0; s < samplesPerConcept; s++) {
        const x = rand() * 2 - 1;
        const y = rand() * 2 - 1;
        const t = rand();
        this.encodeFeaturesInto(x, y, t, coords, s * D);
        const [tr, tg, tb] = concept.evalPixel(x, y, t);
        const tOff = s * 3;
        targets[tOff] = tr;
        targets[tOff + 1] = tg;
        targets[tOff + 2] = tb;
      }

      this.forwardBatch(samplesPerConcept, gamma, beta);
      dGamma.fill(0);
      dBeta.fill(0);

      for (let b = 0; b < samplesPerConcept; b++) {
        const hOff = b * H;
        const fOff = b * D;
        const oOff = b * 3;

        for (let c = 0; c < 3; c++) {
          const pred = rgb[oOff + c];
          const diff = pred - targets[oOff + c];
          visualSqErr += diff * diff;
          const dop = diff * invTotalChannels * pred * (1 - pred);
          dOutPre[c] = dop;
          this.grads.bOut[c] += dop;
        }

        for (let i = 0; i < H; i++) {
          let sum = 0;
          const ai3 = a3[hOff + i];
          const wOff = i * 3;
          for (let c = 0; c < 3; c++) {
            const dop = dOutPre[c];
            sum += dop * this.wOut[wOff + c];
            this.grads.wOut[wOff + c] += ai3 * dop;
          }
          dA3[i] = sum;
        }

        for (let j = 0; j < H; j++) {
          const dz = dA3[j] * (z3[hOff + j] > 0 ? 1 : 0.05);
          dZ3[j] = dz;
          this.grads.b3[j] += dz;
        }

        for (let i = 0; i < H; i++) {
          dA1[i] = dZ3[i];
          let sum = 0;
          const ai2 = a2[hOff + i];
          const wOff = i * H;
          for (let j = 0; j < H; j++) {
            const dz = dZ3[j];
            sum += dz * this.w3[wOff + j];
            this.grads.w3[wOff + j] += ai2 * dz;
          }
          dA2[i] = sum;
        }

        for (let j = 0; j < H; j++) {
          const dMod = dA2[j] * (z2[hOff + j] > 0 ? 1 : 0.05);
          dGamma[j] += dMod * z2Pre[hOff + j];
          dBeta[j] += dMod;
          const dz2p = dMod * gamma[j];
          dZ2Pre[j] = dz2p;
          this.grads.b2[j] += dz2p;
        }

        for (let i = 0; i < H; i++) {
          let sum = 0;
          const ai1 = a1[hOff + i];
          const wOff = i * H;
          for (let j = 0; j < H; j++) {
            const dz = dZ2Pre[j];
            sum += dz * this.w2[wOff + j];
            this.grads.w2[wOff + j] += ai1 * dz;
          }
          dA1[i] += sum;
        }

        for (let j = 0; j < H; j++) {
          const dZ1 = dA1[j] * (z1[hOff + j] > 0 ? 1 : 0.05);
          this.grads.b1[j] += dZ1;
          for (let i = 0; i < D; i++) {
            this.grads.w1[i * H + j] += coords[fOff + i] * dZ1;
          }
        }
      }

      for (let d = 0; d < dCond; d++) {
        const off = d * H;
        let dc = 0;
        const cd = cond[d];
        for (let h = 0; h < H; h++) {
          dc += dGamma[h] * this.filmGamma[off + h] + dBeta[h] * this.filmBeta[off + h];
          this.grads.filmGamma[off + h] += cd * dGamma[h];
          this.grads.filmBeta[off + h] += cd * dBeta[h];
        }
        dCondVec[d] = dc;
      }

      // Joint Compositional Scene-Latent Predictor backprop
      const predLatents = this.forwardSceneLatents(cond);
      for (let k = 0; k < SCENE_LATENT_DIM; k++) {
        const p = predLatents[k];
        const diff = p - concept.sceneLatents[k];
        const dsp = diff * invTotalScene * p * (1 - p);
        dSceneOutPre[k] = dsp;
        this.grads.bSceneOut[k] += dsp;
      }
      for (let j = 0; j < S; j++) {
        let sum = 0;
        const hj = this.scratchSceneH[j];
        const wOff = j * SCENE_LATENT_DIM;
        for (let k = 0; k < SCENE_LATENT_DIM; k++) {
          const dsp = dSceneOutPre[k];
          sum += dsp * this.wSceneOut[wOff + k];
          this.grads.wSceneOut[wOff + k] += hj * dsp;
        }
        dSceneH[j] = sum * (this.scratchSceneZ[j] > 0 ? 1 : 0.08);
        this.grads.bScene1[j] += dSceneH[j];
      }
      for (let d = 0; d < dCond; d++) {
        const cd = cond[d];
        const wOff = d * S;
        let dc = 0;
        for (let j = 0; j < S; j++) {
          const dz = dSceneH[j];
          this.grads.wScene1[wOff + j] += cd * dz;
          dc += dz * this.wScene1[wOff + j];
        }
        dCondVec[d] += dc;
      }

      // Joint Neural Audio Head Training across time slices
      for (let s = 0; s < audioSamplesPerConcept; s++) {
        const at = (s + rand()) / audioSamplesPerConcept;
        const predAudio = this.forwardAudioAt(cond, at);
        const targetAudio = concept.evalAudio(at);

        for (let k = 0; k < AUDIO_PARAM_DIM; k++) {
          const p = predAudio[k];
          const diff = p - targetAudio[k];
          audioSqErr += diff * diff;
          const dap = diff * invTotalAudio * p * (1 - p);
          dAudOutPre[k] = dap;
          this.grads.bAudOut[k] += dap;
        }

        for (let j = 0; j < A; j++) {
          let sum = 0;
          const hj = this.scratchAudH[j];
          const wOff = j * AUDIO_PARAM_DIM;
          for (let k = 0; k < AUDIO_PARAM_DIM; k++) {
            const dap = dAudOutPre[k];
            sum += dap * this.wAudOut[wOff + k];
            this.grads.wAudOut[wOff + k] += hj * dap;
          }
          dAudH[j] = sum * (this.scratchAudZ[j] > 0 ? 1 : 0.08);
          this.grads.bAud1[j] += dAudH[j];
        }

        for (let i = 0; i < this.audioInDim; i++) {
          const xi = this.scratchAudIn[i];
          const wOff = i * A;
          let dc = 0;
          for (let j = 0; j < A; j++) {
            const dz = dAudH[j];
            this.grads.wAud1[wOff + j] += xi * dz;
            if (i < dCond) dc += dz * this.wAud1[wOff + j];
          }
          if (i < dCond) dCondVec[i] += dc;
        }
      }

      const invN = 1 / tokenIds.length;
      for (let k = 0; k < tokenIds.length; k++) {
        const off = tokenIds[k] * dCond;
        if (off + dCond <= this.grads.textEmb.length) {
          for (let d = 0; d < dCond; d++) {
            this.grads.textEmb[off + d] += dCondVec[d] * invN;
          }
        }
      }
    }

    const visualSampleCount = totalSamples * 3;
    const audioSampleCount = MULTIMODAL_CONCEPTS.length * audioSamplesPerConcept * AUDIO_PARAM_DIM;

    return {
      grads: entries.map(([name]) => new Float32Array(this.grads[name])),
      sqErrSum: visualSqErr + audioSqErr * 0.4,
      sampleCount: visualSampleCount + audioSampleCount,
      visualSqErrSum: visualSqErr,
      visualSampleCount,
      audioSqErrSum: audioSqErr,
      audioSampleCount,
    };
  }

  public applyExternalGradientsAndStep(
    averagedGrads: Float32Array[],
    sqErrSum: number,
    sampleCount: number,
    visualSqErrSum?: number,
    visualSampleCount?: number,
    audioSqErrSum?: number,
    audioSampleCount?: number,
  ): MultimodalStepMetric {
    const entries = this.paramEntries();
    for (let idx = 0; idx < entries.length; idx++) {
      if (averagedGrads[idx] && averagedGrads[idx].length === this.grads[entries[idx][0]].length) {
        this.grads[entries[idx][0]].set(averagedGrads[idx]);
      }
    }

    const visualMse =
      visualSqErrSum !== undefined && visualSampleCount
        ? visualSqErrSum / Math.max(1, visualSampleCount)
        : undefined;
    const audioMse =
      audioSqErrSum !== undefined && audioSampleCount
        ? audioSqErrSum / Math.max(1, audioSampleCount)
        : undefined;
    const mse = visualMse ?? sqErrSum / Math.max(1, sampleCount);

    this.step += 1;

    const progress = Math.min(1, this.step / Math.max(1, this.hyperparams.totalSteps));
    const lr = this.hyperparams.learningRate * (0.2 + 0.8 * 0.5 * (1 + Math.cos(Math.PI * progress)));
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;
    const bc1 = 1 - Math.pow(beta1, this.step);
    const bc2 = 1 - Math.pow(beta2, this.step);
    const invBc1 = 1 / bc1;
    const invBc2 = 1 / bc2;

    for (let e = 0; e < entries.length; e++) {
      const [name, w] = entries[e];
      const g = this.grads[name];
      const m = this.adamM[name];
      const v = this.adamV[name];
      for (let i = 0; i < w.length; i++) {
        let grad = g[i];
        if (grad > 2) grad = 2;
        else if (grad < -2) grad = -2;
        const mi = beta1 * m[i] + 0.1 * grad;
        const vi = beta2 * v[i] + 0.001 * grad * grad;
        m[i] = mi;
        v[i] = vi;
        w[i] -= (lr * (mi * invBc1)) / (Math.sqrt(vi * invBc2) + eps);
      }
    }

    const psnr = 10 * Math.log10(1 / Math.max(1e-6, mse));
    this.lastLoss = mse;
    const metric: MultimodalStepMetric = {
      step: this.step,
      loss: mse,
      psnr,
      visualLoss: visualMse,
      audioLoss: audioMse,
    };
    this.history.push(metric);
    return metric;
  }

  public trainStep(): MultimodalStepMetric {
    const pkt = this.computeBatchGradients(28);
    return this.applyExternalGradientsAndStep(
      pkt.grads,
      pkt.sqErrSum,
      pkt.sampleCount,
      pkt.visualSqErrSum,
      pkt.visualSampleCount,
      pkt.audioSqErrSum,
      pkt.audioSampleCount,
    );
  }

  public predictAudioTrack(
    prompt: string,
    times: number[],
    cameraMotion: CameraMotionMode = 'auto',
  ): NeuralAudioFrame[] {
    const { cond } = this.poolPrompt(prompt);
    const cam = resolveCameraMotion(prompt, cameraMotion);
    const latents = this.predictSceneLatents(prompt);
    const parsed = parseCinemaPrompt(prompt);
    const hasEngine = parsed.actors.includes('car') || parsed.actors.includes('rocket');
    const tracks: NeuralAudioFrame[] = new Array(times.length);

    for (let i = 0; i < times.length; i++) {
      const t = times[i];
      const out = this.forwardAudioAt(cond, t);
      const camWidthBoost =
        cam === 'orbit'
          ? 0.14
          : cam === 'pan-right'
            ? 0.12
            : cam === 'handheld'
              ? 0.08
              : 0;

      const rainBoost = latents[8] * 0.45;
      const engineBoost = hasEngine ? 0.42 : latents[7] * 0.35;

      tracks[i] = [
        out[0],
        out[1],
        Math.min(1, Math.max(out[2], rainBoost + latents[3] * 0.25)),
        out[3],
        Math.min(1, Math.max(out[4], latents[11] * 0.65)),
        Math.min(1, Math.max(out[5], rainBoost * 0.85 + latents[9] * 0.35)),
        Math.min(1, out[6] + camWidthBoost),
        Math.min(1, Math.max(out[7], engineBoost)),
      ];
    }
    return tracks;
  }

  /**
   * Evaluates Qowe's trained FiLM-conditioned Spatiotemporal Neural Field over a fast
   * atmospheric lighting grid for a batch of frame times.
   */
  private evalNeuralFieldGrid(
    cond: Float32Array,
    gridW: number,
    gridH: number,
    times: number[],
  ): Float32Array[] {
    const { gamma, beta } = this.computeFilm(cond);
    const H = this.hyperparams.hiddenDim;
    const numFreqs = this.hyperparams.numFreqs;
    const N = gridW * gridH;

    const spatialZ1 = new Float32Array(N * H);
    const spatialFeat = new Float32Array(3 + numFreqs * 4);
    const w1 = this.w1;
    const b1 = this.b1;
    const w2 = this.w2;
    const b2 = this.b2;
    const w3 = this.w3;
    const b3 = this.b3;
    const wOut = this.wOut;
    const bOut = this.bOut;
    const freqs = this.freqs;

    let pIdx = 0;
    for (let py = 0; py < gridH; py++) {
      const ny = (py / Math.max(1, gridH - 1)) * 2 - 1;
      for (let px = 0; px < gridW; px++) {
        const nx = (px / Math.max(1, gridW - 1)) * 2 - 1;
        spatialFeat[0] = nx;
        spatialFeat[1] = ny;
        spatialFeat[2] = Math.hypot(nx, ny);
        let sfIdx = 3;
        for (let f = 0; f < numFreqs; f++) {
          const fq = freqs[f];
          spatialFeat[sfIdx++] = Math.sin(fq * nx);
          spatialFeat[sfIdx++] = Math.cos(fq * nx);
          spatialFeat[sfIdx++] = Math.sin(fq * ny);
          spatialFeat[sfIdx++] = Math.cos(fq * ny);
        }

        const outOff = pIdx * H;
        for (let j = 0; j < H; j++) {
          let sum =
            b1[j] +
            spatialFeat[0] * w1[j] +
            spatialFeat[1] * w1[H + j] +
            spatialFeat[2] * w1[2 * H + j];
          let sCursor = 3;
          for (let f = 0; f < numFreqs; f++) {
            const base = (4 + f * 6) * H + j;
            sum +=
              spatialFeat[sCursor++] * w1[base] +
              spatialFeat[sCursor++] * w1[base + H] +
              spatialFeat[sCursor++] * w1[base + 2 * H] +
              spatialFeat[sCursor++] * w1[base + 3 * H];
          }
          spatialZ1[outOff + j] = sum;
        }
        pIdx++;
      }
    }

    const tempBias = new Float32Array(H);
    const a1 = new Float32Array(H);
    const a2 = new Float32Array(H);
    const a3 = new Float32Array(H);

    const grids: Float32Array[] = [];
    for (let fIdx = 0; fIdx < times.length; fIdx++) {
      const tNorm = times[fIdx] * 2 - 1;
      for (let j = 0; j < H; j++) {
        let tb = tNorm * w1[3 * H + j];
        for (let f = 0; f < numFreqs; f++) {
          const base = (4 + f * 6 + 4) * H + j;
          const fq = freqs[f];
          tb += Math.sin(fq * tNorm) * w1[base] + Math.cos(fq * tNorm) * w1[base + H];
        }
        tempBias[j] = tb;
      }

      const rgbGrid = new Float32Array(N * 3);
      for (let p = 0; p < N; p++) {
        const sOff = p * H;
        for (let j = 0; j < H; j++) {
          const z = spatialZ1[sOff + j] + tempBias[j];
          a1[j] = z > 0 ? z : 0.05 * z;
        }

        for (let j = 0; j < H; j++) {
          let s = b2[j];
          for (let i = 0; i < H; i++) s += a1[i] * w2[i * H + j];
          const mod = s * gamma[j] + beta[j];
          a2[j] = mod > 0 ? mod : 0.05 * mod;
        }

        for (let j = 0; j < H; j++) {
          let s = b3[j];
          for (let i = 0; i < H; i++) s += a2[i] * w3[i * H + j];
          const res = s + a1[j];
          a3[j] = res > 0 ? res : 0.05 * res;
        }

        const rOff = p * 3;
        for (let c = 0; c < 3; c++) {
          let s = bOut[c];
          for (let i = 0; i < H; i++) s += a3[i] * wOut[i * 3 + c];
          rgbGrid[rOff + c] = sigmoid(s);
        }
      }
      grids.push(rgbGrid);
    }

    return grids;
  }

  public renderFramesBatch(
    prompt: string,
    width: number,
    height: number,
    times: number[],
    cameraMotion: CameraMotionMode = 'auto',
  ): Uint8ClampedArray[] {
    const { cond } = this.poolPrompt(prompt);
    const resolvedCam = times.length <= 1 ? 'static' : resolveCameraMotion(prompt, cameraMotion);
    const latents = this.predictSceneLatents(prompt);
    const parsed = parseCinemaPrompt(prompt);

    const gridW = Math.min(width, width > height ? 64 : 48);
    const gridH = Math.min(height, width > height ? 36 : 48);
    const neuralGrids = this.evalNeuralFieldGrid(cond, gridW, gridH, times);

    // Neural field confidence scales smoothly with training steps completed
    const neuralBlendWeight = Math.min(0.62, 0.25 + Math.min(1, this.step / 100) * 0.37);

    const results: Uint8ClampedArray[] = new Array(times.length);
    for (let fIdx = 0; fIdx < times.length; fIdx++) {
      const outRgba = new Uint8ClampedArray(width * height * 4);
      composeCinematicFrame({
        width,
        height,
        t: times[fIdx],
        cameraMotion: resolvedCam,
        neuralGrid: neuralGrids[fIdx],
        gridW,
        gridH,
        neuralBlendWeight,
        latents,
        parsed,
        outRgba,
      });
      results[fIdx] = outRgba;
    }

    return results;
  }

  public renderFrame(prompt: string, width: number, height: number, t: number): Uint8ClampedArray {
    return this.renderFramesBatch(prompt, width, height, [t], 'static')[0];
  }

  public generateImage(prompt: string, size = 128): GeneratedMediaFrames {
    const frames = this.renderFramesBatch(prompt, size, size, [0.35], 'static');
    return { width: size, height: size, fps: 1, frames };
  }

  public generateVideo(
    prompt: string,
    size = 80,
    numFrames = 60,
    fps = 60,
    cameraMotion: CameraMotionMode = 'auto',
  ): GeneratedMediaFrames {
    const times: number[] = new Array(numFrames);
    for (let f = 0; f < numFrames; f++) {
      times[f] = f / Math.max(1, numFrames);
    }
    const cam = resolveCameraMotion(prompt, cameraMotion);
    const frames = this.renderFramesBatch(prompt, size, size, times, cam);
    const audioTrack = this.predictAudioTrack(prompt, times, cam);
    return { width: size, height: size, fps, frames, audioTrack, cameraMotion: cam };
  }

  public predictCinemaPayload(
    prompt: string,
    options: {
      aspectRatio?: '16:9' | '1:1';
      durationSec?: number;
      fps?: number;
      cameraMotion?: CameraMotionMode;
      keyframeCount?: number;
      denoisingSteps?: number;
    } = {},
  ): { payload: LtxCinemaPayload; audioTrack: NeuralAudioFrame[] } {
    const aspectRatio = options.aspectRatio ?? '16:9';
    const durationSec = Math.max(2, Math.min(8, options.durationSec ?? 4));
    const fps = options.fps ?? 60;
    const keyframeCount = Math.max(4, options.keyframeCount ?? 8);
    const keyframeW = aspectRatio === '16:9' ? 48 : 36;
    const keyframeH = aspectRatio === '16:9' ? 27 : 36;
    const denoisingSteps = options.denoisingSteps ?? 8;

    const { cond } = this.poolPrompt(prompt);
    const cam = resolveCameraMotion(prompt, options.cameraMotion ?? 'auto');
    const latents = this.predictSceneLatents(prompt);

    const kfTimes = Array.from({ length: keyframeCount }, (_, i) => i / Math.max(1, keyframeCount - 1));
    const baseKeyframes = this.evalNeuralFieldGrid(cond, keyframeW, keyframeH, kfTimes);

    const totalAudioSlices = Math.min(240, Math.round(durationSec * fps));
    const audioTimes = Array.from({ length: totalAudioSlices }, (_, i) => i / totalAudioSlices);
    const audioTrack = this.predictAudioTrack(prompt, audioTimes, cam);

    const { denoisedKeyframes, flowConsistencyScore } = runLtx25DualStreamFlow({
      baseKeyframes,
      keyframeW,
      keyframeH,
      sceneLatents: latents,
      audioTrack,
      config: { denoisingSteps },
    });

    const neuralBlendWeight = Math.min(0.68, 0.25 + Math.min(1, this.step / 100) * 0.43);

    return {
      payload: {
        prompt,
        aspectRatio,
        durationSec,
        fps,
        cameraMotion: cam,
        sceneLatents: Array.from(latents),
        neuralKeyframes: denoisedKeyframes,
        keyframeW,
        keyframeH,
        neuralBlendWeight,
        trainingStep: this.step,
        denoisingSteps,
        flowConsistencyScore,
      },
      audioTrack,
    };
  }

  public generateCinematicVideo(
    prompt: string,
    options: {
      width?: number;
      height?: number;
      numFrames?: number;
      fps?: number;
      cameraMotion?: CameraMotionMode;
    } = {},
  ): GeneratedMediaFrames {
    const width = options.width ?? 320;
    const height = options.height ?? 180;
    const numFrames = options.numFrames ?? 120;
    const fps = options.fps ?? 60;
    const times: number[] = new Array(numFrames);
    for (let f = 0; f < numFrames; f++) {
      times[f] = f / Math.max(1, numFrames);
    }
    const cam = resolveCameraMotion(prompt, options.cameraMotion ?? 'auto');
    const frames = this.renderFramesBatch(prompt, width, height, times, cam);
    const audioTrack = this.predictAudioTrack(prompt, times, cam);
    return { width, height, fps, frames, audioTrack, cameraMotion: cam };
  }

  public exportCheckpoint(): MultimodalCheckpoint {
    return {
      version: 'qowe-veo-3.0-scratch',
      step: this.step,
      loss: this.lastLoss,
      hyperparams: { ...this.hyperparams },
      tokenizer: this.tokenizer,
      weights: {
        textEmb: Array.from(this.textEmb),
        w1: Array.from(this.w1),
        b1: Array.from(this.b1),
        w2: Array.from(this.w2),
        b2: Array.from(this.b2),
        w3: Array.from(this.w3),
        b3: Array.from(this.b3),
        wOut: Array.from(this.wOut),
        bOut: Array.from(this.bOut),
        filmGamma: Array.from(this.filmGamma),
        filmBeta: Array.from(this.filmBeta),
        wAud1: Array.from(this.wAud1),
        bAud1: Array.from(this.bAud1),
        wAudOut: Array.from(this.wAudOut),
        bAudOut: Array.from(this.bAudOut),
        wScene1: Array.from(this.wScene1),
        bScene1: Array.from(this.bScene1),
        wSceneOut: Array.from(this.wSceneOut),
        bSceneOut: Array.from(this.bSceneOut),
      },
      history: [...this.history],
    };
  }

  public static fromCheckpoint(cp: MultimodalCheckpoint): QoweVisionVideoEngine {
    const eng = new QoweVisionVideoEngine(cp.tokenizer, cp.hyperparams);
    eng.step = cp.step;
    eng.lastLoss = cp.loss;
    eng.history = [...cp.history];
    eng.textEmb.set(cp.weights.textEmb);
    eng.w1.set(cp.weights.w1);
    eng.b1.set(cp.weights.b1);
    eng.w2.set(cp.weights.w2);
    eng.b2.set(cp.weights.b2);
    eng.w3.set(cp.weights.w3);
    eng.b3.set(cp.weights.b3);
    eng.wOut.set(cp.weights.wOut);
    eng.bOut.set(cp.weights.bOut);
    eng.filmGamma.set(cp.weights.filmGamma);
    eng.filmBeta.set(cp.weights.filmBeta);
    if (cp.weights.wAud1 && cp.weights.wAud1.length === eng.wAud1.length) {
      eng.wAud1.set(cp.weights.wAud1);
    }
    if (cp.weights.bAud1 && cp.weights.bAud1.length === eng.bAud1.length) {
      eng.bAud1.set(cp.weights.bAud1);
    }
    if (cp.weights.wAudOut && cp.weights.wAudOut.length === eng.wAudOut.length) {
      eng.wAudOut.set(cp.weights.wAudOut);
    }
    if (cp.weights.bAudOut && cp.weights.bAudOut.length === eng.bAudOut.length) {
      eng.bAudOut.set(cp.weights.bAudOut);
    }
    if (cp.weights.wScene1 && cp.weights.wScene1.length === eng.wScene1.length) {
      eng.wScene1.set(cp.weights.wScene1);
    }
    if (cp.weights.bScene1 && cp.weights.bScene1.length === eng.bScene1.length) {
      eng.bScene1.set(cp.weights.bScene1);
    }
    if (cp.weights.wSceneOut && cp.weights.wSceneOut.length === eng.wSceneOut.length) {
      eng.wSceneOut.set(cp.weights.wSceneOut);
    }
    if (cp.weights.bSceneOut && cp.weights.bSceneOut.length === eng.bSceneOut.length) {
      eng.bSceneOut.set(cp.weights.bSceneOut);
    }
    return eng;
  }
}
