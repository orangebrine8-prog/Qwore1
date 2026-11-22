import type { SceneActor, VideoSceneSpec } from './intentAndMedia';

function seededRand(seed: number): number {
  const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453123;
  return x - Math.floor(x);
}

function drawRoundedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
}

function drawActorShape(
  ctx: CanvasRenderingContext2D,
  actor: SceneActor,
  phase: number,
  cx: number,
  cy: number,
  size: number,
): void {
  ctx.save();
  ctx.translate(cx, cy);

  if (actor.trail) {
    const grad = ctx.createLinearGradient(-size * 2.2, 0, 0, 0);
    grad.addColorStop(0, 'rgba(255,255,255,0)');
    grad.addColorStop(1, actor.secondaryColor + '88');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(-size * 2.1, -size * 0.14);
    ctx.lineTo(0, -size * 0.28);
    ctx.lineTo(0, size * 0.28);
    ctx.lineTo(-size * 2.1, size * 0.14);
    ctx.closePath();
    ctx.fill();
  }

  switch (actor.type) {
    case 'rocket': {
      ctx.rotate(-0.18 + Math.sin(phase * 2) * 0.04);
      // Exhaust flame
      const flameLen = size * (0.65 + 0.25 * Math.sin(phase * 18));
      const flameGrad = ctx.createLinearGradient(-size * 0.6, 0, -size * 0.6 - flameLen, 0);
      flameGrad.addColorStop(0, '#fef08a');
      flameGrad.addColorStop(0.5, actor.secondaryColor);
      flameGrad.addColorStop(1, 'rgba(249,115,22,0)');
      ctx.fillStyle = flameGrad;
      ctx.beginPath();
      ctx.moveTo(-size * 0.45, -size * 0.16);
      ctx.lineTo(-size * 0.55 - flameLen, 0);
      ctx.lineTo(-size * 0.45, size * 0.16);
      ctx.closePath();
      ctx.fill();

      // Rocket hull
      ctx.fillStyle = actor.color;
      ctx.beginPath();
      ctx.ellipse(0, 0, size * 0.62, size * 0.24, 0, 0, Math.PI * 2);
      ctx.fill();
      // Nosecone
      ctx.fillStyle = actor.secondaryColor;
      ctx.beginPath();
      ctx.moveTo(size * 0.35, -size * 0.19);
      ctx.lineTo(size * 0.78, 0);
      ctx.lineTo(size * 0.35, size * 0.19);
      ctx.closePath();
      ctx.fill();
      // Window
      ctx.fillStyle = '#0ea5e9';
      ctx.beginPath();
      ctx.arc(size * 0.1, 0, size * 0.11, 0, Math.PI * 2);
      ctx.fill();
      break;
    }

    case 'car': {
      // Headlight beam
      const beam = ctx.createLinearGradient(size * 0.55, 0, size * 2.3, 0);
      beam.addColorStop(0, '#fef08a99');
      beam.addColorStop(1, 'rgba(254,240,138,0)');
      ctx.fillStyle = beam;
      ctx.beginPath();
      ctx.moveTo(size * 0.55, -size * 0.04);
      ctx.lineTo(size * 2.3, -size * 0.28);
      ctx.lineTo(size * 2.3, size * 0.28);
      ctx.lineTo(size * 0.55, size * 0.1);
      ctx.closePath();
      ctx.fill();

      // Chassis
      ctx.fillStyle = actor.color;
      drawRoundedRect(ctx, -size * 0.65, -size * 0.15, size * 1.3, size * 0.34, size * 0.08);
      ctx.fill();
      // Cabin
      ctx.fillStyle = actor.secondaryColor;
      ctx.beginPath();
      ctx.moveTo(-size * 0.35, -size * 0.15);
      ctx.lineTo(-size * 0.15, -size * 0.38);
      ctx.lineTo(size * 0.24, -size * 0.38);
      ctx.lineTo(size * 0.42, -size * 0.15);
      ctx.closePath();
      ctx.fill();
      // Wheels
      ctx.fillStyle = '#020617';
      for (const wx of [-size * 0.36, size * 0.36]) {
        ctx.beginPath();
        ctx.arc(wx, size * 0.2, size * 0.14, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = actor.secondaryColor;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      break;
    }

    case 'cat': {
      const bob = Math.sin(phase * 4) * size * 0.04;
      ctx.translate(0, bob);
      // Tail
      ctx.strokeStyle = actor.color;
      ctx.lineWidth = size * 0.1;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(-size * 0.28, size * 0.22);
      ctx.quadraticCurveTo(
        -size * 0.65,
        -size * 0.1 + Math.sin(phase * 6) * size * 0.2,
        -size * 0.52,
        -size * 0.34,
      );
      ctx.stroke();
      // Body + Head
      ctx.fillStyle = actor.color;
      ctx.beginPath();
      ctx.ellipse(0, size * 0.12, size * 0.34, size * 0.28, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(size * 0.18, -size * 0.2, size * 0.22, 0, Math.PI * 2);
      ctx.fill();
      // Ears
      ctx.beginPath();
      ctx.moveTo(size * 0.04, -size * 0.34);
      ctx.lineTo(size * 0.12, -size * 0.52);
      ctx.lineTo(size * 0.2, -size * 0.36);
      ctx.moveTo(size * 0.22, -size * 0.36);
      ctx.lineTo(size * 0.34, -size * 0.5);
      ctx.lineTo(size * 0.36, -size * 0.3);
      ctx.fill();
      // Eyes
      ctx.fillStyle = actor.secondaryColor;
      ctx.beginPath();
      ctx.arc(size * 0.12, -size * 0.22, size * 0.038, 0, Math.PI * 2);
      ctx.arc(size * 0.27, -size * 0.22, size * 0.038, 0, Math.PI * 2);
      ctx.fill();
      break;
    }

    case 'elephant': {
      const trunkLift = Math.sin(phase * 4) * size * 0.12;
      ctx.fillStyle = actor.color;
      ctx.beginPath();
      ctx.ellipse(-size * 0.08, 0, size * 0.45, size * 0.34, 0, 0, Math.PI * 2);
      ctx.fill();
      // Legs
      ctx.fillRect(-size * 0.36, size * 0.18, size * 0.16, size * 0.34);
      ctx.fillRect(-size * 0.02, size * 0.18, size * 0.16, size * 0.34);
      // Head & Ear
      ctx.beginPath();
      ctx.arc(size * 0.34, -size * 0.12, size * 0.24, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = actor.secondaryColor;
      ctx.beginPath();
      ctx.ellipse(size * 0.18, -size * 0.1, size * 0.14, size * 0.2, -0.2, 0, Math.PI * 2);
      ctx.fill();
      // Trunk
      ctx.strokeStyle = actor.color;
      ctx.lineWidth = size * 0.11;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(size * 0.5, -size * 0.06);
      ctx.quadraticCurveTo(size * 0.72, size * 0.12, size * 0.62, size * 0.32 - trunkLift);
      ctx.stroke();
      break;
    }

    case 'bird':
    case 'dragon': {
      const flap = Math.sin(phase * 10) * size * 0.38;
      ctx.fillStyle = actor.color;
      ctx.beginPath();
      ctx.ellipse(0, 0, size * 0.42, size * 0.15, 0, 0, Math.PI * 2);
      ctx.fill();
      // Wings
      ctx.fillStyle = actor.secondaryColor;
      ctx.beginPath();
      ctx.moveTo(-size * 0.1, 0);
      ctx.quadraticCurveTo(-size * 0.18, -size * 0.5 + flap, size * 0.25, -size * 0.12);
      ctx.closePath();
      ctx.fill();
      break;
    }

    case 'boat': {
      ctx.rotate(Math.sin(phase * 3) * 0.06);
      // Hull
      ctx.fillStyle = actor.color;
      ctx.beginPath();
      ctx.moveTo(-size * 0.6, 0);
      ctx.lineTo(size * 0.6, 0);
      ctx.lineTo(size * 0.42, size * 0.25);
      ctx.lineTo(-size * 0.42, size * 0.25);
      ctx.closePath();
      ctx.fill();
      // Sail
      ctx.fillStyle = actor.secondaryColor;
      ctx.beginPath();
      ctx.moveTo(-size * 0.04, -size * 0.04);
      ctx.lineTo(-size * 0.04, -size * 0.72);
      ctx.lineTo(size * 0.44, -size * 0.08);
      ctx.closePath();
      ctx.fill();
      break;
    }

    case 'banana': {
      ctx.rotate(Math.sin(phase * 2) * 0.15);
      ctx.fillStyle = actor.color;
      ctx.beginPath();
      ctx.arc(0, 0, size * 0.55, 0.4, Math.PI * 1.15);
      ctx.arc(-size * 0.16, -size * 0.12, size * 0.48, Math.PI * 1.12, 0.35, true);
      ctx.closePath();
      ctx.fill();
      break;
    }

    case 'robot':
    case 'astronaut': {
      ctx.fillStyle = actor.color;
      drawRoundedRect(ctx, -size * 0.26, -size * 0.15, size * 0.52, size * 0.55, size * 0.12);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(0, -size * 0.32, size * 0.22, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = actor.secondaryColor;
      drawRoundedRect(ctx, -size * 0.14, -size * 0.38, size * 0.28, size * 0.12, size * 0.05);
      ctx.fill();
      break;
    }

    default: {
      // Glowing energy orb / crystal / tree silhouette
      const glow = ctx.createRadialGradient(0, 0, size * 0.06, 0, 0, size * 0.6);
      glow.addColorStop(0, '#ffffff');
      glow.addColorStop(0.35, actor.color);
      glow.addColorStop(0.75, actor.secondaryColor + '66');
      glow.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(0, 0, size * 0.6, 0, Math.PI * 2);
      ctx.fill();
      break;
    }
  }

  ctx.restore();
}

interface CachedBuilding {
  bxBase: number;
  bw: number;
  by: number;
  h: number;
  isEven: boolean;
  windowColor: string;
  windowsX: number[];
  windowsY: number[];
}

interface CachedSceneAssets {
  width: number;
  height: number;
  horizonY: number;
  starsX: Float32Array;
  starsY: Float32Array;
  buildings: CachedBuilding[];
  mountainLayers: { pxBase: number[]; py: number[] }[];
  weatherSeedX: Float32Array;
  weatherSeedY: Float32Array;
}

const sceneCache = new WeakMap<VideoSceneSpec, CachedSceneAssets>();

function getOrBuildSceneAssets(
  spec: VideoSceneSpec,
  width: number,
  height: number,
): CachedSceneAssets {
  const existing = sceneCache.get(spec);
  if (existing && existing.width === width && existing.height === height) {
    return existing;
  }

  const horizonY = height * 0.64;

  const starCount = 65;
  const starsX = new Float32Array(starCount);
  const starsY = new Float32Array(starCount);
  for (let i = 0; i < starCount; i++) {
    starsX[i] = seededRand(i * 3 + 1) * width;
    starsY[i] = seededRand(i * 3 + 2) * horizonY * 0.85;
  }

  const buildings: CachedBuilding[] = [];
  if (spec.backdrop === 'city') {
    const buildingCount = 16;
    const bw = (width * 1.2) / buildingCount;
    for (let i = 0; i < buildingCount; i++) {
      const h = height * (0.18 + seededRand(i + 11) * 0.34);
      const bxBase = -width * 0.1 + i * bw;
      const by = horizonY - h;
      const windowsX: number[] = [];
      const windowsY: number[] = [];
      for (let wy = by + 10; wy < horizonY - 12; wy += 16) {
        for (let wx = bxBase + 6; wx < bxBase + bw * 0.78; wx += 12) {
          if (seededRand(wx * 0.1 + wy) > 0.38) {
            windowsX.push(wx);
            windowsY.push(wy);
          }
        }
      }
      buildings.push({
        bxBase,
        bw: bw * 0.88,
        by,
        h: h + 4,
        isEven: i % 2 === 0,
        windowColor: i % 3 === 0 ? spec.accent + 'aa' : spec.secondary + '99',
        windowsX,
        windowsY,
      });
    }
  }

  const mountainLayers: { pxBase: number[]; py: number[] }[] = [];
  if (spec.backdrop === 'mountains' || spec.backdrop === 'forest') {
    const peaks = 8;
    for (let layer = 0; layer < 2; layer++) {
      const pxBase: number[] = [];
      const py: number[] = [];
      for (let p = 0; p <= peaks; p++) {
        pxBase.push(-width * 0.1 + (p / peaks) * (width * 1.2));
        py.push(horizonY - height * (layer === 0 ? 0.22 : 0.12) * seededRand(p + layer * 20));
      }
      mountainLayers.push({ pxBase, py });
    }
  }

  const weatherCount = spec.weather === 'rain' ? 95 : 55;
  const weatherSeedX = new Float32Array(weatherCount);
  const weatherSeedY = new Float32Array(weatherCount);
  for (let i = 0; i < weatherCount; i++) {
    weatherSeedX[i] = seededRand(i * 7 + 3);
    weatherSeedY[i] = seededRand(i * 7 + 5);
  }

  const built: CachedSceneAssets = {
    width,
    height,
    horizonY,
    starsX,
    starsY,
    buildings,
    mountainLayers,
    weatherSeedX,
    weatherSeedY,
  };
  sceneCache.set(spec, built);
  return built;
}

export function renderVideoSceneFrame(
  ctx: CanvasRenderingContext2D,
  spec: VideoSceneSpec,
  width: number,
  height: number,
  timeSec: number,
): void {
  const assets = getOrBuildSceneAssets(spec, width, height);
  const duration = Math.max(2, spec.durationSec);
  const progress = (timeSec % duration) / duration;
  const angle = progress * Math.PI * 2;
  const horizonY = assets.horizonY;

  ctx.save();
  const zoomFactor = 1 + (Math.sin(angle) * 0.5 + 0.5) * spec.cameraZoom;
  ctx.translate(width / 2, height / 2);
  ctx.scale(zoomFactor, zoomFactor);
  ctx.translate(-width / 2, -height / 2);

  // 1. Sky gradient
  const skyGrad = ctx.createLinearGradient(0, 0, 0, horizonY);
  skyGrad.addColorStop(0, spec.skyTop);
  skyGrad.addColorStop(0.75, spec.skyBottom);
  skyGrad.addColorStop(1, spec.horizonGlow);
  ctx.fillStyle = skyGrad;
  ctx.fillRect(-width * 0.2, -height * 0.2, width * 1.4, height * 1.4);

  // 2. Twinkling stars (batched into 3 alpha buckets instead of 65 style changes)
  if (spec.backdrop === 'space' || spec.backdrop === 'city' || spec.celestial === 'moon') {
    const starsX = assets.starsX;
    const starsY = assets.starsY;
    const alphas = ['rgba(255,255,255,0.42)', 'rgba(255,255,255,0.68)', 'rgba(255,255,255,0.94)'];
    for (let bucket = 0; bucket < 3; bucket++) {
      ctx.fillStyle = alphas[bucket];
      for (let i = bucket; i < starsX.length; i += 3) {
        const twinkle = 0.5 + 0.5 * Math.sin(angle * 2 + i);
        const s = 1.4 + twinkle * 1.1;
        ctx.fillRect(starsX[i], starsY[i], s, s);
      }
    }
  }

  // 3. Celestial body (sun / moon / planet / nebula)
  if (spec.celestial !== 'none') {
    const cx = width * (0.72 + Math.sin(angle) * 0.02);
    const cy = height * (0.28 + Math.cos(angle) * 0.015);
    const radius = Math.min(width, height) * 0.11;
    const halo = ctx.createRadialGradient(cx, cy, radius * 0.2, cx, cy, radius * 2.8);
    halo.addColorStop(0, spec.horizonGlow + 'ee');
    halo.addColorStop(0.45, spec.accent + '55');
    halo.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(cx, cy, radius * 2.8, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = spec.celestial === 'moon' ? '#f8fafc' : spec.horizonGlow;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();

    if (spec.celestial === 'planet') {
      ctx.strokeStyle = spec.accent + 'aa';
      ctx.lineWidth = Math.max(3, radius * 0.12);
      ctx.beginPath();
      ctx.ellipse(cx, cy, radius * 1.65, radius * 0.32, -0.25, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  // 4. Parallax Backdrop (city / mountains / ocean / forest / space / abstract)
  const panShift = Math.sin(angle) * width * 0.06 * spec.cameraPan;

  if (spec.backdrop === 'city') {
    for (let i = 0; i < assets.buildings.length; i++) {
      const b = assets.buildings[i];
      const bx = b.bxBase - panShift;
      ctx.fillStyle = b.isEven ? '#0b1120' : '#111827';
      ctx.fillRect(bx, b.by, b.bw, b.h);
      ctx.fillStyle = b.windowColor;
      for (let w = 0; w < b.windowsX.length; w++) {
        ctx.fillRect(b.windowsX[w] - panShift, b.windowsY[w], 5, 8);
      }
    }
  } else if (assets.mountainLayers.length > 0) {
    for (let layer = 0; layer < assets.mountainLayers.length; layer++) {
      const ml = assets.mountainLayers[layer];
      const shift = panShift * (layer + 1) * 0.6;
      ctx.fillStyle = layer === 0 ? '#1e293b' : '#0f172a';
      ctx.beginPath();
      ctx.moveTo(-width * 0.2, horizonY + 4);
      for (let p = 0; p < ml.pxBase.length; p++) {
        ctx.lineTo(ml.pxBase[p] - shift, ml.py[p]);
      }
      ctx.lineTo(width * 1.2, horizonY + 4);
      ctx.closePath();
      ctx.fill();
    }
  }

  // 5. Ground / Ocean plane + Shimmering Water Reflections
  const groundGrad = ctx.createLinearGradient(0, horizonY, 0, height);
  groundGrad.addColorStop(0, spec.groundTop);
  groundGrad.addColorStop(1, spec.groundBottom);
  ctx.fillStyle = groundGrad;
  ctx.fillRect(-width * 0.2, horizonY, width * 1.4, height - horizonY + height * 0.2);

  if (spec.waterReflection || spec.backdrop === 'ocean') {
    const evenColor = spec.horizonGlow + '44';
    const oddColor = spec.accent + '44';
    for (let pass = 0; pass < 2; pass++) {
      ctx.fillStyle = pass === 0 ? evenColor : oddColor;
      for (let i = pass; i < 24; i += 2) {
        const y = horizonY + ((i + 0.5) / 24) * (height - horizonY);
        const waveW = width * (0.08 + 0.26 * (1 - i / 24));
        const waveX =
          width * 0.5 +
          Math.sin(angle * 2 + i * 0.7) * width * 0.14 -
          waveW / 2;
        ctx.fillRect(waveX, y, waveW, 2.5);
      }
    }
  }

  // 6. Animated Subjects
  const baseUnit = Math.min(width, height) * 0.13;
  for (let idx = 0; idx < spec.actors.length; idx++) {
    const actor = spec.actors[idx];
    const t = angle * actor.speed + idx * 1.4;
    let ax = actor.x * width;
    let ay = actor.y * height;

    if (actor.motion === 'dash') {
      ax = ((actor.x + progress * actor.speed) % 1.2 - 0.1) * width;
    } else if (actor.motion === 'glide') {
      ax += Math.sin(t) * width * 0.18;
      ay += Math.cos(t * 2) * height * 0.04;
    } else if (actor.motion === 'orbit') {
      ax += Math.cos(t) * width * 0.14;
      ay += Math.sin(t) * height * 0.1;
    } else if (actor.motion === 'hover' || actor.motion === 'bob') {
      ay += Math.sin(t * 2) * height * 0.04;
    }

    drawActorShape(ctx, actor, t, ax, ay, baseUnit * actor.scale);
  }

  // 7. Atmospheric Weather Particles (batched path rendering for 60 fps speed)
  if (spec.weather !== 'none') {
    const sx = assets.weatherSeedX;
    const sy = assets.weatherSeedY;
    const count = sx.length;

    if (spec.weather === 'rain') {
      ctx.strokeStyle = spec.accent + '88';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      const xShift = progress * 0.18;
      const yShift = progress * 3;
      for (let i = 0; i < count; i++) {
        const px = ((sx[i] + xShift) % 1) * width;
        const py = ((sy[i] + yShift) % 1) * height;
        ctx.moveTo(px, py);
        ctx.lineTo(px - 3, py + 14);
      }
      ctx.stroke();
    } else if (spec.weather === 'petals') {
      ctx.fillStyle = '#fbcfe8cc';
      ctx.beginPath();
      const xShift = progress * 0.35;
      for (let i = 0; i < count; i++) {
        const px = ((sx[i] + xShift) % 1) * width;
        const py = ((sy[i] + progress) % 1) * height;
        ctx.moveTo(px + 4, py);
        ctx.ellipse(px, py, 4, 2.2, angle + i, 0, Math.PI * 2);
      }
      ctx.fill();
    } else if (spec.weather === 'embers') {
      const yShift = progress * 1.2;
      const xShift = progress * 0.35;
      for (let pass = 0; pass < 2; pass++) {
        ctx.fillStyle = pass === 0 ? '#fb923ccc' : '#fef08acc';
        ctx.beginPath();
        for (let i = pass; i < count; i += 2) {
          const px = ((sx[i] + xShift) % 1) * width;
          const py = (1 - ((sy[i] + yShift) % 1)) * height;
          ctx.moveTo(px + 2.4, py);
          ctx.arc(px, py, 2.4, 0, Math.PI * 2);
        }
        ctx.fill();
      }
    } else {
      ctx.fillStyle = spec.weather === 'snow' ? '#ffffffcc' : spec.accent + 'bb';
      const r = spec.weather === 'snow' ? 2.6 : 2;
      const xShift = progress * 0.35;
      ctx.beginPath();
      for (let i = 0; i < count; i++) {
        const px = ((sx[i] + xShift) % 1) * width;
        const py = ((sy[i] + progress) % 1) * height;
        ctx.moveTo(px + r, py);
        ctx.arc(px, py, r, 0, Math.PI * 2);
      }
      ctx.fill();
    }
  }

  ctx.restore();
}

export async function recordVideoSceneToWebm(
  spec: VideoSceneSpec,
  width: number,
  height: number,
  fps = 60,
): Promise<Blob> {
  if (typeof MediaRecorder === 'undefined') {
    throw new Error('MediaRecorder is unavailable in this browser.');
  }
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Unable to initialize 2D canvas recorder.');
  }

  const mimeCandidates = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  const mimeType = mimeCandidates.find((m) => MediaRecorder.isTypeSupported(m)) ?? 'video/webm';
  const stream = canvas.captureStream(fps);
  const recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 3_500_000 });
  const chunks: BlobPart[] = [];

  return new Promise<Blob>((resolve, reject) => {
    recorder.ondataavailable = (ev) => {
      if (ev.data && ev.data.size > 0) chunks.push(ev.data);
    };
    recorder.onerror = () => reject(new Error('Failed to encode WebM video stream.'));
    recorder.onstop = () => resolve(new Blob(chunks, { type: mimeType }));

    recorder.start();
    const totalMs = Math.max(2000, Math.min(6000, spec.durationSec * 1000));
    const start = performance.now();

    const step = (now: number) => {
      const elapsedSec = (now - start) / 1000;
      renderVideoSceneFrame(ctx, spec, width, height, elapsedSec);
      if (now - start < totalMs) {
        requestAnimationFrame(step);
      } else {
        recorder.stop();
      }
    };
    requestAnimationFrame(step);
  });
}
