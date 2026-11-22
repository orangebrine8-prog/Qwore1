export type CameraMotionMode =
  | 'auto'
  | 'dolly-in'
  | 'orbit'
  | 'pan-right'
  | 'crane-up'
  | 'handheld'
  | 'static';

export const SCENE_LATENT_DIM = 14;

/**
 * 14 continuous scene & physics latents predicted by Qowe's trained-from-scratch
 * compositional scene-latent head (and blended softly with prompt semantics):
 * [0] warmth (golden sunrise vs cool blue/noir)
 * [1] neonIntensity (cyberpunk/synthwave glow)
 * [2] spaceCosmos (starfield, eclipse corona, deep space)
 * [3] waterPresence (ocean/river/wet asphalt reflections)
 * [4] mountainScale (alpine/crater ridge relief)
 * [5] cityDensity (skyscrapers + glowing windows)
 * [6] forestDensity (pine trees + canopy mist)
 * [7] volcanoLava (magma veins + glowing caldera)
 * [8] rainLevel (perspective diagonal rain + puddle ripples)
 * [9] snowOrFireflies (drifting glow orbs / snow flakes)
 * [10] emberLevel (rising hot embers / thruster sparks)
 * [11] kineticSpeed (camera/subject speed energy)
 * [12] godRays (volumetric atmospheric light shafts)
 * [13] productSlate (clean dark studio turntable spotlight)
 */
export type SceneLatentVector = Float32Array;

export type DynamicActorKind =
  | 'rocket'
  | 'car'
  | 'dragon'
  | 'dog'
  | 'cat'
  | 'bird'
  | 'astronaut'
  | 'robot'
  | 'boat'
  | 'whale'
  | 'banana'
  | 'crystal';

export interface ParsedCinemaPrompt {
  actors: DynamicActorKind[];
  hasSunrise: boolean;
  hasEclipse: boolean;
  hasAurora: boolean;
  hasGrid: boolean;
  hasVortex: boolean;
  hasWaterfall: boolean;
}

export function parseCinemaPrompt(prompt: string): ParsedCinemaPrompt {
  const lower = prompt.toLowerCase();
  const actors: DynamicActorKind[] = [];

  const rules: [RegExp, DynamicActorKind][] = [
    [/\b(rocket|spaceship|shuttle|launch|starship)\b/, 'rocket'],
    [/\b(car|sports car|supercar|vehicle|drift|highway|cyberpunk car)\b/, 'car'],
    [/\b(dragon|wyvern)\b/, 'dragon'],
    [/\b(dog|puppy|retriever|wolf|hound|fox)\b/, 'dog'],
    [/\b(cat|kitten|panther|tiger|lion)\b/, 'cat'],
    [/\b(bird|eagle|hawk|falcon|phoenix|owl)\b/, 'bird'],
    [/\b(astronaut|cosmonaut|spaceman)\b/, 'astronaut'],
    [/\b(robot|mech|android|cyborg)\b/, 'robot'],
    [/\b(boat|ship|yacht|sailboat|vessel)\b/, 'boat'],
    [/\b(whale|dolphin|shark|fish|orca)\b/, 'whale'],
    [/\b(banana|fruit)\b/, 'banana'],
    [/\b(crystal|prism|gem|monolith|diamond)\b/, 'crystal'],
  ];

  for (const [re, kind] of rules) {
    if (re.test(lower)) actors.push(kind);
  }

  return {
    actors: actors.slice(0, 2),
    hasSunrise: /\b(sun|sunrise|sunset|golden hour|dawn|dusk)\b/.test(lower),
    hasEclipse: /\b(eclipse|black hole|corona|moon)\b/.test(lower),
    hasAurora: /\b(aurora|northern lights|ribbons)\b/.test(lower),
    hasGrid: /\b(synthwave|vaporwave|outrun|laser grid|grid)\b/.test(lower),
    hasVortex: /\b(vortex|portal|swirl|plasma)\b/.test(lower),
    hasWaterfall: /\b(waterfall|falls|cascade)\b/.test(lower),
  };
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function hash2(x: number, y: number): number {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453123;
  return s - Math.floor(s);
}

function fbm1(x: number): number {
  return (
    Math.sin(x) * 0.5 +
    Math.sin(x * 2.13 + 1.3) * 0.25 +
    Math.sin(x * 4.41 + 2.7) * 0.125 +
    Math.sin(x * 8.9 + 0.6) * 0.0625
  );
}

/**
 * Renders a crisp high-definition cinematic frame by combining:
 * 1) Qowe's coarse-evaluated from-scratch neural field lighting/color buffer
 * 2) Qowe's trained compositional scene latents (14 continuous scene attributes)
 * 3) Real-time multi-plane 2.5D terrain, volumetric light shafts, water/wet-street reflections,
 *    articulated SDF motion-tracked subjects, perspective weather particles, & ACES filmic grade.
 */
export function composeCinematicFrame(params: {
  width: number;
  height: number;
  t: number;
  cameraMotion: Exclude<CameraMotionMode, 'auto'>;
  neuralGrid: Float32Array;
  gridW: number;
  gridH: number;
  neuralBlendWeight: number;
  latents: SceneLatentVector;
  parsed: ParsedCinemaPrompt;
  outRgba: Uint8ClampedArray;
}): void {
  const {
    width,
    height,
    t,
    cameraMotion,
    neuralGrid,
    gridW,
    gridH,
    neuralBlendWeight,
    latents,
    parsed,
    outRgba,
  } = params;

  const warmth = latents[0];
  const neon = latents[1];
  const space = latents[2];
  const water = latents[3];
  const mountains = latents[4];
  const city = latents[5];
  const forest = latents[6];
  const volcano = latents[7];
  const rain = latents[8];
  const fireflies = latents[9];
  const embers = latents[10];
  const kinetic = latents[11];
  const godRays = latents[12];
  const studioProduct = latents[13];

  const tau = t * Math.PI * 2;
  const maxWm1 = Math.max(1, width - 1);
  const maxHm1 = Math.max(1, height - 1);
  const aspect = width / Math.max(1, height);

  // Smooth camera gimbal parameters over loop
  let camZoom = 1;
  let camPanX = 0;
  let camPanY = 0;
  let camRoll = 0;
  if (cameraMotion === 'dolly-in') {
    camZoom = 1 - 0.18 * (0.5 - 0.5 * Math.cos(tau));
  } else if (cameraMotion === 'orbit') {
    camZoom = 0.94;
    camPanX = 0.14 * Math.sin(tau);
    camPanY = 0.03 * Math.cos(tau);
  } else if (cameraMotion === 'pan-right') {
    camZoom = 0.93;
    camPanX = 0.18 * Math.sin(tau);
  } else if (cameraMotion === 'crane-up') {
    camZoom = 0.95;
    camPanY = 0.1 * Math.cos(tau);
  } else if (cameraMotion === 'handheld') {
    camZoom = 0.96;
    camPanX = 0.024 * Math.sin(tau * 2 + 0.3) + 0.01 * Math.cos(tau * 3);
    camPanY = 0.02 * Math.cos(tau * 2) + 0.008 * Math.sin(tau * 3);
    camRoll = 0.025 * Math.sin(tau * 2);
  }

  const cosR = Math.cos(camRoll);
  const sinR = Math.sin(camRoll);

  // Dynamic horizon height & celestial position
  const horizonY = studioProduct > 0.55 ? 0.35 : 0.12 - camPanY * 0.25;
  const sunX = 0.08 * Math.sin(tau * 0.5) - camPanX * 0.45;
  const sunY = parsed.hasEclipse
    ? -0.22 + 0.04 * Math.sin(tau)
    : -0.26 - 0.05 * Math.sin(tau) - camPanY * 0.3;

  // Dynamic hero actor position along cinematic spline
  const actorKind = parsed.actors[0];
  let actorX = 0;
  let actorY = 0;
  let actorAngle = 0;
  if (actorKind) {
    if (actorKind === 'rocket') {
      actorX = 0.06 * Math.sin(tau);
      actorY = 0.32 - ((t * 1.15) % 1.0) * 0.92;
      actorAngle = 0.08 * Math.cos(tau);
    } else if (actorKind === 'car') {
      actorX = -0.85 + ((t * 1.4) % 1.7);
      actorY = horizonY + 0.28 + 0.015 * Math.sin(tau * 8);
      actorAngle = 0;
    } else if (actorKind === 'dragon' || actorKind === 'bird') {
      actorX = -0.65 + ((t * 1.3) % 1.3);
      actorY = -0.22 + 0.12 * Math.sin(tau * 2);
      actorAngle = 0.15 * Math.cos(tau * 2);
    } else if (actorKind === 'dog' || actorKind === 'cat') {
      actorX = -0.7 + ((t * 1.35) % 1.4);
      actorY = horizonY + 0.26 - Math.abs(Math.sin(t * Math.PI * 6)) * 0.06;
      actorAngle = -0.06 * Math.cos(t * Math.PI * 6);
    } else if (actorKind === 'boat' || actorKind === 'whale') {
      actorX = -0.55 + ((t * 1.1) % 1.1);
      actorY =
        actorKind === 'whale'
          ? horizonY + 0.16 - Math.max(0, Math.sin(t * Math.PI * 2)) * 0.28
          : horizonY + 0.18 + 0.025 * Math.sin(tau * 3);
      actorAngle = actorKind === 'whale' ? -0.35 * Math.cos(t * Math.PI * 2) : 0.04 * Math.cos(tau * 3);
    } else {
      actorX = 0.16 * Math.sin(tau);
      actorY = -0.02 + 0.08 * Math.cos(tau);
      actorAngle = 0.12 * Math.sin(tau);
    }
  }

  // Float buffer for crisp multi-pass FX & horizontal bloom
  const floatRgb = new Float32Array(width * height * 3);

  const sampleNeuralField = (ux: number, uy: number): [number, number, number] => {
    const gx = clamp01(ux) * (gridW - 1);
    const gy = clamp01(uy) * (gridH - 1);
    const x0 = Math.floor(gx);
    const y0 = Math.floor(gy);
    const x1 = Math.min(gridW - 1, x0 + 1);
    const y1 = Math.min(gridH - 1, y0 + 1);
    const fx = gx - x0;
    const fy = gy - y0;
    const w00 = (1 - fx) * (1 - fy);
    const w10 = fx * (1 - fy);
    const w01 = (1 - fx) * fy;
    const w11 = fx * fy;
    const o00 = (y0 * gridW + x0) * 3;
    const o10 = (y0 * gridW + x1) * 3;
    const o01 = (y1 * gridW + x0) * 3;
    const o11 = (y1 * gridW + x1) * 3;
    return [
      neuralGrid[o00] * w00 + neuralGrid[o10] * w10 + neuralGrid[o01] * w01 + neuralGrid[o11] * w11,
      neuralGrid[o00 + 1] * w00 +
        neuralGrid[o10 + 1] * w10 +
        neuralGrid[o01 + 1] * w01 +
        neuralGrid[o11 + 1] * w11,
      neuralGrid[o00 + 2] * w00 +
        neuralGrid[o10 + 2] * w10 +
        neuralGrid[o01 + 2] * w01 +
        neuralGrid[o11 + 2] * w11,
    ];
  };

  // Helper to evaluate full sky/terrain pixel color at warped coordinates (cx, cy)
  const evalEnvironment = (
    cx: number,
    cy: number,
    includeReflection: boolean,
  ): [number, number, number] => {
    // Ground / Reflection surface check
    if (includeReflection && cy > horizonY && studioProduct < 0.6) {
      const depth = Math.max(0.02, cy - horizonY);
      const perspX = cx / (0.35 + depth * 1.45);
      const waveOffset =
        water > 0.35 || rain > 0.3
          ? Math.sin(perspX * 16 + tau * 2.5 + depth * 28) * (0.015 + depth * 0.04) +
            Math.cos(perspX * 9 - tau * 1.5) * 0.01
          : 0;

      // Mirror sample above horizon
      const mirY = horizonY - (cy - horizonY) * (0.85 + 0.1 * waveOffset);
      const [mr, mg, mb] = evalEnvironment(cx + waveOffset, mirY, false);

      // Synthwave / Cyberpunk laser grid overlay on ground plane
      if (parsed.hasGrid || (neon > 0.6 && water < 0.45)) {
        const gz = Math.max(0.05, cy - horizonY);
        const gx = (cx + camPanX * 0.5) / gz;
        const gy = 1.4 / gz - ((t * (2 + kinetic * 2)) % 1);
        const lineX = Math.exp(-Math.pow(Math.sin(gx * Math.PI * 1.15), 2) * 55);
        const lineY = Math.exp(-Math.pow(Math.sin(gy * Math.PI * 1.15), 2) * 55);
        const gridGlow = Math.min(1, lineX + lineY) * Math.min(1, gz * 4);
        return [
          mr * 0.38 + 0.04 + 0.9 * gridGlow,
          mg * 0.35 + 0.02 + 0.22 * gridGlow,
          mb * 0.45 + 0.09 + 0.95 * gridGlow,
        ];
      }

      // Water ocean surf or wet cyberpunk asphalt
      const fresnel = 0.28 + 0.56 * Math.exp(-depth * 3.2);
      const specGlint =
        Math.exp(-Math.pow((cx - sunX + waveOffset * 2) * 8, 2)) *
        (0.5 + 0.5 * warmth) *
        (0.45 + 0.55 * Math.sin(depth * 55 - tau * 2));
      const baseR = water > 0.4 ? 0.02 : 0.05;
      const baseG = water > 0.4 ? 0.12 : 0.06;
      const baseB = water > 0.4 ? 0.24 : 0.11;

      const r = mr * fresnel + baseR * (1 - fresnel) + Math.max(0, specGlint) * (0.85 * warmth + 0.25 * neon);
      const g = mg * fresnel + baseG * (1 - fresnel) + Math.max(0, specGlint) * (0.65 * warmth + 0.45 * neon);
      const b = mb * fresnel + baseB * (1 - fresnel) + Math.max(0, specGlint) * (0.3 * warmth + 0.75 * neon);

      // Puddle ripple rings if raining
      if (rain > 0.25) {
        const ringCellX = Math.floor((cx + 2) * 7);
        const ringCellY = Math.floor((cy + 2) * 14);
        const phase = (hash2(ringCellX, ringCellY) + t * 2) % 1;
        const rx = ((cx + 2) * 7 - ringCellX - 0.5) * 1.2;
        const ry = ((cy + 2) * 14 - ringCellY - 0.5) * 2.2;
        const rd = Math.hypot(rx, ry);
        const rippleGlow = Math.exp(-Math.pow((rd - phase * 0.48) * 22, 2)) * (1 - phase) * rain * 0.45;
        return [r + rippleGlow * 0.6, g + rippleGlow * 0.85, b + rippleGlow];
      }

      return [r, g, b];
    }

    // Studio product lighting backdrop
    if (studioProduct > 0.55) {
      const floor = cy > horizonY;
      const spot = Math.exp(-Math.hypot(cx * 0.85, (cy - 0.1) * 1.4) * 2.4);
      const rim = Math.exp(-Math.hypot(cx, cy + 0.25) * 1.9);
      if (floor) {
        return [0.05 + 0.16 * spot, 0.055 + 0.15 * spot, 0.075 + 0.17 * spot];
      }
      return [0.04 + 0.14 * rim + 0.08 * warmth, 0.045 + 0.12 * rim, 0.07 + 0.15 * rim];
    }

    // Volumetric Sky Layer
    const skyNorm = clamp01((cy + 1) / (horizonY + 1));
    let sr = 0.03 + warmth * (0.28 + 0.62 * Math.pow(skyNorm, 1.45)) + neon * 0.14 * skyNorm;
    let sg = 0.05 + warmth * (0.14 + 0.44 * Math.pow(skyNorm, 1.7)) + neon * (0.06 + 0.14 * skyNorm);
    let sb = 0.15 + (1 - space) * 0.34 * (1 - skyNorm * 0.6) + neon * 0.28 + warmth * 0.1 * skyNorm;

    // Deep space darkening when cosmic
    if (space > 0.35) {
      const sp = clamp01((space - 0.35) * 1.55);
      sr = sr * (1 - sp) + 0.018 * sp;
      sg = sg * (1 - sp) + 0.022 * sp;
      sb = sb * (1 - sp) + 0.065 * sp;
    }

    // Volumetric layered clouds
    if (space < 0.7) {
      const cloudN = fbm1(cx * 3.8 + t * (0.6 + kinetic * 0.8) + cy * 2.2);
      const cloudMask = Math.max(0, cloudN - 0.15) * (1 - skyNorm * 0.45) * 0.36;
      sr += cloudMask * (0.9 * warmth + 0.45 * (1 - warmth));
      sg += cloudMask * (0.68 * warmth + 0.52 * (1 - warmth));
      sb += cloudMask * (0.42 * warmth + 0.72 * (1 - warmth));
    }

    // Crisp Starfield & Nebula
    if (space > 0.2 || neon > 0.4 || parsed.hasAurora) {
      const starDensity = space > 0.4 ? 1.0 : 0.45;
      const sx = Math.floor((cx * aspect + 2) * 85);
      const sy = Math.floor((cy + 2) * 85);
      const h = hash2(sx, sy);
      if (h > 0.992 - 0.008 * starDensity) {
        const twinkle = 0.6 + 0.4 * Math.sin(tau * 2 + h * 120);
        const bright = ((h - 0.984) / 0.016) * twinkle;
        sr += bright * 0.9;
        sg += bright * 0.94;
        sb += bright * 1.05;
      }
    }

    // Aurora Borealis Volumetric Ribbons
    if (parsed.hasAurora || (mountains > 0.45 && space > 0.25 && warmth < 0.4)) {
      const ribbonY = -0.35 + 0.15 * Math.sin(cx * 3.1 + tau) + 0.06 * Math.cos(cx * 6.5 - tau);
      const band = Math.exp(-Math.pow((cy - ribbonY) * 5.2, 2));
      const curtain = 0.35 + 0.65 * Math.max(0, Math.sin(cx * 11 - tau * 1.5));
      const topViolet = Math.exp(-Math.pow((cy + 0.68) * 3.6, 2)) * curtain;
      sr += 0.14 * band * curtain + 0.45 * topViolet;
      sg += 0.92 * band * curtain + 0.12 * topViolet;
      sb += 0.52 * band * curtain + 0.72 * topViolet;
    }

    // Sun / Eclipse Corona / Celestial Body
    const dxS = (cx - sunX) * aspect;
    const dyS = cy - sunY;
    const distS = Math.hypot(dxS, dyS);
    if (parsed.hasEclipse) {
      const ang = Math.atan2(dyS, dxS);
      const flare = 1 + 0.28 * Math.sin(ang * 6 - tau) * Math.cos(ang * 4 + tau);
      const corona = Math.exp(-Math.pow((distS - 0.23) * 14, 2)) * flare;
      const outer = Math.exp(-Math.pow(Math.max(0, distS - 0.21) * 4.5, 2));
      const mask = distS < 0.205 ? 0.04 : 1;
      sr = sr * mask + 1.25 * corona + 0.42 * outer;
      sg = sg * mask + 0.48 * corona + 0.14 * outer;
      sb = sb * mask + 0.24 * corona + 0.18 * outer;
    } else if (parsed.hasVortex) {
      const ang = Math.atan2(dyS, dxS);
      const spiral = 0.5 + 0.5 * Math.sin(distS * 18 - ang * 4 - tau * 2);
      const fade = Math.exp(-distS * 1.6);
      const core = Math.exp(-distS * distS * 38);
      sr += 0.35 * spiral * fade + 0.95 * core;
      sg += 0.75 * spiral * fade + 1.0 * core;
      sb += 1.05 * fade + 1.1 * core;
    } else if (warmth > 0.35 || parsed.hasSunrise || parsed.hasGrid) {
      const sunRadius = parsed.hasGrid ? 0.26 : 0.14;
      const disk = distS < sunRadius ? 1 : Math.exp(-Math.pow((distS - sunRadius) * 26, 2));
      const stripeMask =
        parsed.hasGrid && cy > sunY + 0.02 && Math.sin(cy * 62 + tau) < -0.15 ? 0.15 : 1;
      const halo = Math.exp(-distS * distS * 5.2);
      sr += (1.15 * disk * stripeMask + 0.55 * halo) * (0.45 + 0.65 * warmth);
      sg += (0.78 * disk * stripeMask + 0.32 * halo) * (0.4 + 0.6 * warmth);
      sb += (0.28 * disk * stripeMask + 0.12 * halo) * (0.3 + 0.5 * neon);
    }

    // Volumetric God Rays radiating down from sun
    if (godRays > 0.35 && includeReflection) {
      const ang = Math.atan2(dyS, dxS);
      const rays = Math.pow(Math.max(0, Math.sin(ang * 9 + tau * 0.4)), 4) * Math.exp(-distS * 1.3);
      sr += rays * 0.28 * warmth;
      sg += rays * 0.22 * warmth;
      sb += rays * 0.16;
    }

    // Far & Mid-Ground Atmospheric Mountain Ridges
    if (mountains > 0.25 || volcano > 0.25) {
      const mScale = Math.max(mountains, volcano);
      const farRidge =
        horizonY - 0.16 * mScale - 0.11 * mScale * (0.5 + 0.5 * fbm1((cx + camPanX * 0.25) * 4.5));
      if (cy > farRidge) {
        const haze = 0.55;
        sr = sr * haze + 0.12 * (1 - haze);
        sg = sg * haze + 0.14 * (1 - haze);
        sb = sb * haze + 0.24 * (1 - haze);
      }
      const nearRidge =
        horizonY -
        (volcano > 0.4
          ? Math.max(0, 0.34 - Math.abs(cx + camPanX * 0.5) * 0.65)
          : 0.09 * mScale + 0.075 * mScale * (0.5 + 0.5 * fbm1((cx + camPanX * 0.65) * 7.2 + 2)));
      if (cy > nearRidge) {
        const snowRim = Math.exp(-Math.pow((cy - nearRidge) * 28, 2)) * (volcano > 0.4 ? 0 : 0.28);
        if (volcano > 0.4) {
          const lavaCrack = Math.exp(-Math.pow(Math.sin(cx * 9 + cy * 5 - tau * 1.2), 2) * 32);
          return [0.11 + 1.05 * lavaCrack, 0.04 + 0.42 * lavaCrack, 0.04 + 0.06 * lavaCrack];
        }
        sr = 0.04 + snowRim * (0.7 + 0.5 * warmth) + 0.03 * neon;
        sg = 0.055 + snowRim * (0.8 + 0.2 * warmth) + 0.02 * neon;
        sb = 0.095 + snowRim * 0.95 + 0.05 * neon;
      }
    }

    // Cyberpunk / City Skyline
    if (city > 0.25) {
      const bCoord = (cx + camPanX * 0.7 + 3) * 8.5;
      const bIdx = Math.floor(bCoord);
      const bFrac = bCoord - bIdx;
      const bHeight = 0.12 + 0.42 * hash2(bIdx, 7);
      const roofY = horizonY - bHeight * city;
      if (cy > roofY && bFrac > 0.08 && bFrac < 0.92) {
        const isCyan = bIdx % 2 === 0;
        const winX = Math.sin(bFrac * Math.PI * 6);
        const winY = Math.sin((cy - roofY) * 85);
        const lit = winX > 0.3 && winY > 0.25 && hash2(bIdx, Math.floor(cy * 45)) > 0.35 ? 1 : 0;
        const roofBeacon = Math.hypot(bFrac - 0.5, (cy - roofY) * 4) < 0.12 ? 1 : 0;
        if (isCyan) {
          sr = 0.04 + lit * 0.18 * neon + roofBeacon * 0.9;
          sg = 0.05 + lit * 0.85 * (0.4 + 0.6 * neon) + roofBeacon * 0.15;
          sb = 0.09 + lit * 0.98 + roofBeacon * 0.35;
        } else {
          sr = 0.06 + lit * 0.95 * (0.4 + 0.6 * neon) + roofBeacon * 0.2;
          sg = 0.04 + lit * 0.24 * neon + roofBeacon * 0.8;
          sb = 0.08 + lit * 0.72 + roofBeacon * 0.9;
        }
      }
    }

    // Forest Pine Silhouettes
    if (forest > 0.3) {
      const tCoord = (cx + camPanX * 0.8 + 4) * 14;
      const tIdx = Math.floor(tCoord);
      const tFrac = Math.abs(tCoord - tIdx - 0.5) * 2;
      const treeH = 0.12 + 0.2 * hash2(tIdx, 19);
      const treeTop = horizonY - treeH * forest;
      if (cy > treeTop) {
        const progress = (cy - treeTop) / Math.max(0.05, horizonY - treeTop);
        if (tFrac < progress * 0.9) {
          return [0.025, 0.085 + 0.05 * (1 - progress), 0.055];
        }
      }
    }

    // Cascading Waterfall stream
    if (parsed.hasWaterfall) {
      const wDist = Math.abs(cx + 0.05 * Math.sin(cy * 6));
      if (wDist < 0.18 && cy > -0.2) {
        const rush = 0.5 + 0.5 * Math.sin(cy * 32 - tau * 3 + cx * 12);
        return [0.42 + 0.45 * rush, 0.72 + 0.25 * rush, 0.9 + 0.15 * rush];
      }
    }

    return [sr, sg, sb];
  };

  // Pass 1: Render base neural-conditioned environment & water/asphalt reflection
  for (let py = 0; py < height; py++) {
    const nyBase = (py / maxHm1) * 2 - 1;
    const uy = py / maxHm1;
    for (let px = 0; px < width; px++) {
      const nxBase = (px / maxWm1) * 2 - 1;
      const ux = px / maxWm1;

      // Apply smooth camera zoom, pan & gimbal roll
      const rx = nxBase * cosR - nyBase * sinR;
      const ry = nxBase * sinR + nyBase * cosR;
      const cx = rx * camZoom + camPanX;
      const cy = ry * camZoom + camPanY;

      const [er, eg, eb] = evalEnvironment(cx, cy, true);
      const [nr, ng, nb] = sampleNeuralField(ux, uy);

      // Blend trained spatiotemporal neural lighting cleanly into crisp environment
      const w = neuralBlendWeight;
      const off = (py * width + px) * 3;
      floatRgb[off] = er * (1 - w * 0.45) + nr * (w * 0.55);
      floatRgb[off + 1] = eg * (1 - w * 0.45) + ng * (w * 0.55);
      floatRgb[off + 2] = eb * (1 - w * 0.45) + nb * (w * 0.55);
    }
  }

  // Pass 2: Rasterize Dynamic Articulated Subject & Volumetric Thruster / Headlight FX
  if (actorKind) {
    const cosA = Math.cos(-actorAngle);
    const sinA = Math.sin(-actorAngle);
    const scale = actorKind === 'banana' || actorKind === 'crystal' ? 1.25 : 0.28;

    const minPx = Math.max(0, Math.floor(((actorX - 0.6) * 0.5 + 0.5) * width));
    const maxPx = Math.min(width - 1, Math.ceil(((actorX + 0.6) * 0.5 + 0.5) * width));
    const minPy = Math.max(0, Math.floor(((actorY - 0.55) * 0.5 + 0.5) * height));
    const maxPy = Math.min(height - 1, Math.ceil(((actorY + 0.55) * 0.5 + 0.5) * height));

    for (let py = minPy; py <= maxPy; py++) {
      const ny = (py / maxHm1) * 2 - 1;
      for (let px = minPx; px <= maxPx; px++) {
        const nx = (px / maxWm1) * 2 - 1;
        const dx = (nx - actorX) * aspect;
        const dy = ny - actorY;
        const lx = (dx * cosA - dy * sinA) / scale;
        const ly = (dx * sinA + dy * cosA) / scale;
        const off = (py * width + px) * 3;

        if (actorKind === 'rocket') {
          const inBody = Math.abs(lx) < 0.18 && ly > -0.55 && ly < 0.45;
          const inNose = ly >= -0.95 && ly <= -0.55 && Math.abs(lx) < (ly + 0.95) * 0.45;
          const inFin =
            ly > 0.15 && ly < 0.52 && Math.abs(lx) >= 0.18 && Math.abs(lx) < 0.38 - (0.52 - ly) * 0.25;
          if (inBody || inNose || inFin) {
            const shade = 0.65 + 0.35 * (0.5 - lx * 1.8);
            const win = Math.hypot(lx, ly + 0.25) < 0.085;
            floatRgb[off] = win ? 0.35 : inFin ? 0.92 * shade : 0.92 * shade;
            floatRgb[off + 1] = win ? 0.85 : inFin ? 0.25 * shade : 0.94 * shade;
            floatRgb[off + 2] = win ? 1.15 : inFin ? 0.18 * shade : 0.98 * shade;
          } else if (ly >= 0.45 && ly < 1.35) {
            // Volumetric Mach-diamond plasma exhaust plume
            const plumeW = 0.16 * (1 - (ly - 0.45) * 0.72) + 0.03 * Math.sin(ly * 28 - tau * 6);
            const ex = Math.exp(-Math.pow(lx / Math.max(0.03, plumeW), 2));
            const core = Math.exp(-Math.pow((ly - 0.45) * 1.4, 2)) * ex;
            floatRgb[off] += core * 1.45;
            floatRgb[off + 1] += core * 0.72;
            floatRgb[off + 2] += core * 0.22;
          }
        } else if (actorKind === 'car') {
          const chassis = lx > -0.85 && lx < 0.85 && ly > -0.12 && ly < 0.18;
          const cabin = lx > -0.45 && lx < 0.38 && ly > -0.34 && ly <= -0.1 && Math.abs(lx + 0.05) < 0.48 - (-0.1 - ly) * 0.75;
          const wheelL = Math.hypot(lx + 0.48, ly - 0.18) < 0.16;
          const wheelR = Math.hypot(lx - 0.48, ly - 0.18) < 0.16;
          if (wheelL || wheelR) {
            const rim =
              Math.hypot(lx + 0.48, ly - 0.18) < 0.085 ||
              Math.hypot(lx - 0.48, ly - 0.18) < 0.085;
            floatRgb[off] = rim ? 0.75 : 0.06;
            floatRgb[off + 1] = rim ? 0.78 : 0.06;
            floatRgb[off + 2] = rim ? 0.85 : 0.08;
          } else if (chassis || cabin) {
            const spec = Math.exp(-Math.pow((ly + 0.15) * 12, 2)) * 0.45;
            floatRgb[off] = cabin ? 0.14 + spec : (warmth > 0.5 ? 0.92 : 0.15) + spec;
            floatRgb[off + 1] = cabin ? 0.24 + spec : (warmth > 0.5 ? 0.18 : 0.65) + spec;
            floatRgb[off + 2] = cabin ? 0.38 + spec : 0.92 + spec;
          }
          // Volumetric Headlight Beam & Tail light streak
          if (lx >= 0.8 && lx < 2.2) {
            const beamCone = Math.exp(-Math.pow((ly - 0.02) / (0.08 + (lx - 0.8) * 0.32), 2));
            const falloff = Math.exp(-(lx - 0.8) * 1.4);
            floatRgb[off] += beamCone * falloff * 0.95;
            floatRgb[off + 1] += beamCone * falloff * 0.88;
            floatRgb[off + 2] += beamCone * falloff * 0.65;
          } else if (lx <= -0.82 && lx > -2.0) {
            const trail = Math.exp(-Math.pow((ly - 0.02) * 14, 2)) * Math.exp((lx + 0.82) * 1.8);
            floatRgb[off] += trail * 1.1;
            floatRgb[off + 1] += trail * 0.12;
            floatRgb[off + 2] += trail * 0.28;
          }
        } else if (actorKind === 'dragon' || actorKind === 'bird') {
          const wingFlap = Math.sin(tau * 4) * 0.45;
          const body = Math.hypot(lx * 0.65, (ly - 0.04 * Math.sin(lx * 5 - tau * 3)) * 2.4) < 0.35;
          const wing =
            lx > -0.35 &&
            lx < 0.25 &&
            ly < 0.05 &&
            ly > -0.75 + Math.abs(lx) * 0.9 + wingFlap * (0.4 - Math.abs(lx));
          if (body || wing) {
            floatRgb[off] = actorKind === 'dragon' ? 0.72 : 0.88;
            floatRgb[off + 1] = actorKind === 'dragon' ? 0.16 : 0.9;
            floatRgb[off + 2] = actorKind === 'dragon' ? 0.14 : 0.95;
          } else if (actorKind === 'dragon' && lx > 0.32 && lx < 1.3) {
            // Dragon fire breath
            const cone = Math.exp(-Math.pow(ly / (0.07 + (lx - 0.32) * 0.28), 2)) * Math.exp(-(lx - 0.32) * 1.8);
            floatRgb[off] += cone * 1.4;
            floatRgb[off + 1] += cone * 0.65;
            floatRgb[off + 2] += cone * 0.12;
          }
        } else if (actorKind === 'dog' || actorKind === 'cat') {
          const torso = Math.hypot(lx * 1.0, ly * 2.2) < 0.42;
          const head = Math.hypot(lx - 0.38, ly + 0.24) < 0.22;
          const stride = Math.sin(tau * 6) * 0.18;
          const legF = Math.abs(lx - (0.24 + stride)) < 0.07 && ly > 0.1 && ly < 0.5;
          const legB = Math.abs(lx - (-0.24 - stride)) < 0.07 && ly > 0.1 && ly < 0.5;
          const tail = Math.hypot(lx + 0.45, ly + 0.16 - 0.08 * Math.sin(tau * 6)) < 0.12;
          if (torso || head || legF || legB || tail) {
            const coatR = actorKind === 'dog' ? 0.84 : 0.9;
            const coatG = actorKind === 'dog' ? 0.64 : 0.52;
            const coatB = actorKind === 'dog' ? 0.34 : 0.22;
            floatRgb[off] = coatR;
            floatRgb[off + 1] = coatG;
            floatRgb[off + 2] = coatB;
          }
        } else if (actorKind === 'banana') {
          const rot = 0.16 * Math.sin(tau);
          const rx = lx * Math.cos(rot) - ly * Math.sin(rot);
          const ry = lx * Math.sin(rot) + ly * Math.cos(rot);
          const curveY = 0.35 * rx * rx - 0.15;
          if (Math.abs(rx) < 0.65 && Math.abs(ry - curveY) < 0.22 * (1 - 0.45 * rx * rx)) {
            const hl = Math.exp(-Math.pow((ry - (curveY - 0.08)) * 14, 2));
            floatRgb[off] = Math.min(1.2, 0.88 + 0.35 * hl);
            floatRgb[off + 1] = Math.min(1.2, 0.72 + 0.35 * hl);
            floatRgb[off + 2] = 0.1 + 0.4 * hl;
          }
        } else if (actorKind === 'boat' || actorKind === 'whale') {
          const hull =
            actorKind === 'boat'
              ? ly > 0.02 && ly < 0.24 && Math.abs(lx) < 0.62 - ly * 0.8
              : Math.hypot(lx * 0.85, ly * 2.0) < 0.45;
          const sail = actorKind === 'boat' && ly > -0.75 && ly <= 0.02 && lx > -0.04 && lx < (ly + 0.75) * 0.62;
          if (hull || sail) {
            floatRgb[off] = sail ? 0.96 : actorKind === 'whale' ? 0.16 : 0.72;
            floatRgb[off + 1] = sail ? 0.95 : actorKind === 'whale' ? 0.34 : 0.24;
            floatRgb[off + 2] = sail ? 0.92 : actorKind === 'whale' ? 0.52 : 0.18;
          }
        } else {
          // Astronaut / Robot / Crystal
          const core = Math.hypot(lx, ly) < 0.34;
          const visor = Math.hypot(lx, ly + 0.08) < 0.16;
          if (core) {
            floatRgb[off] = visor ? 0.18 : 0.84;
            floatRgb[off + 1] = visor ? 0.72 : 0.88;
            floatRgb[off + 2] = visor ? 1.1 : 0.95;
          }
        }
      }
    }
  }

  // Pass 3: Perspective Rain Streaks, Rising Embers, & Fireflies
  if (rain > 0.25) {
    const dropCount = Math.floor(rain * 140);
    for (let i = 0; i < dropCount; i++) {
      const rx = hash2(i, 11);
      const speed = 1.6 + hash2(i, 23) * 1.4;
      const ry = (hash2(i, 37) + t * speed) % 1.0;
      const sx = Math.floor(rx * width);
      const sy = Math.floor(ry * height);
      for (let len = 0; len < 6; len++) {
        const px = sx - Math.floor(len * 0.4);
        const py = sy + len;
        if (px >= 0 && px < width && py >= 0 && py < height) {
          const off = (py * width + px) * 3;
          floatRgb[off] += 0.18;
          floatRgb[off + 1] += 0.24;
          floatRgb[off + 2] += 0.34;
        }
      }
    }
  }

  if (embers > 0.25 || volcano > 0.4) {
    const count = Math.floor(Math.max(embers, volcano) * 55);
    for (let i = 0; i < count; i++) {
      const ex = (hash2(i, 51) + 0.05 * Math.sin(tau * 2 + i)) * width;
      const ey = (1 - ((hash2(i, 67) + t * (0.7 + hash2(i, 71))) % 1.0)) * height;
      const cx = Math.floor(ex);
      const cy = Math.floor(ey);
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const px = cx + dx;
          const py = cy + dy;
          if (px >= 0 && px < width && py >= 0 && py < height) {
            const falloff = Math.exp(-(dx * dx + dy * dy) * 0.9);
            const off = (py * width + px) * 3;
            floatRgb[off] += 1.1 * falloff;
            floatRgb[off + 1] += 0.42 * falloff;
            floatRgb[off + 2] += 0.08 * falloff;
          }
        }
      }
    }
  }

  if (fireflies > 0.35) {
    const count = Math.floor(fireflies * 26);
    for (let i = 0; i < count; i++) {
      const fx = (0.1 + 0.8 * hash2(i, 83) + 0.06 * Math.sin(tau + i * 1.9)) * width;
      const fy = (0.15 + 0.75 * hash2(i, 97) + 0.05 * Math.cos(tau + i * 2.3)) * height;
      const pulse = 0.4 + 0.6 * Math.max(0, Math.sin(tau * 2 + i));
      const cx = Math.floor(fx);
      const cy = Math.floor(fy);
      for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          const px = cx + dx;
          const py = cy + dy;
          if (px >= 0 && px < width && py >= 0 && py < height) {
            const g = Math.exp(-(dx * dx + dy * dy) * 0.45) * pulse;
            const off = (py * width + px) * 3;
            floatRgb[off] += 0.72 * g;
            floatRgb[off + 1] += 1.05 * g;
            floatRgb[off + 2] += 0.38 * g;
          }
        }
      }
    }
  }

  // Pass 4: ACES Filmic Tone Mapping + Specular Bloom + Subtle Cinema Vignette
  for (let py = 0; py < height; py++) {
    const ny = (py / maxHm1) * 2 - 1;
    for (let px = 0; px < width; px++) {
      const nx = (px / maxWm1) * 2 - 1;
      const sOff = (py * width + px) * 3;
      const dOff = (py * width + px) * 4;
      const vig = 1 - 0.16 * Math.min(1, (nx * nx + ny * ny) * 0.48);

      const lOff = (py * width + Math.max(0, px - 2)) * 3;
      const rOff = (py * width + Math.min(width - 1, px + 2)) * 3;

      for (let c = 0; c < 3; c++) {
        const val = floatRgb[sOff + c];
        const streak = (floatRgb[lOff + c] + floatRgb[rOff + c]) * 0.5;
        const flare = streak > 0.85 ? (streak - 0.85) * 0.32 : 0;
        const hdr = Math.max(0, (val + flare) * vig);
        // Fast ACES filmic approximation
        const mapped = (hdr * (2.51 * hdr + 0.03)) / (hdr * (2.43 * hdr + 0.59) + 0.14);
        outRgba[dOff + c] = Math.min(255, Math.max(0, Math.round(mapped * 255)));
      }
      outRgba[dOff + 3] = 255;
    }
  }
}
