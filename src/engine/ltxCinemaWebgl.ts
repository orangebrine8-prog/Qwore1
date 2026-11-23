import type { LtxCinemaPayload } from './ltx25DiT';
import { parseCinemaPrompt } from './cinemaCompositor';
import {
  createAudioBufferFromPcm,
  type NeuralAudioFrame,
  renderNeuralAudioPcm,
} from '../utils/neuralAudioSynth';

export function detectActorIdFromPrompt(prompt: string): number {
  const parsed = parseCinemaPrompt(prompt);
  const first = parsed.actors[0];
  if (!first) return 0;
  if (first === 'car') return 1;
  if (first === 'rocket') return 2;
  if (first === 'whale') return 3;
  if (first === 'crystal' || first === 'banana') return 4;
  if (first === 'robot' || first === 'astronaut') return 5;
  if (first === 'dragon' || first === 'bird') return 6;
  return 0;
}

const VERT_SRC = `#version 300 es
precision highp float;
layout(location = 0) in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

const FRAG_SRC = `#version 300 es
precision highp float;

in vec2 vUv;
out vec4 fragColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uCamMode;
uniform vec4 uLatentsA; // warmth, neon, space, water
uniform vec4 uLatentsB; // mountains, city, forest, volcano
uniform vec4 uLatentsC; // rain, fireflies, embers, kinetic
uniform vec4 uLatentsD; // godRays, studioProduct, neuralBlend, keyframeFrac
uniform vec4 uActorFlags; // actorId, hasEclipse, hasAurora, hasGrid
uniform sampler2D uNeuralTex0;
uniform sampler2D uNeuralTex1;

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

float noise2(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash12(i + vec2(0.0, 0.0)), hash12(i + vec2(1.0, 0.0)), u.x),
    mix(hash12(i + vec2(0.0, 1.0)), hash12(i + vec2(1.0, 1.0)), u.x),
    u.y
  );
}

float fbm4(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  mat2 rot = mat2(0.8, 0.6, -0.6, 0.8);
  for (int i = 0; i < 4; i++) {
    v += a * noise2(p);
    p = rot * p * 2.03 + vec2(17.1, 31.7);
    a *= 0.5;
  }
  return v;
}

float ridgedFbm(vec2 p) {
  float v = 0.0;
  float a = 0.52;
  float prev = 1.0;
  mat2 rot = mat2(0.8, -0.6, 0.6, 0.8);
  for (int i = 0; i < 5; i++) {
    float n = 1.0 - abs(noise2(p) * 2.0 - 1.0);
    n = n * n;
    v += n * a * prev;
    prev = clamp(n * 1.4, 0.2, 1.0);
    p = rot * p * 2.07 + vec2(11.3, 7.9);
    a *= 0.5;
  }
  return v;
}

float smin(float a, float b, float k) {
  float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

float sdBox(vec3 p, vec3 b) {
  vec3 q = abs(p) - b;
  return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
  vec3 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
}

mat2 rot2(float a) {
  float c = cos(a);
  float s = sin(a);
  return mat2(c, -s, s, c);
}

float terrainHeight(vec2 xz) {
  float mountains = uLatentsB.x;
  float volcano = uLatentsB.w;
  float water = uLatentsA.w;
  float city = uLatentsB.y;
  float studio = uLatentsD.y;

  if (studio > 0.58) {
    return -0.25 + 0.015 * sin(xz.x * 0.4) * cos(xz.y * 0.4);
  }
  if (city > 0.35 && mountains < 0.35) {
    // Urban boulevard plane
    return -0.1;
  }

  float scale = mix(0.2, 3.2, max(mountains, volcano));
  float ridge = ridgedFbm(xz * 0.16);
  float roll = fbm4(xz * 0.08 + vec2(4.2, 1.7));

  float h = (ridge * 0.72 + roll * 0.28) * scale;

  // Carve valley / river channel when water is present
  if (water > 0.3) {
    float channel = smoothstep(1.2, 7.5, abs(xz.x - sin(xz.y * 0.15) * 2.4));
    h *= mix(0.18, 1.0, channel);
    if (water > 0.75 && mountains < 0.35) {
      h = -0.35 + ridge * 0.22;
    }
  }

  // Volcanic cone silhouette in midground
  if (volcano > 0.4) {
    float d = length(xz - vec2(0.0, 14.0));
    float cone = max(0.0, 4.8 - d * 0.42);
    float caldera = exp(-d * d * 0.18) * 1.4;
    h = max(h * 0.5, cone - caldera);
  }

  return h - 0.55;
}

vec3 terrainNormal(vec2 xz, float tDist) {
  float eps = max(0.02, 0.0025 * tDist);
  float hL = terrainHeight(xz - vec2(eps, 0.0));
  float hR = terrainHeight(xz + vec2(eps, 0.0));
  float hD = terrainHeight(xz - vec2(0.0, eps));
  float hU = terrainHeight(xz + vec2(0.0, eps));
  return normalize(vec3(hL - hR, 2.0 * eps, hD - hU));
}

// 3D Raymarched Urban Skyline distance field (only active when city > 0.25)
vec2 mapCity(vec3 p) {
  float roadW = 3.2;
  if (abs(p.x) < roadW) {
    return vec2(p.y + 0.1, 1.0);
  }
  vec2 cell = floor(p.xz / 4.2);
  vec2 q = mod(p.xz, 4.2) - 2.1;
  float hHash = hash12(cell + vec2(19.0, 7.0));
  float bH = (2.2 + hHash * 8.5) * uLatentsB.y;
  float bW = 1.2 + 0.5 * hash12(cell + vec2(3.0, 11.0));
  float d = sdBox(vec3(q.x, p.y - bH * 0.5 + 0.1, q.y), vec3(bW, bH * 0.5, bW));
  return vec2(d, 2.0 + hHash);
}

// 3D Smooth-Min Sculpted Hero Subject SDF (strictly 3D volumetric geometry)
vec2 mapHeroActor(vec3 p, float actorId, float tau) {
  if (actorId < 0.5) return vec2(1e5, 0.0);

  if (actorId < 1.5) {
    // 1: 3D Hypercar gliding along wet avenue/coastal highway
    vec3 q = p - vec3(0.0, 0.36 + 0.015 * sin(tau * 6.0), 4.8 + 1.6 * sin(tau));
    q.xz *= rot2(0.06 * cos(tau));
    float body = sdBox(q, vec3(0.88, 0.22, 1.95)) - 0.12;
    float cabin = sdBox(q - vec3(0.0, 0.28, -0.1), vec3(0.64, 0.18, 0.92)) - 0.12;
    float chassis = smin(body, cabin, 0.16);
    float w1 = length(vec2(length(q.yz - vec2(-0.1, 1.2)) - 0.28, max(0.0, abs(q.x) - 0.92))) - 0.09;
    float w2 = length(vec2(length(q.yz - vec2(-0.1, -1.2)) - 0.28, max(0.0, abs(q.x) - 0.92))) - 0.09;
    float wheels = min(w1, w2);
    if (wheels < chassis) return vec2(wheels, 11.0);
    return vec2(chassis, 10.0);
  }

  if (actorId < 2.5) {
    // 2: 3D Ascending Rocket Starship
    float lift = mod(uTime * 1.15, 1.0) * 5.2 - 1.4;
    vec3 q = p - vec3(0.25 * sin(tau), 1.8 + lift, 8.5);
    q.xy *= rot2(0.08 * cos(tau));
    float hull = sdCapsule(q, vec3(0.0, -1.2, 0.0), vec3(0.0, 1.1, 0.0), 0.36);
    float nose = sdCapsule(q, vec3(0.0, 0.9, 0.0), vec3(0.0, 1.85, 0.0), mix(0.36, 0.04, clamp((q.y - 0.9) / 0.95, 0.0, 1.0)));
    float fins = sdBox(vec3(abs(q.x), q.y + 0.95, abs(q.z)), vec3(0.68, 0.32, 0.05)) - 0.02;
    float d = smin(min(hull, nose), fins, 0.08);
    return vec2(d, 12.0);
  }

  if (actorId < 3.5) {
    // 3: 3D Breaching Ocean Whale
    float arc = sin(tau);
    vec3 q = p - vec3(0.0, -0.15 + max(0.0, arc) * 1.45, 7.2);
    q.yz *= rot2(-0.55 * cos(tau));
    float spine = sdCapsule(q, vec3(0.0, -0.1, -1.2), vec3(0.0, 0.2, 1.15), 0.44 - 0.14 * abs(q.z));
    float tail = sdBox(q - vec3(0.0, -0.05, -1.45), vec3(0.68, 0.06, 0.28)) - 0.04;
    return vec2(smin(spine, tail, 0.15), 13.0);
  }

  if (actorId < 4.5) {
    // 4: 3D Iridescent Floating Crystal Monolith
    vec3 q = p - vec3(0.0, 1.35 + 0.16 * sin(tau), 5.6);
    q.xz *= rot2(tau);
    q.xy *= rot2(0.25 * sin(tau));
    float d = (abs(q.x) + abs(q.y * 0.55) + abs(q.z) - 0.95) * 0.58;
    return vec2(d, 14.0);
  }

  if (actorId < 5.5) {
    // 5: 3D Hovering Autonomous Recon Drone / Mech Sphere
    vec3 q = p - vec3(0.45 * sin(tau), 1.45 + 0.18 * cos(tau * 2.0), 5.4);
    q.xz *= rot2(0.4 * sin(tau));
    float core = length(q) - 0.56;
    float ring = length(vec2(length(q.xz) - 0.85, q.y)) - 0.09;
    return vec2(smin(core, ring, 0.1), 15.0);
  }

  // 6: 3D Soaring Dragon / Raptor
  vec3 q = p - vec3(0.75 * sin(tau), 2.15 + 0.3 * cos(tau * 2.0), 8.2);
  q.xz *= rot2(-0.35 * cos(tau));
  float body = sdCapsule(q, vec3(0.0, 0.0, -0.9), vec3(0.0, 0.1, 0.85), 0.24);
  float flap = sin(tau * 4.0 - abs(q.x) * 1.4) * 0.35;
  float wings = sdBox(vec3(abs(q.x), q.y - flap * abs(q.x) * 0.45, q.z), vec3(1.65, 0.05, 0.42)) - 0.03;
  return vec2(smin(body, wings, 0.12), 16.0);
}

vec3 calcHeroNormal(vec3 p, float actorId, float tau) {
  vec2 e = vec2(0.008, 0.0);
  return normalize(vec3(
    mapHeroActor(p + e.xyy, actorId, tau).x - mapHeroActor(p - e.xyy, actorId, tau).x,
    mapHeroActor(p + e.yxy, actorId, tau).x - mapHeroActor(p - e.yxy, actorId, tau).x,
    mapHeroActor(p + e.yyx, actorId, tau).x - mapHeroActor(p - e.yyx, actorId, tau).x
  ));
}

// Physical Rayleigh + Mie sky + volumetric clouds + aurora
vec3 evalSkyAndClouds(vec3 ro, vec3 rd, vec3 sunDir, float tau) {
  float warmth = uLatentsA.x;
  float neon = uLatentsA.y;
  float space = uLatentsA.z;
  float hasEclipse = uActorFlags.y;
  float hasAurora = uActorFlags.z;

  float yUp = max(0.0, rd.y);
  vec3 zenithCol = mix(vec3(0.06, 0.14, 0.34), vec3(0.02, 0.03, 0.08), clamp(space + neon * 0.45, 0.0, 1.0));
  vec3 horizCol = mix(
    vec3(0.45, 0.62, 0.82),
    mix(vec3(0.96, 0.48, 0.22), vec3(0.28, 0.12, 0.48), neon),
    clamp(warmth + neon * 0.5, 0.0, 1.0)
  );
  if (space > 0.55) {
    horizCol = mix(horizCol, vec3(0.14, 0.06, 0.12), (space - 0.55) * 1.8);
  }

  vec3 col = mix(horizCol, zenithCol, pow(yUp, 0.55));

  // Solar / Eclipse Mie scattering glow
  float sunDot = max(dot(rd, sunDir), 0.0);
  if (hasEclipse > 0.5) {
    float ang = acos(clamp(sunDot, -1.0, 1.0));
    float corona = exp(-pow((ang - 0.095) * 32.0, 2.0));
    float outer = exp(-ang * 7.5);
    col += vec3(1.4, 0.38, 0.16) * corona + vec3(0.45, 0.12, 0.18) * outer;
    if (ang < 0.088) col *= 0.03;
  } else {
    float mieGlow = pow(sunDot, 10.0) * (0.45 + 0.55 * warmth);
    float sunDisk = smoothstep(0.9985, 0.9994, sunDot);
    vec3 sunTint = mix(vec3(0.85, 0.92, 1.0), vec3(1.25, 0.72, 0.36), warmth);
    col += sunTint * mieGlow + sunTint * sunDisk * 3.8;
  }

  // Deep Space Starfield
  if (space > 0.25 || neon > 0.35 || hasAurora > 0.5) {
    vec2 starUv = floor(rd.xz / (abs(rd.y) + 0.35) * 140.0);
    float sh = hash12(starUv);
    if (sh > 0.992) {
      float twinkle = 0.65 + 0.35 * sin(tau * 2.0 + sh * 80.0);
      col += vec3(0.85, 0.9, 1.0) * smoothstep(0.992, 1.0, sh) * twinkle * smoothstep(0.02, 0.25, yUp);
    }
  }

  // Volumetric Aurora Borealis Curtains
  if (hasAurora > 0.5 || (uLatentsB.x > 0.45 && space > 0.2 && warmth < 0.4)) {
    if (rd.y > 0.02) {
      vec2 pA = rd.xz / (rd.y + 0.16);
      float ribbon = sin(pA.x * 1.4 + tau * 0.5 + fbm4(pA * 0.45 + vec2(uTime * 0.3, 0.0)) * 3.2);
      float curtain = exp(-abs(ribbon) * 4.2) * smoothstep(0.02, 0.22, rd.y) * smoothstep(0.92, 0.25, rd.y);
      vec3 aurCol = mix(vec3(0.1, 0.95, 0.52), vec3(0.55, 0.18, 0.95), smoothstep(0.18, 0.55, rd.y));
      col += aurCol * curtain * 0.85;
    }
  }

  // Volumetric 3D Raymarched Clouds (10 optical scattering steps)
  if (rd.y > 0.015 && space < 0.72) {
    float cloudBase = 18.0;
    float cloudTop = 27.0;
    float t0 = (cloudBase - ro.y) / max(0.03, rd.y);
    float t1 = (cloudTop - ro.y) / max(0.03, rd.y);
    if (t0 > 0.0 && t1 > t0) {
      float dt = (t1 - t0) / 10.0;
      float transmittance = 1.0;
      vec3 cloudLight = vec3(0.0);
      float hgPhase = 0.35 + 0.65 * pow(sunDot, 5.0);
      vec3 cSunCol = mix(vec3(0.92, 0.95, 1.0), vec3(1.15, 0.62, 0.32), warmth);
      vec3 cAmbCol = mix(vec3(0.28, 0.36, 0.52), vec3(0.22, 0.14, 0.32), neon);

      for (int i = 0; i < 10; i++) {
        float tCur = t0 + (float(i) + 0.5) * dt;
        vec3 pos = ro + rd * tCur;
        float hNorm = (pos.y - cloudBase) / (cloudTop - cloudBase);
        float dens = fbm4(pos.xz * 0.045 + vec2(uTime * 0.45, uTime * 0.12));
        dens = smoothstep(0.46, 0.82, dens) * sin(hNorm * 3.14159) * 0.55;
        if (dens > 0.005) {
          float sunDensity = fbm4((pos.xz + sunDir.xz * 2.2) * 0.045 + vec2(uTime * 0.45, 0.0));
          float selfShadow = exp(-max(0.0, sunDensity - 0.42) * 3.5);
          vec3 sampleLight = cAmbCol * (0.6 + 0.4 * hNorm) + cSunCol * selfShadow * (0.7 + 0.9 * hgPhase);
          float alpha = 1.0 - exp(-dens * dt * 0.12);
          cloudLight += transmittance * sampleLight * alpha;
          transmittance *= (1.0 - alpha);
          if (transmittance < 0.04) break;
        }
      }
      float distFade = exp(-t0 * 0.0022);
      col = mix(col, cloudLight + col * transmittance, distFade * (1.0 - space));
    }
  }

  return col;
}

// 4-Wave Gerstner Ocean Surface Normal
vec3 evalWaterNormal(vec2 p, float tau) {
  float eps = 0.035;
  float h0 =
    0.045 * sin(p.x * 1.9 + tau * 2.0 + p.y * 0.7) +
    0.028 * cos(p.x * 3.3 - p.y * 2.6 + tau * 3.0) +
    0.015 * sin(p.x * 6.8 + p.y * 5.2 - tau * 4.0) +
    0.01 * fbm4(p * 2.2 + vec2(tau * 0.25, -tau * 0.2));
  float hx =
    0.045 * sin((p.x + eps) * 1.9 + tau * 2.0 + p.y * 0.7) +
    0.028 * cos((p.x + eps) * 3.3 - p.y * 2.6 + tau * 3.0) +
    0.015 * sin((p.x + eps) * 6.8 + p.y * 5.2 - tau * 4.0) +
    0.01 * fbm4((p + vec2(eps, 0.0)) * 2.2 + vec2(tau * 0.25, -tau * 0.2));
  float hz =
    0.045 * sin(p.x * 1.9 + tau * 2.0 + (p.y + eps) * 0.7) +
    0.028 * cos(p.x * 3.3 - (p.y + eps) * 2.6 + tau * 3.0) +
    0.015 * sin(p.x * 6.8 + (p.y + eps) * 5.2 - tau * 4.0) +
    0.01 * fbm4((p + vec2(0.0, eps)) * 2.2 + vec2(tau * 0.25, -tau * 0.2));
  return normalize(vec3(h0 - hx, eps, h0 - hz));
}

float calcTerrainShadow(vec3 ro, vec3 sunDir) {
  float res = 1.0;
  float t = 0.18;
  for (int i = 0; i < 16; i++) {
    vec3 p = ro + sunDir * t;
    float h = terrainHeight(p.xz);
    float d = p.y - h;
    if (d < 0.01) return 0.12;
    res = min(res, 7.5 * d / t);
    t += clamp(d * 0.75, 0.25, 1.8);
    if (t > 24.0) break;
  }
  return clamp(res, 0.12, 1.0);
}

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution) / max(1.0, uResolution.y);
  float tau = uTime * 6.28318530718;

  float warmth = uLatentsA.x;
  float neon = uLatentsA.y;
  float space = uLatentsA.z;
  float water = uLatentsA.w;
  float mountains = uLatentsB.x;
  float city = uLatentsB.y;
  float forest = uLatentsB.z;
  float volcano = uLatentsB.w;
  float rain = uLatentsC.x;
  float fireflies = uLatentsC.y;
  float embers = uLatentsC.z;
  float godRays = uLatentsD.x;
  float studio = uLatentsD.y;
  float neuralBlend = uLatentsD.z;
  float kfFrac = uLatentsD.w;
  float actorId = uActorFlags.x;

  // True 3D Camera Trajectory Matrix
  vec3 ro = vec3(0.0, 1.55, -2.5);
  vec3 ta = vec3(0.0, 0.85, 8.0);
  int camMode = int(uCamMode + 0.5);
  if (camMode == 1) {
    // dolly-in
    ro.z = -2.8 + 2.4 * (0.5 - 0.5 * cos(tau));
    ro.y = 1.45 - 0.25 * (0.5 - 0.5 * cos(tau));
  } else if (camMode == 2) {
    // orbit
    ro.x = 2.6 * sin(tau);
    ro.z = -2.2 + 0.8 * cos(tau);
    ro.y = 1.7 + 0.25 * sin(tau);
  } else if (camMode == 3) {
    // pan-right
    ro.x = 2.4 * sin(tau);
    ta.x = 2.8 * sin(tau);
  } else if (camMode == 4) {
    // crane-up
    ro.y = 1.15 + 1.45 * (0.5 - 0.5 * cos(tau));
    ta.y = 0.65 + 0.45 * (0.5 - 0.5 * cos(tau));
  } else if (camMode == 5) {
    // handheld cinema stabilizer
    ro.x = 0.18 * sin(tau * 2.0) + 0.07 * cos(tau * 3.0);
    ro.y = 1.5 + 0.14 * cos(tau * 2.0);
  }

  // Ensure camera always clears local terrain height
  ro.y = max(ro.y, terrainHeight(ro.xz) + 0.75);

  vec3 ww = normalize(ta - ro);
  vec3 uu = normalize(cross(vec3(0.0, 1.0, 0.0), ww));
  vec3 vv = cross(ww, uu);
  vec3 rd = normalize(uv.x * uu + uv.y * vv + 1.35 * ww);

  vec3 sunDir = normalize(vec3(
    0.35 * sin(tau * 0.5) + 0.22,
    mix(0.32, 0.14, warmth),
    0.92
  ));

  vec3 skyCol = evalSkyAndClouds(ro, rd, sunDir, tau);
  vec3 col = skyCol;
  float bestT = 100.0;
  int hitType = 0; // 0=sky, 1=terrain, 2=water, 3=city, 4=hero
  vec3 hitPos = ro;
  vec3 hitNorm = vec3(0.0, 1.0, 0.0);
  float cityHash = 0.0;
  float heroMatId = 0.0;

  // 1) Raymarch 3D Terrain Heightfield
  if (city < 0.35 || mountains > 0.35) {
    float t = 0.2;
    for (int i = 0; i < 64; i++) {
      vec3 p = ro + rd * t;
      float h = terrainHeight(p.xz);
      float d = p.y - h;
      if (d < 0.008 * t) {
        bestT = t;
        hitPos = p;
        hitNorm = terrainNormal(p.xz, t);
        hitType = 1;
        break;
      }
      t += max(0.06, d * 0.48);
      if (t > 65.0) break;
    }
  }

  // 2) Raymarch 3D Urban Skyline (if city > 0.25)
  if (city > 0.25) {
    float t = 0.3;
    for (int i = 0; i < 52; i++) {
      vec3 p = ro + rd * t;
      vec2 res = mapCity(p);
      if (res.x < 0.015) {
        if (t < bestT) {
          bestT = t;
          hitPos = p;
          cityHash = res.y;
          vec2 e = vec2(0.02, 0.0);
          hitNorm = normalize(vec3(
            mapCity(p + e.xyy).x - mapCity(p - e.xyy).x,
            mapCity(p + e.yxy).x - mapCity(p - e.yxy).x,
            mapCity(p + e.yyx).x - mapCity(p - e.yyx).x
          ));
          hitType = (res.y < 1.5) ? 2 : 3;
        }
        break;
      }
      t += max(0.05, res.x * 0.72);
      if (t > 55.0) break;
    }
  }

  // 3) Analytic Water / Wet Reflection Plane Intersection
  float waterLevel = -0.1;
  if ((water > 0.22 || rain > 0.3) && studio < 0.55 && rd.y < -0.002) {
    float tw = (waterLevel - ro.y) / rd.y;
    if (tw > 0.1 && tw < bestT) {
      bestT = tw;
      hitPos = ro + rd * tw;
      hitType = 2;
    }
  }

  // 4) Raymarch 3D Sculpted Hero Actor (only if explicitly prompted)
  if (actorId > 0.5) {
    float t = 0.2;
    for (int i = 0; i < 48; i++) {
      vec3 p = ro + rd * t;
      vec2 res = mapHeroActor(p, actorId, tau);
      if (res.x < 0.01) {
        if (t < bestT) {
          bestT = t;
          hitPos = p;
          hitNorm = calcHeroNormal(p, actorId, tau);
          heroMatId = res.y;
          hitType = 4;
        }
        break;
      }
      t += max(0.04, res.x * 0.8);
      if (t > 28.0) break;
    }
  }

  // PBR Shading Stage
  vec3 sunCol = mix(vec3(0.95, 0.97, 1.0), vec3(1.35, 0.76, 0.42), warmth);
  vec3 skyAmb = mix(vec3(0.16, 0.24, 0.38), vec3(0.24, 0.14, 0.32), neon);

  if (hitType == 1) {
    // 3D Mountain / Alpine / Forest / Volcano Terrain PBR
    float slope = hitNorm.y;
    float elev = hitPos.y;
    float rockNoise = fbm4(hitPos.xz * 2.5);
    vec3 rockAlbedo = mix(vec3(0.09, 0.095, 0.11), vec3(0.2, 0.17, 0.15), rockNoise);

    // Forest biome
    float treeMask = smoothstep(0.62, 0.82, slope) * smoothstep(-0.1, 0.7, elev) * forest;
    vec3 forestAlbedo = vec3(0.05, 0.14, 0.07) * (0.7 + 0.6 * rockNoise);
    vec3 albedo = mix(rockAlbedo, forestAlbedo, treeMask);

    // Alpine snow caps
    float snowMask = smoothstep(0.66, 0.85, slope) * smoothstep(0.8, 1.85, elev + 0.3 * rockNoise) * (1.0 - volcano);
    albedo = mix(albedo, vec3(0.86, 0.9, 0.95), snowMask);

    // Studio product dark brushed slate
    if (studio > 0.55) {
      albedo = vec3(0.06, 0.065, 0.08) * (0.85 + 0.3 * rockNoise);
    }

    float sh = calcTerrainShadow(hitPos + hitNorm * 0.04, sunDir);
    float diff = max(dot(hitNorm, sunDir), 0.0);
    float amb = 0.35 + 0.65 * max(0.0, hitNorm.y);
    col = albedo * (sunCol * diff * sh * 1.45 + skyAmb * amb * 0.75);

    // Glowing molten magma rivers on volcanoes
    if (volcano > 0.4) {
      float fissure = ridgedFbm(hitPos.xz * 1.4 - vec2(0.0, uTime * 0.35));
      float magma = smoothstep(0.72, 0.94, fissure) * smoothstep(2.2, 0.1, abs(hitPos.x));
      col += vec3(1.8, 0.42, 0.05) * magma * 1.35;
    }

    // Distance atmospheric aerial perspective fog
    float fog = 1.0 - exp(-bestT * 0.022);
    col = mix(col, skyCol, clamp(fog, 0.0, 0.94));
  } else if (hitType == 2) {
    // Physically Based Ocean / Wet Road Fresnel Reflection
    vec3 wNorm = evalWaterNormal(hitPos.xz, tau);
    if (city > 0.35 && water < 0.5) {
      // Rainy wet urban asphalt with micro-bump & puddle ripples
      float micro = (noise2(hitPos.xz * 18.0) - 0.5) * 0.08;
      wNorm = normalize(vec3(micro, 1.0, micro * 0.7));
      if (rain > 0.2) {
        vec2 cell = floor(hitPos.xz * 2.5);
        vec2 f = fract(hitPos.xz * 2.5) - 0.5;
        float ph = fract(hash12(cell) + uTime * 2.2);
        float d = length(f);
        float ripple = exp(-pow((d - ph * 0.45) * 28.0, 2.0)) * (1.0 - ph) * rain;
        wNorm = normalize(wNorm + vec3(f.x, 0.0, f.y) * ripple * 0.4);
      }
    }

    vec3 reflDir = reflect(rd, wNorm);
    reflDir.y = max(0.02, reflDir.y);
    vec3 reflCol = evalSkyAndClouds(hitPos, reflDir, sunDir, tau);

    // Reflect cyber skyscrapers on wet road
    if (city > 0.35) {
      float neonBands = 0.5 + 0.5 * sin(hitPos.x * 2.8 + fbm4(hitPos.xz * 0.8) * 3.0);
      reflCol += mix(vec3(0.05, 0.65, 1.0), vec3(1.0, 0.15, 0.6), neonBands) * 0.25 * neon;
    }

    // Schlick Fresnel
    float cosTheta = clamp(dot(-rd, wNorm), 0.0, 1.0);
    float fresnel = 0.04 + 0.96 * pow(1.0 - cosTheta, 4.5);

    vec3 deepWater = water > 0.3 ? vec3(0.015, 0.09, 0.15) : vec3(0.025, 0.028, 0.04);
    float specSun = pow(max(dot(reflDir, sunDir), 0.0), 120.0) * 3.2;
    float foam = water > 0.4 ? smoothstep(0.84, 0.96, fbm4(hitPos.xz * 3.5 + vec2(tau * 0.2, 0.0))) * 0.16 : 0.0;

    col = mix(deepWater, reflCol, fresnel) + sunCol * specSun + vec3(0.75, 0.85, 0.92) * foam;
    float fog = 1.0 - exp(-bestT * 0.024);
    col = mix(col, skyCol, clamp(fog, 0.0, 0.95));
  } else if (hitType == 3) {
    // 3D Raymarched Skyscraper Architecture
    float diff = max(dot(hitNorm, sunDir), 0.0);
    vec3 facade = vec3(0.04, 0.048, 0.07) * (0.4 + 0.6 * diff) + skyAmb * 0.3;
    float isWindowWall = abs(hitNorm.y) < 0.5 ? 1.0 : 0.0;
    vec2 winUv = vec2(hitPos.x + hitPos.z, hitPos.y) * vec2(2.6, 2.1);
    vec2 winCell = floor(winUv);
    vec2 winFrac = fract(winUv);
    float pane = step(0.18, winFrac.x) * step(winFrac.x, 0.82) * step(0.2, winFrac.y) * step(winFrac.y, 0.82);
    float lit = step(0.45, hash12(winCell + vec2(cityHash * 17.0, 3.0)));
    vec3 winCol = mod(cityHash * 10.0, 2.0) > 1.0
      ? vec3(0.18, 0.82, 1.0)
      : mix(vec3(1.0, 0.68, 0.35), vec3(1.0, 0.22, 0.65), neon);
    facade += winCol * pane * lit * isWindowWall * (0.55 + 0.75 * neon);
    float fog = 1.0 - exp(-bestT * 0.032);
    col = mix(facade, skyCol, clamp(fog, 0.0, 0.95));
  } else if (hitType == 4) {
    // 3D PBR Sculpted Hero Subject Shading
    float diff = max(dot(hitNorm, sunDir), 0.0);
    float fresnel = pow(1.0 - max(dot(-rd, hitNorm), 0.0), 4.0);
    vec3 refl = evalSkyAndClouds(hitPos, reflect(rd, hitNorm), sunDir, tau);
    vec3 baseAlbedo = vec3(0.12, 0.14, 0.18);
    float metallic = 0.82;
    if (heroMatId > 10.5 && heroMatId < 11.5) {
      baseAlbedo = vec3(0.04, 0.04, 0.045);
      metallic = 0.25;
    } else if (heroMatId > 12.5 && heroMatId < 13.5) {
      baseAlbedo = vec3(0.08, 0.16, 0.24);
      metallic = 0.35;
    } else if (heroMatId > 13.5 && heroMatId < 14.5) {
      baseAlbedo = vec3(0.18, 0.65, 0.95);
      metallic = 0.9;
    } else if (heroMatId > 15.5) {
      baseAlbedo = vec3(0.28, 0.14, 0.12);
      metallic = 0.35;
    }
    col = baseAlbedo * (sunCol * diff * 0.9 + skyAmb * 0.55) + refl * mix(fresnel, 0.65, metallic);
    float fog = 1.0 - exp(-bestT * 0.02);
    col = mix(col, skyCol, clamp(fog, 0.0, 0.85));
  }

  // Volumetric Atmospheric God-Rays
  if (godRays > 0.3) {
    float rayAngle = atan(rd.y - sunDir.y, rd.x - sunDir.x);
    float beams = pow(max(0.0, sin(rayAngle * 7.0 + tau * 0.35)), 4.0) * pow(max(dot(rd, sunDir), 0.0), 2.5);
    col += sunCol * beams * 0.22 * godRays;
  }

  // Subtle LTX-2.5 Neural Latent Lighting Coupling (interpolated in time between denoised keyframes)
  vec3 nf0 = texture(uNeuralTex0, vec2(vUv.x, 1.0 - vUv.y)).rgb;
  vec3 nf1 = texture(uNeuralTex1, vec2(vUv.x, 1.0 - vUv.y)).rgb;
  vec3 neuralTint = mix(nf0, nf1, kfFrac);
  col = mix(col, col * (0.72 + 0.58 * neuralTint) + neuralTint * 0.08, neuralBlend * 0.42);

  // Perspective Rain Streaks & Bokeh Embers / Fireflies
  if (rain > 0.28) {
    vec2 rUv = vec2(vUv.x * 46.0 + vUv.y * 6.0, vUv.y * 12.0 + uTime * 9.0);
    vec2 rCell = floor(rUv);
    vec2 rF = fract(rUv);
    float rDrop = step(0.92, hash12(rCell)) * exp(-abs(rF.x - 0.5) * 18.0) * smoothstep(0.0, 0.4, rF.y) * smoothstep(1.0, 0.4, rF.y);
    col += vec3(0.32, 0.45, 0.65) * rDrop * rain * 0.65;
  }
  if (embers > 0.3 || volcano > 0.4) {
    vec2 eUv = vec2(vUv.x * 18.0 + sin(uTime * 3.0 + vUv.y * 4.0) * 0.35, vUv.y * 10.0 - uTime * 3.5);
    vec2 eCell = floor(eUv);
    vec2 eF = fract(eUv) - 0.5;
    float eOn = step(0.88, hash12(eCell + 5.1));
    float glow = exp(-dot(eF, eF) * 28.0) * eOn;
    col += vec3(1.5, 0.48, 0.1) * glow * max(embers, volcano);
  }
  if (fireflies > 0.35) {
    vec2 fUv = vec2(vUv.x * 11.0 + 0.4 * sin(tau + vUv.y * 3.0), vUv.y * 7.0 + 0.3 * cos(tau + vUv.x * 3.0));
    vec2 fCell = floor(fUv);
    vec2 fF = fract(fUv) - 0.5;
    float fOn = step(0.9, hash12(fCell + 12.7)) * (0.5 + 0.5 * sin(tau * 2.0 + hash12(fCell) * 12.0));
    float glow = exp(-dot(fF, fF) * 18.0) * fOn;
    col += vec3(0.75, 1.15, 0.42) * glow * fireflies;
  }

  // ACES Filmic Tone Mapping + Optical Vignette + Organic 35mm Micro-Grain
  float vig = 1.0 - 0.22 * dot(uv, uv);
  col = max(vec3(0.0), col * vig);
  col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);
  float grain = (hash12(gl_FragCoord.xy + fract(uTime) * 317.0) - 0.5) * 0.018;
  col = clamp(pow(max(col, vec3(0.0)), vec3(0.94)) + grain, 0.0, 1.0);

  fragColor = vec4(col, 1.0);
}
`;

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error('Unable to create WebGL2 shader');
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader) || 'Unknown shader compile error';
    gl.deleteShader(shader);
    throw new Error(info);
  }
  return shader;
}

function cameraModeToFloat(mode: LtxCinemaPayload['cameraMotion']): number {
  if (mode === 'dolly-in') return 1;
  if (mode === 'orbit') return 2;
  if (mode === 'pan-right') return 3;
  if (mode === 'crane-up') return 4;
  if (mode === 'handheld') return 5;
  return 0;
}

export class LtxWebglRenderer {
  private readonly gl: WebGL2RenderingContext;
  private readonly program: WebGLProgram;
  private readonly vao: WebGLVertexArrayObject;
  private readonly vbo: WebGLBuffer;
  private readonly tex0: WebGLTexture;
  private readonly tex1: WebGLTexture;
  private readonly uniforms: Record<string, WebGLUniformLocation | null>;
  private uploadedKeyframePair = '';
  private disposed = false;

  constructor(canvas: HTMLCanvasElement | OffscreenCanvas) {
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
    }) as WebGL2RenderingContext | null;
    if (!gl) {
      throw new Error('WebGL2 unavailable');
    }
    this.gl = gl;

    const vs = compileShader(gl, gl.VERTEX_SHADER, VERT_SRC);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, FRAG_SRC);
    const prog = gl.createProgram();
    if (!prog) throw new Error('Unable to create WebGL2 program');
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(prog) || 'WebGL2 link error';
      gl.deleteProgram(prog);
      throw new Error(info);
    }
    this.program = prog;

    const vao = gl.createVertexArray();
    const vbo = gl.createBuffer();
    if (!vao || !vbo) throw new Error('Unable to allocate VAO/VBO');
    this.vao = vao;
    this.vbo = vbo;

    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
      gl.STATIC_DRAW,
    );
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    const createKfTex = () => {
      const tex = gl.createTexture();
      if (!tex) throw new Error('Unable to create texture');
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return tex;
    };
    this.tex0 = createKfTex();
    this.tex1 = createKfTex();

    this.uniforms = {
      uResolution: gl.getUniformLocation(prog, 'uResolution'),
      uTime: gl.getUniformLocation(prog, 'uTime'),
      uCamMode: gl.getUniformLocation(prog, 'uCamMode'),
      uLatentsA: gl.getUniformLocation(prog, 'uLatentsA'),
      uLatentsB: gl.getUniformLocation(prog, 'uLatentsB'),
      uLatentsC: gl.getUniformLocation(prog, 'uLatentsC'),
      uLatentsD: gl.getUniformLocation(prog, 'uLatentsD'),
      uActorFlags: gl.getUniformLocation(prog, 'uActorFlags'),
      uNeuralTex0: gl.getUniformLocation(prog, 'uNeuralTex0'),
      uNeuralTex1: gl.getUniformLocation(prog, 'uNeuralTex1'),
    };
  }

  private uploadKeyframe(tex: WebGLTexture, kf: Float32Array | undefined, w: number, h: number): void {
    const gl = this.gl;
    const rgba = new Uint8Array(w * h * 4);
    if (kf && kf.length >= w * h * 3) {
      for (let i = 0, p = 0; i < w * h; i++, p += 3) {
        const d = i * 4;
        rgba[d] = Math.min(255, Math.max(0, Math.round(kf[p] * 255)));
        rgba[d + 1] = Math.min(255, Math.max(0, Math.round(kf[p + 1] * 255)));
        rgba[d + 2] = Math.min(255, Math.max(0, Math.round(kf[p + 2] * 255)));
        rgba[d + 3] = 255;
      }
    } else {
      rgba.fill(128);
    }
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
  }

  public render(payload: LtxCinemaPayload, tNorm: number, width: number, height: number): void {
    if (this.disposed) return;
    const gl = this.gl;
    const t = ((tNorm % 1) + 1) % 1;
    const kfs = payload.neuralKeyframes;
    const numKfs = Math.max(1, kfs.length);
    const scaledT = t * (numKfs - 1);
    const k0 = Math.min(numKfs - 1, Math.floor(scaledT));
    const k1 = Math.min(numKfs - 1, k0 + 1);
    const kFrac = scaledT - k0;

    const pairKey = `${payload.prompt}:${payload.cameraMotion}:${payload.trainingStep}:${k0}:${k1}:${payload.keyframeW}x${payload.keyframeH}`;
    if (this.uploadedKeyframePair !== pairKey) {
      this.uploadKeyframe(this.tex0, kfs[k0], payload.keyframeW, payload.keyframeH);
      this.uploadKeyframe(this.tex1, kfs[k1], payload.keyframeW, payload.keyframeH);
      this.uploadedKeyframePair = pairKey;
    }

    const L = payload.sceneLatents;
    const parsed = parseCinemaPrompt(payload.prompt);
    const actorId = detectActorIdFromPrompt(payload.prompt);

    gl.viewport(0, 0, width, height);
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);

    gl.uniform2f(this.uniforms.uResolution, width, height);
    gl.uniform1f(this.uniforms.uTime, t);
    gl.uniform1f(this.uniforms.uCamMode, cameraModeToFloat(payload.cameraMotion));
    gl.uniform4f(this.uniforms.uLatentsA, L[0] ?? 0.5, L[1] ?? 0.2, L[2] ?? 0.2, L[3] ?? 0.4);
    gl.uniform4f(this.uniforms.uLatentsB, L[4] ?? 0.3, L[5] ?? 0.0, L[6] ?? 0.1, L[7] ?? 0.0);
    gl.uniform4f(this.uniforms.uLatentsC, L[8] ?? 0.0, L[9] ?? 0.1, L[10] ?? 0.0, L[11] ?? 0.5);
    gl.uniform4f(
      this.uniforms.uLatentsD,
      L[12] ?? 0.4,
      L[13] ?? 0.0,
      payload.neuralBlendWeight,
      kFrac,
    );
    gl.uniform4f(
      this.uniforms.uActorFlags,
      actorId,
      parsed.hasEclipse ? 1 : 0,
      parsed.hasAurora ? 1 : 0,
      parsed.hasGrid ? 1 : 0,
    );

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex0);
    gl.uniform1i(this.uniforms.uNeuralTex0, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.tex1);
    gl.uniform1i(this.uniforms.uNeuralTex1, 1);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
  }

  public dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    const gl = this.gl;
    gl.deleteTexture(this.tex0);
    gl.deleteTexture(this.tex1);
    gl.deleteBuffer(this.vbo);
    gl.deleteVertexArray(this.vao);
    gl.deleteProgram(this.program);
  }
}

export function renderLtxStillToDataUrl(
  payload: LtxCinemaPayload,
  tNorm = 0.35,
  width = payload.aspectRatio === '16:9' ? 960 : 640,
  height = payload.aspectRatio === '16:9' ? 540 : 640,
): string | null {
  if (typeof document === 'undefined') return null;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  let renderer: LtxWebglRenderer | null = null;
  try {
    renderer = new LtxWebglRenderer(canvas);
    renderer.render(payload, tNorm, width, height);
    return canvas.toDataURL('image/png');
  } catch {
    return null;
  } finally {
    renderer?.dispose();
  }
}

export async function exportLtxCinemaToWebm(
  payload: LtxCinemaPayload,
  audioTrack?: NeuralAudioFrame[],
  outWidth = payload.aspectRatio === '16:9' ? 960 : 640,
  outHeight = payload.aspectRatio === '16:9' ? 540 : 640,
): Promise<Blob> {
  if (typeof MediaRecorder === 'undefined') {
    throw new Error('Video recording is not supported in this browser.');
  }
  const canvas = document.createElement('canvas');
  canvas.width = outWidth;
  canvas.height = outHeight;
  if (typeof canvas.captureStream !== 'function') {
    throw new Error('Canvas stream recording is unavailable in this browser.');
  }

  const renderer = new LtxWebglRenderer(canvas);
  const fps = Math.max(24, payload.fps || 60);
  const stream = canvas.captureStream(fps);
  const tracks: MediaStreamTrack[] = [...stream.getVideoTracks()];

  let audioCtx: AudioContext | null = null;
  let audioSource: AudioBufferSourceNode | null = null;
  const cycleSec = Math.max(2, payload.durationSec || 4);
  const totalDurationMs = cycleSec * 1000;

  if (Array.isArray(audioTrack) && audioTrack.length > 0 && typeof AudioContext !== 'undefined') {
    try {
      audioCtx = new AudioContext();
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume().catch(() => {});
      }
      const pcm = renderNeuralAudioPcm(audioTrack, cycleSec, audioCtx.sampleRate);
      const buffer = createAudioBufferFromPcm(ctxOrThrow(audioCtx), pcm);
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

  const combined = new MediaStream(tracks);
  const mimeCandidates = audioSource
    ? ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm']
    : ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  const mimeType = mimeCandidates.find((m) => MediaRecorder.isTypeSupported(m)) ?? '';

  const recorder = mimeType
    ? new MediaRecorder(combined, { mimeType, videoBitsPerSecond: 6_000_000 })
    : new MediaRecorder(combined);

  const chunks: BlobPart[] = [];
  recorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) chunks.push(e.data);
  };

  const stopped = new Promise<Blob>((resolve, reject) => {
    recorder.onstop = () => resolve(new Blob(chunks, { type: mimeType || 'video/webm' }));
    recorder.onerror = () => reject(new Error('MediaRecorder failed while encoding video.'));
  });

  try {
    renderer.render(payload, 0, outWidth, outHeight);
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
        const tNorm = (elapsed / totalDurationMs) % 1;
        renderer.render(payload, tNorm, outWidth, outHeight);
        requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });

    if (audioSource) {
      try {
        audioSource.stop();
      } catch {
        // Ignore
      }
    }
    recorder.stop();
    return await stopped;
  } finally {
    renderer.dispose();
    for (const track of combined.getTracks()) {
      track.stop();
    }
    if (audioCtx) {
      await audioCtx.close().catch(() => {});
    }
  }
}

function ctxOrThrow(ctx: AudioContext): AudioContext {
  return ctx;
}
