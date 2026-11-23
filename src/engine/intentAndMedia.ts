export type ResolvedChatIntent = 'text' | 'build' | 'image' | 'video';

export type SceneBackdrop = 'city' | 'mountains' | 'ocean' | 'forest' | 'space' | 'abstract';
export type SceneCelestial = 'sun' | 'moon' | 'planet' | 'nebula' | 'none';
export type SceneWeather = 'none' | 'rain' | 'snow' | 'sparkles' | 'embers' | 'petals';
export type SceneActorType =
  | 'rocket'
  | 'car'
  | 'cat'
  | 'elephant'
  | 'bird'
  | 'fish'
  | 'robot'
  | 'boat'
  | 'astronaut'
  | 'orb'
  | 'dragon'
  | 'banana'
  | 'tree'
  | 'crystal';
export type SceneMotion = 'glide' | 'bob' | 'orbit' | 'dash' | 'hover' | 'pulse';

export interface SceneActor {
  type: SceneActorType;
  label: string;
  x: number;
  y: number;
  scale: number;
  color: string;
  secondaryColor: string;
  motion: SceneMotion;
  speed: number;
  trail: boolean;
}

export interface VideoSceneSpec {
  title: string;
  mood: string;
  aspectRatio: '16:9' | '1:1';
  durationSec: number;
  skyTop: string;
  skyBottom: string;
  horizonGlow: string;
  groundTop: string;
  groundBottom: string;
  accent: string;
  secondary: string;
  celestial: SceneCelestial;
  backdrop: SceneBackdrop;
  weather: SceneWeather;
  waterReflection: boolean;
  cameraPan: number;
  cameraZoom: number;
  actors: SceneActor[];
  caption: string;
}

const VALID_BACKDROPS = new Set<SceneBackdrop>([
  'city',
  'mountains',
  'ocean',
  'forest',
  'space',
  'abstract',
]);
const VALID_CELESTIAL = new Set<SceneCelestial>(['sun', 'moon', 'planet', 'nebula', 'none']);
const VALID_WEATHER = new Set<SceneWeather>(['none', 'rain', 'snow', 'sparkles', 'embers', 'petals']);
const VALID_ACTORS = new Set<SceneActorType>([
  'rocket',
  'car',
  'cat',
  'elephant',
  'bird',
  'fish',
  'robot',
  'boat',
  'astronaut',
  'orb',
  'dragon',
  'banana',
  'tree',
  'crystal',
]);
const VALID_MOTIONS = new Set<SceneMotion>(['glide', 'bob', 'orbit', 'dash', 'hover', 'pulse']);

export function detectChatIntent(
  prompt: string,
  explicitMode: ResolvedChatIntent,
): ResolvedChatIntent {
  if (explicitMode !== 'text') return explicitMode;

  const trimmed = prompt.trim();
  if (!trimmed) return 'text';

  const isCodeQuestion =
    /\b(code|html|css|svg|python|javascript|typescript|react|component|function|script|api|algorithm|how does|explain|what is)\b/i.test(
      trimmed,
    );

  const isExplanationQuestion =
    /^(how|what|why|when|where|who|explain|describe|compare|difference)\b/i.test(trimmed);

  const asksVideo =
    /\b(generate|create|make|render|produce|show)\b.*\b(video|animation|animated clip|movie|timelapse|60\s*fps)\b/i.test(
      trimmed,
    ) || /^(animate)\b/i.test(trimmed);

  if (asksVideo && !isCodeQuestion) {
    return 'video';
  }

  const asksImage =
    /\b(generate|create|make|draw|paint|render|sketch|illustrate|design|show)\b.*\b(image|picture|photo|artwork|illustration|portrait|poster|wallpaper|scene|painting)\b/i.test(
      trimmed,
    ) || /^(draw|paint|illustrate|sketch)\s+(a|an|the|me)\b/i.test(trimmed);

  if (asksImage && !isCodeQuestion) {
    return 'image';
  }

  const asksFullStackBuild =
    !isExplanationQuestion &&
    (/\b(build|create|make|develop|architect|scaffold)\b.*\b(full[- ]?stack|web ?app|application|dashboard|portal|platform|saas|crm|kanban|ecommerce|storefront|tracker|crud|backend|express|database)\b/i.test(
      trimmed,
    ) ||
      (/\b(build|create|make|develop)\b/i.test(trimmed) &&
        /\b(api|database|sql|backend|server|auth)\b/i.test(trimmed)));

  if (asksFullStackBuild) {
    return 'build';
  }

  return 'text';
}

export function sanitizeGeneratedSvg(rawOutput: string, fallbackTitle = 'Qowe 1.0 Artwork'): string {
  const cleaned = rawOutput
    .replace(/```(?:svg|xml|html)?\s*/gi, '')
    .replace(/```/g, '')
    .trim();

  const match = cleaned.match(/<svg[\s\S]*?<\/svg>/i);
  if (!match) {
    return buildFallbackSvg(fallbackTitle);
  }

  let svg = match[0];

  // Strip potentially unsafe active elements or inline JS schemes
  svg = svg
    .replace(/<script[\s\S]*?<\/script>/gi, '')
    .replace(/<foreignObject[\s\S]*?<\/foreignObject>/gi, '')
    .replace(/<iframe[\s\S]*?<\/iframe>/gi, '')
    .replace(/<object[\s\S]*?<\/object>/gi, '')
    .replace(/<embed[^>]*\/?>/gi, '')
    .replace(/\son[a-z]+\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+)/gi, '')
    .replace(/javascript:/gi, '')
    .replace(/data:text\/html/gi, '');

  if (!/\bxmlns\s*=/.test(svg)) {
    svg = svg.replace(/<svg\b/i, '<svg xmlns="http://www.w3.org/2000/svg"');
  }
  if (!/\bviewBox\s*=/.test(svg)) {
    svg = svg.replace(/<svg\b/i, '<svg viewBox="0 0 1024 1024"');
  }

  return svg;
}

function escapeXml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

export function buildFallbackSvg(prompt: string): string {
  const title = escapeXml(prompt.slice(0, 64) || 'Qowe 1.0 Studio Render');
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="1024" height="1024">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#0b132b"/>
      <stop offset="55%" stop-color="#1c2541"/>
      <stop offset="100%" stop-color="#3a506b"/>
    </linearGradient>
    <radialGradient id="glow" cx="50%" cy="42%" r="42%">
      <stop offset="0%" stop-color="#6fffe9" stop-opacity="0.9"/>
      <stop offset="45%" stop-color="#3b82f6" stop-opacity="0.45"/>
      <stop offset="100%" stop-color="#0b132b" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="1024" height="1024" fill="url(#bg)"/>
  <circle cx="512" cy="430" r="320" fill="url(#glow)"/>
  <path d="M0 760 Q 256 660 512 730 T 1024 700 L 1024 1024 L 0 1024 Z" fill="#0f172a" opacity="0.9"/>
  <circle cx="512" cy="430" r="120" fill="#e2e8f0" opacity="0.92"/>
  <text x="512" y="920" text-anchor="middle" fill="#cbd5e1" font-family="sans-serif" font-size="26">${title}</text>
</svg>`;
}

function sanitizeHexColor(input: unknown, fallback: string): string {
  if (typeof input === 'string' && /^#[0-9a-fA-F]{3,8}$/.test(input.trim())) {
    return input.trim();
  }
  return fallback;
}

function clampNum(value: unknown, min: number, max: number, fallback: number): number {
  const n = typeof value === 'number' && Number.isFinite(value) ? value : fallback;
  return Math.max(min, Math.min(max, n));
}

function inferDefaultSpec(prompt: string): VideoSceneSpec {
  const lower = prompt.toLowerCase();
  const isCyber = /cyber|neon|city|rain|night|synth|street/.test(lower);
  const isOcean = /ocean|sea|wave|beach|boat|ship|water|sunset/.test(lower);
  const isSpace = /space|rocket|mars|moon|galaxy|planet|star|astronaut/.test(lower);
  const isNature = /forest|mountain|cherry|blossom|garden|nature|tree|bird/.test(lower);

  const actors: SceneActor[] = [];
  const candidates: [RegExp, SceneActorType, string, string, SceneMotion][] = [
    [/rocket|spaceship/, 'rocket', '#f8fafc', '#f97316', 'glide'],
    [/car|vehicle|cyber/, 'car', '#38bdf8', '#f43f5e', 'dash'],
    [/cat|kitten/, 'cat', '#fb923c', '#fde68a', 'bob'],
    [/elephant/, 'elephant', '#94a3b8', '#f472b6', 'bob'],
    [/dragon/, 'dragon', '#ef4444', '#facc15', 'glide'],
    [/robot/, 'robot', '#60a5fa', '#22d3ee', 'hover'],
    [/boat|ship/, 'boat', '#f8fafc', '#ef4444', 'bob'],
    [/astronaut/, 'astronaut', '#f8fafc', '#38bdf8', 'hover'],
    [/fish|whale|dolphin/, 'fish', '#38bdf8', '#2dd4bf', 'glide'],
    [/bird|eagle/, 'bird', '#f8fafc', '#fbbf24', 'glide'],
    [/banana/, 'banana', '#facc15', '#fef08a', 'orbit'],
  ];

  for (const [re, type, color, secondaryColor, motion] of candidates) {
    if (re.test(lower)) {
      actors.push({
        type,
        label: type,
        x: 0.45,
        y: type === 'car' || type === 'elephant' || type === 'cat' ? 0.72 : 0.46,
        scale: 1,
        color,
        secondaryColor,
        motion,
        speed: 1,
        trail: type === 'rocket' || type === 'dragon' || type === 'orb',
      });
    }
  }

  if (actors.length === 0) {
    actors.push({
      type: isSpace ? 'rocket' : isOcean ? 'boat' : isCyber ? 'car' : 'orb',
      label: 'primary',
      x: 0.45,
      y: isSpace ? 0.48 : 0.68,
      scale: 1,
      color: isCyber ? '#22d3ee' : '#fbbf24',
      secondaryColor: '#f43f5e',
      motion: isSpace ? 'glide' : isCyber ? 'dash' : 'bob',
      speed: 1,
      trail: true,
    });
  }

  return {
    title: prompt.slice(0, 60) || 'Qowe 1.0 Video Scene',
    mood: isCyber ? 'neon-noir' : isSpace ? 'cosmic' : isOcean ? 'golden-hour' : 'cinematic',
    aspectRatio: '16:9',
    durationSec: 4,
    skyTop: isCyber ? '#070b1a' : isSpace ? '#020617' : isNature ? '#1e293b' : '#0f172a',
    skyBottom: isCyber ? '#3b0764' : isSpace ? '#1e1b4b' : isOcean ? '#f97316' : '#334155',
    horizonGlow: isCyber ? '#ec4899' : isOcean ? '#fbbf24' : '#38bdf8',
    groundTop: isCyber ? '#090d16' : isOcean ? '#0284c7' : '#0f172a',
    groundBottom: isCyber ? '#020617' : isOcean ? '#0c4a6e' : '#020617',
    accent: isCyber ? '#22d3ee' : '#fbbf24',
    secondary: '#ec4899',
    celestial: isSpace ? 'planet' : isCyber ? 'moon' : 'sun',
    backdrop: isSpace
      ? 'space'
      : isCyber
        ? 'city'
        : isOcean
          ? 'ocean'
          : isNature
            ? 'forest'
            : 'mountains',
    weather: /rain/.test(lower)
      ? 'rain'
      : /snow/.test(lower)
        ? 'snow'
        : /blossom|petal/.test(lower)
          ? 'petals'
          : /fire|ember|lava/.test(lower)
            ? 'embers'
            : 'sparkles',
    waterReflection: isCyber || isOcean || /lake|pond|reflection/.test(lower),
    cameraPan: 0.45,
    cameraZoom: 0.08,
    actors,
    caption: '',
  };
}

export function normalizeVideoSceneSpec(raw: unknown, prompt: string): VideoSceneSpec {
  const defaults = inferDefaultSpec(prompt);
  if (!raw || typeof raw !== 'object') return defaults;

  const obj = raw as Record<string, unknown>;
  const rawActors = Array.isArray(obj.actors) ? obj.actors.slice(0, 4) : defaults.actors;

  const actors: SceneActor[] = rawActors.map((item, idx) => {
    const a = (item && typeof item === 'object' ? item : {}) as Record<string, unknown>;
    const actorType = VALID_ACTORS.has(a.type as SceneActorType)
      ? (a.type as SceneActorType)
      : defaults.actors[0]?.type ?? 'orb';
    const motion = VALID_MOTIONS.has(a.motion as SceneMotion)
      ? (a.motion as SceneMotion)
      : 'glide';
    return {
      type: actorType,
      label: typeof a.label === 'string' && a.label.trim() ? a.label.trim().slice(0, 32) : `subject-${idx + 1}`,
      x: clampNum(a.x, 0.1, 0.9, 0.3 + idx * 0.2),
      y: clampNum(a.y, 0.15, 0.85, 0.55),
      scale: clampNum(a.scale, 0.45, 1.8, 1),
      color: sanitizeHexColor(a.color, defaults.accent),
      secondaryColor: sanitizeHexColor(a.secondaryColor, defaults.secondary),
      motion,
      speed: clampNum(a.speed, 0.25, 2.5, 1),
      trail: typeof a.trail === 'boolean' ? a.trail : true,
    };
  });

  if (actors.length === 0) {
    actors.push(...defaults.actors);
  }

  return {
    title:
      typeof obj.title === 'string' && obj.title.trim()
        ? obj.title.trim().slice(0, 80)
        : defaults.title,
    mood:
      typeof obj.mood === 'string' && obj.mood.trim()
        ? obj.mood.trim().slice(0, 40)
        : defaults.mood,
    aspectRatio: obj.aspectRatio === '1:1' ? '1:1' : '16:9',
    durationSec: clampNum(obj.durationSec, 2, 10, 4),
    skyTop: sanitizeHexColor(obj.skyTop, defaults.skyTop),
    skyBottom: sanitizeHexColor(obj.skyBottom, defaults.skyBottom),
    horizonGlow: sanitizeHexColor(obj.horizonGlow, defaults.horizonGlow),
    groundTop: sanitizeHexColor(obj.groundTop, defaults.groundTop),
    groundBottom: sanitizeHexColor(obj.groundBottom, defaults.groundBottom),
    accent: sanitizeHexColor(obj.accent, defaults.accent),
    secondary: sanitizeHexColor(obj.secondary, defaults.secondary),
    celestial: VALID_CELESTIAL.has(obj.celestial as SceneCelestial)
      ? (obj.celestial as SceneCelestial)
      : defaults.celestial,
    backdrop: VALID_BACKDROPS.has(obj.backdrop as SceneBackdrop)
      ? (obj.backdrop as SceneBackdrop)
      : defaults.backdrop,
    weather: VALID_WEATHER.has(obj.weather as SceneWeather)
      ? (obj.weather as SceneWeather)
      : defaults.weather,
    waterReflection:
      typeof obj.waterReflection === 'boolean' ? obj.waterReflection : defaults.waterReflection,
    cameraPan: clampNum(obj.cameraPan, -1, 1, defaults.cameraPan),
    cameraZoom: clampNum(obj.cameraZoom, 0, 0.25, defaults.cameraZoom),
    actors,
    caption: typeof obj.caption === 'string' ? obj.caption.trim().slice(0, 80) : '',
  };
}
