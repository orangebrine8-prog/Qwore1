import 'dotenv/config';
import crypto from 'node:crypto';
import fs from 'node:fs';
import express, { type NextFunction, type Request, type Response } from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { GoogleGenAI, type GenerateContentResponse, ThinkingLevel, Type } from '@google/genai';
import {
  QoweApiKeyStore,
  extractApiKeyFromHeaders,
  type PublicQoweApiKeyRecord,
} from './server/apiKeyService.ts';
import { type ApiEndpointCategory, QoweUsageTracker } from './server/apiUsageTracker.ts';
import { type BaseUrlResolution, resolveBaseUrls } from './server/urlUtils.ts';
import {
  normalizeVideoSceneSpec,
  sanitizeGeneratedSvg,
  type VideoSceneSpec,
} from './src/engine/intentAndMedia.ts';
import { generateLocalQoweResponse } from './server/localQoweFallback.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.set('trust proxy', 1);
app.use(express.json({ limit: '2mb' }));

const keyStore = new QoweApiKeyStore();
const usageTracker = new QoweUsageTracker();

function classifyEndpointCategory(reqPath: string, bodyEndpoint?: unknown): ApiEndpointCategory {
  if (reqPath.includes('/images/') || bodyEndpoint === 'image') return 'image';
  if (reqPath.includes('/videos/') || bodyEndpoint === 'video') return 'video';
  if (reqPath.includes('/models') || bodyEndpoint === 'models') return 'models';
  return 'chat';
}

interface AuthenticatedRequest extends Request {
  qoweKey?: PublicQoweApiKeyRecord;
}

function createGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
    throw new Error('GEMINI_API_KEY is not configured. Add a valid key in Settings > Secrets.');
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      timeout: 45000,
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function formatUpstreamError(err: unknown): string {
  const raw = err instanceof Error ? err.message : String(err);
  try {
    const outer = JSON.parse(raw) as { error?: { message?: string; code?: number } };
    const innerRaw = outer?.error?.message;
    if (typeof innerRaw === 'string') {
      try {
        const inner = JSON.parse(innerRaw) as { error?: { message?: string; code?: number } };
        if (inner?.error?.message) return inner.error.message;
      } catch {
        return innerRaw;
      }
    }
  } catch {
    // Raw is plain text
  }
  return raw || 'Failed to generate response.';
}

function isTransientUpstreamError(err: unknown): boolean {
  const msg = formatUpstreamError(err).toLowerCase();
  const status = (err as { status?: number })?.status;
  if (status === 401 || status === 403) return false;
  return (
    status === 408 ||
    status === 429 ||
    (typeof status === 'number' && status >= 500) ||
    msg.includes('500') ||
    msg.includes('502') ||
    msg.includes('503') ||
    msg.includes('504') ||
    msg.includes('high demand') ||
    msg.includes('try again later') ||
    msg.includes('temporarily') ||
    msg.includes('quota') ||
    msg.includes('resource_exhausted') ||
    msg.includes('rate limit') ||
    msg.includes('too many requests') ||
    msg.includes('capacity') ||
    msg.includes('unavailable') ||
    msg.includes('internal') ||
    msg.includes('overloaded') ||
    msg.includes('timeout') ||
    msg.includes('aborted') ||
    msg.includes('fetch failed') ||
    msg.includes('not configured')
  );
}

const UPSTREAM_MODEL_CHAIN: { model: string; useThinking: boolean }[] = [
  { model: 'gemini-4-hydrogen', useThinking: true },
  { model: 'gemini-3.1-flash-lite', useThinking: false },
  { model: 'gemini-2.5-flash', useThinking: false },
];

async function generateWithHydrogenFallback<T>(
  runner: (model: string, useThinking: boolean) => Promise<T>,
): Promise<T> {
  let lastErr: unknown;
  for (const candidate of UPSTREAM_MODEL_CHAIN) {
    try {
      return await runner(candidate.model, candidate.useThinking);
    } catch (err) {
      lastErr = err;
      if (!isTransientUpstreamError(err)) {
        throw err;
      }
    }
  }
  throw lastErr;
}

function getDimensionsForRatio(aspectRatio: '1:1' | '16:9' | '9:16' = '1:1') {
  return aspectRatio === '16:9'
    ? { width: 1024, height: 576 }
    : aspectRatio === '9:16'
      ? { width: 576, height: 1024 }
      : { width: 1024, height: 1024 };
}

async function generateStudioImageSvg(
  ai: GoogleGenAI,
  prompt: string,
  aspectRatio: '1:1' | '16:9' | '9:16' = '1:1',
): Promise<{ svg: string; width: number; height: number }> {
  const dimensions =
    aspectRatio === '16:9'
      ? { width: 1024, height: 576 }
      : aspectRatio === '9:16'
        ? { width: 576, height: 1024 }
        : { width: 1024, height: 1024 };

  const svgPrompt = `Create a richly detailed, studio-quality vector illustration in pure SVG for: "${prompt}".
Canvas requirements:
- Output ONLY raw <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${dimensions.width} ${dimensions.height}" width="${dimensions.width}" height="${dimensions.height}">...</svg> with no markdown fences and no commentary.
- Craft cohesive atmospheric backgrounds using multi-stop <linearGradient> and <radialGradient> lighting, soft glows, depth layers (far backdrop, midground environment, crisp hero foreground subject), expressive silhouettes/shapes, specular highlights, and realistic cast shadows.
- Never use <script>, <foreignObject>, external image URLs, or event attributes.`;

  const response = await generateWithHydrogenFallback((model, useThinking) =>
    ai.models.generateContent({
      model,
      contents: svgPrompt,
      config: {
        systemInstruction:
          'You are an elite digital illustrator for Qowe 1.0. Produce vivid, detailed, visually cohesive SVG artwork matching the exact subject, lighting, mood, and composition requested.',
        ...(useThinking ? { thinkingConfig: { thinkingLevel: ThinkingLevel.LOW } } : {}),
      },
    }),
  );

  const svg = sanitizeGeneratedSvg(response.text ?? '', prompt);
  return { svg, ...dimensions };
}

async function generateVideoSceneSpec(
  ai: GoogleGenAI,
  prompt: string,
  aspectRatio: '16:9' | '1:1' = '16:9',
  durationSec = 4,
): Promise<VideoSceneSpec> {
  const response = await generateWithHydrogenFallback((model, useThinking) =>
    ai.models.generateContent({
      model,
      contents: `Design a vivid 60 fps animated cinematic scene specification for the prompt: "${prompt}".
Desired aspectRatio: "${aspectRatio}", durationSec: ${Math.max(2, Math.min(10, Math.round(durationSec)))}.
Pick rich hex colors, the best matching backdrop, celestial lighting, atmospheric weather, waterReflection, and 1-3 animated actors matching the subjects in the prompt.`,
      config: {
        systemInstruction:
          'You are the 60 fps motion scene director for Qowe 1.0. Output clean JSON matching the exact scene schema.',
        responseMimeType: 'application/json',
        ...(useThinking ? { thinkingConfig: { thinkingLevel: ThinkingLevel.LOW } } : {}),
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            title: { type: Type.STRING },
            mood: { type: Type.STRING },
            aspectRatio: { type: Type.STRING },
            durationSec: { type: Type.NUMBER },
            skyTop: { type: Type.STRING },
            skyBottom: { type: Type.STRING },
            horizonGlow: { type: Type.STRING },
            groundTop: { type: Type.STRING },
            groundBottom: { type: Type.STRING },
            accent: { type: Type.STRING },
            secondary: { type: Type.STRING },
            celestial: { type: Type.STRING },
            backdrop: { type: Type.STRING },
            weather: { type: Type.STRING },
            waterReflection: { type: Type.BOOLEAN },
            cameraPan: { type: Type.NUMBER },
            cameraZoom: { type: Type.NUMBER },
            caption: { type: Type.STRING },
            actors: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  type: { type: Type.STRING },
                  label: { type: Type.STRING },
                  x: { type: Type.NUMBER },
                  y: { type: Type.NUMBER },
                  scale: { type: Type.NUMBER },
                  color: { type: Type.STRING },
                  secondaryColor: { type: Type.STRING },
                  motion: { type: Type.STRING },
                  speed: { type: Type.NUMBER },
                  trail: { type: Type.BOOLEAN },
                },
              },
            },
          },
        },
      },
    }),
  );

  let parsed: unknown = null;
  try {
    parsed = JSON.parse((response.text ?? '{}').trim());
  } catch {
    parsed = null;
  }
  const normalized = normalizeVideoSceneSpec(parsed, prompt);
  normalized.aspectRatio = aspectRatio;
  normalized.durationSec = Math.max(2, Math.min(10, Math.round(durationSec)));
  return normalized;
}

function getBaseUrls(req: Request): BaseUrlResolution {
  const host = req.get('x-forwarded-host') || req.get('host') || 'localhost:3000';
  const proto = (req.get('x-forwarded-proto') || req.protocol || 'http').split(',')[0].trim();
  const current = `${proto}://${host}`;
  return resolveBaseUrls({
    requestOrigin: current,
    rawAppUrl: process.env.APP_URL,
  });
}

async function requireQoweApiKey(
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction,
): Promise<void> {
  const rawKey = extractApiKeyFromHeaders(req.headers);
  const record = await keyStore.verifyAndTouch(rawKey);
  if (!record) {
    res.status(401).json({
      error: {
        message:
          'Invalid or missing Qowe API key. Pass Authorization: Bearer qowe_sk_... or x-api-key header.',
        type: 'invalid_request_error',
        code: 'invalid_api_key',
      },
    });
    return;
  }
  req.qoweKey = record;
  const startedAt = performance.now();
  const endpointPath = req.originalUrl.split('?')[0] || req.path;
  const bodyObj = (req.body ?? {}) as { model?: unknown; endpoint?: unknown };
  const category = classifyEndpointCategory(endpointPath, bodyObj.endpoint);
  const requestedModel =
    category === 'image'
      ? 'qowe-1.0-vision'
      : category === 'video'
        ? 'qowe-1.0-video'
        : category === 'models'
          ? 'qowe-1.0-models'
          : typeof bodyObj.model === 'string' && bodyObj.model.trim()
            ? bodyObj.model.trim().slice(0, 64)
            : 'qowe-1.0-ultra';

  res.once('finish', () => {
    const latencyMs = Math.max(1, Math.round(performance.now() - startedAt));
    void keyStore.recordLatency(record.id, latencyMs);
    void usageTracker.recordRequest({
      endpoint: `${req.method} ${endpointPath}`,
      category,
      model: requestedModel,
      keyPrefix: record.prefix,
      keyName: record.name,
      status: res.statusCode,
      latencyMs,
    });
  });

  next();
}

app.get('/api/health', (_req, res) => {
  res.json({
    ok: true,
    model: 'Qowe 1.0',
    hasKey: Boolean(process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== 'MY_GEMINI_API_KEY'),
  });
});

// ---- Qowe API Key Management & Real-Time Usage Stats ----
app.get('/api/usage', async (_req, res) => {
  try {
    const usage = await usageTracker.getSummary();
    res.json({ usage });
  } catch (err) {
    res.status(500).json({ error: formatUpstreamError(err) });
  }
});

app.get('/api/keys', async (req, res) => {
  try {
    const [keys, usage] = await Promise.all([keyStore.listKeys(), usageTracker.getSummary()]);
    res.json({ keys, usage, baseUrls: getBaseUrls(req) });
  } catch (err) {
    res.status(500).json({ error: formatUpstreamError(err) });
  }
});

app.post('/api/keys', async (req, res) => {
  try {
    const { name } = (req.body ?? {}) as { name?: unknown };
    const created = await keyStore.createKey(name);
    res.status(201).json({ key: created.record, rawKey: created.rawKey });
  } catch (err) {
    res.status(400).json({ error: err instanceof Error ? err.message : 'Unable to create API key.' });
  }
});

app.delete('/api/keys/:id', async (req, res) => {
  const id = String(req.params.id || '').trim();
  if (!id) {
    res.status(400).json({ error: 'Key ID is required.' });
    return;
  }
  const removed = await keyStore.revokeKey(id);
  if (!removed) {
    res.status(404).json({ error: 'API key not found.' });
    return;
  }
  res.json({ ok: true });
});

// ---- Studio Internal Endpoints ----
interface IncomingMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

app.post('/api/chat', async (req, res) => {
  const {
    messages,
    architectureName,
    totalParamsFormatted,
    trainingContext,
    checkpointStep,
    checkpointLoss,
    reasoningDepth,
    outputMode,
  } = req.body as {
    messages?: IncomingMessage[];
    architectureName?: string;
    totalParamsFormatted?: string;
    trainingContext?: string;
    checkpointStep?: number;
    checkpointLoss?: number;
    reasoningDepth?: 'standard' | 'deep';
    outputMode?: 'text' | 'build';
  };

  if (!Array.isArray(messages) || messages.length === 0) {
    res.status(400).json({ error: 'A non-empty messages array is required.' });
    return;
  }

  let ai: GoogleGenAI | null = null;
  try {
    ai = createGeminiClient();
  } catch {
    ai = null;
  }

  res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders?.();
  res.write(': stream-open\n\n');

  let clientClosed = false;
  let activeAbortCtrl: AbortController | null = null;
  res.on('close', () => {
    if (!res.writableEnded) {
      clientClosed = true;
      activeAbortCtrl?.abort();
    }
  });

  const streamLocalQoweFallback = async () => {
    const localText = generateLocalQoweResponse({
      messages,
      architectureName,
      totalParamsFormatted,
      trainingContext,
      checkpointStep,
      checkpointLoss,
      outputMode,
    });
    const chunkSize = 48;
    for (let i = 0; i < localText.length; i += chunkSize) {
      if (clientClosed || res.writableEnded) return;
      res.write(`data: ${JSON.stringify({ text: localText.slice(i, i + chunkSize) })}\n\n`);
    }
  };

  const archLabel = architectureName || 'Qowe 1.0 Ultra 800B';
  const paramLabel = totalParamsFormatted || '801.32B';
  const corpusNote =
    typeof trainingContext === 'string' && trainingContext.trim().length > 0
      ? `\n\nThe user has also loaded the following custom training corpus into your Qowe 1.0 workspace (use it as domain context when relevant):\n"""\n${trainingContext.slice(0, 3000)}\n"""`
      : '';
  const checkpointNote =
    typeof checkpointStep === 'number' && checkpointStep > 0
      ? `\nActive local from-scratch training checkpoint: step ${checkpointStep}, cross-entropy loss ${typeof checkpointLoss === 'number' ? checkpointLoss.toFixed(3) : 'N/A'}.`
      : '';

  const buildModePrompt =
    outputMode === 'build'
      ? `\n\nFULL-STACK CODE & BUILD MODE ACTIVE (Hydrogen 4 Architect):
The user wants you to architect and build a complete, immediately runnable FULL-STACK application (Frontend UI + Backend API + SQL Database Schema & Seed Data).
1. Start with a concise "### Build Task Plan" listing 4-5 completed engineering milestones using markdown checkboxes (\`- [x] ...\`).
2. Output clean multi-file fenced code blocks using exact \`file="..."\` attributes in the fence info string in this order:
   - \`\`\`json file="package.json"\`\`\` — includes \`"type": "module"\`, \`"scripts": { "dev": "tsx server.ts", "start": "node server.ts" }\`, \`"engines": { "node": ">=22.18.0" }\`, and \`express\`.
   - \`\`\`sql file="db/schema.sql"\`\`\` — clean \`CREATE TABLE IF NOT EXISTS ...;\` DDL and 3-5 realistic \`INSERT INTO ... VALUES (...);\` seed rows so the app starts populated with real data.
   - \`\`\`ts file="server.ts"\`\`\` — production Express Node.js backend serving \`public/index.html\` and clean REST endpoints (\`GET /api/...\`, \`POST /api/...\`, \`PATCH /api/.../:id\`, \`DELETE /api/.../:id\`, \`GET /api/stats\`, \`GET /api/health\`) matching the table names in \`db/schema.sql\`, returning arrays cleanly for collection GETs (or consistent JSON objects) with real validation and persistent storage.
   - \`\`\`html file="public/index.html"\`\`\` — complete, responsive, refined UI using clean inline \`<style>\` and real \`fetch('/api/...')\` calls (for reading, creating, updating/toggling, filtering, and deleting records) with zero external CDN dependencies and zero \`alert()\`/\`confirm()\`/\`prompt()\` dialogs.
Every REST endpoint your frontend calls MUST match a REST collection/resource in \`db/schema.sql\` and \`server.ts\` so the live workspace full-stack sandbox AND the exported .zip project both work immediately.`
      : '';

  const systemInstruction = `You are Qowe 1.0 (${archLabel}, ${paramLabel} parameter multimodal causal transformer), an elite analytical, full-stack software architect, and reasoning AI engine built in the Qowe 1.0 studio.
Understand the user's underlying intent deeply before responding. For reasoning, math, architecture, or code tasks: verify edge cases internally, decompose complex ideas cleanly, and provide exact, high-signal answers with runnable fenced code blocks when helpful.${checkpointNote}${corpusNote}${buildModePrompt}`;

  const contents = messages.map((m) => ({
    role: m.role === 'assistant' ? 'model' : 'user',
    parts: [{ text: m.content }],
  }));

  const thinkingLevel = reasoningDepth === 'standard' ? ThinkingLevel.MEDIUM : ThinkingLevel.HIGH;
  let streamedAnyChunk = false;

  const runStream = async (model: string, level: ThinkingLevel | null, timeoutMs?: number) => {
    if (!ai) throw new Error('upstream unavailable');
    const abortCtrl = new AbortController();
    activeAbortCtrl = abortCtrl;
    const timer = timeoutMs ? setTimeout(() => abortCtrl.abort(), timeoutMs) : undefined;
    try {
      const stream = await ai.models.generateContentStream({
        model,
        contents,
        config: {
          systemInstruction,
          abortSignal: abortCtrl.signal,
          ...(level ? { thinkingConfig: { thinkingLevel: level } } : {}),
        },
      });

      let receivedAny = false;
      for await (const chunk of stream) {
        if (!receivedAny && timer) {
          clearTimeout(timer);
        }
        receivedAny = true;
        if (clientClosed) break;
        const c = chunk as GenerateContentResponse;
        if (c.text) {
          streamedAnyChunk = true;
          res.write(`data: ${JSON.stringify({ text: c.text })}\n\n`);
        }
      }
    } finally {
      if (timer) clearTimeout(timer);
      if (activeAbortCtrl === abortCtrl) {
        activeAbortCtrl = null;
      }
    }
  };

  try {
    let completed = false;
    if (ai) {
      for (const candidate of UPSTREAM_MODEL_CHAIN) {
        if (clientClosed) break;
        try {
          await runStream(
            candidate.model,
            candidate.useThinking ? thinkingLevel : null,
            outputMode === 'build' ? 30000 : 12000,
          );
          if (streamedAnyChunk) {
            completed = true;
            break;
          }
        } catch (upstreamErr) {
          if (clientClosed || streamedAnyChunk) break;
          if (!isTransientUpstreamError(upstreamErr)) {
            throw upstreamErr;
          }
        }
      }
    }

    if (!completed && !clientClosed && !streamedAnyChunk) {
      await streamLocalQoweFallback();
    }

    if (!clientClosed && !res.writableEnded) {
      res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
      res.end();
    }
  } catch (err) {
    if (!clientClosed && !res.writableEnded) {
      if (!streamedAnyChunk) {
        await streamLocalQoweFallback();
        res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
      } else {
        res.write(`data: ${JSON.stringify({ error: formatUpstreamError(err) })}\n\n`);
      }
      res.end();
    }
  }
});

app.post('/api/generate-image', async (req, res) => {
  const { prompt, aspectRatio } = (req.body ?? {}) as {
    prompt?: string;
    aspectRatio?: '1:1' | '16:9' | '9:16';
  };
  if (typeof prompt !== 'string' || !prompt.trim()) {
    res.status(400).json({ error: 'A non-empty image prompt is required.' });
    return;
  }

  const safeRatio = aspectRatio === '16:9' || aspectRatio === '9:16' ? aspectRatio : '1:1';
  const cleanPrompt = prompt.trim().slice(0, 1000);
  try {
    const ai = createGeminiClient();
    const result = await generateStudioImageSvg(ai, cleanPrompt, safeRatio);
    res.json({
      model: 'qowe-1.0-vision',
      ...result,
      mimeType: 'image/svg+xml',
    });
  } catch {
    const dims = getDimensionsForRatio(safeRatio);
    res.json({
      model: 'qowe-1.0-vision',
      svg: sanitizeGeneratedSvg('', cleanPrompt),
      ...dims,
      mimeType: 'image/svg+xml',
    });
  }
});

app.post('/api/generate-video', async (req, res) => {
  const { prompt, aspectRatio, durationSec } = (req.body ?? {}) as {
    prompt?: string;
    aspectRatio?: '16:9' | '1:1';
    durationSec?: number;
  };
  if (typeof prompt !== 'string' || !prompt.trim()) {
    res.status(400).json({ error: 'A non-empty video prompt is required.' });
    return;
  }

  const safeRatio = aspectRatio === '1:1' ? '1:1' : '16:9';
  const safeDur = typeof durationSec === 'number' ? durationSec : 4;
  const cleanPrompt = prompt.trim().slice(0, 1000);
  try {
    const ai = createGeminiClient();
    const sceneSpec = await generateVideoSceneSpec(ai, cleanPrompt, safeRatio, safeDur);
    res.json({
      model: 'qowe-1.0-video',
      fps: 60,
      sceneSpec,
    });
  } catch {
    const fallbackSpec = normalizeVideoSceneSpec(null, cleanPrompt);
    fallbackSpec.aspectRatio = safeRatio;
    fallbackSpec.durationSec = Math.max(2, Math.min(10, Math.round(safeDur)));
    res.json({
      model: 'qowe-1.0-video',
      fps: 60,
      sceneSpec: fallbackSpec,
    });
  }
});

// ---- Studio Same-Origin API Playground Bridge (avoids Authorization Bearer proxy conflicts) ----
app.post('/api/playground/test', requireQoweApiKey, async (req, res) => {
  const { endpoint, prompt } = (req.body ?? {}) as {
    endpoint?: string;
    prompt?: string;
  };

  if (endpoint === 'models') {
    res.json({
      object: 'list',
      data: [
        {
          id: 'qowe-1.0-ultra',
          object: 'model',
          owned_by: 'qowe',
          capabilities: ['chat', 'reasoning', 'code', 'build'],
          parameters: '801.32B',
        },
        {
          id: 'qowe-1.0-vision',
          object: 'model',
          owned_by: 'qowe',
          capabilities: ['text-to-image'],
          format: 'image/svg+xml',
        },
        {
          id: 'qowe-1.0-video',
          object: 'model',
          owned_by: 'qowe',
          capabilities: ['text-to-video'],
          fps: 60,
        },
      ],
    });
    return;
  }

  if (typeof prompt !== 'string' || !prompt.trim()) {
    res.status(400).json({
      error: { message: 'prompt must be a non-empty string.', type: 'invalid_request_error' },
    });
    return;
  }

  let ai: GoogleGenAI | null = null;
  try {
    ai = createGeminiClient();
  } catch {
    ai = null;
  }

  const cleanPrompt = prompt.trim().slice(0, 1000);
  if (endpoint === 'image') {
    try {
      if (!ai) throw new Error('local fallback');
      const result = await generateStudioImageSvg(ai, cleanPrompt, '1:1');
      res.json({
        created: Math.floor(Date.now() / 1000),
        model: 'qowe-1.0-vision',
        data: [
          {
            prompt: cleanPrompt,
            mime_type: 'image/svg+xml',
            width: result.width,
            height: result.height,
            svg: result.svg,
          },
        ],
      });
    } catch {
      res.json({
        created: Math.floor(Date.now() / 1000),
        model: 'qowe-1.0-vision',
        data: [
          {
            prompt: cleanPrompt,
            mime_type: 'image/svg+xml',
            width: 1024,
            height: 1024,
            svg: sanitizeGeneratedSvg('', cleanPrompt),
          },
        ],
      });
    }
    return;
  }

  if (endpoint === 'video') {
    let sceneSpec: VideoSceneSpec;
    try {
      if (!ai) throw new Error('local fallback');
      sceneSpec = await generateVideoSceneSpec(ai, cleanPrompt, '16:9', 4);
    } catch {
      sceneSpec = normalizeVideoSceneSpec(null, cleanPrompt);
      sceneSpec.aspectRatio = '16:9';
      sceneSpec.durationSec = 4;
    }
    res.json({
      id: `vid_${crypto.randomBytes(10).toString('hex')}`,
      created: Math.floor(Date.now() / 1000),
      model: 'qowe-1.0-video',
      fps: 60,
      duration_sec: sceneSpec.durationSec,
      aspect_ratio: sceneSpec.aspectRatio,
      scene_spec: sceneSpec,
    });
    return;
  }

  let contentText = '';
  try {
    if (!ai) throw new Error('local fallback');
    const response = await generateWithHydrogenFallback((upstreamModel, useThinking) =>
      ai.models.generateContent({
        model: upstreamModel,
        contents: [{ role: 'user', parts: [{ text: cleanPrompt }] }],
        config: {
          systemInstruction:
            'You are Qowe 1.0 Ultra (801.32B parameter multimodal causal transformer). Provide clear, accurate, well-structured responses.',
          ...(useThinking ? { thinkingConfig: { thinkingLevel: ThinkingLevel.MEDIUM } } : {}),
        },
      }),
    );
    contentText = response.text ?? '';
  } catch {
    contentText = generateLocalQoweResponse({
      messages: [{ role: 'user', content: cleanPrompt }],
    });
  }

  res.json({
    id: `chatcmpl-${crypto.randomBytes(10).toString('hex')}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: 'qowe-1.0-ultra',
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: contentText },
        finish_reason: 'stop',
      },
    ],
  });
});

// ---- Public Qowe 1.0 API Endpoints (x-api-key or Bearer qowe_sk_... Authenticated) ----
app.use('/api/v1', (req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, x-api-key, x-qowe-api-key');
  res.setHeader('Access-Control-Max-Age', '86400');
  if (req.method === 'OPTIONS') {
    res.status(204).end();
    return;
  }
  next();
});

app.get('/api/v1/health', (_req, res) => {
  res.json({
    ok: true,
    service: 'qowe-1.0-api',
    version: '1.0',
    authHeaders: ['x-api-key', 'x-qowe-api-key', 'Authorization: Bearer'],
  });
});

app.get('/api/v1/models', requireQoweApiKey, (_req, res) => {
  res.json({
    object: 'list',
    data: [
      {
        id: 'qowe-1.0-ultra',
        object: 'model',
        owned_by: 'qowe',
        capabilities: ['chat', 'reasoning', 'code', 'build'],
        parameters: '801.32B',
      },
      {
        id: 'qowe-1.0-vision',
        object: 'model',
        owned_by: 'qowe',
        capabilities: ['text-to-image'],
        format: 'image/svg+xml',
      },
      {
        id: 'qowe-1.0-video',
        object: 'model',
        owned_by: 'qowe',
        capabilities: ['text-to-video'],
        fps: 60,
      },
    ],
  });
});

app.post('/api/v1/chat/completions', requireQoweApiKey, async (req, res) => {
  const { model, messages, stream, reasoning_effort } = (req.body ?? {}) as {
    model?: string;
    messages?: IncomingMessage[];
    stream?: boolean;
    reasoning_effort?: 'standard' | 'deep';
  };

  if (!Array.isArray(messages) || messages.length === 0) {
    res.status(400).json({
      error: { message: 'messages must be a non-empty array.', type: 'invalid_request_error' },
    });
    return;
  }

  let ai: GoogleGenAI | null = null;
  try {
    ai = createGeminiClient();
  } catch {
    ai = null;
  }

  const systemParts = messages
    .filter((m) => m.role === 'system' && typeof m.content === 'string')
    .map((m) => m.content);
  const systemInstruction =
    systemParts.join('\n\n') ||
    'You are Qowe 1.0 Ultra (801.32B parameter multimodal causal transformer). Provide clear, accurate, well-structured responses.';

  const chatContents = messages
    .filter((m) => m.role !== 'system' && typeof m.content === 'string' && m.content.trim())
    .map((m) => ({
      role: m.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: m.content }],
    }));

  if (chatContents.length === 0) {
    res.status(400).json({
      error: { message: 'At least one user or assistant message with content is required.', type: 'invalid_request_error' },
    });
    return;
  }

  const thinkingLevel =
    reasoning_effort === 'standard' ? ThinkingLevel.MEDIUM : ThinkingLevel.HIGH;
  const responseModelName = typeof model === 'string' && model.trim() ? model.trim() : 'qowe-1.0-ultra';
  const completionId = `chatcmpl-${crypto.randomBytes(10).toString('hex')}`;
  const created = Math.floor(Date.now() / 1000);

  if (stream) {
    res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders?.();

    let clientClosed = false;
    let sentAny = false;
    const abortCtrl = new AbortController();
    res.on('close', () => {
      if (!res.writableEnded) {
        clientClosed = true;
        abortCtrl.abort();
      }
    });

    try {
      if (!ai) throw new Error('local fallback');
      const streamResp = await generateWithHydrogenFallback((upstreamModel, useThinking) =>
        ai.models.generateContentStream({
          model: upstreamModel,
          contents: chatContents,
          config: {
            systemInstruction,
            abortSignal: abortCtrl.signal,
            ...(useThinking ? { thinkingConfig: { thinkingLevel } } : {}),
          },
        }),
      );

      for await (const chunk of streamResp) {
        if (clientClosed) break;
        const c = chunk as GenerateContentResponse;
        if (c.text) {
          sentAny = true;
          res.write(
            `data: ${JSON.stringify({
              id: completionId,
              object: 'chat.completion.chunk',
              created,
              model: responseModelName,
              choices: [{ index: 0, delta: { content: c.text }, finish_reason: null }],
            })}\n\n`,
          );
        }
      }
    } catch {
      // Fall through to local fallback below if nothing streamed yet
    }

    if (!sentAny && !clientClosed && !res.writableEnded) {
      const localText = generateLocalQoweResponse({ messages });
      res.write(
        `data: ${JSON.stringify({
          id: completionId,
          object: 'chat.completion.chunk',
          created,
          model: responseModelName,
          choices: [{ index: 0, delta: { content: localText }, finish_reason: null }],
        })}\n\n`,
      );
    }

    if (!clientClosed && !res.writableEnded) {
      res.write(
        `data: ${JSON.stringify({
          id: completionId,
          object: 'chat.completion.chunk',
          created,
          model: responseModelName,
          choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
        })}\n\n`,
      );
      res.write('data: [DONE]\n\n');
      res.end();
    }
    return;
  }

  let contentText = '';
  try {
    if (!ai) throw new Error('local fallback');
    const response = await generateWithHydrogenFallback((upstreamModel, useThinking) =>
      ai.models.generateContent({
        model: upstreamModel,
        contents: chatContents,
        config: {
          systemInstruction,
          ...(useThinking ? { thinkingConfig: { thinkingLevel } } : {}),
        },
      }),
    );
    contentText = response.text ?? '';
  } catch {
    contentText = generateLocalQoweResponse({ messages });
  }

  res.json({
    id: completionId,
    object: 'chat.completion',
    created,
    model: responseModelName,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: contentText },
        finish_reason: 'stop',
      },
    ],
    usage: {
      prompt_tokens: Math.ceil(JSON.stringify(chatContents).length / 4),
      completion_tokens: Math.ceil(contentText.length / 4),
      total_tokens:
        Math.ceil(JSON.stringify(chatContents).length / 4) +
        Math.ceil(contentText.length / 4),
    },
  });
});

app.post('/api/v1/images/generations', requireQoweApiKey, async (req, res) => {
  const { prompt, aspect_ratio } = (req.body ?? {}) as {
    prompt?: string;
    aspect_ratio?: '1:1' | '16:9' | '9:16';
  };
  if (typeof prompt !== 'string' || !prompt.trim()) {
    res.status(400).json({
      error: { message: 'prompt must be a non-empty string.', type: 'invalid_request_error' },
    });
    return;
  }

  const safeRatio = aspect_ratio === '16:9' || aspect_ratio === '9:16' ? aspect_ratio : '1:1';
  const cleanPrompt = prompt.trim().slice(0, 1000);
  let result: { svg: string; width: number; height: number };
  try {
    const ai = createGeminiClient();
    result = await generateStudioImageSvg(ai, cleanPrompt, safeRatio);
  } catch {
    const dims = getDimensionsForRatio(safeRatio);
    result = { svg: sanitizeGeneratedSvg('', cleanPrompt), ...dims };
  }
  res.json({
    created: Math.floor(Date.now() / 1000),
    model: 'qowe-1.0-vision',
    data: [
      {
        prompt: prompt.trim(),
        mime_type: 'image/svg+xml',
        width: result.width,
        height: result.height,
        svg: result.svg,
        b64_json: Buffer.from(result.svg, 'utf8').toString('base64'),
      },
    ],
  });
});

app.post('/api/v1/videos/generations', requireQoweApiKey, async (req, res) => {
  const { prompt, aspect_ratio, duration_sec } = (req.body ?? {}) as {
    prompt?: string;
    aspect_ratio?: '16:9' | '1:1';
    duration_sec?: number;
  };
  if (typeof prompt !== 'string' || !prompt.trim()) {
    res.status(400).json({
      error: { message: 'prompt must be a non-empty string.', type: 'invalid_request_error' },
    });
    return;
  }

  const safeRatio = aspect_ratio === '1:1' ? '1:1' : '16:9';
  const safeDur = typeof duration_sec === 'number' ? duration_sec : 4;
  const cleanPrompt = prompt.trim().slice(0, 1000);
  let sceneSpec: VideoSceneSpec;
  try {
    const ai = createGeminiClient();
    sceneSpec = await generateVideoSceneSpec(ai, cleanPrompt, safeRatio, safeDur);
  } catch {
    sceneSpec = normalizeVideoSceneSpec(null, cleanPrompt);
    sceneSpec.aspectRatio = safeRatio;
    sceneSpec.durationSec = Math.max(2, Math.min(10, Math.round(safeDur)));
  }
  res.json({
    id: `vid_${crypto.randomBytes(10).toString('hex')}`,
    created: Math.floor(Date.now() / 1000),
    model: 'qowe-1.0-video',
    fps: 60,
    duration_sec: sceneSpec.durationSec,
    aspect_ratio: sceneSpec.aspectRatio,
    scene_spec: sceneSpec,
  });
});

app.use('/api', (_req, res) => {
  res.status(404).json({
    error: {
      message: 'API endpoint not found.',
      type: 'invalid_request_error',
      code: 'endpoint_not_found',
    },
  });
});

async function startServer() {
  const distDir = path.resolve(__dirname, 'dist');
  const hasDistBuild = fs.existsSync(path.join(distDir, 'index.html'));
  const isProd =
    process.env.NODE_ENV === 'production' ||
    process.env.RENDER === 'true' ||
    (process.env.npm_lifecycle_event === 'start' && hasDistBuild);

  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      if (fs.existsSync(path.join(distDir, 'index.html'))) {
        res.sendFile(path.join(distDir, 'index.html'));
      } else {
        res.status(503).send('Build output not found. Run npm run build first.');
      }
    });
  }

  const port = Number(process.env.PORT) || 3000;
  app.listen(port, '0.0.0.0', () => {
    console.log(`Qowe 1.0 server listening on http://localhost:${port}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
