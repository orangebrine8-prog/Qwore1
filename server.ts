import 'dotenv/config';
import crypto from 'node:crypto';
import fs from 'node:fs';
import express, { type NextFunction, type Request, type Response } from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
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
import { generateQoweAssistantResponse } from './src/engine/qoweAssistantEngine.ts';

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

function formatServerError(err: unknown): string {
  return err instanceof Error ? err.message : String(err || 'Unexpected server error.');
}

function getDimensionsForRatio(aspectRatio: '1:1' | '16:9' | '9:16' = '1:1') {
  return aspectRatio === '16:9'
    ? { width: 1024, height: 576 }
    : aspectRatio === '9:16'
      ? { width: 576, height: 1024 }
      : { width: 1024, height: 1024 };
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

const QOWE_MODELS_LIST = [
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
];

app.get('/api/health', (_req, res) => {
  res.json({
    ok: true,
    model: 'Qowe 1.0',
    engine: 'qowe-1.0-self-hosted',
    hasKey: true,
  });
});

// ---- Qowe API Key Management & Real-Time Usage Stats ----
app.get('/api/usage', async (_req, res) => {
  try {
    const usage = await usageTracker.getSummary();
    res.json({ usage });
  } catch (err) {
    res.status(500).json({ error: formatServerError(err) });
  }
});

app.get('/api/keys', async (req, res) => {
  try {
    const [keys, usage] = await Promise.all([keyStore.listKeys(), usageTracker.getSummary()]);
    res.json({ keys, usage, baseUrls: getBaseUrls(req) });
  } catch (err) {
    res.status(500).json({ error: formatServerError(err) });
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

// ---- Studio Internal Endpoints (100% Self-Hosted Qowe 1.0 Engine) ----
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
  } = (req.body ?? {}) as {
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

  res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders?.();
  res.write(': stream-open\n\n');

  let clientClosed = false;
  res.on('close', () => {
    if (!res.writableEnded) {
      clientClosed = true;
    }
  });

  try {
    const responseText = generateQoweAssistantResponse({
      messages,
      architectureName,
      totalParamsFormatted,
      trainingContext,
      checkpointStep,
      checkpointLoss,
      reasoningDepth,
      outputMode,
    });

    const chunkSize = 48;
    for (let i = 0; i < responseText.length; i += chunkSize) {
      if (clientClosed || res.writableEnded) break;
      res.write(`data: ${JSON.stringify({ text: responseText.slice(i, i + chunkSize) })}\n\n`);
    }

    if (!clientClosed && !res.writableEnded) {
      res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
      res.end();
    }
  } catch (err) {
    if (!clientClosed && !res.writableEnded) {
      res.write(`data: ${JSON.stringify({ error: formatServerError(err) })}\n\n`);
      res.end();
    }
  }
});

app.post('/api/generate-image', (req, res) => {
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
  const dims = getDimensionsForRatio(safeRatio);
  res.json({
    model: 'qowe-1.0-vision',
    svg: sanitizeGeneratedSvg('', cleanPrompt),
    ...dims,
    mimeType: 'image/svg+xml',
  });
});

app.post('/api/generate-video', (req, res) => {
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
  const sceneSpec: VideoSceneSpec = normalizeVideoSceneSpec(null, cleanPrompt);
  sceneSpec.aspectRatio = safeRatio;
  sceneSpec.durationSec = Math.max(2, Math.min(10, Math.round(safeDur)));
  res.json({
    model: 'qowe-1.0-video',
    fps: 60,
    sceneSpec,
  });
});

// ---- Studio Same-Origin API Playground Bridge ----
app.post('/api/playground/test', requireQoweApiKey, (req, res) => {
  const { endpoint, prompt } = (req.body ?? {}) as {
    endpoint?: string;
    prompt?: string;
  };

  if (endpoint === 'models') {
    res.json({
      object: 'list',
      data: QOWE_MODELS_LIST,
    });
    return;
  }

  if (typeof prompt !== 'string' || !prompt.trim()) {
    res.status(400).json({
      error: { message: 'prompt must be a non-empty string.', type: 'invalid_request_error' },
    });
    return;
  }

  const cleanPrompt = prompt.trim().slice(0, 1000);
  if (endpoint === 'image') {
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
    return;
  }

  if (endpoint === 'video') {
    const sceneSpec = normalizeVideoSceneSpec(null, cleanPrompt);
    sceneSpec.aspectRatio = '16:9';
    sceneSpec.durationSec = 4;
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

  const contentText = generateQoweAssistantResponse({
    messages: [{ role: 'user', content: cleanPrompt }],
  });

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
    engine: 'qowe-1.0-self-hosted',
    version: '1.0',
    authHeaders: ['x-api-key', 'x-qowe-api-key', 'Authorization: Bearer'],
  });
});

app.get('/api/v1/models', requireQoweApiKey, (_req, res) => {
  res.json({
    object: 'list',
    data: QOWE_MODELS_LIST,
  });
});

app.post('/api/v1/chat/completions', requireQoweApiKey, (req, res) => {
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

  const chatMessages = messages.filter(
    (m) => m && typeof m.content === 'string' && m.content.trim().length > 0,
  );
  if (chatMessages.length === 0) {
    res.status(400).json({
      error: {
        message: 'At least one message with non-empty content is required.',
        type: 'invalid_request_error',
      },
    });
    return;
  }

  const responseModelName =
    typeof model === 'string' && model.trim() ? model.trim() : 'qowe-1.0-ultra';
  const completionId = `chatcmpl-${crypto.randomBytes(10).toString('hex')}`;
  const created = Math.floor(Date.now() / 1000);
  const contentText = generateQoweAssistantResponse({
    messages: chatMessages,
    reasoningDepth: reasoning_effort === 'standard' ? 'standard' : 'deep',
  });

  if (stream) {
    res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders?.();

    const chunkSize = 48;
    for (let i = 0; i < contentText.length; i += chunkSize) {
      res.write(
        `data: ${JSON.stringify({
          id: completionId,
          object: 'chat.completion.chunk',
          created,
          model: responseModelName,
          choices: [{ index: 0, delta: { content: contentText.slice(i, i + chunkSize) }, finish_reason: null }],
        })}\n\n`,
      );
    }

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
    return;
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
      prompt_tokens: Math.ceil(JSON.stringify(chatMessages).length / 4),
      completion_tokens: Math.ceil(contentText.length / 4),
      total_tokens:
        Math.ceil(JSON.stringify(chatMessages).length / 4) +
        Math.ceil(contentText.length / 4),
    },
  });
});

app.post('/api/v1/images/generations', requireQoweApiKey, (req, res) => {
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
  const dims = getDimensionsForRatio(safeRatio);
  const svg = sanitizeGeneratedSvg('', cleanPrompt);

  res.json({
    created: Math.floor(Date.now() / 1000),
    model: 'qowe-1.0-vision',
    data: [
      {
        prompt: prompt.trim(),
        mime_type: 'image/svg+xml',
        width: dims.width,
        height: dims.height,
        svg,
        b64_json: Buffer.from(svg, 'utf8').toString('base64'),
      },
    ],
  });
});

app.post('/api/v1/videos/generations', requireQoweApiKey, (req, res) => {
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
  const sceneSpec = normalizeVideoSceneSpec(null, cleanPrompt);
  sceneSpec.aspectRatio = safeRatio;
  sceneSpec.durationSec = Math.max(2, Math.min(10, Math.round(safeDur)));

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
