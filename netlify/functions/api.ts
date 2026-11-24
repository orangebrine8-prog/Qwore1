import crypto from 'node:crypto';
import {
  QoweApiKeyStore,
  extractApiKeyFromHeaders,
  type PublicQoweApiKeyRecord,
} from '../../server/apiKeyService.ts';
import { type ApiEndpointCategory, QoweUsageTracker } from '../../server/apiUsageTracker.ts';
import { resolveBaseUrls } from '../../server/urlUtils.ts';
import {
  normalizeVideoSceneSpec,
  sanitizeGeneratedSvg,
} from '../../src/engine/intentAndMedia.ts';
import {
  generateQoweAssistantResponse,
  type LocalQoweMessage,
} from '../../src/engine/qoweAssistantEngine.ts';

const keyStore = new QoweApiKeyStore('/tmp/qowe-keys.json', { allowStatelessSigned: true });
const usageTracker = new QoweUsageTracker('/tmp/qowe-usage.json');

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

const CORS_HEADERS: Record<string, string> = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-api-key, x-qowe-api-key',
  'Access-Control-Max-Age': '86400',
};

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      ...CORS_HEADERS,
    },
  });
}

function classifyEndpointCategory(reqPath: string, bodyEndpoint?: unknown): ApiEndpointCategory {
  if (reqPath.includes('/images/') || bodyEndpoint === 'image') return 'image';
  if (reqPath.includes('/videos/') || bodyEndpoint === 'video') return 'video';
  if (reqPath.includes('/models') || bodyEndpoint === 'models') return 'models';
  return 'chat';
}

function getDimensionsForRatio(aspectRatio: '1:1' | '16:9' | '9:16' = '1:1') {
  return aspectRatio === '16:9'
    ? { width: 1024, height: 576 }
    : aspectRatio === '9:16'
      ? { width: 576, height: 1024 }
      : { width: 1024, height: 1024 };
}

function normalizeApiPath(rawPathname: string): string {
  if (rawPathname.startsWith('/.netlify/functions/api')) {
    const rest = rawPathname.slice('/.netlify/functions/api'.length) || '/';
    return rest.startsWith('/api') ? rest : `/api${rest === '/' ? '' : rest}`;
  }
  return rawPathname;
}

async function parseJsonBody<T = Record<string, unknown>>(req: Request): Promise<T> {
  try {
    const text = await req.text();
    if (!text.trim()) return {} as T;
    return JSON.parse(text) as T;
  } catch {
    return {} as T;
  }
}

async function authenticateApiKey(req: Request): Promise<PublicQoweApiKeyRecord | null> {
  const rawKey = extractApiKeyFromHeaders({
    authorization: req.headers.get('authorization') ?? undefined,
    'x-api-key': req.headers.get('x-api-key') ?? undefined,
    'x-qowe-api-key': req.headers.get('x-qowe-api-key') ?? undefined,
  });
  return keyStore.verifyAndTouch(rawKey);
}

async function recordUsage(
  record: PublicQoweApiKeyRecord,
  method: string,
  endpointPath: string,
  status: number,
  startedAt: number,
  bodyObj?: { model?: unknown; endpoint?: unknown },
): Promise<void> {
  const latencyMs = Math.max(1, Math.round(performance.now() - startedAt));
  const category = classifyEndpointCategory(endpointPath, bodyObj?.endpoint);
  const requestedModel =
    category === 'image'
      ? 'qowe-1.0-vision'
      : category === 'video'
        ? 'qowe-1.0-video'
        : category === 'models'
          ? 'qowe-1.0-models'
          : typeof bodyObj?.model === 'string' && bodyObj.model.trim()
            ? bodyObj.model.trim().slice(0, 64)
            : 'qowe-1.0-ultra';

  await Promise.all([
    keyStore.recordLatency(record.id, latencyMs),
    usageTracker.recordRequest({
      endpoint: `${method} ${endpointPath}`,
      category,
      model: requestedModel,
      keyPrefix: record.prefix,
      keyName: record.name,
      status,
      latencyMs,
    }),
  ]);
}

export default async function handler(req: Request): Promise<Response> {
  if (req.method === 'OPTIONS') {
    return new Response(null, { status: 204, headers: CORS_HEADERS });
  }

  const url = new URL(req.url);
  const pathname = normalizeApiPath(url.pathname);
  const method = req.method.toUpperCase();

  // ---- Health ----
  if (method === 'GET' && pathname === '/api/health') {
    return jsonResponse({
      ok: true,
      model: 'Qowe 1.0',
      engine: 'qowe-1.0-self-hosted',
      runtime: 'netlify-functions',
      hasKey: true,
    });
  }

  if (method === 'GET' && pathname === '/api/v1/health') {
    return jsonResponse({
      ok: true,
      service: 'qowe-1.0-api',
      engine: 'qowe-1.0-self-hosted',
      version: '1.0',
      authHeaders: ['x-api-key', 'x-qowe-api-key', 'Authorization: Bearer'],
    });
  }

  // ---- Keys & Usage ----
  if (method === 'GET' && pathname === '/api/usage') {
    const usage = await usageTracker.getSummary();
    return jsonResponse({ usage });
  }

  if (method === 'GET' && pathname === '/api/keys') {
    const host = req.headers.get('x-forwarded-host') || req.headers.get('host') || url.host;
    const proto = (req.headers.get('x-forwarded-proto') || url.protocol.replace(':', '') || 'https')
      .split(',')[0]
      .trim();
    const [keys, usage] = await Promise.all([keyStore.listKeys(), usageTracker.getSummary()]);
    return jsonResponse({
      keys,
      usage,
      baseUrls: resolveBaseUrls({
        requestOrigin: `${proto}://${host}`,
        rawAppUrl: process.env.URL || process.env.APP_URL,
      }),
    });
  }

  if (method === 'POST' && pathname === '/api/keys') {
    try {
      const body = await parseJsonBody<{ name?: unknown }>(req);
      const created = await keyStore.createKey(body.name);
      return jsonResponse({ key: created.record, rawKey: created.rawKey }, 201);
    } catch (err) {
      return jsonResponse(
        { error: err instanceof Error ? err.message : 'Unable to create API key.' },
        400,
      );
    }
  }

  if (method === 'DELETE' && pathname.startsWith('/api/keys/')) {
    const id = decodeURIComponent(pathname.slice('/api/keys/'.length)).trim();
    if (!id) return jsonResponse({ error: 'Key ID is required.' }, 400);
    const removed = await keyStore.revokeKey(id);
    if (!removed) return jsonResponse({ error: 'API key not found.' }, 404);
    return jsonResponse({ ok: true });
  }

  // ---- Studio Chat Streaming (/api/chat) ----
  if (method === 'POST' && pathname === '/api/chat') {
    const body = await parseJsonBody<{
      messages?: LocalQoweMessage[];
      architectureName?: string;
      totalParamsFormatted?: string;
      trainingContext?: string;
      checkpointStep?: number;
      checkpointLoss?: number;
      reasoningDepth?: 'standard' | 'deep';
      outputMode?: 'text' | 'build';
    }>(req);

    if (!Array.isArray(body.messages) || body.messages.length === 0) {
      return jsonResponse({ error: 'A non-empty messages array is required.' }, 400);
    }

    const responseText = generateQoweAssistantResponse({
      messages: body.messages,
      architectureName: body.architectureName,
      totalParamsFormatted: body.totalParamsFormatted,
      trainingContext: body.trainingContext,
      checkpointStep: body.checkpointStep,
      checkpointLoss: body.checkpointLoss,
      reasoningDepth: body.reasoningDepth,
      outputMode: body.outputMode,
    });

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(': stream-open\n\n'));
        const chunkSize = 48;
        for (let i = 0; i < responseText.length; i += chunkSize) {
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ text: responseText.slice(i, i + chunkSize) })}\n\n`),
          );
        }
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ done: true })}\n\n`));
        controller.close();
      },
    });

    return new Response(stream, {
      status: 200,
      headers: {
        'Content-Type': 'text/event-stream; charset=utf-8',
        'Cache-Control': 'no-cache, no-transform',
        ...CORS_HEADERS,
      },
    });
  }

  // ---- Studio Media Synthesis ----
  if (method === 'POST' && pathname === '/api/generate-image') {
    const body = await parseJsonBody<{ prompt?: string; aspectRatio?: '1:1' | '16:9' | '9:16' }>(req);
    if (typeof body.prompt !== 'string' || !body.prompt.trim()) {
      return jsonResponse({ error: 'A non-empty image prompt is required.' }, 400);
    }
    const safeRatio =
      body.aspectRatio === '16:9' || body.aspectRatio === '9:16' ? body.aspectRatio : '1:1';
    const cleanPrompt = body.prompt.trim().slice(0, 1000);
    const dims = getDimensionsForRatio(safeRatio);
    return jsonResponse({
      model: 'qowe-1.0-vision',
      svg: sanitizeGeneratedSvg('', cleanPrompt),
      ...dims,
      mimeType: 'image/svg+xml',
    });
  }

  if (method === 'POST' && pathname === '/api/generate-video') {
    const body = await parseJsonBody<{
      prompt?: string;
      aspectRatio?: '16:9' | '1:1';
      durationSec?: number;
    }>(req);
    if (typeof body.prompt !== 'string' || !body.prompt.trim()) {
      return jsonResponse({ error: 'A non-empty video prompt is required.' }, 400);
    }
    const safeRatio = body.aspectRatio === '1:1' ? '1:1' : '16:9';
    const safeDur = typeof body.durationSec === 'number' ? body.durationSec : 4;
    const cleanPrompt = body.prompt.trim().slice(0, 1000);
    const sceneSpec = normalizeVideoSceneSpec(null, cleanPrompt);
    sceneSpec.aspectRatio = safeRatio;
    sceneSpec.durationSec = Math.max(2, Math.min(10, Math.round(safeDur)));
    return jsonResponse({
      model: 'qowe-1.0-video',
      fps: 60,
      sceneSpec,
    });
  }

  // ---- Authenticated Endpoints ----
  const isAuthedRoute =
    pathname === '/api/playground/test' ||
    pathname === '/api/v1/models' ||
    pathname === '/api/v1/chat/completions' ||
    pathname === '/api/v1/images/generations' ||
    pathname === '/api/v1/videos/generations';

  if (isAuthedRoute) {
    const keyRecord = await authenticateApiKey(req);
    if (!keyRecord) {
      return jsonResponse(
        {
          error: {
            message:
              'Invalid or missing Qowe API key. Pass Authorization: Bearer qowe_sk_... or x-api-key header.',
            type: 'invalid_request_error',
            code: 'invalid_api_key',
          },
        },
        401,
      );
    }
    const startedAt = performance.now();

    if (method === 'GET' && pathname === '/api/v1/models') {
      await recordUsage(keyRecord, method, pathname, 200, startedAt);
      return jsonResponse({ object: 'list', data: QOWE_MODELS_LIST });
    }

    if (method === 'POST' && pathname === '/api/playground/test') {
      const body = await parseJsonBody<{ endpoint?: string; prompt?: string }>(req);
      if (body.endpoint === 'models') {
        await recordUsage(keyRecord, method, pathname, 200, startedAt, body);
        return jsonResponse({ object: 'list', data: QOWE_MODELS_LIST });
      }
      if (typeof body.prompt !== 'string' || !body.prompt.trim()) {
        await recordUsage(keyRecord, method, pathname, 400, startedAt, body);
        return jsonResponse(
          { error: { message: 'prompt must be a non-empty string.', type: 'invalid_request_error' } },
          400,
        );
      }

      const cleanPrompt = body.prompt.trim().slice(0, 1000);
      if (body.endpoint === 'image') {
        await recordUsage(keyRecord, method, pathname, 200, startedAt, body);
        return jsonResponse({
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

      if (body.endpoint === 'video') {
        const sceneSpec = normalizeVideoSceneSpec(null, cleanPrompt);
        sceneSpec.aspectRatio = '16:9';
        sceneSpec.durationSec = 4;
        await recordUsage(keyRecord, method, pathname, 200, startedAt, body);
        return jsonResponse({
          id: `vid_${crypto.randomBytes(10).toString('hex')}`,
          created: Math.floor(Date.now() / 1000),
          model: 'qowe-1.0-video',
          fps: 60,
          duration_sec: sceneSpec.durationSec,
          aspect_ratio: sceneSpec.aspectRatio,
          scene_spec: sceneSpec,
        });
      }

      const contentText = generateQoweAssistantResponse({
        messages: [{ role: 'user', content: cleanPrompt }],
      });
      await recordUsage(keyRecord, method, pathname, 200, startedAt, body);
      return jsonResponse({
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
    }

    if (method === 'POST' && pathname === '/api/v1/chat/completions') {
      const body = await parseJsonBody<{
        model?: string;
        messages?: LocalQoweMessage[];
        stream?: boolean;
        reasoning_effort?: 'standard' | 'deep';
      }>(req);

      if (!Array.isArray(body.messages) || body.messages.length === 0) {
        await recordUsage(keyRecord, method, pathname, 400, startedAt, body);
        return jsonResponse(
          { error: { message: 'messages must be a non-empty array.', type: 'invalid_request_error' } },
          400,
        );
      }

      const chatMessages = body.messages.filter(
        (m) => m && typeof m.content === 'string' && m.content.trim().length > 0,
      );
      if (chatMessages.length === 0) {
        await recordUsage(keyRecord, method, pathname, 400, startedAt, body);
        return jsonResponse(
          {
            error: {
              message: 'At least one message with non-empty content is required.',
              type: 'invalid_request_error',
            },
          },
          400,
        );
      }

      const responseModelName =
        typeof body.model === 'string' && body.model.trim() ? body.model.trim() : 'qowe-1.0-ultra';
      const completionId = `chatcmpl-${crypto.randomBytes(10).toString('hex')}`;
      const created = Math.floor(Date.now() / 1000);
      const contentText = generateQoweAssistantResponse({
        messages: chatMessages,
        reasoningDepth: body.reasoning_effort === 'standard' ? 'standard' : 'deep',
      });

      await recordUsage(keyRecord, method, pathname, 200, startedAt, body);

      if (body.stream) {
        const encoder = new TextEncoder();
        const sse = new ReadableStream({
          start(controller) {
            const chunkSize = 48;
            for (let i = 0; i < contentText.length; i += chunkSize) {
              controller.enqueue(
                encoder.encode(
                  `data: ${JSON.stringify({
                    id: completionId,
                    object: 'chat.completion.chunk',
                    created,
                    model: responseModelName,
                    choices: [
                      { index: 0, delta: { content: contentText.slice(i, i + chunkSize) }, finish_reason: null },
                    ],
                  })}\n\n`,
                ),
              );
            }
            controller.enqueue(
              encoder.encode(
                `data: ${JSON.stringify({
                  id: completionId,
                  object: 'chat.completion.chunk',
                  created,
                  model: responseModelName,
                  choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
                })}\n\n`,
              ),
            );
            controller.enqueue(encoder.encode('data: [DONE]\n\n'));
            controller.close();
          },
        });
        return new Response(sse, {
          status: 200,
          headers: {
            'Content-Type': 'text/event-stream; charset=utf-8',
            'Cache-Control': 'no-cache, no-transform',
            ...CORS_HEADERS,
          },
        });
      }

      return jsonResponse({
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
    }

    if (method === 'POST' && pathname === '/api/v1/images/generations') {
      const body = await parseJsonBody<{
        prompt?: string;
        aspect_ratio?: '1:1' | '16:9' | '9:16';
        model?: unknown;
      }>(req);
      if (typeof body.prompt !== 'string' || !body.prompt.trim()) {
        await recordUsage(keyRecord, method, pathname, 400, startedAt, { endpoint: 'image', model: body.model });
        return jsonResponse(
          { error: { message: 'prompt must be a non-empty string.', type: 'invalid_request_error' } },
          400,
        );
      }
      const safeRatio =
        body.aspect_ratio === '16:9' || body.aspect_ratio === '9:16' ? body.aspect_ratio : '1:1';
      const cleanPrompt = body.prompt.trim().slice(0, 1000);
      const dims = getDimensionsForRatio(safeRatio);
      const svg = sanitizeGeneratedSvg('', cleanPrompt);
      await recordUsage(keyRecord, method, pathname, 200, startedAt, { endpoint: 'image', model: body.model });
      return jsonResponse({
        created: Math.floor(Date.now() / 1000),
        model: 'qowe-1.0-vision',
        data: [
          {
            prompt: body.prompt.trim(),
            mime_type: 'image/svg+xml',
            width: dims.width,
            height: dims.height,
            svg,
            b64_json: Buffer.from(svg, 'utf8').toString('base64'),
          },
        ],
      });
    }

    if (method === 'POST' && pathname === '/api/v1/videos/generations') {
      const body = await parseJsonBody<{
        prompt?: string;
        aspect_ratio?: '16:9' | '1:1';
        duration_sec?: number;
        model?: unknown;
      }>(req);
      if (typeof body.prompt !== 'string' || !body.prompt.trim()) {
        await recordUsage(keyRecord, method, pathname, 400, startedAt, { endpoint: 'video', model: body.model });
        return jsonResponse(
          { error: { message: 'prompt must be a non-empty string.', type: 'invalid_request_error' } },
          400,
        );
      }
      const safeRatio = body.aspect_ratio === '1:1' ? '1:1' : '16:9';
      const safeDur = typeof body.duration_sec === 'number' ? body.duration_sec : 4;
      const cleanPrompt = body.prompt.trim().slice(0, 1000);
      const sceneSpec = normalizeVideoSceneSpec(null, cleanPrompt);
      sceneSpec.aspectRatio = safeRatio;
      sceneSpec.durationSec = Math.max(2, Math.min(10, Math.round(safeDur)));
      await recordUsage(keyRecord, method, pathname, 200, startedAt, { endpoint: 'video', model: body.model });
      return jsonResponse({
        id: `vid_${crypto.randomBytes(10).toString('hex')}`,
        created: Math.floor(Date.now() / 1000),
        model: 'qowe-1.0-video',
        fps: 60,
        duration_sec: sceneSpec.durationSec,
        aspect_ratio: sceneSpec.aspectRatio,
        scene_spec: sceneSpec,
      });
    }
  }

  return jsonResponse(
    {
      error: {
        message: 'API endpoint not found.',
        type: 'invalid_request_error',
        code: 'endpoint_not_found',
      },
    },
    404,
  );
}
