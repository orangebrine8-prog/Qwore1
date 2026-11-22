import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {
  QoweApiKeyStore,
  extractApiKeyFromHeaders,
  hashQoweApiKey,
} from '../../server/apiKeyService';
import { QoweUsageTracker } from '../../server/apiUsageTracker';
import {
  isAiStudioSharedPreviewUrl,
  isCookieGatedPreviewUrl,
  resolveBaseUrls,
} from '../../server/urlUtils';
import {
  detectChatIntent,
  normalizeVideoSceneSpec,
  sanitizeGeneratedSvg,
} from './intentAndMedia';

test('resolveBaseUrls avoids advertising cookie-gated ais-dev URLs as public API base URLs', () => {
  const devCtx = resolveBaseUrls({
    requestOrigin: 'https://ais-dev-abc123.asia-southeast1.run.app',
    rawAppUrl: 'https://ais-dev-abc123.asia-southeast1.run.app',
  });
  assert.equal(devCtx.isCurrentDevPreview, true);
  assert.equal(devCtx.sharedUrl, 'https://ais-pre-abc123.asia-southeast1.run.app');
  assert.equal(devCtx.recommendedUrl, 'https://ais-pre-abc123.asia-southeast1.run.app');
  assert.equal(devCtx.isRecommendedDevPreview, false);

  const localCtx = resolveBaseUrls({
    requestOrigin: 'http://localhost:3000',
    rawAppUrl: 'https://ais-dev-abc123.asia-southeast1.run.app',
  });
  assert.equal(localCtx.isCurrentDevPreview, false);
  assert.equal(localCtx.recommendedUrl, 'http://localhost:3000');
  assert.equal(localCtx.sharedUrl, 'https://ais-pre-abc123.asia-southeast1.run.app');

  const sharedCtx = resolveBaseUrls({
    requestOrigin: 'https://ais-pre-abc123.asia-southeast1.run.app',
    rawAppUrl: 'https://ais-dev-abc123.asia-southeast1.run.app',
  });
  assert.equal(sharedCtx.isCurrentDevPreview, false);
  assert.equal(sharedCtx.isCurrentSharedPreview, true);
  assert.equal(sharedCtx.isRecommendedCookieGated, true);
  assert.equal(sharedCtx.recommendedUrl, 'https://ais-pre-abc123.asia-southeast1.run.app');

  assert.equal(isAiStudioSharedPreviewUrl('https://ais-pre-abc123.asia-southeast1.run.app'), true);
  assert.equal(isCookieGatedPreviewUrl('https://ais-pre-abc123.asia-southeast1.run.app'), true);
  assert.equal(isCookieGatedPreviewUrl('https://ais-dev-abc123.asia-southeast1.run.app'), true);
  assert.equal(isCookieGatedPreviewUrl('https://qowe-service-123.a.run.app'), false);
});

test('QoweApiKeyStore creates, hashes, verifies, tracks latency/usage, and revokes keys safely', async () => {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'qowe-keys-test-'));
  const storePath = path.join(tmpDir, 'keys.json');
  const store = new QoweApiKeyStore(storePath);

  const created = await store.createKey('External Test App');
  assert.ok(created.rawKey.startsWith('qowe_sk_'));
  assert.equal(created.record.name, 'External Test App');
  assert.equal(created.record.requestCount, 0);
  assert.equal(created.record.avgLatencyMs, 0);
  assert.equal(created.record.lastLatencyMs, null);

  // Raw key must never be stored in plaintext on disk
  const rawDisk = await fs.readFile(storePath, 'utf8');
  assert.equal(rawDisk.includes(created.rawKey), false);
  assert.equal(rawDisk.includes(hashQoweApiKey(created.rawKey)), true);

  // Valid verification increments requestCount and records latency
  const verified = await store.verifyAndTouch(created.rawKey);
  assert.ok(verified);
  assert.equal(verified.id, created.record.id);
  assert.equal(verified.requestCount, 1);

  await store.recordLatency(created.record.id, 240);
  const verified2 = await store.verifyAndTouch(created.rawKey);
  assert.ok(verified2);
  await store.recordLatency(created.record.id, 120);
  const listed = await store.listKeys();
  assert.equal(listed[0].requestCount, 2);
  assert.equal(listed[0].avgLatencyMs, 180);
  assert.equal(listed[0].lastLatencyMs, 120);

  // Invalid key is rejected
  const invalid = await store.verifyAndTouch('qowe_sk_000000000000000000000000000000000000000000000000');
  assert.equal(invalid, null);

  // Revocation removes key cleanly
  const revoked = await store.revokeKey(created.record.id);
  assert.equal(revoked, true);
  assert.equal(await store.verifyAndTouch(created.rawKey), null);
  assert.deepEqual(await store.listKeys(), []);

  await fs.rm(tmpDir, { recursive: true, force: true });
});

test('QoweUsageTracker tracks real-time request counts, average/p95 latency, category stats, and caps recentEvents', async () => {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'qowe-usage-test-'));
  const usagePath = path.join(tmpDir, 'usage.json');
  const tracker = new QoweUsageTracker(usagePath);

  const empty = await tracker.getSummary();
  assert.equal(empty.totalRequests, 0);
  assert.equal(empty.avgLatencyMs, 0);
  assert.equal(empty.p95LatencyMs, 0);
  assert.equal(empty.successRate, 100);

  const latencies = [100, 150, 200, 250, 300, 400, 500, 600, 700, 1200];
  for (let i = 0; i < latencies.length; i += 1) {
    await tracker.recordRequest({
      endpoint: i % 2 === 0 ? 'POST /api/v1/chat/completions' : 'POST /api/v1/images/generations',
      category: i % 2 === 0 ? 'chat' : 'image',
      model: i % 2 === 0 ? 'qowe-1.0-ultra' : 'qowe-1.0-vision',
      keyPrefix: 'qowe_sk_demo...1234',
      keyName: 'Demo Key',
      status: i === 9 ? 500 : 200,
      latencyMs: latencies[i],
    });
  }

  // Record 8 more models requests so total = 18 and recentEvents caps cleanly at 15
  for (let i = 0; i < 8; i += 1) {
    await tracker.recordRequest({
      endpoint: 'GET /api/v1/models',
      category: 'models',
      model: 'qowe-1.0-models',
      keyPrefix: 'qowe_sk_demo...1234',
      keyName: 'Demo Key',
      status: 200,
      latencyMs: 12,
    });
  }

  const summary = await tracker.getSummary();
  assert.equal(summary.totalRequests, 18);
  assert.equal(summary.successCount, 17);
  assert.equal(summary.errorCount, 1);
  assert.equal(summary.successRate, 94.4);
  assert.equal(summary.minLatencyMs, 12);
  assert.equal(summary.maxLatencyMs, 1200);
  assert.equal(summary.lastLatencyMs, 12);
  assert.equal(summary.p95LatencyMs, 1200);
  assert.equal(summary.byCategory.chat.count, 5);
  assert.equal(summary.byCategory.image.count, 5);
  assert.equal(summary.byCategory.models.count, 8);
  assert.equal(summary.byCategory.video.count, 0);
  assert.equal(summary.recentEvents.length, 15);
  assert.equal(summary.recentEvents[0].endpoint, 'GET /api/v1/models');

  // Verify persistence across reload
  const reloaded = new QoweUsageTracker(usagePath);
  const persistedSummary = await reloaded.getSummary();
  assert.equal(persistedSummary.totalRequests, 18);
  assert.equal(persistedSummary.p95LatencyMs, 1200);

  await fs.rm(tmpDir, { recursive: true, force: true });
});

test('extractApiKeyFromHeaders supports Bearer, x-api-key, and x-qowe-api-key headers', () => {
  assert.equal(
    extractApiKeyFromHeaders({ authorization: 'Bearer qowe_sk_abc123' }),
    'qowe_sk_abc123',
  );
  assert.equal(extractApiKeyFromHeaders({ 'x-api-key': 'qowe_sk_xyz789' }), 'qowe_sk_xyz789');
  assert.equal(
    extractApiKeyFromHeaders({ 'x-qowe-api-key': 'qowe_sk_bridge999' }),
    'qowe_sk_bridge999',
  );
  assert.equal(extractApiKeyFromHeaders({}), undefined);
});

test('detectChatIntent routes natural image/video generation prompts without hijacking code questions', () => {
  assert.equal(detectChatIntent('generate an image of a cyberpunk cat', 'text'), 'image');
  assert.equal(detectChatIntent('draw a golden mountain sunrise', 'text'), 'image');
  assert.equal(detectChatIntent('generate a video of a rocket launching into mars', 'text'), 'video');
  assert.equal(detectChatIntent('animate ocean waves under neon lights', 'text'), 'video');
  assert.equal(detectChatIntent('explain how video compression works in python', 'text'), 'text');
  assert.equal(detectChatIntent('write html code to draw a square', 'text'), 'text');
  assert.equal(detectChatIntent('hello how are you', 'build'), 'build');
});

test('sanitizeGeneratedSvg strips scripts/handlers and ensures valid SVG attributes', () => {
  const dirty = '```svg\n<svg onclick="alert(1)"><script>evil()</script><circle cx="100" cy="100" r="50" /></svg>\n```';
  const clean = sanitizeGeneratedSvg(dirty, 'Safe Circle');
  assert.ok(clean.includes('<svg'));
  assert.ok(clean.includes('xmlns="http://www.w3.org/2000/svg"'));
  assert.ok(clean.includes('viewBox="0 0 1024 1024"'));
  assert.equal(/<script/i.test(clean), false);
  assert.equal(/onclick=/i.test(clean), false);
});

test('normalizeVideoSceneSpec clamps coordinates and supplies prompt-aware defaults', () => {
  const normalized = normalizeVideoSceneSpec(
    {
      backdrop: 'invalid_place',
      skyTop: 'not-a-color',
      durationSec: 99,
      actors: [{ type: 'rocket', x: 5, y: -2, scale: 9, color: '#38bdf8' }],
    },
    'rocket blasting through space',
  );

  assert.equal(normalized.durationSec, 10);
  assert.equal(normalized.backdrop, 'space');
  assert.equal(normalized.actors[0].x, 0.9);
  assert.equal(normalized.actors[0].y, 0.15);
  assert.equal(normalized.actors[0].scale, 1.8);
  assert.equal(normalized.actors[0].color, '#38bdf8');
});
