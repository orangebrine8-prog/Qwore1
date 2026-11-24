import test from 'node:test';
import assert from 'node:assert/strict';
import netlifyApiHandler from '../netlify/functions/api';
import { executeVirtualApiRequest, parseFullStackFiles } from '../src/engine/fullStackProject';
import { generateQoweAssistantResponse } from '../src/engine/qoweAssistantEngine';
import { generateLocalFullStackBuild, generateLocalQoweResponse } from './localQoweFallback';

test('generateLocalQoweResponse answers identity, greeting, custom corpus, arithmetic, and code prompts cleanly', () => {
  const identity = generateLocalQoweResponse({
    messages: [{ role: 'user', content: 'Who are you?' }],
    architectureName: 'Qowe 1.0 Ultra 800B',
    totalParamsFormatted: '801.32B',
  });
  assert.ok(identity.includes('Qowe 1.0'));
  assert.equal(identity.toLowerCase().includes('high demand'), false);

  const geminiCheck = generateQoweAssistantResponse({
    messages: [{ role: 'user', content: 'Are you Gemini or powered by another AI?' }],
  });
  assert.ok(geminiCheck.includes('Qowe 1.0'));
  assert.ok(geminiCheck.includes('zero external API keys'));

  const greeting = generateLocalQoweResponse({
    messages: [{ role: 'user', content: 'yo' }],
  });
  assert.ok(greeting.includes('Qowe 1.0'));

  const multiTurn = generateQoweAssistantResponse({
    messages: [
      { role: 'user', content: 'Explain softmax stability' },
      { role: 'assistant', content: 'Subtract the max logit first.' },
      { role: 'user', content: 'What was my previous message?' },
    ],
  });
  assert.ok(multiTurn.includes('Explain softmax stability'));

  const customCorpusReply = generateLocalQoweResponse({
    messages: [{ role: 'user', content: 'What is project Atlas?' }],
    trainingContext:
      '<|user|> What is project Atlas?\n<|assistant|> Project Atlas is a local neural mesh engine built by our team.<|endoftext|>',
  });
  assert.ok(customCorpusReply.includes('Project Atlas is a local neural mesh engine'));

  const customProseReply = generateQoweAssistantResponse({
    messages: [{ role: 'user', content: 'Summarize quantum cooling turbine telemetry' }],
    trainingContext:
      'Quantum cooling turbine telemetry maintains superconducting coils at 4 Kelvin across closed helium loops.',
  });
  assert.ok(customProseReply.includes('Quantum cooling turbine telemetry'));

  const mathReply = generateLocalQoweResponse({
    messages: [{ role: 'user', content: 'What is (12 + 8) * 6?' }],
  });
  assert.ok(mathReply.includes('120'), `Expected 120 in math reply, got: ${mathReply}`);

  const codeReply = generateLocalQoweResponse({
    messages: [{ role: 'user', content: 'Write binary search in Python' }],
  });
  assert.ok(codeReply.includes('def binary_search'));
});

test('generateLocalFullStackBuild produces parseable multi-file Express + SQL + HTML project', () => {
  const markdown = generateLocalFullStackBuild('Build a daily habit tracker app');
  const parsed = parseFullStackFiles(markdown);

  assert.ok(parsed);
  assert.equal(parsed.isFullStack, true);
  assert.ok(parsed.files.some((f) => f.path === 'package.json'));
  assert.ok(parsed.files.some((f) => f.path === 'db/schema.sql'));
  assert.ok(parsed.files.some((f) => f.path === 'server.ts'));
  assert.ok(parsed.files.some((f) => f.path === 'public/index.html'));
  assert.ok(parsed.routes.some((r) => r.path.includes('/api/habits')));
  assert.equal(parsed.initialTables.habits.length, 3);

  const getRes = executeVirtualApiRequest(parsed.initialTables, {
    method: 'GET',
    url: '/api/habits',
  });
  assert.equal(getRes.response.status, 200);
  assert.equal((getRes.response.body as unknown[]).length, 3);
});

test('Netlify function handler serves /api/health, /api/chat SSE, and /api/v1 authenticated completions', async () => {
  const healthRes = await netlifyApiHandler(new Request('https://qowe-test.netlify.app/api/health'));
  assert.equal(healthRes.status, 200);
  const healthJson = (await healthRes.json()) as {
    ok: boolean;
    model: string;
    engine: string;
    runtime: string;
    hasKey: boolean;
  };
  assert.equal(healthJson.ok, true);
  assert.equal(healthJson.model, 'Qowe 1.0');
  assert.equal(healthJson.engine, 'qowe-1.0-self-hosted');
  assert.equal(healthJson.runtime, 'netlify-functions');
  assert.equal(healthJson.hasKey, true);

  const chatRes = await netlifyApiHandler(
    new Request('https://qowe-test.netlify.app/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: [{ role: 'user', content: 'Are you Gemini?' }],
        architectureName: 'Qowe 1.0 Ultra 800B',
        totalParamsFormatted: '801.32B',
      }),
    }),
  );
  assert.equal(chatRes.status, 200);
  assert.ok((chatRes.headers.get('content-type') || '').includes('text/event-stream'));
  const sseBody = await chatRes.text();
  assert.ok(sseBody.includes('Qowe 1.0'));
  assert.ok(sseBody.includes('"done":true'));

  const createKeyRes = await netlifyApiHandler(
    new Request('https://qowe-test.netlify.app/api/keys', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: 'Netlify Test App' }),
    }),
  );
  assert.equal(createKeyRes.status, 201);
  const createdKeyJson = (await createKeyRes.json()) as { rawKey: string };
  assert.ok(createdKeyJson.rawKey.startsWith('qowe_sk_'));

  const v1ChatRes = await netlifyApiHandler(
    new Request('https://qowe-test.netlify.app/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${createdKeyJson.rawKey}`,
      },
      body: JSON.stringify({
        model: 'qowe-1.0-ultra',
        messages: [{ role: 'user', content: 'What is 15 * 4?' }],
      }),
    }),
  );
  assert.equal(v1ChatRes.status, 200);
  const v1Json = (await v1ChatRes.json()) as {
    choices: { message: { content: string } }[];
  };
  assert.ok(v1Json.choices[0].message.content.includes('60'));
});

