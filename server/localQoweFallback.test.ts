import test from 'node:test';
import assert from 'node:assert/strict';
import { executeVirtualApiRequest, parseFullStackFiles } from '../src/engine/fullStackProject';
import { generateLocalFullStackBuild, generateLocalQoweResponse } from './localQoweFallback';

test('generateLocalQoweResponse answers identity, greeting, custom corpus, arithmetic, and code prompts cleanly', () => {
  const identity = generateLocalQoweResponse({
    messages: [{ role: 'user', content: 'Who are you?' }],
    architectureName: 'Qowe 1.0 Ultra 800B',
    totalParamsFormatted: '801.32B',
  });
  assert.ok(identity.includes('Qowe 1.0'));
  assert.equal(identity.toLowerCase().includes('high demand'), false);

  const greeting = generateLocalQoweResponse({
    messages: [{ role: 'user', content: 'Hi' }],
  });
  assert.ok(greeting.includes('Hello') || greeting.includes('Qowe 1.0'));

  const customCorpusReply = generateLocalQoweResponse({
    messages: [{ role: 'user', content: 'What is project Atlas?' }],
    trainingContext:
      '<|user|> What is project Atlas?\n<|assistant|> Project Atlas is a local neural mesh engine built by our team.<|endoftext|>',
  });
  assert.ok(customCorpusReply.includes('Project Atlas is a local neural mesh engine'));

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
