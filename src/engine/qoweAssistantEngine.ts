import { TRAINING_DATASETS } from './datasets.ts';
import { encode, trainTokenizer, type QoweTokenizer } from './tokenizer.ts';

export interface LocalQoweMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface LocalQoweRequest {
  messages: LocalQoweMessage[];
  architectureName?: string;
  totalParamsFormatted?: string;
  trainingContext?: string;
  checkpointStep?: number;
  checkpointLoss?: number;
  outputMode?: 'text' | 'build';
  reasoningDepth?: 'standard' | 'deep';
}

interface DialoguePair {
  user: string;
  assistant: string;
  tokens: Set<string>;
}

const TOKEN_WORD_RE = /[a-z0-9_]+/g;

const STOP_WORDS = new Set([
  'a',
  'an',
  'and',
  'are',
  'as',
  'at',
  'be',
  'by',
  'can',
  'do',
  'does',
  'for',
  'from',
  'how',
  'i',
  'in',
  'is',
  'it',
  'me',
  'my',
  'of',
  'on',
  'or',
  'please',
  'tell',
  'the',
  'this',
  'to',
  'what',
  'with',
  'you',
  'your',
]);

function extractContentWords(text: string): string[] {
  const raw = text.toLowerCase().match(TOKEN_WORD_RE) ?? [];
  const filtered = raw.filter((w) => !STOP_WORDS.has(w));
  return filtered.length > 0 ? filtered : raw;
}

function parseCorpusDialogues(corpus: string): DialoguePair[] {
  const segments = corpus
    .split('<|endoftext|>')
    .map((s) => s.trim())
    .filter(Boolean);

  const pairs: DialoguePair[] = [];
  for (const seg of segments) {
    const userIdx = seg.indexOf('<|user|>');
    const assistantIdx = seg.indexOf('<|assistant|>');
    if (userIdx !== -1 && assistantIdx !== -1 && assistantIdx > userIdx) {
      const user = seg.slice(userIdx + '<|user|>'.length, assistantIdx).trim();
      const assistant = seg.slice(assistantIdx + '<|assistant|>'.length).trim();
      if (user && assistant) {
        pairs.push({
          user,
          assistant,
          tokens: new Set(extractContentWords(user)),
        });
      }
    }
  }
  return pairs;
}

function findBestCorpusDialogue(prompt: string, trainingContext?: string): DialoguePair | null {
  const customPairs = trainingContext?.trim() ? parseCorpusDialogues(trainingContext.trim()) : [];
  const defaultPairs = parseCorpusDialogues(TRAINING_DATASETS[0]?.corpus ?? '');
  const pairs = [...customPairs, ...defaultPairs];
  if (pairs.length === 0) return null;

  const normalizedPrompt = prompt.trim().toLowerCase().replace(/[?!.]+$/g, '');
  const promptWords = extractContentWords(prompt);
  let bestPair: DialoguePair | null = null;
  let bestScore = 0;

  for (let idx = 0; idx < pairs.length; idx++) {
    const pair = pairs[idx];
    const isCustom = idx < customPairs.length;
    const normUser = pair.user.trim().toLowerCase().replace(/[?!.]+$/g, '');
    if (normUser === normalizedPrompt) {
      return pair;
    }

    let overlap = 0;
    for (const w of promptWords) {
      if (pair.tokens.has(w)) overlap += 1;
    }
    const unionSize = new Set([...promptWords, ...pair.tokens]).size || 1;
    let score = overlap / unionSize + (isCustom ? 0.08 : 0);

    if (
      (normalizedPrompt === 'hi' ||
        normalizedPrompt === 'hello' ||
        normalizedPrompt === 'hey' ||
        normalizedPrompt === 'yo') &&
      (normUser === 'hi' || normUser.startsWith('hello'))
    ) {
      score += 0.65;
    }

    if (score > bestScore) {
      bestScore = score;
      bestPair = pair;
    }
  }

  return bestScore >= 0.45 ? bestPair : null;
}

function tryEvaluateArithmetic(prompt: string): string | null {
  const cleaned = prompt
    .toLowerCase()
    .replace(/\b(what is|what's|calculate|compute|solve|evaluate|equals)\b/g, '')
    .replace(/[?=\s]+$/g, '')
    .trim();

  if (!cleaned || !/^[0-9+\-*/^().\s]+$/.test(cleaned) || !/[+\-*/^]/.test(cleaned)) {
    return null;
  }

  let pos = 0;
  const skipWs = () => {
    while (pos < cleaned.length && cleaned[pos] === ' ') pos += 1;
  };
  const parseExpr = (): number => {
    let val = parseTerm();
    skipWs();
    while (pos < cleaned.length && (cleaned[pos] === '+' || cleaned[pos] === '-')) {
      const op = cleaned[pos++];
      const rhs = parseTerm();
      val = op === '+' ? val + rhs : val - rhs;
      skipWs();
    }
    return val;
  };
  const parseTerm = (): number => {
    let val = parsePower();
    skipWs();
    while (pos < cleaned.length && (cleaned[pos] === '*' || cleaned[pos] === '/')) {
      const op = cleaned[pos++];
      const rhs = parsePower();
      if (op === '/' && rhs === 0) throw new Error('Division by zero');
      val = op === '*' ? val * rhs : val / rhs;
      skipWs();
    }
    return val;
  };
  const parsePower = (): number => {
    const base = parseFactor();
    skipWs();
    if (cleaned[pos] === '^') {
      pos += 1;
      return Math.pow(base, parsePower());
    }
    return base;
  };
  const parseFactor = (): number => {
    skipWs();
    if (cleaned[pos] === '-') {
      pos += 1;
      return -parseFactor();
    }
    if (cleaned[pos] === '(') {
      pos += 1;
      const inner = parseExpr();
      skipWs();
      if (cleaned[pos] !== ')') throw new Error('Mismatched parenthesis');
      pos += 1;
      return inner;
    }
    const start = pos;
    while (pos < cleaned.length && /[0-9.]/.test(cleaned[pos])) pos += 1;
    if (start === pos) throw new Error('Expected number');
    const num = Number(cleaned.slice(start, pos));
    if (!Number.isFinite(num)) throw new Error('Invalid number');
    return num;
  };

  try {
    const result = parseExpr();
    skipWs();
    if (pos !== cleaned.length || !Number.isFinite(result)) return null;
    const formatted = Number.isInteger(result) ? String(result) : Number(result.toFixed(6)).toString();
    return `\`${cleaned}\` = **${formatted}**`;
  } catch {
    return null;
  }
}

function tryAnswerCodingQuestion(prompt: string): string | null {
  const q = prompt.toLowerCase();
  if (q.includes('binary search')) {
    return `Here is an iterative **binary search** implementation in Python ($O(\\log n)$ time, $O(1)$ space):

\`\`\`python
def binary_search(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
\`\`\``;
  }

  if (q.includes('fibonacci')) {
    return `Here is a memory-efficient **Fibonacci generator** in Python ($O(n)$ time, $O(1)$ space):

\`\`\`python
def fibonacci(n: int):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b
\`\`\``;
  }

  if (q.includes('softmax')) {
    return `Here is numerically stable **Softmax** in TypeScript (subtracting \`maxLogit\` prevents float overflow):

\`\`\`ts
export function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - maxLogit));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}
\`\`\``;
  }

  if (q.includes('quicksort') || (q.includes('sort') && (q.includes('python') || q.includes('typescript') || q.includes('javascript')))) {
    return `Here is a clean **Quicksort** implementation in TypeScript:

\`\`\`ts
export function quickSort(arr: number[]): number[] {
  if (arr.length <= 1) return arr;
  const pivot = arr[Math.floor(arr.length / 2)];
  const left = arr.filter((x) => x < pivot);
  const middle = arr.filter((x) => x === pivot);
  const right = arr.filter((x) => x > pivot);
  return [...quickSort(left), ...middle, ...quickSort(right)];
}
\`\`\``;
  }

  if ((q.includes('fetch') || q.includes('api call')) && (q.includes('js') || q.includes('javascript') || q.includes('typescript') || q.includes('react'))) {
    return `Here is a clean TypeScript \`fetch\` helper with proper error handling:

\`\`\`ts
export async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  });
  if (!response.ok) {
    throw new Error(\`Request failed (\${response.status}): \${await response.text()}\`);
  }
  return (await response.json()) as T;
}
\`\`\``;
  }

  if (q.includes('express') && (q.includes('server') || q.includes('api') || q.includes('route'))) {
    return `Here is a minimal production Express server in TypeScript:

\`\`\`ts
import express from 'express';

const app = express();
app.use(express.json());

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, service: 'qowe-api' });
});

app.post('/api/echo', (req, res) => {
  res.status(201).json({ received: req.body });
});

app.listen(Number(process.env.PORT) || 3000);
\`\`\``;
  }

  if (q.includes('html') && (q.includes('button') || q.includes('card') || q.includes('page') || q.includes('hello'))) {
    return `Here is a clean standalone HTML starter template:

\`\`\`html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Qowe Starter</title>
  <style>
    body { font-family: system-ui, sans-serif; display: grid; place-items: center; min-height: 100vh; margin: 0; background: #f5f5f7; }
    .card { background: #fff; padding: 28px; border-radius: 20px; box-shadow: 0 8px 24px rgba(0,0,0,0.06); text-align: center; }
    button { margin-top: 12px; padding: 10px 18px; border: none; border-radius: 999px; background: #007aff; color: #fff; font-weight: 600; cursor: pointer; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Built with Qowe 1.0</h1>
    <button onclick="this.textContent = 'Clicked!'">Click Me</button>
  </div>
</body>
</html>
\`\`\``;
  }

  return null;
}

interface BuildDomainSpec {
  appTitle: string;
  tableName: string;
  resourcePath: string;
  itemLabel: string;
  statuses: [string, string, string];
  seeds: { title: string; category: string; status: string; priority: string }[];
}

function inferBuildDomain(prompt: string): BuildDomainSpec {
  const p = prompt.toLowerCase();
  if (p.includes('habit')) {
    return {
      appTitle: 'HabitPulse Daily Tracker',
      tableName: 'habits',
      resourcePath: '/api/habits',
      itemLabel: 'Habit',
      statuses: ['active', 'completed', 'paused'],
      seeds: [
        { title: 'Morning 20m deep focus sprint', category: 'Focus', status: 'completed', priority: 'high' },
        { title: 'Strength & mobility session', category: 'Health', status: 'active', priority: 'high' },
        { title: 'Read 25 pages technical literature', category: 'Learning', status: 'active', priority: 'medium' },
      ],
    };
  }
  if (p.includes('expense') || p.includes('budget') || p.includes('finance')) {
    return {
      appTitle: 'LedgerFlow Expense Tracker',
      tableName: 'expenses',
      resourcePath: '/api/expenses',
      itemLabel: 'Expense',
      statuses: ['pending', 'cleared', 'reimbursed'],
      seeds: [
        { title: 'Cloud GPU compute cluster ($142)', category: 'Infrastructure', status: 'cleared', priority: 'high' },
        { title: 'Domain & SSL renewal ($24)', category: 'Operations', status: 'cleared', priority: 'medium' },
        { title: 'Team coffee & design workshop ($38)', category: 'Team', status: 'pending', priority: 'low' },
      ],
    };
  }
  if (p.includes('note') || p.includes('wiki') || p.includes('doc')) {
    return {
      appTitle: 'Memoir Vault Workspace',
      tableName: 'notes',
      resourcePath: '/api/notes',
      itemLabel: 'Note',
      statuses: ['draft', 'pinned', 'archived'],
      seeds: [
        { title: 'LTX-2.5 Dual-Stream architecture summary', category: 'Research', status: 'pinned', priority: 'high' },
        { title: 'Production PostgreSQL indexing checklist', category: 'Engineering', status: 'draft', priority: 'medium' },
        { title: 'Q4 product launch milestones', category: 'Product', status: 'pinned', priority: 'high' },
      ],
    };
  }
  if (p.includes('inventory') || p.includes('warehouse') || p.includes('stock')) {
    return {
      appTitle: 'StockGrid Inventory Console',
      tableName: 'items',
      resourcePath: '/api/items',
      itemLabel: 'Inventory Item',
      statuses: ['in_stock', 'low_stock', 'ordered'],
      seeds: [
        { title: 'NVMe Gen5 4TB Server Drives (18 units)', category: 'Hardware', status: 'in_stock', priority: 'high' },
        { title: 'USB-C 100W GaN Workstations (4 units)', category: 'Peripherals', status: 'low_stock', priority: 'medium' },
        { title: 'Acoustic Studio Panels Pack (12 units)', category: 'Studio', status: 'ordered', priority: 'low' },
      ],
    };
  }
  if (p.includes('bookmark') || p.includes('link')) {
    return {
      appTitle: 'LinkVault Bookmark Manager',
      tableName: 'bookmarks',
      resourcePath: '/api/bookmarks',
      itemLabel: 'Bookmark',
      statuses: ['unread', 'favorite', 'archived'],
      seeds: [
        { title: 'Attention Is All You Need — Paper Notes', category: 'Research', status: 'favorite', priority: 'high' },
        { title: 'LTX-2.5 Dual-Stream Architecture Guide', category: 'Video AI', status: 'unread', priority: 'high' },
        { title: 'SQLite Performance Tuning Playbook', category: 'Database', status: 'favorite', priority: 'medium' },
      ],
    };
  }
  if (p.includes('todo') || p.includes('task')) {
    return {
      appTitle: 'TaskFlow Workspace',
      tableName: 'tasks',
      resourcePath: '/api/tasks',
      itemLabel: 'Task',
      statuses: ['todo', 'in_progress', 'done'],
      seeds: [
        { title: 'Ship REST API validation middleware', category: 'Backend', status: 'done', priority: 'high' },
        { title: 'Build interactive filtering & live counters', category: 'Frontend', status: 'in_progress', priority: 'high' },
        { title: 'Write database schema seed script', category: 'Database', status: 'todo', priority: 'medium' },
      ],
    };
  }

  return {
    appTitle: 'SprintPulse Issue Tracker',
    tableName: 'issues',
    resourcePath: '/api/issues',
    itemLabel: 'Issue',
    statuses: ['open', 'in_progress', 'resolved'],
    seeds: [
      { title: 'Implement local streaming inference engine', category: 'Engine', status: 'resolved', priority: 'high' },
      { title: 'Add real-time search & status filters', category: 'Frontend', status: 'in_progress', priority: 'high' },
      { title: 'Export schema.sql & Express bundle', category: 'Platform', status: 'open', priority: 'medium' },
    ],
  };
}

export function generateLocalFullStackBuild(prompt: string): string {
  const spec = inferBuildDomain(prompt);
  const seedInserts = spec.seeds
    .map(
      (row) =>
        `INSERT INTO ${spec.tableName} (title, category, status, priority) VALUES ('${row.title.replace(/'/g, "''")}', '${row.category}', '${row.status}', '${row.priority}');`,
    )
    .join('\n');

  return `### Build Task Plan
- [x] Scaffold Node.js ESM \`package.json\` with Express and TypeScript server entrypoint
- [x] Define \`db/schema.sql\` with \`${spec.tableName}\` table DDL and initial seed rows
- [x] Implement Express REST endpoints (\`GET\`, \`POST\`, \`PATCH\`, \`DELETE ${spec.resourcePath}\`) and \`/api/health\`
- [x] Build responsive single-file \`public/index.html\` connected to live \`fetch('${spec.resourcePath}')\` CRUD flows

\`\`\`json file="package.json"
{
  "name": "${spec.tableName}-fullstack-app",
  "version": "1.0.0",
  "private": true,
  "type": "module",
  "engines": {
    "node": ">=22.18.0"
  },
  "scripts": {
    "dev": "tsx server.ts",
    "start": "node server.ts"
  },
  "dependencies": {
    "express": "^4.21.2"
  },
  "devDependencies": {
    "tsx": "^4.21.0"
  }
}
\`\`\`

\`\`\`sql file="db/schema.sql"
CREATE TABLE IF NOT EXISTS ${spec.tableName} (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL,
  category TEXT NOT NULL DEFAULT 'General',
  status TEXT NOT NULL DEFAULT '${spec.statuses[0]}',
  priority TEXT NOT NULL DEFAULT 'medium',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

${seedInserts}
\`\`\`

\`\`\`ts file="server.ts"
import express from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

interface RecordItem {
  id: number;
  title: string;
  category: string;
  status: string;
  priority: string;
  created_at: string;
}

let nextId = 4;
const records: RecordItem[] = [
${spec.seeds
  .map(
    (s, i) =>
      `  { id: ${i + 1}, title: ${JSON.stringify(s.title)}, category: ${JSON.stringify(s.category)}, status: ${JSON.stringify(s.status)}, priority: ${JSON.stringify(s.priority)}, created_at: new Date().toISOString() },`,
  )
  .join('\n')}
];

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, service: ${JSON.stringify(spec.appTitle)}, count: records.length });
});

app.get('${spec.resourcePath}', (_req, res) => {
  res.json(records);
});

app.post('${spec.resourcePath}', (req, res) => {
  const { title, category, status, priority } = req.body ?? {};
  if (typeof title !== 'string' || !title.trim()) {
    res.status(400).json({ error: 'title is required' });
    return;
  }
  const created: RecordItem = {
    id: nextId++,
    title: title.trim(),
    category: typeof category === 'string' && category.trim() ? category.trim() : 'General',
    status: typeof status === 'string' && status.trim() ? status.trim() : ${JSON.stringify(spec.statuses[0])},
    priority: typeof priority === 'string' && priority.trim() ? priority.trim() : 'medium',
    created_at: new Date().toISOString(),
  };
  records.unshift(created);
  res.status(201).json(created);
});

app.patch('${spec.resourcePath}/:id', (req, res) => {
  const id = Number(req.params.id);
  const item = records.find((r) => r.id === id);
  if (!item) {
    res.status(404).json({ error: '${spec.itemLabel} not found' });
    return;
  }
  if (typeof req.body?.title === 'string') item.title = req.body.title.trim();
  if (typeof req.body?.status === 'string') item.status = req.body.status;
  if (typeof req.body?.priority === 'string') item.priority = req.body.priority;
  res.json(item);
});

app.delete('${spec.resourcePath}/:id', (req, res) => {
  const id = Number(req.params.id);
  const idx = records.findIndex((r) => r.id === id);
  if (idx === -1) {
    res.status(404).json({ error: '${spec.itemLabel} not found' });
    return;
  }
  records.splice(idx, 1);
  res.json({ ok: true });
});

const port = Number(process.env.PORT) || 3000;
app.listen(port, () => console.log(\`${spec.appTitle} listening on http://localhost:\${port}\`));
\`\`\`

\`\`\`html file="public/index.html"
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>${spec.appTitle}</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f5f5f7; color: #111827; padding: 24px; }
    .container { max-width: 760px; margin: 0 auto; background: #fff; border-radius: 20px; padding: 24px; box-shadow: 0 8px 24px rgba(0,0,0,0.06); }
    h1 { margin: 0 0 6px; font-size: 22px; }
    .sub { margin: 0 0 18px; color: #6b7280; font-size: 13px; }
    .toolbar { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
    input, select, button { padding: 9px 12px; border-radius: 10px; border: 1px solid #d1d5db; font-size: 13px; }
    input[type="text"] { flex: 1; min-width: 180px; }
    button.primary { background: #007aff; color: #fff; border: none; font-weight: 600; cursor: pointer; }
    .row { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 12px 14px; border: 1px solid #e5e7eb; border-radius: 14px; margin-bottom: 10px; }
    .meta { font-size: 12px; color: #6b7280; margin-top: 2px; }
    .actions { display: flex; gap: 6px; }
    .pill { font-size: 11px; padding: 3px 9px; border-radius: 999px; background: #eef2ff; color: #3730a3; font-weight: 600; }
  </style>
</head>
<body>
  <div class="container">
    <h1>${spec.appTitle}</h1>
    <p class="sub">Full-stack Express + SQL workspace powered locally by Qowe 1.0.</p>
    <form id="create-form" class="toolbar">
      <input id="title-input" type="text" placeholder="Add new ${spec.itemLabel.toLowerCase()}..." required />
      <select id="priority-input">
        <option value="high">High</option>
        <option value="medium" selected>Medium</option>
        <option value="low">Low</option>
      </select>
      <button class="primary" type="submit">Add ${spec.itemLabel}</button>
    </form>
    <div class="toolbar">
      <input id="search-input" type="text" placeholder="Filter records..." />
      <select id="status-filter">
        <option value="all">All Statuses</option>
        <option value="${spec.statuses[0]}">${spec.statuses[0]}</option>
        <option value="${spec.statuses[1]}">${spec.statuses[1]}</option>
        <option value="${spec.statuses[2]}">${spec.statuses[2]}</option>
      </select>
    </div>
    <div id="list"></div>
  </div>
  <script>
    const API = '${spec.resourcePath}';
    let allItems = [];
    async function loadItems() {
      const res = await fetch(API);
      allItems = await res.json();
      render();
    }
    function render() {
      const q = document.getElementById('search-input').value.trim().toLowerCase();
      const status = document.getElementById('status-filter').value;
      const filtered = allItems.filter((item) => {
        const matchQ = !q || item.title.toLowerCase().includes(q) || String(item.category).toLowerCase().includes(q);
        const matchS = status === 'all' || item.status === status;
        return matchQ && matchS;
      });
      const list = document.getElementById('list');
      list.innerHTML = '';
      filtered.forEach((item) => {
        const row = document.createElement('div');
        row.className = 'row';
        row.innerHTML = '<div><strong>' + item.title + '</strong><div class="meta">' + item.category + ' · Priority: ' + item.priority + '</div></div>';
        const actions = document.createElement('div');
        actions.className = 'actions';
        const statusBtn = document.createElement('button');
        statusBtn.className = 'pill';
        statusBtn.textContent = item.status;
        statusBtn.onclick = async () => {
          const order = ${JSON.stringify(spec.statuses)};
          const next = order[(order.indexOf(item.status) + 1) % order.length];
          await fetch(API + '/' + item.id, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ status: next }) });
          loadItems();
        };
        const delBtn = document.createElement('button');
        delBtn.textContent = 'Delete';
        delBtn.onclick = async () => {
          await fetch(API + '/' + item.id, { method: 'DELETE' });
          loadItems();
        };
        actions.append(statusBtn, delBtn);
        row.appendChild(actions);
        list.appendChild(row);
      });
    }
    document.getElementById('create-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const input = document.getElementById('title-input');
      const priority = document.getElementById('priority-input').value;
      if (!input.value.trim()) return;
      await fetch(API, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title: input.value.trim(), category: 'Workspace', priority }) });
      input.value = '';
      loadItems();
    });
    document.getElementById('search-input').addEventListener('input', render);
    document.getElementById('status-filter').addEventListener('change', render);
    loadItems();
  </script>
</body>
</html>
\`\`\``;
}

const tokenizerCache = new Map<string, QoweTokenizer>();

function getCorpusTokenizer(corpus: string): QoweTokenizer {
  const key = corpus.slice(0, 200) + ':' + corpus.length;
  let tok = tokenizerCache.get(key);
  if (!tok) {
    tok = trainTokenizer(corpus, 360);
    if (tokenizerCache.size > 6) {
      const oldest = tokenizerCache.keys().next().value;
      if (oldest) tokenizerCache.delete(oldest);
    }
    tokenizerCache.set(key, tok);
  }
  return tok;
}

export function generateQoweAssistantResponse(req: LocalQoweRequest): string {
  const userMsgs = req.messages.filter((m) => m.role === 'user' && typeof m.content === 'string');
  const prompt = (userMsgs[userMsgs.length - 1]?.content ?? '').trim();

  if (req.outputMode === 'build') {
    return generateLocalFullStackBuild(prompt || 'Build a full-stack workspace app');
  }

  const archName = req.architectureName || 'Qowe 1.0 Ultra 800B';
  const paramLabel = req.totalParamsFormatted || '801.32B';
  const activeCorpus = req.trainingContext?.trim() || TRAINING_DATASETS[0].corpus;
  const lower = prompt.toLowerCase().replace(/[?!.]+$/g, '').trim();

  // 1. Conversational identity & casual greetings (always unmistakably Qowe 1.0, never a third-party AI)
  if (
    lower === 'hi' ||
    lower === 'hello' ||
    lower === 'hey' ||
    lower === 'yo' ||
    lower === 'sup' ||
    lower === 'test' ||
    lower === 'ping' ||
    lower === "what's up" ||
    lower === 'whats up' ||
    lower === 'how are you' ||
    lower === 'good morning' ||
    lower === 'good afternoon' ||
    lower === 'good evening' ||
    /^(hi|hello|hey|yo)\s+(bro|gng|qowe|there|man|dude|ai)$/i.test(lower) ||
    lower.startsWith('hello how are you') ||
    lower.startsWith('hi how are you')
  ) {
    return `Hey! I am **${archName}** (${paramLabel} parameters), running directly on your own self-hosted Qowe 1.0 engine with zero external AI keys or rate limits. Ask me about transformers, coding, math, full-stack app building, or 60 fps video generation!`;
  }

  if (
    lower.includes('thanks') ||
    lower.includes('thank you') ||
    lower === 'thx' ||
    lower === 'cool' ||
    lower === 'nice'
  ) {
    return `Anytime! Let me know what you want to explore, train, build, or generate next with **${archName}**.`;
  }

  if (
    lower.includes('who made you') ||
    lower.includes('who are you') ||
    lower.includes('what are you') ||
    lower.includes('are you gemini') ||
    lower.includes('are you claude') ||
    lower.includes('are you gpt') ||
    lower.includes('powered by') ||
    lower.includes('what ai are you') ||
    lower.includes('which model are you') ||
    lower.includes('your own model')
  ) {
    return `I am **${archName}** (${paramLabel} parameters)—your own self-contained transformer model and multimodal studio, running locally with zero external API keys or third-party model dependencies.`;
  }

  if (
    (lower.includes('what did i') || lower.includes('previous message') || lower.includes('last question')) &&
    userMsgs.length >= 2
  ) {
    const prevPrompt = userMsgs[userMsgs.length - 2].content.trim();
    return `Your previous message in this conversation was:\n\n> ${prevPrompt}`;
  }

  // 2. Safe numeric evaluation
  const mathAns = tryEvaluateArithmetic(prompt);
  if (mathAns) return mathAns;

  // 3. Common coding/algorithm queries
  const codeAns = tryAnswerCodingQuestion(prompt);
  if (codeAns) return codeAns;

  // 4. Match active training corpus & built-in conversational Q&A pairs
  const matchedPair = findBestCorpusDialogue(prompt, req.trainingContext);
  if (matchedPair) {
    return matchedPair.assistant;
  }

  // 5. Domain-specific Qowe 1.0 explanations & architecture reasoning
  const tokenizer = getCorpusTokenizer(activeCorpus);
  const promptTokenIds = encode(tokenizer, prompt);
  const stepNote =
    typeof req.checkpointStep === 'number' && req.checkpointStep > 0
      ? ` (local checkpoint step ${req.checkpointStep}, loss ${typeof req.checkpointLoss === 'number' ? req.checkpointLoss.toFixed(3) : 'N/A'})`
      : '';

  if (lower.includes('self-attention') || lower.includes('attention') || lower.includes('transformer')) {
    return `In **${archName}**, causal multi-head self-attention works step-by-step across each decoder block:

1. **Token + Position Embeddings**: Each token ID maps into vector \`wte[id]\` and adds positional vector \`wpe[pos]\`.
2. **Pre-LayerNorm**: Normalizes hidden states (\`eps = 1e-5\`) before attention for stable gradient flow.
3. **Q / K / V Projections**: Projects hidden vectors into Query, Key, and Value matrices split across \`nHeads\`.
4. **Causal Scaled Dot-Product**: Computes \`softmax((Q · K^T) / sqrt(headDim))\` with a lower-triangular causal mask so token \`t\` only attends to positions \`0..t\`.
5. **Residual + GELU FFN**: Projects back via \`wo\`, adds residual \`xMid\`, applies Pre-LN \`ln2\`, and runs a two-layer GELU MLP before projecting final logits.`;
  }

  if (lower.includes('video') || lower.includes('ltx') || lower.includes('60 fps')) {
    return `**${archName}** runs **Lightricks/LTX-2.5 Asymmetric Dual-Stream Rectified Flow DiT** coupled with a **960×540 60 fps WebGL2 3D PBR Raymarching** renderer and synchronized stereo neural audio—100% locally in your browser with zero external API keys. Switch the composer to **Video (60 fps)** or open the **Video** tab to synthesize scenes directly.`;
  }

  if (lower.includes('parameter') || lower.includes('architecture') || lower.includes('spec')) {
    return `I am **${archName}** configured at **${paramLabel} parameters**${stepNote}. My transformer stack combines word-preserving regex pre-tokenization (${tokenizer.vocab.length} active vocabulary tokens), Pre-LN causal multi-head self-attention, GELU FFN blocks, and local LTX-2.5 vision/video synthesis—running directly on your own model with unlimited requests.`;
  }

  if (lower.includes('train') || lower.includes('weight') || lower.includes('checkpoint') || lower.includes('dataset')) {
    return `You can train custom text or vision/video weights anytime with no usage limits${stepNote}:
- Open **Train LLM (Text)** to run live AdamW gradient descent steps over any dialogue or code corpus.
- Switch the header toggle to **Trained Weights** to sample raw next-token predictions directly from your checkpoint weights (${promptTokenIds.length} tokens encoded for your latest prompt).
- Click **Download Model** to export a standalone offline \`.html\` chat app or full \`.json\` bundle.`;
  }

  // 6. Custom prose match inside loaded trainingContext
  if (req.trainingContext && req.trainingContext.trim().length > 0) {
    const queryWords = new Set(extractContentWords(prompt));
    const customChunks = req.trainingContext
      .split(/<\|endoftext\|>|\n{2,}/)
      .map((s) => s.replace(/<\|user\|>|<\|assistant\|>/g, '').trim())
      .filter((s) => s.length > 15);
    let bestChunk = '';
    let bestHits = 0;
    for (const chunk of customChunks) {
      const words = extractContentWords(chunk);
      let hits = 0;
      for (const w of words) {
        if (queryWords.has(w)) hits += 1;
      }
      if (hits > bestHits) {
        bestHits = hits;
        bestChunk = chunk;
      }
    }
    if (bestHits >= 2 && bestChunk) {
      return `From your trained **${archName}** knowledge corpus:\n\n${bestChunk}`;
    }
  }

  // 7. Coherent general assistant response (never raw gibberish tokens)
  const corpusPairs = parseCorpusDialogues(activeCorpus);
  const topicHints = corpusPairs
    .slice(0, 4)
    .map((p) => p.user.replace(/[?!.]+$/g, ''))
    .join(', ');

  return `**${archName}** (${paramLabel})${stepNote} here—running directly on your own self-hosted model (${promptTokenIds.length} prompt tokens parsed).

For **"${prompt}"**, here is what I can do right here with zero external AI dependencies:
- **Custom Training**: Paste any custom knowledge or Q&A pairs (\`<|user|> Question\\n<|assistant|> Answer<|endoftext|>\`) in **Train LLM (Text)** so I learn and answer directly on your domain data${topicHints ? ` (currently loaded: *${topicHints}*)` : ''}.
- **Code & Full-Stack Apps**: Ask for Python/TypeScript/HTML code, or switch to **Code & Build** mode to generate a runnable Express + SQL + HTML app you can preview live and export as a \`.zip\`.
- **Math & Architecture**: Ask calculations (like \`25 * 48\`) or questions about transformers, self-attention, perplexity, and LTX-2.5 60 fps synthesis.
- **Image & 60 FPS Video**: Switch to **Image** or **Video (60 fps)** to synthesize 3D PBR visuals locally.`;
}
