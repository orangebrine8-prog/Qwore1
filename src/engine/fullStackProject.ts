export interface FullStackProjectFile {
  path: string;
  language: string;
  content: string;
}

export interface DetectedApiRoute {
  method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  path: string;
}

export interface ParsedFullStackProject {
  id: string;
  title: string;
  files: FullStackProjectFile[];
  entryHtmlFile: FullStackProjectFile | null;
  serverFile: FullStackProjectFile | null;
  schemaFile: FullStackProjectFile | null;
  packageFile: FullStackProjectFile | null;
  routes: DetectedApiRoute[];
  initialTables: Record<string, Record<string, unknown>[]>;
  expectsArrayCollections: boolean;
  isFullStack: boolean;
  strippedRanges: { start: number; end: number }[];
}

export interface VirtualApiRequest {
  method: string;
  url: string;
  bodyText?: string;
}

export interface VirtualApiResponse {
  status: number;
  body: unknown;
}

function hashString(str: string): string {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0).toString(36);
}

function normalizeFilePath(rawPath: string): string {
  return rawPath
    .trim()
    .replace(/^["'`]+|["'`]+$/g, '')
    .replace(/^\.\//, '')
    .replace(/^\/+/, '');
}

function extractFileMeta(
  infoString: string,
  body: string,
  usedPaths: Set<string>,
): { path: string | null; language: string; cleanedBody: string } {
  const tokens = infoString.trim();
  const langMatch = tokens.match(/^([a-zA-Z0-9_+-]+)/);
  let language = (langMatch?.[1] ?? 'text').toLowerCase();

  // Check explicit file="..." / path="..." / filename="..." or lang:path
  let explicitPath: string | null = null;
  const quotedFile = tokens.match(/(?:file|path|filename)\s*=\s*["']([^"']+)["']/i);
  const bareFile = tokens.match(/(?:file|path|filename)\s*=\s*([^\s]+)/i);
  const colonFile = tokens.match(/^[a-zA-Z0-9_+-]+:([^\s]+)/);

  if (quotedFile?.[1]) {
    explicitPath = normalizeFilePath(quotedFile[1]);
  } else if (bareFile?.[1]) {
    explicitPath = normalizeFilePath(bareFile[1]);
  } else if (colonFile?.[1]) {
    explicitPath = normalizeFilePath(colonFile[1]);
  }

  // Fallback: check first line comment inside block for file path
  let cleanedBody = body.replace(/\n$/, '');
  if (!explicitPath) {
    const firstLineMatch = cleanedBody.match(
      /^(?:\/\/\s*(?:file|path)\s*:\s*([^\n]+)|<!--\s*(?:file|path)\s*:\s*([^>]+?)\s*-->|--\s*(?:file|path)\s*:\s*([^\n]+)|#\s*(?:file|path)\s*:\s*([^\n]+))\n/i,
    );
    const candidate =
      firstLineMatch?.[1] ?? firstLineMatch?.[2] ?? firstLineMatch?.[3] ?? firstLineMatch?.[4];
    if (firstLineMatch && candidate) {
      explicitPath = normalizeFilePath(candidate);
      cleanedBody = cleanedBody.slice(firstLineMatch[0].length);
    }
  }

  if (!explicitPath) {
    // Infer sensible path based on language & content when multi-file context uses untagged fences
    if (language === 'html' || /<!doctype html/i.test(cleanedBody)) {
      language = 'html';
      explicitPath = usedPaths.has('public/index.html') ? 'index.html' : 'public/index.html';
    } else if (language === 'sql' || /\bcreate\s+table\b/i.test(cleanedBody)) {
      language = 'sql';
      explicitPath = 'db/schema.sql';
    } else if (
      (language === 'ts' || language === 'typescript' || language === 'js' || language === 'javascript') &&
      (/\bexpress\b/i.test(cleanedBody) || /\bapp\.(get|post|put|patch|delete|listen)\s*\(/i.test(cleanedBody))
    ) {
      explicitPath = language.startsWith('js') ? 'server.js' : 'server.ts';
    } else if (language === 'json' && /"dependencies"|"scripts"/.test(cleanedBody)) {
      explicitPath = 'package.json';
    } else if (/GEMINI_API_KEY|PORT=|DATABASE_URL=/m.test(cleanedBody)) {
      explicitPath = '.env.example';
    }
  }

  return { path: explicitPath, language, cleanedBody };
}

function splitTopLevelSqlTuple(tupleStr: string): string[] {
  const values: string[] = [];
  let current = '';
  let inQuote: string | null = null;
  let parenDepth = 0;

  for (let i = 0; i < tupleStr.length; i++) {
    const ch = tupleStr[i];
    if (inQuote) {
      if (ch === inQuote && tupleStr[i + 1] === inQuote) {
        current += ch;
        i++;
      } else if (ch === inQuote && tupleStr[i - 1] !== '\\') {
        inQuote = null;
        current += ch;
      } else {
        current += ch;
      }
      continue;
    }

    if (ch === "'" || ch === '"') {
      inQuote = ch;
      current += ch;
    } else if (ch === '(') {
      parenDepth++;
      current += ch;
    } else if (ch === ')') {
      parenDepth = Math.max(0, parenDepth - 1);
      current += ch;
    } else if (ch === ',' && parenDepth === 0) {
      values.push(current.trim());
      current = '';
    } else {
      current += ch;
    }
  }
  if (current.trim().length > 0) {
    values.push(current.trim());
  }
  return values;
}

function parseSqlLiteral(raw: string): unknown {
  const t = raw.trim();
  if (/^null$/i.test(t)) return null;
  if (/^true$/i.test(t)) return true;
  if (/^false$/i.test(t)) return false;
  if (/^-?\d+(?:\.\d+)?$/.test(t)) return Number(t);
  if (
    (t.startsWith("'") && t.endsWith("'")) ||
    (t.startsWith('"') && t.endsWith('"'))
  ) {
    return t.slice(1, -1).replace(/''/g, "'").replace(/\\"/g, '"');
  }
  if (/^CURRENT_TIMESTAMP$/i.test(t) || /^datetime\(/i.test(t)) {
    return new Date().toISOString();
  }
  return t;
}

function extractSqlSchemaAndSeeds(sqlText: string): {
  columnsByTable: Record<string, string[]>;
  tables: Record<string, Record<string, unknown>[]>;
} {
  const columnsByTable: Record<string, string[]> = {};
  const tables: Record<string, Record<string, unknown>[]> = {};

  // Parse CREATE TABLE statements
  const createRe = /CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`"']?([a-zA-Z_][a-zA-Z0-9_]*)[`"']?\s*\(([\s\S]*?)\)\s*;/gi;
  let cMatch: RegExpExecArray | null;
  while ((cMatch = createRe.exec(sqlText)) !== null) {
    const tableName = cMatch[1].toLowerCase();
    const colsRaw = splitTopLevelSqlTuple(cMatch[2]);
    const cols: string[] = [];
    for (const def of colsRaw) {
      const trimmed = def.trim();
      if (!trimmed || /^(PRIMARY\s+KEY|FOREIGN\s+KEY|UNIQUE|CONSTRAINT|CHECK|INDEX)/i.test(trimmed)) {
        continue;
      }
      const colMatch = trimmed.match(/^[`"']?([a-zA-Z_][a-zA-Z0-9_]*)[`"']?\s+/);
      if (colMatch?.[1]) {
        cols.push(colMatch[1]);
      }
    }
    columnsByTable[tableName] = cols;
    if (!tables[tableName]) tables[tableName] = [];
  }

  // Parse INSERT INTO ... VALUES ... statements
  const insertRe =
    /INSERT\s+(?:OR\s+IGNORE\s+)?INTO\s+[`"']?([a-zA-Z_][a-zA-Z0-9_]*)[`"']?\s*(?:\(([^)]+)\))?\s*VALUES\s*([\s\S]*?)\s*;/gi;
  let iMatch: RegExpExecArray | null;
  while ((iMatch = insertRe.exec(sqlText)) !== null) {
    const tableName = iMatch[1].toLowerCase();
    const explicitCols = iMatch[2]
      ? iMatch[2].split(',').map((c) => c.trim().replace(/[`"']/g, ''))
      : columnsByTable[tableName] ?? [];
    if (!tables[tableName]) tables[tableName] = [];

    const valuesSection = iMatch[3];
    // Extract each (...) tuple
    let depth = 0;
    let inQ: string | null = null;
    let startIdx = -1;
    for (let i = 0; i < valuesSection.length; i++) {
      const ch = valuesSection[i];
      if (inQ) {
        if (ch === inQ && valuesSection[i - 1] !== '\\') {
          if (valuesSection[i + 1] === inQ) {
            i++;
          } else {
            inQ = null;
          }
        }
        continue;
      }
      if (ch === "'" || ch === '"') {
        inQ = ch;
      } else if (ch === '(') {
        if (depth === 0) startIdx = i + 1;
        depth++;
      } else if (ch === ')') {
        depth--;
        if (depth === 0 && startIdx >= 0) {
          const tupleRaw = valuesSection.slice(startIdx, i);
          const parsedVals = splitTopLevelSqlTuple(tupleRaw).map(parseSqlLiteral);
          const row: Record<string, unknown> = {};
          const nextId = tables[tableName].length + 1;
          parsedVals.forEach((val, idx) => {
            const colName = explicitCols[idx] || `col_${idx + 1}`;
            row[colName] = val;
          });
          if (row.id === undefined) {
            row.id = nextId;
          }
          tables[tableName].push(row);
          startIdx = -1;
        }
      }
    }
  }

  return { columnsByTable, tables };
}

function extractRoutesFromServer(serverCode: string): DetectedApiRoute[] {
  const routes: DetectedApiRoute[] = [];
  const seen = new Set<string>();
  const routeRe = /\b(?:app|router)\.(get|post|put|patch|delete)\s*\(\s*['"`]([^'"`]+)['"`]/gi;
  let m: RegExpExecArray | null;
  while ((m = routeRe.exec(serverCode)) !== null) {
    const method = m[1].toUpperCase() as DetectedApiRoute['method'];
    const routePath = m[2];
    const key = `${method}:${routePath}`;
    if (!seen.has(key)) {
      seen.add(key);
      routes.push({ method, path: routePath });
    }
  }
  return routes;
}

function extractSeedArraysFromServer(serverCode: string): Record<string, Record<string, unknown>[]> {
  const tables: Record<string, Record<string, unknown>[]> = {};
  const arrRe = /(?:const|let|var)\s+(?:initial|seed|default)?([A-Za-z0-9_]+)\s*(?::[^=]+)?=\s*(\[\s*\{[\s\S]*?\}\s*\])/g;
  let m: RegExpExecArray | null;
  while ((m = arrRe.exec(serverCode)) !== null) {
    const name = m[1].toLowerCase().replace(/^seed|^initial|^default/, '') || 'items';
    try {
      // Safe JSON-like normalization for literal arrays of objects
      const jsonCandidate = m[2]
        .replace(/,\s*([}\]])/g, '$1')
        .replace(/([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:/g, '$1"$2":')
        .replace(/'([^'\\]*(\\.[^'\\]*)*)'/g, '"$1"');
      const parsed = JSON.parse(jsonCandidate) as unknown;
      if (Array.isArray(parsed) && parsed.length > 0 && typeof parsed[0] === 'object') {
        tables[name] = parsed as Record<string, unknown>[];
      }
    } catch {
      // Skip non-JSON expressions cleanly
    }
  }
  return tables;
}

export function parseFullStackFiles(markdown: string): ParsedFullStackProject | null {
  const fenceRe = /```([^\n]*)\n([\s\S]*?)```/g;
  const rawBlocks: { start: number; end: number; info: string; body: string }[] = [];
  let match: RegExpExecArray | null;
  while ((match = fenceRe.exec(markdown)) !== null) {
    rawBlocks.push({
      start: match.index,
      end: fenceRe.lastIndex,
      info: match[1] ?? '',
      body: match[2] ?? '',
    });
  }

  if (rawBlocks.length === 0) return null;

  const files: FullStackProjectFile[] = [];
  const usedPaths = new Set<string>();
  const strippedRanges: { start: number; end: number }[] = [];

  for (const blk of rawBlocks) {
    const meta = extractFileMeta(blk.info, blk.body, usedPaths);
    if (meta.path) {
      let uniquePath = meta.path;
      if (usedPaths.has(uniquePath)) {
        // Replace earlier version if re-emitted, or keep latest
        const existingIdx = files.findIndex((f) => f.path === uniquePath);
        if (existingIdx >= 0) {
          files[existingIdx] = { path: uniquePath, language: meta.language, content: meta.cleanedBody };
          strippedRanges.push({ start: blk.start, end: blk.end });
          continue;
        }
        uniquePath = `${usedPaths.size}-${meta.path}`;
      }
      usedPaths.add(uniquePath);
      files.push({
        path: uniquePath,
        language: meta.language,
        content: meta.cleanedBody,
      });
      strippedRanges.push({ start: blk.start, end: blk.end });
    }
  }

  const entryHtmlFile =
    files.find((f) => f.language === 'html' || f.path.endsWith('.html')) ?? null;
  const serverFile =
    files.find((f) => /server\.(ts|js|mjs)$/i.test(f.path) || f.path.includes('routes')) ?? null;
  const schemaFile =
    files.find((f) => f.language === 'sql' || f.path.endsWith('.sql')) ?? null;
  const packageFile =
    files.find((f) => f.path.endsWith('package.json')) ?? null;

  const isFullStack =
    files.length >= 2 &&
    Boolean(entryHtmlFile) &&
    Boolean(serverFile || schemaFile || packageFile);

  if (!isFullStack) {
    return null;
  }

  const routes = serverFile ? extractRoutesFromServer(serverFile.content) : [];
  const sqlExtracted = schemaFile ? extractSqlSchemaAndSeeds(schemaFile.content) : { tables: {} };
  const serverSeeds = serverFile ? extractSeedArraysFromServer(serverFile.content) : {};

  const initialTables: Record<string, Record<string, unknown>[]> = {
    ...serverSeeds,
    ...sqlExtracted.tables,
  };

  // Ensure every route's primary collection exists as an initialized table
  for (const r of routes) {
    const cleanPath = r.path.replace(/^\/api\//, '').split('/')[0];
    if (
      cleanPath &&
      !['health', 'stats', 'summary', 'analytics', 'auth', 'login', 'register'].includes(cleanPath) &&
      !initialTables[cleanPath]
    ) {
      initialTables[cleanPath] = [];
    }
  }

  // Detect whether client/server returns bare arrays or wrapped objects { items / [table] }
  const combinedCode = `${serverFile?.content ?? ''}\n${entryHtmlFile?.content ?? ''}`;
  const expectsWrapped =
    /\bdata\.(items|rows|records|results)\b/.test(entryHtmlFile?.content ?? '') ||
    /res\.json\(\s*\{\s*(items|rows|records|results)\s*:/.test(serverFile?.content ?? '');

  let title = 'Full-Stack Application';
  if (entryHtmlFile) {
    const tMatch = entryHtmlFile.content.match(/<title>([^<]+)<\/title>/i);
    if (tMatch?.[1]?.trim()) {
      title = tMatch[1].trim().slice(0, 64);
    }
  }
  if (title === 'Full-Stack Application' && packageFile) {
    try {
      const pkg = JSON.parse(packageFile.content) as { name?: string };
      if (pkg.name) title = pkg.name;
    } catch {
      // Ignore
    }
  }

  const idSeed = files.map((f) => `${f.path}:${f.content.length}`).join('|') + combinedCode.slice(0, 120);

  return {
    id: `fs-${hashString(idSeed)}`,
    title,
    files,
    entryHtmlFile,
    serverFile,
    schemaFile,
    packageFile,
    routes,
    initialTables,
    expectsArrayCollections: !expectsWrapped,
    isFullStack: true,
    strippedRanges,
  };
}

function findBestTable(
  tables: Record<string, Record<string, unknown>[]>,
  resourceSegment: string,
): string {
  const seg = resourceSegment.toLowerCase();
  if (tables[seg]) return seg;
  // Handle singular/plural variations (e.g. task vs tasks)
  if (tables[`${seg}s`]) return `${seg}s`;
  if (seg.endsWith('s') && tables[seg.slice(0, -1)]) return seg.slice(0, -1);
  if (seg.endsWith('ies') && tables[`${seg.slice(0, -3)}y`]) return `${seg.slice(0, -3)}y`;
  tables[seg] = [];
  return seg;
}

export function executeVirtualApiRequest(
  tables: Record<string, Record<string, unknown>[]>,
  req: VirtualApiRequest,
  expectsArrayCollections = true,
): { response: VirtualApiResponse; nextTables: Record<string, Record<string, unknown>[]>; mutated: boolean } {
  const method = (req.method || 'GET').toUpperCase();
  const parsedUrl = new URL(req.url, 'http://qowe.sandbox');
  const pathname = parsedUrl.pathname.replace(/\/+$/, '') || '/';
  const apiSub = pathname.replace(/^\/api\/?/, '');
  const segments = apiSub.split('/').filter(Boolean);

  let parsedBody: Record<string, unknown> = {};
  if (req.bodyText && req.bodyText.trim().length > 0) {
    try {
      const candidate = JSON.parse(req.bodyText) as unknown;
      if (candidate && typeof candidate === 'object') {
        parsedBody = candidate as Record<string, unknown>;
      }
    } catch {
      return {
        response: { status: 400, body: { error: 'Invalid JSON request body.' } },
        nextTables: tables,
        mutated: false,
      };
    }
  }

  // Deep clone tables for immutable React updates when mutating
  const cloneTables = (): Record<string, Record<string, unknown>[]> => {
    const out: Record<string, Record<string, unknown>[]> = {};
    for (const [k, rows] of Object.entries(tables)) {
      out[k] = rows.map((r) => ({ ...r }));
    }
    return out;
  };

  if (segments.length === 0 || segments[0] === 'health') {
    return {
      response: {
        status: 200,
        body: { ok: true, runtime: 'qowe-fullstack-sandbox', tables: Object.keys(tables) },
      },
      nextTables: tables,
      mutated: false,
    };
  }

  if (segments[0] === 'stats' || segments[0] === 'summary' || segments[0] === 'analytics') {
    const summary: Record<string, unknown> = {};
    let totalRecords = 0;
    for (const [tName, rows] of Object.entries(tables)) {
      totalRecords += rows.length;
      summary[tName] = rows.length;
    }
    return {
      response: { status: 200, body: { totalRecords, tables: summary } },
      nextTables: tables,
      mutated: false,
    };
  }

  if (segments[0] === 'auth') {
    const action = segments[1] || 'login';
    const nextTables = cloneTables();
    const userTable = findBestTable(nextTables, 'users');
    const users = nextTables[userTable];
    const email = String(parsedBody.email ?? parsedBody.username ?? '').trim().toLowerCase();

    if (action === 'register' || action === 'signup') {
      if (!email) {
        return {
          response: { status: 400, body: { error: 'Email or username is required.' } },
          nextTables: tables,
          mutated: false,
        };
      }
      const existing = users.find((u) => String(u.email ?? u.username ?? '').toLowerCase() === email);
      if (existing) {
        return {
          response: { status: 409, body: { error: 'Account already exists.' } },
          nextTables: tables,
          mutated: false,
        };
      }
      const newUser: Record<string, unknown> = {
        id: users.length + 1,
        name: parsedBody.name ?? email.split('@')[0],
        email,
        role: 'member',
        created_at: new Date().toISOString(),
      };
      users.push(newUser);
      return {
        response: { status: 201, body: { user: newUser, token: `qowe_session_${newUser.id}` } },
        nextTables,
        mutated: true,
      };
    }

    const found = users.find((u) => String(u.email ?? u.username ?? '').toLowerCase() === email) ?? users[0];
    if (!found) {
      return {
        response: { status: 401, body: { error: 'Invalid credentials.' } },
        nextTables: tables,
        mutated: false,
      };
    }
    return {
      response: { status: 200, body: { user: found, token: `qowe_session_${found.id ?? 1}` } },
      nextTables: tables,
      mutated: false,
    };
  }

  const nextTables = cloneTables();
  const tableName = findBestTable(nextTables, segments[0]);
  const rows = nextTables[tableName];
  const itemId = segments[1];
  const subAction = segments[2];

  if (method === 'GET') {
    if (itemId) {
      const found = rows.find((r) => String(r.id) === String(itemId));
      if (!found) {
        return {
          response: { status: 404, body: { error: 'Record not found.' } },
          nextTables: tables,
          mutated: false,
        };
      }
      return { response: { status: 200, body: found }, nextTables: tables, mutated: false };
    }

    let filtered = [...rows];
    const q = (parsedUrl.searchParams.get('q') ?? parsedUrl.searchParams.get('search') ?? '')
      .trim()
      .toLowerCase();
    if (q) {
      filtered = filtered.filter((r) => JSON.stringify(r).toLowerCase().includes(q));
    }

    for (const [paramKey, paramVal] of parsedUrl.searchParams.entries()) {
      if (['q', 'search', 'sort', 'limit', 'offset'].includes(paramKey) || !paramVal || paramVal === 'all') {
        continue;
      }
      filtered = filtered.filter((r) => {
        const rv = r[paramKey];
        if (rv === undefined) return true;
        return String(rv).toLowerCase() === paramVal.toLowerCase();
      });
    }

    return {
      response: {
        status: 200,
        body: expectsArrayCollections
          ? filtered
          : { items: filtered, [tableName]: filtered, total: filtered.length },
      },
      nextTables: tables,
      mutated: false,
    };
  }

  if (method === 'POST') {
    if (itemId && subAction) {
      const idx = rows.findIndex((r) => String(r.id) === String(itemId));
      if (idx < 0) {
        return {
          response: { status: 404, body: { error: 'Record not found.' } },
          nextTables: tables,
          mutated: false,
        };
      }
      const target = rows[idx];
      if (subAction === 'toggle') {
        if (typeof target.completed === 'boolean') target.completed = !target.completed;
        else if (typeof target.done === 'boolean') target.done = !target.done;
        else if (typeof target.status === 'string') {
          target.status = target.status === 'done' || target.status === 'completed' ? 'open' : 'done';
        }
      } else if (subAction === 'upvote' || subAction === 'like') {
        const cur = Number(target.votes ?? target.likes ?? 0);
        if (target.votes !== undefined) target.votes = cur + 1;
        else target.likes = cur + 1;
      } else {
        Object.assign(target, parsedBody);
      }
      return { response: { status: 200, body: target }, nextTables, mutated: true };
    }

    const numericIds = rows.map((r) => Number(r.id)).filter((n) => Number.isFinite(n));
    const nextId = (numericIds.length > 0 ? Math.max(...numericIds) : rows.length) + 1;
    const created: Record<string, unknown> = {
      id: nextId,
      ...parsedBody,
      created_at: parsedBody.created_at ?? new Date().toISOString(),
    };
    rows.unshift(created);
    return { response: { status: 201, body: created }, nextTables, mutated: true };
  }

  if (method === 'PUT' || method === 'PATCH') {
    const idx = rows.findIndex((r) => String(r.id) === String(itemId));
    if (idx < 0) {
      return {
        response: { status: 404, body: { error: 'Record not found.' } },
        nextTables: tables,
        mutated: false,
      };
    }
    const updated = { ...rows[idx], ...parsedBody, updated_at: new Date().toISOString() };
    rows[idx] = updated;
    return { response: { status: 200, body: updated }, nextTables, mutated: true };
  }

  if (method === 'DELETE') {
    if (!itemId) {
      nextTables[tableName] = [];
      return { response: { status: 200, body: { cleared: true } }, nextTables, mutated: true };
    }
    const idx = rows.findIndex((r) => String(r.id) === String(itemId));
    if (idx < 0) {
      return {
        response: { status: 404, body: { error: 'Record not found.' } },
        nextTables: tables,
        mutated: false,
      };
    }
    const [removed] = rows.splice(idx, 1);
    return { response: { status: 200, body: { deleted: true, id: removed.id } }, nextTables, mutated: true };
  }

  return {
    response: { status: 405, body: { error: `Method ${method} not allowed` } },
    nextTables: tables,
    mutated: false,
  };
}

export function buildFullStackSandboxSrcDoc(htmlCode: string, bridgeChannelId: string): string {
  const bridgeScript = `<script>
(function(){
  const CHANNEL = ${JSON.stringify(bridgeChannelId)};
  const nativeFetch = window.fetch.bind(window);
  const pending = new Map();
  let seq = 0;

  window.addEventListener('message', function(ev) {
    const d = ev.data;
    if (!d || d.channel !== CHANNEL || d.kind !== 'api-res') return;
    const entry = pending.get(d.reqId);
    if (!entry) return;
    pending.delete(d.reqId);
    entry.resolve(d);
  });

  window.fetch = function(input, init) {
    const rawUrl = typeof input === 'string' ? input : (input && input.url) || '';
    const isRelativeApi = rawUrl.startsWith('/api') || rawUrl.startsWith('api/');
    let resolvedApiPath = null;
    if (isRelativeApi) {
      resolvedApiPath = rawUrl.startsWith('/') ? rawUrl : '/' + rawUrl;
    } else if (/^https?:\\/\\//i.test(rawUrl)) {
      try {
        const u = new URL(rawUrl);
        if (u.pathname.startsWith('/api')) {
          resolvedApiPath = u.pathname + u.search;
        }
      } catch {}
    }

    if (!resolvedApiPath) {
      return nativeFetch(input, init);
    }

    const reqId = ++seq;
    const method = ((init && init.method) || (typeof input !== 'string' && input && input.method) || 'GET').toUpperCase();
    let bodyText = '';
    if (init && typeof init.body === 'string') {
      bodyText = init.body;
    }

    return new Promise(function(resolve) {
      pending.set(reqId, {
        resolve: function(resPayload) {
          const jsonStr = JSON.stringify(resPayload.body === undefined ? null : resPayload.body);
          resolve(new Response(jsonStr, {
            status: resPayload.status || 200,
            headers: { 'Content-Type': 'application/json' }
          }));
        }
      });
      window.parent.postMessage({
        channel: CHANNEL,
        kind: 'api-req',
        reqId: reqId,
        method: method,
        url: resolvedApiPath,
        bodyText: bodyText
      }, '*');
    });
  };
})();
</script>`;

  if (/<head[^>]*>/i.test(htmlCode)) {
    return htmlCode.replace(/<head([^>]*)>/i, `<head$1>\n${bridgeScript}`);
  }
  if (/<body[^>]*>/i.test(htmlCode)) {
    return htmlCode.replace(/<body([^>]*)>/i, `<head>${bridgeScript}</head><body$1>`);
  }
  return `<!doctype html><html><head>${bridgeScript}</head><body>${htmlCode}</body></html>`;
}

// Standard CRC-32 + Store (method 0) PKZIP encoder with zero external dependencies
const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) {
      c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    }
    table[i] = c >>> 0;
  }
  return table;
})();

function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

export function createProjectZipBytes(files: FullStackProjectFile[]): Uint8Array {
  const encoder = new TextEncoder();
  const normalizedFiles = [...files];
  if (!normalizedFiles.some((f) => f.path === '.env.example')) {
    normalizedFiles.push({
      path: '.env.example',
      language: 'text',
      content: 'PORT=3000\nNODE_ENV=production\n',
    });
  }

  const localParts: Uint8Array[] = [];
  const centralParts: Uint8Array[] = [];
  let offset = 0;

  for (const file of normalizedFiles) {
    const nameBytes = encoder.encode(file.path);
    const dataBytes = encoder.encode(file.content);
    const crc = crc32(dataBytes);

    const localHeader = new Uint8Array(30 + nameBytes.length);
    const lv = new DataView(localHeader.buffer);
    lv.setUint32(0, 0x04034b50, true); // Local file header signature
    lv.setUint16(4, 20, true); // Version needed
    lv.setUint16(6, 0x0800, true); // UTF-8 flag
    lv.setUint16(8, 0, true); // Store (no compression)
    lv.setUint16(10, 0, true);
    lv.setUint16(12, 0, true);
    lv.setUint32(14, crc, true);
    lv.setUint32(18, dataBytes.length, true);
    lv.setUint32(22, dataBytes.length, true);
    lv.setUint16(26, nameBytes.length, true);
    lv.setUint16(28, 0, true);
    localHeader.set(nameBytes, 30);

    localParts.push(localHeader, dataBytes);

    const centralHeader = new Uint8Array(46 + nameBytes.length);
    const cv = new DataView(centralHeader.buffer);
    cv.setUint32(0, 0x02014b50, true); // Central directory signature
    cv.setUint16(4, 20, true);
    cv.setUint16(6, 20, true);
    cv.setUint16(8, 0x0800, true);
    cv.setUint16(10, 0, true);
    cv.setUint16(12, 0, true);
    cv.setUint16(14, 0, true);
    cv.setUint32(16, crc, true);
    cv.setUint32(20, dataBytes.length, true);
    cv.setUint32(24, dataBytes.length, true);
    cv.setUint16(28, nameBytes.length, true);
    cv.setUint16(30, 0, true);
    cv.setUint16(32, 0, true);
    cv.setUint16(34, 0, true);
    cv.setUint16(36, 0, true);
    cv.setUint32(38, 0, true);
    cv.setUint32(42, offset, true);
    centralHeader.set(nameBytes, 46);

    centralParts.push(centralHeader);
    offset += localHeader.length + dataBytes.length;
  }

  const centralSize = centralParts.reduce((s, p) => s + p.length, 0);
  const eocd = new Uint8Array(22);
  const ev = new DataView(eocd.buffer);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(4, 0, true);
  ev.setUint16(6, 0, true);
  ev.setUint16(8, normalizedFiles.length, true);
  ev.setUint16(10, normalizedFiles.length, true);
  ev.setUint32(12, centralSize, true);
  ev.setUint32(16, offset, true);
  ev.setUint16(20, 0, true);

  const totalLen = offset + centralSize + eocd.length;
  const result = new Uint8Array(totalLen);
  let cursor = 0;
  for (const part of localParts) {
    result.set(part, cursor);
    cursor += part.length;
  }
  for (const part of centralParts) {
    result.set(part, cursor);
    cursor += part.length;
  }
  result.set(eocd, cursor);
  return result;
}

export function createProjectZipBlob(files: FullStackProjectFile[]): Blob {
  const bytes = createProjectZipBytes(files);
  return new Blob([bytes.buffer as ArrayBuffer], { type: 'application/zip' });
}
