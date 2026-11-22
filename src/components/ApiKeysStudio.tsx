import React, { useCallback, useEffect, useState } from 'react';
import {
  Activity,
  Check,
  Copy,
  Globe,
  KeyRound,
  Play,
  Plus,
  RefreshCw,
  ShieldCheck,
  Terminal,
  Trash2,
} from 'lucide-react';

interface PublicKeyRecord {
  id: string;
  name: string;
  prefix: string;
  createdAt: string;
  lastUsedAt: string | null;
  requestCount: number;
  avgLatencyMs?: number;
  lastLatencyMs?: number | null;
}

type ApiEndpointCategory = 'chat' | 'image' | 'video' | 'models';

interface CategoryStats {
  count: number;
  avgLatencyMs: number;
  lastLatencyMs: number | null;
}

interface ApiRequestEvent {
  id: string;
  endpoint: string;
  category: ApiEndpointCategory;
  model: string;
  keyPrefix: string;
  keyName: string;
  status: number;
  latencyMs: number;
  timestamp: string;
}

interface ApiUsageSummary {
  totalRequests: number;
  successCount: number;
  errorCount: number;
  successRate: number;
  avgLatencyMs: number;
  p95LatencyMs: number;
  minLatencyMs: number | null;
  maxLatencyMs: number | null;
  lastLatencyMs: number | null;
  byCategory: Record<ApiEndpointCategory, CategoryStats>;
  recentEvents: ApiRequestEvent[];
  updatedAt: string;
}

interface BaseUrlResolution {
  current: string;
  appUrl: string | null;
  sharedUrl: string | null;
  recommendedUrl: string;
  isCurrentDevPreview: boolean;
  isCurrentSharedPreview?: boolean;
  isRecommendedDevPreview: boolean;
  isRecommendedCookieGated?: boolean;
}

interface ApiKeysStudioProps {
  architectureName: string;
  totalParamsFormatted: string;
}

type SnippetLang = 'curl' | 'javascript' | 'python';
type SnippetAuthMode = 'x-api-key' | 'bearer';
type TestEndpointKind = 'chat' | 'image' | 'video' | 'models';

const CUSTOM_URL_STORAGE_KEY = 'qowe_custom_base_url_v1';

const EMPTY_USAGE: ApiUsageSummary = {
  totalRequests: 0,
  successCount: 0,
  errorCount: 0,
  successRate: 100,
  avgLatencyMs: 0,
  p95LatencyMs: 0,
  minLatencyMs: null,
  maxLatencyMs: null,
  lastLatencyMs: null,
  byCategory: {
    chat: { count: 0, avgLatencyMs: 0, lastLatencyMs: null },
    image: { count: 0, avgLatencyMs: 0, lastLatencyMs: null },
    video: { count: 0, avgLatencyMs: 0, lastLatencyMs: null },
    models: { count: 0, avgLatencyMs: 0, lastLatencyMs: null },
  },
  recentEvents: [],
  updatedAt: new Date().toISOString(),
};

const CATEGORY_META: {
  key: ApiEndpointCategory;
  label: string;
  barClass: string;
}[] = [
  { key: 'chat', label: 'Chat Completions', barClass: 'bg-[#007aff] dark:bg-[#0a84ff]' },
  { key: 'image', label: 'Studio Image', barClass: 'bg-indigo-500 dark:bg-indigo-400' },
  { key: 'video', label: '60 FPS Video', barClass: 'bg-teal-500 dark:bg-teal-400' },
  { key: 'models', label: 'Model Registry', barClass: 'bg-neutral-500 dark:bg-neutral-400' },
];

function normalizeCustomHttpUrl(raw: string): string | null {
  const trimmed = raw.trim().replace(/\/+$/, '');
  if (!trimmed) return null;
  try {
    const parsed = new URL(trimmed);
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      return null;
    }
    return parsed.origin;
  } catch {
    return null;
  }
}

function isCookieGatedHost(urlStr: string): boolean {
  return /https?:\/\/ais-(dev|pre)-/i.test(urlStr);
}

export const ApiKeysStudio: React.FC<ApiKeysStudioProps> = ({
  architectureName,
  totalParamsFormatted,
}) => {
  const [keys, setKeys] = useState<PublicKeyRecord[]>([]);
  const [usage, setUsage] = useState<ApiUsageSummary>(EMPTY_USAGE);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshingUsage, setIsRefreshingUsage] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const [newKeyName, setNewKeyName] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const [revealedKey, setRevealedKey] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [customUrlInput, setCustomUrlInput] = useState(() => {
    try {
      return localStorage.getItem(CUSTOM_URL_STORAGE_KEY) || '';
    } catch {
      return '';
    }
  });
  const [baseUrl, setBaseUrl] = useState(() => {
    try {
      const saved = normalizeCustomHttpUrl(localStorage.getItem(CUSTOM_URL_STORAGE_KEY) || '');
      if (saved) return saved;
    } catch {
      // Ignore storage access errors
    }
    return typeof window !== 'undefined' ? window.location.origin : 'http://localhost:3000';
  });
  const [baseUrlsInfo, setBaseUrlsInfo] = useState<BaseUrlResolution | null>(null);
  const [hasCustomUrlChoice, setHasCustomUrlChoice] = useState(() => {
    try {
      return Boolean(normalizeCustomHttpUrl(localStorage.getItem(CUSTOM_URL_STORAGE_KEY) || ''));
    } catch {
      return false;
    }
  });
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const [healthProbe, setHealthProbe] = useState<{
    checking: boolean;
    ok: boolean | null;
    message: string | null;
  }>({ checking: false, ok: null, message: null });

  const [snippetLang, setSnippetLang] = useState<SnippetLang>('curl');
  const [snippetEndpoint, setSnippetEndpoint] = useState<'chat' | 'image' | 'video'>('chat');
  const [snippetAuthMode, setSnippetAuthMode] = useState<SnippetAuthMode>('x-api-key');

  const [testKey, setTestKey] = useState('');
  const [testEndpoint, setTestEndpoint] = useState<TestEndpointKind>('chat');
  const [testPrompt, setTestPrompt] = useState('Introduce Qowe 1.0 in two sentences.');
  const [isTesting, setIsTesting] = useState(false);
  const [testOutput, setTestOutput] = useState<{
    status: number;
    latencyMs: number;
    routeLabel: string;
    body: string;
  } | null>(null);

  const loadKeysAndUsage = useCallback(
    async (silent = false) => {
      if (!silent) {
        setIsLoading(true);
      } else {
        setIsRefreshingUsage(true);
      }
      try {
        const resp = await fetch('/api/keys', { credentials: 'include' });
        const data = (await resp.json()) as {
          keys?: PublicKeyRecord[];
          usage?: ApiUsageSummary;
          baseUrls?: BaseUrlResolution;
          error?: string;
        };
        if (!resp.ok) throw new Error(data.error || 'Failed to load API keys.');
        setKeys(Array.isArray(data.keys) ? data.keys : []);
        if (data.usage) {
          setUsage(data.usage);
        }
        if (data.baseUrls) {
          setBaseUrlsInfo(data.baseUrls);
          if (!hasCustomUrlChoice) {
            setBaseUrl(data.baseUrls.recommendedUrl || data.baseUrls.current);
          }
        } else if (!hasCustomUrlChoice && typeof window !== 'undefined') {
          setBaseUrl(window.location.origin);
        }
      } catch (err) {
        if (!silent) {
          setErrorMsg(err instanceof Error ? err.message : 'Unable to fetch API keys.');
        }
      } finally {
        setIsLoading(false);
        setIsRefreshingUsage(false);
      }
    },
    [hasCustomUrlChoice],
  );

  useEffect(() => {
    void loadKeysAndUsage(false);
  }, [loadKeysAndUsage]);

  useEffect(() => {
    if (!autoRefresh) return;
    const poll = () => {
      if (typeof document !== 'undefined' && document.hidden) return;
      void loadKeysAndUsage(true);
    };
    const interval = window.setInterval(poll, 4500);
    const onVisibility = () => {
      if (!document.hidden) void loadKeysAndUsage(true);
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      window.clearInterval(interval);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [autoRefresh, loadKeysAndUsage]);

  const copyText = async (value: string, id: string) => {
    try {
      await navigator.clipboard.writeText(value);
      setCopiedId(id);
      window.setTimeout(() => setCopiedId((prev) => (prev === id ? null : prev)), 1800);
    } catch {
      // Clipboard blocked in restricted contexts
    }
  };

  const handleApplyCustomUrl = (e: React.FormEvent) => {
    e.preventDefault();
    const normalized = normalizeCustomHttpUrl(customUrlInput);
    if (!normalized) {
      try {
        localStorage.removeItem(CUSTOM_URL_STORAGE_KEY);
      } catch {
        // Ignore
      }
      setHasCustomUrlChoice(false);
      if (baseUrlsInfo) {
        setBaseUrl(baseUrlsInfo.recommendedUrl || baseUrlsInfo.current);
      }
      setHealthProbe({ checking: false, ok: null, message: null });
      return;
    }
    try {
      localStorage.setItem(CUSTOM_URL_STORAGE_KEY, normalized);
    } catch {
      // Ignore
    }
    setCustomUrlInput(normalized);
    setHasCustomUrlChoice(true);
    setBaseUrl(normalized);
    setHealthProbe({ checking: false, ok: null, message: null });
  };

  const handleCheckPublicReachability = async () => {
    setHealthProbe({ checking: true, ok: null, message: null });
    try {
      const resp = await fetch(`${baseUrl}/api/v1/health`, {
        method: 'GET',
        mode: 'cors',
        credentials: 'omit',
      });
      const text = await resp.text();
      const ct = resp.headers.get('content-type') || '';
      if (
        (resp.redirected && resp.url.includes('__cookie_check')) ||
        (ct.includes('text/html') && text.includes('__cookie_check'))
      ) {
        setHealthProbe({
          checking: false,
          ok: false,
          message:
            'HTTP 302 → /__cookie_check.html: External callers are currently blocked by AI Studio’s preview login proxy. Deploy the app to Cloud Run and paste your deployed *.run.app URL above.',
        });
        return;
      }
      if (!resp.ok) {
        setHealthProbe({
          checking: false,
          ok: false,
          message: `Endpoint responded with HTTP ${resp.status}. Verify the deployed URL serves this app.`,
        });
        return;
      }
      const parsed = JSON.parse(text) as { ok?: boolean; service?: string };
      if (parsed.ok) {
        setHealthProbe({
          checking: false,
          ok: true,
          message: `Publicly reachable (HTTP 200 · ${parsed.service || 'qowe-1.0-api'}) — external cURL, Python, and mobile apps can connect without cookie redirects.`,
        });
      } else {
        setHealthProbe({
          checking: false,
          ok: false,
          message: 'Unexpected health payload from URL.',
        });
      }
    } catch {
      setHealthProbe({
        checking: false,
        ok: false,
        message: isCookieGatedHost(baseUrl)
          ? 'Blocked by AI Studio preview proxy cookie redirect (/__cookie_check.html). Deploy to Cloud Run and enter the public *.run.app URL.'
          : 'Network or CORS error reaching /api/v1/health at this URL.',
      });
    }
  };

  const handleCreateKey = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = newKeyName.trim();
    if (!trimmed || isCreating) return;
    setErrorMsg(null);
    setIsCreating(true);
    try {
      const resp = await fetch('/api/keys', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: trimmed }),
      });
      const data = (await resp.json()) as {
        key?: PublicKeyRecord;
        rawKey?: string;
        error?: string;
      };
      if (!resp.ok || !data.key || !data.rawKey) {
        throw new Error(data.error || 'Failed to generate API key.');
      }
      setKeys((prev) => [data.key as PublicKeyRecord, ...prev]);
      setRevealedKey(data.rawKey);
      setTestKey(data.rawKey);
      setNewKeyName('');
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Unable to create API key.');
    } finally {
      setIsCreating(false);
    }
  };

  const handleRevokeKey = async (id: string) => {
    setErrorMsg(null);
    try {
      const resp = await fetch(`/api/keys/${encodeURIComponent(id)}`, {
        method: 'DELETE',
        credentials: 'include',
      });
      if (!resp.ok) {
        const data = (await resp.json().catch(() => ({}))) as { error?: string };
        throw new Error(data.error || 'Failed to revoke key.');
      }
      setKeys((prev) => prev.filter((k) => k.id !== id));
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Unable to revoke API key.');
    }
  };

  const handleRunTest = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isTesting) return;
    setIsTesting(true);
    setTestOutput(null);

    const apiKey = testKey.trim();
    const endpointPath =
      testEndpoint === 'models'
        ? '/api/v1/models'
        : testEndpoint === 'chat'
          ? '/api/v1/chat/completions'
          : testEndpoint === 'image'
            ? '/api/v1/images/generations'
            : '/api/v1/videos/generations';

    const method = testEndpoint === 'models' ? 'GET' : 'POST';
    const payload =
      testEndpoint === 'chat'
        ? { model: 'qowe-1.0-ultra', messages: [{ role: 'user', content: testPrompt }] }
        : testEndpoint === 'image'
          ? { prompt: testPrompt, aspect_ratio: '1:1' }
          : { prompt: testPrompt, aspect_ratio: '16:9', duration_sec: 4 };

    const t0 = performance.now();
    try {
      // Send x-api-key (not Authorization: Bearer) so Google preview proxies do not intercept/302-redirect
      const directResp = await fetch(endpointPath, {
        method,
        credentials: 'include',
        headers: {
          ...(method === 'POST' ? { 'Content-Type': 'application/json' } : {}),
          'x-api-key': apiKey,
        },
        ...(method === 'POST' ? { body: JSON.stringify(payload) } : {}),
      });

      const directRaw = await directResp.text();
      const directCt = directResp.headers.get('content-type') || '';
      const hitCookieCheck =
        (directResp.redirected && directResp.url.includes('__cookie_check')) ||
        (directCt.includes('text/html') && directRaw.includes('__cookie_check'));

      let finalStatus = directResp.status;
      let finalRaw = directRaw;
      let routeLabel = `Direct ${endpointPath}`;

      if (hitCookieCheck) {
        const bridgeResp = await fetch('/api/playground/test', {
          method: 'POST',
          credentials: 'include',
          headers: {
            'Content-Type': 'application/json',
            'x-qowe-api-key': apiKey,
          },
          body: JSON.stringify({
            endpoint: testEndpoint,
            prompt: testPrompt,
          }),
        });
        finalStatus = bridgeResp.status;
        finalRaw = await bridgeResp.text();
        routeLabel = 'Studio Bridge (/api/playground/test)';
      }

      const clientLatencyMs = Math.max(1, Math.round(performance.now() - t0));
      let pretty = finalRaw;
      try {
        pretty = JSON.stringify(JSON.parse(finalRaw), null, 2);
      } catch {
        // Leave text as-is
      }
      setTestOutput({
        status: finalStatus,
        latencyMs: clientLatencyMs,
        routeLabel,
        body: pretty.slice(0, 4000),
      });
      window.setTimeout(() => {
        void loadKeysAndUsage(true);
      }, 60);
    } catch (err) {
      // If cross-origin or proxy redirect failed at fetch level, try same-origin playground bridge once
      try {
        const bridgeResp = await fetch('/api/playground/test', {
          method: 'POST',
          credentials: 'include',
          headers: {
            'Content-Type': 'application/json',
            'x-qowe-api-key': apiKey,
          },
          body: JSON.stringify({
            endpoint: testEndpoint,
            prompt: testPrompt,
          }),
        });
        const bridgeRaw = await bridgeResp.text();
        const clientLatencyMs = Math.max(1, Math.round(performance.now() - t0));
        let pretty = bridgeRaw;
        try {
          pretty = JSON.stringify(JSON.parse(bridgeRaw), null, 2);
        } catch {
          // Leave text as-is
        }
        setTestOutput({
          status: bridgeResp.status,
          latencyMs: clientLatencyMs,
          routeLabel: 'Studio Bridge (/api/playground/test)',
          body: pretty.slice(0, 4000),
        });
        window.setTimeout(() => {
          void loadKeysAndUsage(true);
        }, 60);
      } catch (bridgeErr) {
        setTestOutput({
          status: 0,
          latencyMs: Math.max(1, Math.round(performance.now() - t0)),
          routeLabel: 'Network Error',
          body:
            bridgeErr instanceof Error
              ? bridgeErr.message
              : err instanceof Error
                ? err.message
                : 'Network error testing API endpoint.',
        });
      }
    } finally {
      setIsTesting(false);
    }
  };

  const keyPlaceholder = revealedKey || 'YOUR_QOWE_API_KEY';

  const getSnippet = (): string => {
    const endpoint =
      snippetEndpoint === 'chat'
        ? `${baseUrl}/api/v1/chat/completions`
        : snippetEndpoint === 'image'
          ? `${baseUrl}/api/v1/images/generations`
          : `${baseUrl}/api/v1/videos/generations`;

    const bodyObj =
      snippetEndpoint === 'chat'
        ? {
            model: 'qowe-1.0-ultra',
            messages: [{ role: 'user', content: 'Write a haiku about neural networks.' }],
          }
        : snippetEndpoint === 'image'
          ? { prompt: 'Studio golden banana on dark slate', aspect_ratio: '1:1' }
          : {
              prompt: 'Neon cyberpunk city in rain at night',
              aspect_ratio: '16:9',
              duration_sec: 4,
            };

    const headerKey = snippetAuthMode === 'x-api-key' ? 'x-api-key' : 'Authorization';
    const headerVal =
      snippetAuthMode === 'x-api-key' ? keyPlaceholder : `Bearer ${keyPlaceholder}`;

    if (snippetLang === 'curl') {
      return `curl -X POST "${endpoint}" \\\n  -H "Content-Type: application/json" \\\n  -H "${headerKey}: ${headerVal}" \\\n  -d '${JSON.stringify(bodyObj)}'`;
    }
    if (snippetLang === 'javascript') {
      return `const response = await fetch("${endpoint}", {\n  method: "POST",\n  headers: {\n    "Content-Type": "application/json",\n    "${headerKey}": "${headerVal}"\n  },\n  body: JSON.stringify(${JSON.stringify(bodyObj, null, 2)})\n});\n\nconst data = await response.json();\nconsole.log(data);`;
    }
    return `import requests\n\nresp = requests.post(\n    "${endpoint}",\n    headers={\n        "Content-Type": "application/json",\n        "${headerKey}": "${headerVal}",\n    },\n    json=${JSON.stringify(bodyObj, null, 4)}\n)\nprint(resp.json())`;
  };

  const endpointRows = [
    { method: 'GET', path: '/api/v1/health', desc: 'Unauthenticated API reachability probe' },
    { method: 'GET', path: '/api/v1/models', desc: 'List Qowe 1.0 chat, vision, and video models' },
    {
      method: 'POST',
      path: '/api/v1/chat/completions',
      desc: 'OpenAI-compatible chat & code completions (JSON or SSE stream)',
    },
    {
      method: 'POST',
      path: '/api/v1/images/generations',
      desc: 'Generate vector studio artwork (SVG + base64) from a text prompt',
    },
    {
      method: 'POST',
      path: '/api/v1/videos/generations',
      desc: 'Generate 60 fps declarative video scene specification',
    },
  ];

  const surfaceCard =
    'rounded-[22px] bg-white dark:bg-[#1c1c1e] border border-black/[0.06] dark:border-white/[0.08] shadow-[0_1px_2px_rgba(0,0,0,0.03),0_8px_24px_rgba(0,0,0,0.03)]';

  const currentUrlCookieGated = isCookieGatedHost(baseUrl);

  return (
    <div className="flex-1 overflow-y-auto bg-[#f5f5f7] dark:bg-[#0b0b0d] p-4 sm:p-6">
      <div className="max-w-5xl mx-auto space-y-6">
        <header className="flex flex-col sm:flex-row sm:items-end justify-between gap-3">
          <div className="space-y-1">
            <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-neutral-900 dark:text-white text-balance">
              Qowe 1.0 API Keys &amp; Telemetry
            </h1>
            <p className="text-sm text-neutral-600 dark:text-neutral-400">
              Monitor real-time request volume and response latency across {architectureName} ({totalParamsFormatted}) endpoints.
            </p>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={() => setAutoRefresh((prev) => !prev)}
              className={`px-3.5 py-1.5 rounded-full text-xs font-medium transition-colors whitespace-nowrap ${
                autoRefresh
                  ? 'bg-[#007aff]/10 dark:bg-[#0a84ff]/15 text-[#007aff] dark:text-[#0a84ff]'
                  : 'bg-black/[0.05] dark:bg-white/[0.08] text-neutral-600 dark:text-neutral-400'
              }`}
            >
              {autoRefresh ? 'Live Polling On' : 'Live Polling Paused'}
            </button>
            <button
              type="button"
              onClick={() => void loadKeysAndUsage(true)}
              aria-label="Refresh usage statistics"
              className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-white dark:bg-[#2c2c2e] border border-black/[0.08] dark:border-white/[0.1] text-xs font-medium text-neutral-700 dark:text-neutral-200 hover:bg-neutral-50 dark:hover:bg-[#3a3a3c] transition-colors whitespace-nowrap"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isRefreshingUsage ? 'animate-spin' : ''}`} />
              <span>Refresh</span>
            </button>
          </div>
        </header>

        {/* Real-Time API Usage & Latency KPIs */}
        <section aria-label="Real-time API usage and latency metrics" className="space-y-4">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3.5">
            <div className={`${surfaceCard} p-4 sm:p-5`}>
              <div className="text-xs font-medium text-neutral-500 dark:text-neutral-400">
                Total API Requests
              </div>
              <div
                data-testid="usage-total-requests"
                className="mt-1.5 text-2xl font-semibold text-neutral-900 dark:text-white tabular-nums"
              >
                {usage.totalRequests.toLocaleString()}
              </div>
              <div className="mt-1 text-xs text-neutral-500 dark:text-neutral-400 tabular-nums">
                {usage.successCount.toLocaleString()} passed · {usage.errorCount.toLocaleString()} errors
              </div>
            </div>

            <div className={`${surfaceCard} p-4 sm:p-5`}>
              <div className="text-xs font-medium text-neutral-500 dark:text-neutral-400">
                Average Latency
              </div>
              <div
                data-testid="usage-avg-latency"
                className="mt-1.5 text-2xl font-semibold text-neutral-900 dark:text-white tabular-nums"
              >
                {usage.totalRequests > 0 ? `${usage.avgLatencyMs} ms` : '—'}
              </div>
              <div className="mt-1 text-xs text-neutral-500 dark:text-neutral-400 tabular-nums">
                Min {usage.minLatencyMs !== null ? `${usage.minLatencyMs} ms` : '—'} · Max{' '}
                {usage.maxLatencyMs !== null ? `${usage.maxLatencyMs} ms` : '—'}
              </div>
            </div>

            <div className={`${surfaceCard} p-4 sm:p-5`}>
              <div className="text-xs font-medium text-neutral-500 dark:text-neutral-400">
                P95 Latency
              </div>
              <div
                data-testid="usage-p95-latency"
                className="mt-1.5 text-2xl font-semibold text-neutral-900 dark:text-white tabular-nums"
              >
                {usage.totalRequests > 0 ? `${usage.p95LatencyMs} ms` : '—'}
              </div>
              <div className="mt-1 text-xs text-neutral-500 dark:text-neutral-400 tabular-nums">
                Last response {usage.lastLatencyMs !== null ? `${usage.lastLatencyMs} ms` : '—'}
              </div>
            </div>

            <div className={`${surfaceCard} p-4 sm:p-5`}>
              <div className="text-xs font-medium text-neutral-500 dark:text-neutral-400">
                Success Rate
              </div>
              <div
                data-testid="usage-success-rate"
                className="mt-1.5 text-2xl font-semibold text-neutral-900 dark:text-white tabular-nums"
              >
                {usage.successRate.toFixed(1)}%
              </div>
              <div className="mt-1 text-xs text-neutral-500 dark:text-neutral-400 tabular-nums">
                Updated {new Date(usage.updatedAt).toLocaleTimeString()}
              </div>
            </div>
          </div>

          {/* Endpoint Breakdown + Real-Time Request Log */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
            <div className={`${surfaceCard} lg:col-span-5 p-5 space-y-4`}>
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-[#007aff] dark:text-[#0a84ff]" />
                  <h2 className="text-sm font-semibold text-neutral-900 dark:text-white">
                    Usage by Endpoint
                  </h2>
                </div>
                <span className="text-xs text-neutral-500 dark:text-neutral-400 tabular-nums">
                  {usage.totalRequests} total
                </span>
              </div>

              <div className="space-y-3.5">
                {CATEGORY_META.map((cat) => {
                  const stat = usage.byCategory[cat.key];
                  const sharePct =
                    usage.totalRequests > 0
                      ? Math.round((stat.count / usage.totalRequests) * 100)
                      : 0;
                  return (
                    <div key={cat.key} className="space-y-1.5">
                      <div className="flex items-center justify-between gap-2 text-xs">
                        <span className="font-medium text-neutral-800 dark:text-neutral-200">
                          {cat.label}
                        </span>
                        <span className="font-mono text-neutral-500 dark:text-neutral-400 tabular-nums">
                          {stat.count} req · {stat.count > 0 ? `${stat.avgLatencyMs} ms avg` : '—'}
                        </span>
                      </div>
                      <div className="h-2 w-full rounded-full bg-black/[0.06] dark:bg-white/[0.08] overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-300 ${cat.barClass}`}
                          style={{ width: `${sharePct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className={`${surfaceCard} lg:col-span-7 p-5 space-y-3`}>
              <div className="flex items-center justify-between gap-2">
                <h2 className="text-sm font-semibold text-neutral-900 dark:text-white">
                  Recent Authenticated Requests
                </h2>
                <span className="text-xs text-neutral-500 dark:text-neutral-400 tabular-nums">
                  Latest {usage.recentEvents.length} events
                </span>
              </div>

              {usage.recentEvents.length === 0 ? (
                <div className="py-8 text-center space-y-2">
                  <p className="text-xs text-neutral-500 dark:text-neutral-400">
                    No authenticated API requests recorded yet. Send a test call below to observe live latency and request counts.
                  </p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse text-xs">
                    <thead>
                      <tr className="border-b border-black/[0.06] dark:border-white/[0.08] text-neutral-500 dark:text-neutral-400">
                        <th className="py-2 pr-3 font-medium">Endpoint</th>
                        <th className="py-2 px-3 font-medium">Key</th>
                        <th className="py-2 px-3 font-medium">Status</th>
                        <th className="py-2 px-3 text-right font-medium">Latency</th>
                        <th className="py-2 pl-3 text-right font-medium">Time</th>
                      </tr>
                    </thead>
                    <tbody
                      data-testid="recent-api-events"
                      className="divide-y divide-black/[0.05] dark:divide-white/[0.06]"
                    >
                      {usage.recentEvents.slice(0, 8).map((ev) => {
                        const ok = ev.status >= 200 && ev.status < 400;
                        return (
                          <tr key={ev.id} className="hover:bg-black/[0.02] dark:hover:bg-white/[0.03]">
                            <td className="py-2.5 pr-3 font-mono text-neutral-900 dark:text-neutral-100 whitespace-nowrap">
                              {ev.endpoint}
                            </td>
                            <td className="py-2.5 px-3 text-neutral-600 dark:text-neutral-400 whitespace-nowrap">
                              {ev.keyName}
                            </td>
                            <td
                              className={`py-2.5 px-3 font-mono font-medium tabular-nums whitespace-nowrap ${
                                ok
                                  ? 'text-emerald-600 dark:text-emerald-400'
                                  : 'text-amber-600 dark:text-amber-400'
                              }`}
                            >
                              {ev.status} {ok ? 'OK' : 'ERR'}
                            </td>
                            <td className="py-2.5 px-3 text-right font-mono tabular-nums text-neutral-900 dark:text-neutral-100 whitespace-nowrap">
                              {ev.latencyMs} ms
                            </td>
                            <td className="py-2.5 pl-3 text-right font-mono tabular-nums text-neutral-500 dark:text-neutral-400 whitespace-nowrap">
                              {new Date(ev.timestamp).toLocaleTimeString()}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Endpoint Base URL & Routes */}
        <section className={`${surfaceCard} p-5 space-y-4`}>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="flex items-center gap-2.5">
              <Globe className="w-4 h-4 text-[#007aff] dark:text-[#0a84ff] shrink-0" />
              <div>
                <h2 className="text-sm font-semibold text-neutral-900 dark:text-white">
                  Public Base URL
                </h2>
                <p className="text-xs text-neutral-500 dark:text-neutral-400">
                  {currentUrlCookieGated
                    ? 'Preview URL is behind Google’s login gate for external tools — deploy to Cloud Run for unauthenticated external calls.'
                    : 'Use this Base URL with your x-api-key or Bearer qowe_sk_... key.'}
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2 min-w-0">
              <code
                data-testid="public-base-url"
                className="px-3.5 py-1.5 rounded-full bg-[#f5f5f7] dark:bg-[#121214] border border-black/[0.06] dark:border-white/[0.08] text-xs font-mono text-neutral-800 dark:text-neutral-200 truncate max-w-full sm:max-w-md"
              >
                {baseUrl}
              </code>
              <button
                type="button"
                onClick={() => copyText(baseUrl, 'base-url')}
                className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-[#f5f5f7] dark:bg-[#2c2c2e] text-xs font-medium text-neutral-800 dark:text-neutral-100 hover:bg-black/[0.06] dark:hover:bg-[#3a3a3c] transition-colors whitespace-nowrap shrink-0"
              >
                {copiedId === 'base-url' ? (
                  <Check className="w-3.5 h-3.5 text-emerald-500" />
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
                <span>{copiedId === 'base-url' ? 'Copied' : 'Copy URL'}</span>
              </button>
              <button
                type="button"
                disabled={healthProbe.checking}
                onClick={() => void handleCheckPublicReachability()}
                className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full border border-black/[0.08] dark:border-white/[0.1] text-xs font-medium text-neutral-700 dark:text-neutral-200 hover:bg-black/[0.04] dark:hover:bg-white/[0.08] disabled:opacity-50 transition-colors whitespace-nowrap shrink-0"
              >
                <ShieldCheck className="w-3.5 h-3.5" />
                <span>{healthProbe.checking ? 'Checking...' : 'Check Public Reachability'}</span>
              </button>
            </div>
          </div>

          <div className="space-y-3">
            {baseUrlsInfo && (
              <div className="flex flex-wrap items-center gap-1.5 text-xs">
                {baseUrlsInfo.sharedUrl && (
                  <button
                    type="button"
                    onClick={() => {
                      try {
                        localStorage.removeItem(CUSTOM_URL_STORAGE_KEY);
                      } catch {
                        // Ignore
                      }
                      setCustomUrlInput('');
                      setHasCustomUrlChoice(true);
                      setBaseUrl(baseUrlsInfo.sharedUrl as string);
                      setHealthProbe({ checking: false, ok: null, message: null });
                    }}
                    className={`px-3 py-1 rounded-full font-medium transition-colors ${
                      baseUrl === baseUrlsInfo.sharedUrl
                        ? 'bg-[#007aff] text-white dark:bg-[#0a84ff]'
                        : 'bg-black/[0.05] dark:bg-white/[0.08] text-neutral-600 dark:text-neutral-300'
                    }`}
                  >
                    Shared URL (ais-pre-*)
                  </button>
                )}
                {baseUrlsInfo.current !== baseUrlsInfo.sharedUrl && (
                  <button
                    type="button"
                    onClick={() => {
                      try {
                        localStorage.removeItem(CUSTOM_URL_STORAGE_KEY);
                      } catch {
                        // Ignore
                      }
                      setCustomUrlInput('');
                      setHasCustomUrlChoice(true);
                      setBaseUrl(baseUrlsInfo.current);
                      setHealthProbe({ checking: false, ok: null, message: null });
                    }}
                    className={`px-3 py-1 rounded-full font-medium transition-colors ${
                      baseUrl === baseUrlsInfo.current
                        ? 'bg-[#007aff] text-white dark:bg-[#0a84ff]'
                        : 'bg-black/[0.05] dark:bg-white/[0.08] text-neutral-600 dark:text-neutral-300'
                    }`}
                  >
                    {baseUrlsInfo.isCurrentDevPreview ? 'Dev Preview (ais-dev-*)' : 'Current Origin'}
                  </button>
                )}
              </div>
            )}

            <form onSubmit={handleApplyCustomUrl} className="flex flex-col sm:flex-row gap-2">
              <label htmlFor="custom-base-url-input" className="sr-only">
                Deployed or custom Base URL
              </label>
              <input
                id="custom-base-url-input"
                type="url"
                value={customUrlInput}
                onChange={(e) => setCustomUrlInput(e.target.value)}
                placeholder="Paste deployed Cloud Run URL (e.g. https://your-service.run.app)"
                className="flex-1 px-3.5 py-2 rounded-2xl border border-black/[0.08] dark:border-white/[0.1] bg-[#f5f5f7] dark:bg-[#121214] text-xs font-mono text-neutral-900 dark:text-white focus:outline-none focus:border-[#007aff]"
              />
              <button
                type="submit"
                className="px-4 py-2 rounded-full bg-neutral-900 hover:bg-neutral-800 dark:bg-white dark:hover:bg-neutral-200 text-white dark:text-neutral-900 text-xs font-medium transition-colors whitespace-nowrap shrink-0"
              >
                {customUrlInput.trim() ? 'Use Custom URL' : 'Reset Auto URL'}
              </button>
            </form>

            {healthProbe.message && (
              <div
                data-testid="health-probe-result"
                className={`p-3.5 rounded-2xl border text-xs leading-relaxed ${
                  healthProbe.ok
                    ? 'bg-emerald-50/90 dark:bg-emerald-950/30 border-emerald-200 dark:border-emerald-800/60 text-emerald-900 dark:text-emerald-200'
                    : 'bg-amber-50/90 dark:bg-amber-950/30 border-amber-200 dark:border-amber-800/60 text-amber-900 dark:text-amber-200'
                }`}
              >
                {healthProbe.message}
              </div>
            )}

            {currentUrlCookieGated && (
              <div
                data-testid="dev-proxy-notice"
                className="p-3.5 rounded-2xl bg-amber-50/90 dark:bg-amber-950/30 border border-amber-200/80 dark:border-amber-800/60 text-xs text-amber-900 dark:text-amber-200 leading-relaxed"
              >
                <strong>Why external tools see HTTP 302 → /__cookie_check.html on ais-dev-* / ais-pre-* preview URLs:</strong>{' '}
                Both AI Studio preview hostnames sit behind Google’s session-cookie proxy, so external tools (<code className="font-mono">curl</code>, Python, mobile/backend apps) without browser login cookies get redirected before reaching Express. Click <strong>Deploy App</strong> in AI Studio to publish a direct Cloud Run <code className="font-mono">*.run.app</code> service URL (and paste it above), or use <strong>Test Live API Request</strong> below right now.
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {endpointRows.map((row) => {
              const fullUrl = `${baseUrl}${row.path}`;
              return (
                <div
                  key={row.path}
                  className="p-3.5 rounded-2xl bg-[#f5f5f7] dark:bg-[#121214] flex items-start justify-between gap-2"
                >
                  <div className="min-w-0 space-y-1">
                    <div className="flex items-center gap-2 text-xs">
                      <span className="font-mono font-semibold text-[#007aff] dark:text-[#0a84ff]">
                        {row.method}
                      </span>
                      <span aria-hidden="true" className="text-neutral-400">·</span>
                      <code className="font-mono text-neutral-900 dark:text-neutral-100 truncate">
                        {row.path}
                      </code>
                    </div>
                    <p className="text-xs text-neutral-500 dark:text-neutral-400">{row.desc}</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => copyText(fullUrl, `ep-${row.path}`)}
                    aria-label={`Copy ${row.path} URL`}
                    className="p-2 rounded-full text-neutral-500 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-white hover:bg-black/[0.05] dark:hover:bg-white/[0.08] transition-colors shrink-0"
                  >
                    {copiedId === `ep-${row.path}` ? (
                      <Check className="w-3.5 h-3.5 text-emerald-500" />
                    ) : (
                      <Copy className="w-3.5 h-3.5" />
                    )}
                  </button>
                </div>
              );
            })}
          </div>
        </section>

        {/* Generate & Manage API Keys */}
        <section className={`${surfaceCard} p-5 space-y-4`}>
          <div className="flex items-center gap-2">
            <KeyRound className="w-4 h-4 text-[#007aff] dark:text-[#0a84ff]" />
            <h2 className="text-sm font-semibold text-neutral-900 dark:text-white">
              Create &amp; Manage Qowe API Keys
            </h2>
          </div>

          <form onSubmit={handleCreateKey} className="flex flex-col sm:flex-row gap-2.5">
            <label htmlFor="new-api-key-name" className="sr-only">
              API key name
            </label>
            <input
              id="new-api-key-name"
              type="text"
              value={newKeyName}
              onChange={(e) => setNewKeyName(e.target.value)}
              placeholder="Key name (e.g. Friend Access, Mobile App, Discord Bot)"
              maxLength={60}
              className="flex-1 px-4 py-2.5 rounded-2xl border border-black/[0.08] dark:border-white/[0.1] bg-[#f5f5f7] dark:bg-[#121214] text-sm text-neutral-900 dark:text-white focus:outline-none focus:border-[#007aff] dark:focus:border-[#0a84ff]"
            />
            <button
              type="submit"
              disabled={!newKeyName.trim() || isCreating}
              className="inline-flex items-center justify-center gap-1.5 px-5 py-2.5 rounded-full bg-[#007aff] hover:bg-[#0066d6] dark:bg-[#0a84ff] dark:hover:bg-[#1a8cff] disabled:opacity-50 text-white text-sm font-medium transition-colors whitespace-nowrap"
            >
              <Plus className="w-4 h-4" />
              <span>{isCreating ? 'Creating...' : 'Create API Key'}</span>
            </button>
          </form>

          {revealedKey && (
            <div className="p-4 rounded-2xl border border-emerald-300/70 dark:border-emerald-800/70 bg-emerald-50/80 dark:bg-emerald-950/30 space-y-2.5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-xs font-semibold text-emerald-900 dark:text-emerald-200">
                  New API Key Generated (copy it now — only its SHA-256 hash is kept on the server)
                </span>
                <button
                  type="button"
                  onClick={() => copyText(revealedKey, 'new-key')}
                  className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-700 hover:bg-emerald-800 text-white text-xs font-medium whitespace-nowrap"
                >
                  {copiedId === 'new-key' ? (
                    <Check className="w-3.5 h-3.5" />
                  ) : (
                    <Copy className="w-3.5 h-3.5" />
                  )}
                  <span>{copiedId === 'new-key' ? 'Copied' : 'Copy Key'}</span>
                </button>
              </div>
              <code
                data-testid="revealed-qowe-key"
                className="block p-3 rounded-xl bg-white dark:bg-[#0b0b0d] border border-emerald-200 dark:border-emerald-900 font-mono text-xs text-neutral-900 dark:text-emerald-200 break-all"
              >
                {revealedKey}
              </code>
            </div>
          )}

          {errorMsg && (
            <div className="p-3.5 rounded-2xl bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-900 text-xs text-red-700 dark:text-red-300">
              {errorMsg}
            </div>
          )}

          {isLoading ? (
            <p className="text-xs text-neutral-500">Loading active keys...</p>
          ) : keys.length === 0 ? (
            <div className="p-5 rounded-2xl bg-[#f5f5f7] dark:bg-[#121214] text-center text-xs text-neutral-500 dark:text-neutral-400">
              No Qowe API keys created yet. Create one above to authenticate external requests.
            </div>
          ) : (
            <div className="divide-y divide-black/[0.06] dark:divide-white/[0.08] rounded-2xl bg-[#f5f5f7] dark:bg-[#121214] overflow-hidden">
              {keys.map((k) => (
                <div
                  key={k.id}
                  className="px-4 py-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-2"
                >
                  <div className="space-y-0.5 min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-sm font-medium text-neutral-900 dark:text-white">
                        {k.name}
                      </span>
                      <span aria-hidden="true" className="text-neutral-400">·</span>
                      <code className="text-xs font-mono text-neutral-500 dark:text-neutral-400">
                        {k.prefix}
                      </code>
                    </div>
                    <div className="text-xs text-neutral-500 dark:text-neutral-400 tabular-nums">
                      Requests: {k.requestCount} · Avg Latency:{' '}
                      {k.avgLatencyMs && k.avgLatencyMs > 0 ? `${k.avgLatencyMs} ms` : '—'} · Last:{' '}
                      {k.lastLatencyMs !== null && k.lastLatencyMs !== undefined
                        ? `${k.lastLatencyMs} ms`
                        : '—'}{' '}
                      · Created {new Date(k.createdAt).toLocaleDateString()}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => handleRevokeKey(k.id)}
                    aria-label={`Revoke key ${k.name}`}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-red-500/10 hover:bg-red-500/20 text-xs font-medium text-red-600 dark:text-red-400 self-start sm:self-auto whitespace-nowrap transition-colors"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    <span>Revoke</span>
                  </button>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Code Snippets & Live Tester */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <section className={`${surfaceCard} p-5 space-y-3.5`}>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <Terminal className="w-4 h-4 text-[#007aff] dark:text-[#0a84ff]" />
                <h2 className="text-sm font-semibold text-neutral-900 dark:text-white">
                  Integration Snippets
                </h2>
              </div>

              <div className="flex items-center gap-1 p-1 rounded-full bg-black/[0.05] dark:bg-white/[0.08] text-xs">
                {(['curl', 'javascript', 'python'] as SnippetLang[]).map((lang) => (
                  <button
                    key={lang}
                    type="button"
                    onClick={() => setSnippetLang(lang)}
                    className={`px-3 py-1 rounded-full font-medium capitalize transition-colors ${
                      snippetLang === lang
                        ? 'bg-white dark:bg-[#2c2c2e] text-neutral-900 dark:text-white shadow-xs'
                        : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white'
                    }`}
                  >
                    {lang === 'curl' ? 'cURL' : lang}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
              <div className="flex items-center gap-1.5">
                {(['chat', 'image', 'video'] as const).map((kind) => (
                  <button
                    key={kind}
                    type="button"
                    onClick={() => setSnippetEndpoint(kind)}
                    className={`px-3.5 py-1.5 rounded-full font-medium transition-colors ${
                      snippetEndpoint === kind
                        ? 'bg-[#007aff] text-white dark:bg-[#0a84ff]'
                        : 'bg-black/[0.05] dark:bg-white/[0.08] text-neutral-600 dark:text-neutral-300 hover:bg-black/[0.08] dark:hover:bg-white/[0.12]'
                    }`}
                  >
                    {kind === 'chat' ? 'Chat' : kind === 'image' ? 'Image' : 'Video (60 fps)'}
                  </button>
                ))}
              </div>

              <div
                role="group"
                aria-label="Snippet auth header format"
                className="flex items-center gap-1 p-0.5 rounded-full bg-black/[0.05] dark:bg-white/[0.08]"
              >
                <button
                  type="button"
                  onClick={() => setSnippetAuthMode('x-api-key')}
                  className={`px-2.5 py-1 rounded-full font-mono text-[11px] font-medium transition-colors ${
                    snippetAuthMode === 'x-api-key'
                      ? 'bg-white dark:bg-[#2c2c2e] text-neutral-900 dark:text-white shadow-xs'
                      : 'text-neutral-600 dark:text-neutral-400'
                  }`}
                >
                  x-api-key
                </button>
                <button
                  type="button"
                  onClick={() => setSnippetAuthMode('bearer')}
                  className={`px-2.5 py-1 rounded-full font-mono text-[11px] font-medium transition-colors ${
                    snippetAuthMode === 'bearer'
                      ? 'bg-white dark:bg-[#2c2c2e] text-neutral-900 dark:text-white shadow-xs'
                      : 'text-neutral-600 dark:text-neutral-400'
                  }`}
                >
                  Bearer
                </button>
              </div>
            </div>

            <div className="rounded-2xl bg-[#121214] border border-white/[0.08] overflow-hidden">
              <div className="px-3.5 py-2 border-b border-white/[0.08] flex items-center justify-between gap-2 text-xs font-mono">
                <span className="text-neutral-400 truncate">
                  {snippetLang === 'curl' ? 'cURL' : snippetLang === 'javascript' ? 'JavaScript' : 'Python'} ·{' '}
                  {snippetEndpoint === 'chat'
                    ? '/api/v1/chat/completions'
                    : snippetEndpoint === 'image'
                      ? '/api/v1/images/generations'
                      : '/api/v1/videos/generations'}
                </span>
                <button
                  type="button"
                  onClick={() => copyText(getSnippet(), 'snippet-copy')}
                  className="inline-flex items-center gap-1 px-3 py-1 rounded-full bg-white/10 hover:bg-white/20 text-xs font-sans font-medium text-neutral-200 transition-colors shrink-0"
                >
                  {copiedId === 'snippet-copy' ? (
                    <Check className="w-3 h-3 text-emerald-400" />
                  ) : (
                    <Copy className="w-3 h-3" />
                  )}
                  <span>{copiedId === 'snippet-copy' ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
              <pre className="p-4 overflow-x-auto text-xs font-mono text-neutral-200 leading-relaxed">
                <code>{getSnippet()}</code>
              </pre>
            </div>
          </section>

          <section className={`${surfaceCard} p-5 space-y-3.5`}>
            <h2 className="text-sm font-semibold text-neutral-900 dark:text-white">
              Test Live API Request
            </h2>

            <form onSubmit={handleRunTest} className="space-y-3">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                <div>
                  <label
                    htmlFor="api-test-endpoint"
                    className="block text-xs font-medium text-neutral-600 dark:text-neutral-400 mb-1"
                  >
                    Endpoint
                  </label>
                  <select
                    id="api-test-endpoint"
                    value={testEndpoint}
                    onChange={(e) => setTestEndpoint(e.target.value as TestEndpointKind)}
                    className="w-full px-3 py-2 rounded-2xl border border-black/[0.08] dark:border-white/[0.1] bg-[#f5f5f7] dark:bg-[#121214] text-xs text-neutral-900 dark:text-white focus:outline-none focus:border-[#007aff]"
                  >
                    <option value="chat">POST /api/v1/chat/completions</option>
                    <option value="image">POST /api/v1/images/generations</option>
                    <option value="video">POST /api/v1/videos/generations</option>
                    <option value="models">GET /api/v1/models</option>
                  </select>
                </div>

                <div>
                  <label
                    htmlFor="api-test-key"
                    className="block text-xs font-medium text-neutral-600 dark:text-neutral-400 mb-1"
                  >
                    Qowe API Key (x-api-key)
                  </label>
                  <input
                    id="api-test-key"
                    type="text"
                    value={testKey}
                    onChange={(e) => setTestKey(e.target.value)}
                    placeholder="Paste qowe_sk_... key"
                    className="w-full px-3 py-2 rounded-2xl border border-black/[0.08] dark:border-white/[0.1] bg-[#f5f5f7] dark:bg-[#121214] text-xs font-mono text-neutral-900 dark:text-white focus:outline-none focus:border-[#007aff]"
                  />
                </div>
              </div>

              <div>
                <label
                  htmlFor="api-test-prompt"
                  className="block text-xs font-medium text-neutral-600 dark:text-neutral-400 mb-1"
                >
                  {testEndpoint === 'models' ? 'Request Body (None for GET)' : 'Prompt'}
                </label>
                <div className="flex gap-2">
                  <input
                    id="api-test-prompt"
                    type="text"
                    disabled={testEndpoint === 'models'}
                    value={testEndpoint === 'models' ? 'GET /api/v1/models' : testPrompt}
                    onChange={(e) => setTestPrompt(e.target.value)}
                    className="flex-1 px-3.5 py-2 rounded-2xl border border-black/[0.08] dark:border-white/[0.1] bg-[#f5f5f7] dark:bg-[#121214] disabled:opacity-60 text-xs text-neutral-900 dark:text-white focus:outline-none focus:border-[#007aff]"
                  />
                  <button
                    type="submit"
                    disabled={isTesting || (testEndpoint !== 'models' && !testPrompt.trim())}
                    className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full bg-[#007aff] hover:bg-[#0066d6] dark:bg-[#0a84ff] dark:hover:bg-[#1a8cff] disabled:opacity-50 text-white text-xs font-medium transition-colors whitespace-nowrap"
                  >
                    <Play className="w-3.5 h-3.5" />
                    <span>{isTesting ? 'Calling...' : 'Send Test'}</span>
                  </button>
                </div>
              </div>
            </form>

            {testOutput && (
              <div className="rounded-2xl border border-white/[0.08] bg-[#121214] overflow-hidden">
                <div className="px-3.5 py-2 border-b border-white/[0.08] flex flex-wrap items-center justify-between gap-2 text-xs font-mono tabular-nums">
                  <span className="text-neutral-400">
                    {testOutput.routeLabel} · {testOutput.latencyMs} ms round-trip
                  </span>
                  <span
                    className={
                      testOutput.status >= 200 && testOutput.status < 300
                        ? 'text-emerald-400'
                        : 'text-amber-400'
                    }
                  >
                    HTTP {testOutput.status}
                  </span>
                </div>
                <pre
                  data-testid="api-test-output"
                  className="p-3.5 max-h-56 overflow-auto text-xs font-mono text-neutral-200 leading-relaxed"
                >
                  {testOutput.body}
                </pre>
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
};
