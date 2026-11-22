import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';

export type ApiEndpointCategory = 'chat' | 'image' | 'video' | 'models';

export interface ApiRequestEvent {
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

export interface CategoryStats {
  count: number;
  avgLatencyMs: number;
  lastLatencyMs: number | null;
}

export interface ApiUsageSummary {
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

interface PersistedUsageState {
  totalRequests: number;
  successCount: number;
  errorCount: number;
  latencySamples: number[];
  categoryCounts: Record<ApiEndpointCategory, number>;
  categoryLatencySamples: Record<ApiEndpointCategory, number[]>;
  recentEvents: ApiRequestEvent[];
  updatedAt: string;
}

const MAX_SAMPLES = 200;
const MAX_EVENTS = 15;

function createEmptyState(): PersistedUsageState {
  return {
    totalRequests: 0,
    successCount: 0,
    errorCount: 0,
    latencySamples: [],
    categoryCounts: { chat: 0, image: 0, video: 0, models: 0 },
    categoryLatencySamples: { chat: [], image: [], video: [], models: [] },
    recentEvents: [],
    updatedAt: new Date().toISOString(),
  };
}

export function computePercentile(values: number[], percentile: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((percentile / 100) * sorted.length) - 1));
  return Math.round(sorted[idx]);
}

export function computeAverage(values: number[]): number {
  if (values.length === 0) return 0;
  const sum = values.reduce((acc, v) => acc + v, 0);
  return Math.round(sum / values.length);
}

export class QoweUsageTracker {
  private readonly filePath: string;
  private state: PersistedUsageState = createEmptyState();
  private loaded = false;
  private initPromise: Promise<void> | null = null;
  private writeChain: Promise<void> = Promise.resolve();

  constructor(storeFilePath?: string) {
    this.filePath = storeFilePath ?? path.resolve(process.cwd(), '.qowe-usage.json');
  }

  private async ensureLoaded(): Promise<void> {
    if (this.loaded) return;
    if (!this.initPromise) {
      this.initPromise = (async () => {
        try {
          const raw = await fs.readFile(this.filePath, 'utf8');
          const parsed = JSON.parse(raw) as Partial<PersistedUsageState>;
          if (parsed && typeof parsed.totalRequests === 'number') {
            this.state = {
              ...createEmptyState(),
              ...parsed,
              categoryCounts: {
                ...createEmptyState().categoryCounts,
                ...(parsed.categoryCounts ?? {}),
              },
              categoryLatencySamples: {
                ...createEmptyState().categoryLatencySamples,
                ...(parsed.categoryLatencySamples ?? {}),
              },
              latencySamples: Array.isArray(parsed.latencySamples)
                ? parsed.latencySamples.slice(-MAX_SAMPLES)
                : [],
              recentEvents: Array.isArray(parsed.recentEvents)
                ? parsed.recentEvents.slice(0, MAX_EVENTS)
                : [],
            };
          }
        } catch {
          this.state = createEmptyState();
        } finally {
          this.loaded = true;
        }
      })();
    }
    await this.initPromise;
  }

  private async persist(): Promise<void> {
    this.writeChain = this.writeChain.then(async () => {
      try {
        await fs.mkdir(path.dirname(this.filePath), { recursive: true });
        await fs.writeFile(this.filePath, JSON.stringify(this.state, null, 2), {
          encoding: 'utf8',
          mode: 0o600,
        });
      } catch {
        // Non-fatal persistence error
      }
    });
    await this.writeChain;
  }

  async recordRequest(input: {
    endpoint: string;
    category: ApiEndpointCategory;
    model: string;
    keyPrefix: string;
    keyName: string;
    status: number;
    latencyMs: number;
  }): Promise<void> {
    if (!this.loaded) {
      await this.ensureLoaded();
    }
    const roundedLatency = Math.max(1, Math.round(input.latencyMs));
    this.state.totalRequests += 1;
    if (input.status >= 200 && input.status < 400) {
      this.state.successCount += 1;
    } else {
      this.state.errorCount += 1;
    }

    this.state.latencySamples.push(roundedLatency);
    if (this.state.latencySamples.length > MAX_SAMPLES) {
      this.state.latencySamples.shift();
    }

    this.state.categoryCounts[input.category] = (this.state.categoryCounts[input.category] ?? 0) + 1;
    const catSamples = this.state.categoryLatencySamples[input.category] ?? [];
    catSamples.push(roundedLatency);
    if (catSamples.length > MAX_SAMPLES) {
      catSamples.shift();
    }
    this.state.categoryLatencySamples[input.category] = catSamples;

    const event: ApiRequestEvent = {
      id: `req_${crypto.randomBytes(6).toString('hex')}`,
      endpoint: input.endpoint,
      category: input.category,
      model: input.model,
      keyPrefix: input.keyPrefix,
      keyName: input.keyName,
      status: input.status,
      latencyMs: roundedLatency,
      timestamp: new Date().toISOString(),
    };
    this.state.recentEvents.unshift(event);
    if (this.state.recentEvents.length > MAX_EVENTS) {
      this.state.recentEvents.length = MAX_EVENTS;
    }
    this.state.updatedAt = event.timestamp;
    await this.persist();
  }

  async getSummary(): Promise<ApiUsageSummary> {
    await this.ensureLoaded();
    const samples = this.state.latencySamples;
    const minLatencyMs = samples.length > 0 ? Math.min(...samples) : null;
    const maxLatencyMs = samples.length > 0 ? Math.max(...samples) : null;
    const lastLatencyMs = samples.length > 0 ? samples[samples.length - 1] : null;
    const successRate =
      this.state.totalRequests > 0
        ? Math.round((this.state.successCount / this.state.totalRequests) * 1000) / 10
        : 100;

    const makeCategoryStats = (cat: ApiEndpointCategory): CategoryStats => {
      const catSamples = this.state.categoryLatencySamples[cat] ?? [];
      return {
        count: this.state.categoryCounts[cat] ?? 0,
        avgLatencyMs: computeAverage(catSamples),
        lastLatencyMs: catSamples.length > 0 ? catSamples[catSamples.length - 1] : null,
      };
    };

    return {
      totalRequests: this.state.totalRequests,
      successCount: this.state.successCount,
      errorCount: this.state.errorCount,
      successRate,
      avgLatencyMs: computeAverage(samples),
      p95LatencyMs: computePercentile(samples, 95),
      minLatencyMs,
      maxLatencyMs,
      lastLatencyMs,
      byCategory: {
        chat: makeCategoryStats('chat'),
        image: makeCategoryStats('image'),
        video: makeCategoryStats('video'),
        models: makeCategoryStats('models'),
      },
      recentEvents: this.state.recentEvents,
      updatedAt: this.state.updatedAt,
    };
  }
}
