import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';

export interface QoweApiKeyRecord {
  id: string;
  name: string;
  prefix: string;
  keyHash: string;
  createdAt: string;
  lastUsedAt: string | null;
  requestCount: number;
  totalLatencyMs?: number;
  lastLatencyMs?: number | null;
}

export interface PublicQoweApiKeyRecord {
  id: string;
  name: string;
  prefix: string;
  createdAt: string;
  lastUsedAt: string | null;
  requestCount: number;
  avgLatencyMs: number;
  lastLatencyMs: number | null;
}

const KEY_PREFIX = 'qowe_sk_';
const MAX_KEYS = 25;

export function hashQoweApiKey(rawKey: string): string {
  return crypto.createHash('sha256').update(rawKey, 'utf8').digest('hex');
}

export function sanitizeKeyName(name: unknown): string {
  if (typeof name !== 'string') return '';
  return name.trim().replace(/\s+/g, ' ').slice(0, 60);
}

function toPublicRecord(rec: QoweApiKeyRecord): PublicQoweApiKeyRecord {
  const totalLat = typeof rec.totalLatencyMs === 'number' ? rec.totalLatencyMs : 0;
  return {
    id: rec.id,
    name: rec.name,
    prefix: rec.prefix,
    createdAt: rec.createdAt,
    lastUsedAt: rec.lastUsedAt,
    requestCount: rec.requestCount,
    avgLatencyMs: rec.requestCount > 0 && totalLat > 0 ? Math.round(totalLat / rec.requestCount) : 0,
    lastLatencyMs: typeof rec.lastLatencyMs === 'number' ? rec.lastLatencyMs : null,
  };
}

export interface QoweKeyStoreOptions {
  allowStatelessSigned?: boolean;
  signingSecret?: string;
}

export class QoweApiKeyStore {
  private readonly filePath: string;
  private readonly allowStatelessSigned: boolean;
  private readonly signingSecret: string;
  private records: QoweApiKeyRecord[] = [];
  private loaded = false;
  private writeChain: Promise<void> = Promise.resolve();

  constructor(storeFilePath?: string, options?: QoweKeyStoreOptions) {
    this.filePath = storeFilePath ?? path.resolve(process.cwd(), '.qowe-keys.json');
    this.allowStatelessSigned = Boolean(options?.allowStatelessSigned);
    this.signingSecret =
      options?.signingSecret ||
      process.env.QOWE_KEY_SECRET ||
      process.env.SITE_ID ||
      'qowe-1.0-self-hosted-signing-secret';
  }

  private signNonce(nonceHex: string): string {
    return crypto
      .createHmac('sha256', this.signingSecret)
      .update(nonceHex, 'utf8')
      .digest('hex')
      .slice(0, 16);
  }

  private verifySignedStatelessKey(rawKey: string): PublicQoweApiKeyRecord | null {
    if (!this.allowStatelessSigned) return null;
    if (!rawKey.startsWith(KEY_PREFIX)) return null;
    const body = rawKey.slice(KEY_PREFIX.length);
    const parts = body.split('_');
    if (parts.length !== 2) return null;
    const [nonceHex, providedSig] = parts;
    if (!/^[a-f0-9]{24}$/i.test(nonceHex) || !/^[a-f0-9]{16}$/i.test(providedSig)) {
      return null;
    }
    const expectedSig = this.signNonce(nonceHex);
    const a = Buffer.from(providedSig, 'hex');
    const b = Buffer.from(expectedSig, 'hex');
    if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) {
      return null;
    }
    return {
      id: `key_${nonceHex.slice(0, 16)}`,
      name: 'Qowe Netlify Key',
      prefix: `${rawKey.slice(0, 14)}...${rawKey.slice(-4)}`,
      createdAt: new Date().toISOString(),
      lastUsedAt: new Date().toISOString(),
      requestCount: 1,
      avgLatencyMs: 0,
      lastLatencyMs: null,
    };
  }

  private async ensureLoaded(): Promise<void> {
    if (this.loaded) return;
    this.loaded = true;
    try {
      const raw = await fs.readFile(this.filePath, 'utf8');
      const parsed = JSON.parse(raw) as QoweApiKeyRecord[];
      if (Array.isArray(parsed)) {
        this.records = parsed.filter(
          (r) =>
            r &&
            typeof r.id === 'string' &&
            typeof r.name === 'string' &&
            typeof r.prefix === 'string' &&
            typeof r.keyHash === 'string' &&
            r.keyHash.length === 64,
        );
      }
    } catch {
      this.records = [];
    }
  }

  private async persist(): Promise<void> {
    const snapshot = JSON.stringify(this.records, null, 2);
    this.writeChain = this.writeChain.then(async () => {
      try {
        const dir = path.dirname(this.filePath);
        await fs.mkdir(dir, { recursive: true });
        await fs.writeFile(this.filePath, snapshot, {
          encoding: 'utf8',
          mode: 0o600,
        });
      } catch {
        // Non-fatal in read-only environments
      }
    });
    await this.writeChain;
  }

  async listKeys(): Promise<PublicQoweApiKeyRecord[]> {
    await this.ensureLoaded();
    return this.records.map(toPublicRecord);
  }

  async createKey(nameInput: unknown): Promise<{ record: PublicQoweApiKeyRecord; rawKey: string }> {
    await this.ensureLoaded();
    const cleanName = sanitizeKeyName(nameInput);
    if (!cleanName) {
      throw new Error('API key name is required (1-60 characters).');
    }
    if (this.records.length >= MAX_KEYS) {
      throw new Error(`Maximum limit of ${MAX_KEYS} active API keys reached. Revoke an unused key first.`);
    }

    const rawKey = this.allowStatelessSigned
      ? (() => {
          const nonceHex = crypto.randomBytes(12).toString('hex');
          const sigHex = this.signNonce(nonceHex);
          return `${KEY_PREFIX}${nonceHex}_${sigHex}`;
        })()
      : `${KEY_PREFIX}${crypto.randomBytes(24).toString('hex')}`;

    const keyHash = hashQoweApiKey(rawKey);
    const prefix = `${rawKey.slice(0, 14)}...${rawKey.slice(-4)}`;

    const record: QoweApiKeyRecord = {
      id: `key_${crypto.randomBytes(8).toString('hex')}`,
      name: cleanName,
      prefix,
      keyHash,
      createdAt: new Date().toISOString(),
      lastUsedAt: null,
      requestCount: 0,
      totalLatencyMs: 0,
      lastLatencyMs: null,
    };

    this.records.unshift(record);
    await this.persist();

    return {
      record: toPublicRecord(record),
      rawKey,
    };
  }

  async revokeKey(id: string): Promise<boolean> {
    await this.ensureLoaded();
    const idx = this.records.findIndex((r) => r.id === id);
    if (idx === -1) return false;
    this.records.splice(idx, 1);
    await this.persist();
    return true;
  }

  async verifyAndTouch(rawKey: string | undefined): Promise<PublicQoweApiKeyRecord | null> {
    if (!rawKey || !rawKey.startsWith(KEY_PREFIX) || rawKey.length < 24) {
      return null;
    }
    await this.ensureLoaded();
    const incomingHashBuf = Buffer.from(hashQoweApiKey(rawKey), 'hex');

    let matched: QoweApiKeyRecord | null = null;
    for (const rec of this.records) {
      const storedBuf = Buffer.from(rec.keyHash, 'hex');
      if (
        storedBuf.length === incomingHashBuf.length &&
        crypto.timingSafeEqual(storedBuf, incomingHashBuf)
      ) {
        matched = rec;
      }
    }

    if (matched) {
      matched.requestCount += 1;
      matched.lastUsedAt = new Date().toISOString();
      void this.persist();
      return toPublicRecord(matched);
    }

    return this.verifySignedStatelessKey(rawKey);
  }

  async recordLatency(id: string, latencyMs: number): Promise<void> {
    await this.ensureLoaded();
    const rec = this.records.find((r) => r.id === id);
    if (!rec) return;
    const safeLat = Math.max(1, Math.round(latencyMs));
    rec.totalLatencyMs = (rec.totalLatencyMs ?? 0) + safeLat;
    rec.lastLatencyMs = safeLat;
    void this.persist();
  }
}

export function extractApiKeyFromHeaders(headers: {
  authorization?: string | string[];
  'x-api-key'?: string | string[];
  'x-qowe-api-key'?: string | string[];
}): string | undefined {
  const qKey = Array.isArray(headers['x-qowe-api-key'])
    ? headers['x-qowe-api-key'][0]
    : headers['x-qowe-api-key'];
  if (qKey && qKey.trim()) return qKey.trim();

  const xKey = Array.isArray(headers['x-api-key']) ? headers['x-api-key'][0] : headers['x-api-key'];
  if (xKey && xKey.trim()) return xKey.trim();

  const auth = Array.isArray(headers.authorization) ? headers.authorization[0] : headers.authorization;
  if (auth && /^Bearer\s+/i.test(auth)) {
    return auth.replace(/^Bearer\s+/i, '').trim();
  }
  return undefined;
}
