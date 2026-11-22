import { decode, encode, QoweTokenizer } from './tokenizer';

export interface QoweTrainingHyperparams {
  dModel: number;
  nLayers: number;
  nHeads: number;
  dFF: number;
  contextLength: number;
  learningRate: number;
  totalSteps: number;
  seed: number;
}

export interface QoweCheckpoint {
  version: string;
  createdAt: string;
  step: number;
  loss: number;
  perplexity: number;
  hyperparams: QoweTrainingHyperparams;
  vocabSize: number;
  activeParamCount: number;
  tokenizer: QoweTokenizer;
  weights: {
    wte: number[];
    wpe: number[];
    lnFGamma: number[];
    lnFBeta: number[];
    lmHead: number[];
    lmBias: number[];
    blocks: {
      ln1Gamma: number[];
      ln1Beta: number[];
      wq: number[];
      wk: number[];
      wv: number[];
      wo: number[];
      ln2Gamma: number[];
      ln2Beta: number[];
      w1: number[];
      b1: number[];
      w2: number[];
      b2: number[];
    }[];
  };
  history: TrainingStepMetric[];
}

export interface TrainingStepMetric {
  step: number;
  loss: number;
  perplexity: number;
  gradNorm: number;
  learningRate: number;
  tokensPerSec: number;
}

export interface TokenCandidate {
  id: number;
  token: string;
  probability: number;
}

export interface GenerationResult {
  text: string;
  generatedTokens: number;
  topCandidatesAtLastStep: TokenCandidate[];
}

interface BlockBuffers {
  ln1Gamma: Float32Array;
  ln1Beta: Float32Array;
  wq: Float32Array;
  wk: Float32Array;
  wv: Float32Array;
  wo: Float32Array;
  ln2Gamma: Float32Array;
  ln2Beta: Float32Array;
  w1: Float32Array;
  b1: Float32Array;
  w2: Float32Array;
  b2: Float32Array;
}

interface ModelState {
  vocabSize: number;
  hyperparams: QoweTrainingHyperparams;
  wte: Float32Array;
  wpe: Float32Array;
  lnFGamma: Float32Array;
  lnFBeta: Float32Array;
  lmHead: Float32Array;
  lmBias: Float32Array;
  blocks: BlockBuffers[];
}

function createRng(seed: number): () => number {
  let s = (seed >>> 0) || 1337;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function fillGaussian(arr: Float32Array, std: number, rand: () => number): void {
  for (let i = 0; i < arr.length; i += 2) {
    const u1 = Math.max(1e-7, rand());
    const u2 = rand();
    const mag = std * Math.sqrt(-2.0 * Math.log(u1));
    const z0 = mag * Math.cos(2.0 * Math.PI * u2);
    const z1 = mag * Math.sin(2.0 * Math.PI * u2);
    arr[i] = z0;
    if (i + 1 < arr.length) arr[i + 1] = z1;
  }
}

function gelu(x: number): number {
  const c = 0.7978845608028654;
  const inner = c * (x + 0.044715 * x * x * x);
  return 0.5 * x * (1 + Math.tanh(inner));
}

function geluBackward(x: number): number {
  const c = 0.7978845608028654;
  const x3 = x * x * x;
  const inner = c * (x + 0.044715 * x3);
  const t = Math.tanh(inner);
  const sech2 = 1 - t * t;
  const dInner = c * (1 + 3 * 0.044715 * x * x);
  return 0.5 * (1 + t) + 0.5 * x * sech2 * dInner;
}

export function countKernelParameters(vocabSize: number, hp: QoweTrainingHyperparams): number {
  const { dModel, nLayers, dFF, contextLength } = hp;
  const emb = vocabSize * dModel + contextLength * dModel;
  const perLayer = 4 * dModel * dModel + 2 * dModel * dFF + dFF + dModel + 4 * dModel;
  const lnF = 2 * dModel;
  const head = vocabSize * dModel + vocabSize;
  return emb + nLayers * perLayer + lnF + head;
}

export class QoweTrainerEngine {
  public readonly tokenizer: QoweTokenizer;
  public readonly hyperparams: QoweTrainingHyperparams;
  public readonly windows: number[][];
  public step = 0;
  public lastLoss = 0;
  public history: TrainingStepMetric[] = [];

  private state: ModelState;
  private grads: ModelState;
  private adamM: ModelState;
  private adamV: ModelState;
  private paramTuples: [Float32Array, Float32Array, Float32Array, Float32Array, boolean][];
  private windowIdx = 0;
  private rng: () => number;

  constructor(tokenizer: QoweTokenizer, corpus: string, hyperparams: QoweTrainingHyperparams) {
    this.tokenizer = tokenizer;
    this.hyperparams = { ...hyperparams };
    if (this.hyperparams.dModel % this.hyperparams.nHeads !== 0) {
      this.hyperparams.nHeads = Math.max(1, Math.min(4, this.hyperparams.dModel));
    }

    this.windows = this.buildTrainingWindows(corpus);
    this.rng = createRng(this.hyperparams.seed);

    const vocabSize = tokenizer.vocab.length;
    this.state = this.allocateModel(vocabSize, true);
    this.grads = this.allocateModel(vocabSize, false);
    this.adamM = this.allocateModel(vocabSize, false);
    this.adamV = this.allocateModel(vocabSize, false);
    this.paramTuples = this.collectParamTuples();
    this.lastLoss = Math.log(vocabSize);
  }

  private buildTrainingWindows(corpus: string): number[][] {
    const seqCap = this.hyperparams.contextLength + 1;
    const segments = corpus
      .split('<|endoftext|>')
      .map((s) => s.trim())
      .filter((s) => s.length > 0);

    const windows: number[][] = [];
    for (const seg of segments) {
      const ids = encode(this.tokenizer, seg);
      ids.push(this.tokenizer.specialTokens.endoftext);
      if (ids.length <= seqCap) {
        if (ids.length >= 3) windows.push(ids);
      } else {
        const stride = Math.max(4, Math.floor(this.hyperparams.contextLength / 2));
        for (let start = 0; start < ids.length - 2; start += stride) {
          const slice = ids.slice(start, start + seqCap);
          if (slice.length >= 3) windows.push(slice);
          if (start + seqCap >= ids.length) break;
        }
      }
    }

    if (windows.length === 0) {
      const fallback = encode(this.tokenizer, corpus || 'Hello Qowe 1.0');
      fallback.push(this.tokenizer.specialTokens.endoftext);
      while (fallback.length < 4) fallback.push(this.tokenizer.specialTokens.endoftext);
      windows.push(fallback.slice(0, seqCap));
    }

    return windows;
  }

  private allocateModel(vocabSize: number, initWeights: boolean): ModelState {
    const { dModel, nLayers, dFF, contextLength } = this.hyperparams;
    const wte = new Float32Array(vocabSize * dModel);
    const wpe = new Float32Array(contextLength * dModel);
    const lnFGamma = new Float32Array(dModel);
    const lnFBeta = new Float32Array(dModel);
    const lmHead = new Float32Array(vocabSize * dModel);
    const lmBias = new Float32Array(vocabSize);

    if (initWeights) {
      fillGaussian(wte, 0.08, this.rng);
      fillGaussian(wpe, 0.05, this.rng);
      fillGaussian(lmHead, 0.08, this.rng);
      lnFGamma.fill(1);
    }

    const projStd = 0.08 / Math.sqrt(Math.max(1, nLayers));
    const blocks: BlockBuffers[] = [];
    for (let l = 0; l < nLayers; l++) {
      const block: BlockBuffers = {
        ln1Gamma: new Float32Array(dModel),
        ln1Beta: new Float32Array(dModel),
        wq: new Float32Array(dModel * dModel),
        wk: new Float32Array(dModel * dModel),
        wv: new Float32Array(dModel * dModel),
        wo: new Float32Array(dModel * dModel),
        ln2Gamma: new Float32Array(dModel),
        ln2Beta: new Float32Array(dModel),
        w1: new Float32Array(dModel * dFF),
        b1: new Float32Array(dFF),
        w2: new Float32Array(dFF * dModel),
        b2: new Float32Array(dModel),
      };
      if (initWeights) {
        block.ln1Gamma.fill(1);
        block.ln2Gamma.fill(1);
        fillGaussian(block.wq, 0.08, this.rng);
        fillGaussian(block.wk, 0.08, this.rng);
        fillGaussian(block.wv, 0.08, this.rng);
        fillGaussian(block.wo, projStd, this.rng);
        fillGaussian(block.w1, 0.08, this.rng);
        fillGaussian(block.w2, projStd, this.rng);
      }
      blocks.push(block);
    }

    return {
      vocabSize,
      hyperparams: this.hyperparams,
      wte,
      wpe,
      lnFGamma,
      lnFBeta,
      lmHead,
      lmBias,
      blocks,
    };
  }

  private zeroGrads(): void {
    this.grads.wte.fill(0);
    this.grads.wpe.fill(0);
    this.grads.lnFGamma.fill(0);
    this.grads.lnFBeta.fill(0);
    this.grads.lmHead.fill(0);
    this.grads.lmBias.fill(0);
    for (const b of this.grads.blocks) {
      b.ln1Gamma.fill(0);
      b.ln1Beta.fill(0);
      b.wq.fill(0);
      b.wk.fill(0);
      b.wv.fill(0);
      b.wo.fill(0);
      b.ln2Gamma.fill(0);
      b.ln2Beta.fill(0);
      b.w1.fill(0);
      b.b1.fill(0);
      b.w2.fill(0);
      b.b2.fill(0);
    }
  }

  private layerNormForward(
    x: Float32Array,
    T: number,
    D: number,
    gamma: Float32Array,
    beta: Float32Array,
  ): { out: Float32Array; xHat: Float32Array; invStd: Float32Array } {
    const out = new Float32Array(T * D);
    const xHat = new Float32Array(T * D);
    const invStd = new Float32Array(T);
    const eps = 1e-5;

    for (let t = 0; t < T; t++) {
      const off = t * D;
      let mean = 0;
      for (let d = 0; d < D; d++) mean += x[off + d];
      mean /= D;

      let variance = 0;
      for (let d = 0; d < D; d++) {
        const diff = x[off + d] - mean;
        variance += diff * diff;
      }
      variance /= D;
      const istd = 1 / Math.sqrt(variance + eps);
      invStd[t] = istd;

      for (let d = 0; d < D; d++) {
        const norm = (x[off + d] - mean) * istd;
        xHat[off + d] = norm;
        out[off + d] = norm * gamma[d] + beta[d];
      }
    }
    return { out, xHat, invStd };
  }

  private layerNormBackward(
    dOut: Float32Array,
    xHat: Float32Array,
    invStd: Float32Array,
    T: number,
    D: number,
    gamma: Float32Array,
    dGamma: Float32Array,
    dBeta: Float32Array,
  ): Float32Array {
    const dX = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      const off = t * D;
      let sumDY = 0;
      let sumDYHat = 0;
      for (let d = 0; d < D; d++) {
        const dy = dOut[off + d];
        const xh = xHat[off + d];
        dGamma[d] += dy * xh;
        dBeta[d] += dy;
        const dyNorm = dy * gamma[d];
        sumDY += dyNorm;
        sumDYHat += dyNorm * xh;
      }
      const scale = invStd[t] / D;
      for (let d = 0; d < D; d++) {
        const dyNorm = dOut[off + d] * gamma[d];
        dX[off + d] = scale * (D * dyNorm - sumDY - xHat[off + d] * sumDYHat);
      }
    }
    return dX;
  }

  private forwardSequence(inputs: number[]) {
    const T = inputs.length;
    const { dModel: D, nLayers, nHeads, dFF } = this.hyperparams;
    const headDim = D / nHeads;
    const scale = 1 / Math.sqrt(headDim);

    let x = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      const tokOff = inputs[t] * D;
      const posOff = t * D;
      for (let d = 0; d < D; d++) {
        x[posOff + d] = this.state.wte[tokOff + d] + this.state.wpe[posOff + d];
      }
    }
    const x0 = x;

    const layerCaches = [];
    for (let l = 0; l < nLayers; l++) {
      const b = this.state.blocks[l];
      const xIn = x;
      const ln1 = this.layerNormForward(xIn, T, D, b.ln1Gamma, b.ln1Beta);

      const q = new Float32Array(T * D);
      const k = new Float32Array(T * D);
      const v = new Float32Array(T * D);

      for (let t = 0; t < T; t++) {
        const xOff = t * D;
        for (let j = 0; j < D; j++) {
          let sq = 0;
          let sk = 0;
          let sv = 0;
          for (let i = 0; i < D; i++) {
            const xi = ln1.out[xOff + i];
            const wIdx = i * D + j;
            sq += xi * b.wq[wIdx];
            sk += xi * b.wk[wIdx];
            sv += xi * b.wv[wIdx];
          }
          q[xOff + j] = sq;
          k[xOff + j] = sk;
          v[xOff + j] = sv;
        }
      }

      const attnProbs = new Float32Array(nHeads * T * T);
      const attnCtx = new Float32Array(T * D);

      for (let h = 0; h < nHeads; h++) {
        const hOff = h * headDim;
        const pHeadOff = h * T * T;
        for (let t = 0; t < T; t++) {
          const qOff = t * D + hOff;
          let maxScore = -Infinity;
          const scores = new Float32Array(t + 1);
          for (let s = 0; s <= t; s++) {
            const kOff = s * D + hOff;
            let dot = 0;
            for (let d = 0; d < headDim; d++) {
              dot += q[qOff + d] * k[kOff + d];
            }
            const sc = dot * scale;
            scores[s] = sc;
            if (sc > maxScore) maxScore = sc;
          }

          let expSum = 0;
          for (let s = 0; s <= t; s++) {
            const e = Math.exp(scores[s] - maxScore);
            scores[s] = e;
            expSum += e;
          }
          const invSum = 1 / expSum;
          const pRowOff = pHeadOff + t * T;
          for (let s = 0; s <= t; s++) {
            const p = scores[s] * invSum;
            attnProbs[pRowOff + s] = p;
            const vOff = s * D + hOff;
            for (let d = 0; d < headDim; d++) {
              attnCtx[t * D + hOff + d] += p * v[vOff + d];
            }
          }
        }
      }

      const xMid = new Float32Array(T * D);
      for (let t = 0; t < T; t++) {
        const off = t * D;
        for (let j = 0; j < D; j++) {
          let sum = xIn[off + j];
          for (let i = 0; i < D; i++) {
            sum += attnCtx[off + i] * b.wo[i * D + j];
          }
          xMid[off + j] = sum;
        }
      }

      const ln2 = this.layerNormForward(xMid, T, D, b.ln2Gamma, b.ln2Beta);

      const ffPre = new Float32Array(T * dFF);
      const ffAct = new Float32Array(T * dFF);
      for (let t = 0; t < T; t++) {
        const xOff = t * D;
        const fOff = t * dFF;
        for (let j = 0; j < dFF; j++) {
          let sum = b.b1[j];
          for (let i = 0; i < D; i++) {
            sum += ln2.out[xOff + i] * b.w1[i * dFF + j];
          }
          ffPre[fOff + j] = sum;
          ffAct[fOff + j] = gelu(sum);
        }
      }

      const xOut = new Float32Array(T * D);
      for (let t = 0; t < T; t++) {
        const xOff = t * D;
        const fOff = t * dFF;
        for (let j = 0; j < D; j++) {
          let sum = xMid[xOff + j] + b.b2[j];
          for (let i = 0; i < dFF; i++) {
            sum += ffAct[fOff + i] * b.w2[i * D + j];
          }
          xOut[xOff + j] = sum;
        }
      }

      layerCaches.push({
        xIn,
        ln1,
        q,
        k,
        v,
        attnProbs,
        attnCtx,
        xMid,
        ln2,
        ffPre,
        ffAct,
      });
      x = xOut;
    }

    const lnFCache = this.layerNormForward(x, T, D, this.state.lnFGamma, this.state.lnFBeta);

    const V = this.state.vocabSize;
    const logits = new Float32Array(T * V);
    for (let t = 0; t < T; t++) {
      const hOff = t * D;
      const lOff = t * V;
      for (let vIdx = 0; vIdx < V; vIdx++) {
        const wOff = vIdx * D;
        let dot = this.state.lmBias[vIdx];
        for (let d = 0; d < D; d++) {
          dot += lnFCache.out[hOff + d] * this.state.lmHead[wOff + d];
        }
        logits[lOff + vIdx] = dot;
      }
    }

    return {
      logits,
      cache: { T, inputs, x0, lnFCache, layerCaches },
    };
  }

  public trainStep(): TrainingStepMetric {
    const startMs = performance.now();
    this.zeroGrads();

    // Accumulate gradients across a micro-batch of up to 4 dialogue windows per optimizer step
    const batchWindows = Math.min(4, this.windows.length);
    let totalTokens = 0;
    let totalLoss = 0;

    for (let bIdx = 0; bIdx < batchWindows; bIdx++) {
      const win = this.windows[this.windowIdx % this.windows.length];
      this.windowIdx += 1;

      const inputs = win.slice(0, win.length - 1);
      const targets = win.slice(1);
      const { logits, cache } = this.forwardSequence(inputs);
      const { T, lnFCache, layerCaches } = cache;
      totalTokens += T;

      const V = this.state.vocabSize;
      const { dModel: D, nLayers, nHeads, dFF } = this.hyperparams;
      const headDim = D / nHeads;
      const scale = 1 / Math.sqrt(headDim);

      const dLogits = new Float32Array(T * V);
      const invBatchT = 1 / (T * batchWindows);

      let winLossSum = 0;
      for (let t = 0; t < T; t++) {
        const lOff = t * V;
        let maxL = -Infinity;
        for (let v = 0; v < V; v++) {
          if (logits[lOff + v] > maxL) maxL = logits[lOff + v];
        }
        let expSum = 0;
        for (let v = 0; v < V; v++) {
          const e = Math.exp(logits[lOff + v] - maxL);
          dLogits[lOff + v] = e;
          expSum += e;
        }
        const invSum = 1 / expSum;
        const targetId = targets[t];
        const targetProb = Math.max(1e-12, dLogits[lOff + targetId] * invSum);
        winLossSum += -Math.log(targetProb);

        for (let v = 0; v < V; v++) {
          const p = dLogits[lOff + v] * invSum;
          dLogits[lOff + v] = (p - (v === targetId ? 1 : 0)) * invBatchT;
        }
      }
      totalLoss += winLossSum / T;

      // Backprop through dedicated LM head
      const dLnFOut = new Float32Array(T * D);
      for (let t = 0; t < T; t++) {
        const hOff = t * D;
        const lOff = t * V;
        for (let v = 0; v < V; v++) {
          const dl = dLogits[lOff + v];
          if (dl === 0) continue;
          this.grads.lmBias[v] += dl;
          const wOff = v * D;
          for (let d = 0; d < D; d++) {
            dLnFOut[hOff + d] += dl * this.state.lmHead[wOff + d];
            this.grads.lmHead[wOff + d] += dl * lnFCache.out[hOff + d];
          }
        }
      }

      let dX = this.layerNormBackward(
        dLnFOut,
        lnFCache.xHat,
        lnFCache.invStd,
        T,
        D,
        this.state.lnFGamma,
        this.grads.lnFGamma,
        this.grads.lnFBeta,
      );

      for (let l = nLayers - 1; l >= 0; l--) {
        const b = this.state.blocks[l];
        const gb = this.grads.blocks[l];
        const lc = layerCaches[l];

        const dXMid = new Float32Array(dX);
        const dFfAct = new Float32Array(T * dFF);
        for (let t = 0; t < T; t++) {
          const xOff = t * D;
          const fOff = t * dFF;
          for (let j = 0; j < D; j++) {
            const dy = dX[xOff + j];
            gb.b2[j] += dy;
            for (let i = 0; i < dFF; i++) {
              dFfAct[fOff + i] += dy * b.w2[i * D + j];
              gb.w2[i * D + j] += lc.ffAct[fOff + i] * dy;
            }
          }
        }

        const dLn2Out = new Float32Array(T * D);
        for (let t = 0; t < T; t++) {
          const xOff = t * D;
          const fOff = t * dFF;
          for (let j = 0; j < dFF; j++) {
            const dPre = dFfAct[fOff + j] * geluBackward(lc.ffPre[fOff + j]);
            gb.b1[j] += dPre;
            for (let i = 0; i < D; i++) {
              dLn2Out[xOff + i] += dPre * b.w1[i * dFF + j];
              gb.w1[i * dFF + j] += lc.ln2.out[xOff + i] * dPre;
            }
          }
        }

        const dLn2In = this.layerNormBackward(
          dLn2Out,
          lc.ln2.xHat,
          lc.ln2.invStd,
          T,
          D,
          b.ln2Gamma,
          gb.ln2Gamma,
          gb.ln2Beta,
        );
        for (let i = 0; i < dXMid.length; i++) {
          dXMid[i] += dLn2In[i];
        }

        const dXIn = new Float32Array(dXMid);
        const dAttnCtx = new Float32Array(T * D);
        for (let t = 0; t < T; t++) {
          const off = t * D;
          for (let j = 0; j < D; j++) {
            const dy = dXMid[off + j];
            for (let i = 0; i < D; i++) {
              dAttnCtx[off + i] += dy * b.wo[i * D + j];
              gb.wo[i * D + j] += lc.attnCtx[off + i] * dy;
            }
          }
        }

        const dQ = new Float32Array(T * D);
        const dK = new Float32Array(T * D);
        const dV = new Float32Array(T * D);

        for (let h = 0; h < nHeads; h++) {
          const hOff = h * headDim;
          const pHeadOff = h * T * T;
          for (let t = 0; t < T; t++) {
            const pRowOff = pHeadOff + t * T;
            const dProbs = new Float32Array(t + 1);
            let dotProbs = 0;

            for (let s = 0; s <= t; s++) {
              const p = lc.attnProbs[pRowOff + s];
              let dp = 0;
              for (let d = 0; d < headDim; d++) {
                const dCtx = dAttnCtx[t * D + hOff + d];
                dp += dCtx * lc.v[s * D + hOff + d];
                dV[s * D + hOff + d] += p * dCtx;
              }
              dProbs[s] = dp;
              dotProbs += p * dp;
            }

            for (let s = 0; s <= t; s++) {
              const p = lc.attnProbs[pRowOff + s];
              const dScore = p * (dProbs[s] - dotProbs) * scale;
              for (let d = 0; d < headDim; d++) {
                dQ[t * D + hOff + d] += dScore * lc.k[s * D + hOff + d];
                dK[s * D + hOff + d] += dScore * lc.q[t * D + hOff + d];
              }
            }
          }
        }

        const dLn1Out = new Float32Array(T * D);
        for (let t = 0; t < T; t++) {
          const off = t * D;
          for (let j = 0; j < D; j++) {
            const dq = dQ[off + j];
            const dk = dK[off + j];
            const dv = dV[off + j];
            for (let i = 0; i < D; i++) {
              const xi = lc.ln1.out[off + i];
              const wIdx = i * D + j;
              dLn1Out[off + i] += dq * b.wq[wIdx] + dk * b.wk[wIdx] + dv * b.wv[wIdx];
              gb.wq[wIdx] += xi * dq;
              gb.wk[wIdx] += xi * dk;
              gb.wv[wIdx] += xi * dv;
            }
          }
        }

        const dLn1In = this.layerNormBackward(
          dLn1Out,
          lc.ln1.xHat,
          lc.ln1.invStd,
          T,
          D,
          b.ln1Gamma,
          gb.ln1Gamma,
          gb.ln1Beta,
        );
        for (let i = 0; i < dXIn.length; i++) {
          dXIn[i] += dLn1In[i];
        }
        dX = dXIn;
      }

      for (let t = 0; t < T; t++) {
        const tokOff = inputs[t] * D;
        const posOff = t * D;
        for (let d = 0; d < D; d++) {
          this.grads.wte[tokOff + d] += dX[posOff + d];
          this.grads.wpe[posOff + d] += dX[posOff + d];
        }
      }
    }

    const loss = totalLoss / batchWindows;
    const gradNorm = this.computeAndClipGradNorm(2.0);

    this.step += 1;
    const total = Math.max(10, this.hyperparams.totalSteps);
    const warmup = Math.min(6, Math.floor(total * 0.08));
    let lr = this.hyperparams.learningRate;
    if (this.step < warmup) {
      lr = lr * (this.step / Math.max(1, warmup));
    } else {
      const progress = Math.min(1, (this.step - warmup) / Math.max(1, total - warmup));
      lr = lr * (0.2 + 0.8 * 0.5 * (1 + Math.cos(Math.PI * progress)));
    }

    this.applyAdamW(lr);

    const elapsedSec = Math.max(0.001, (performance.now() - startMs) / 1000);
    const perplexity = Math.exp(Math.min(12, loss));
    this.lastLoss = loss;

    const metric: TrainingStepMetric = {
      step: this.step,
      loss,
      perplexity,
      gradNorm,
      learningRate: lr,
      tokensPerSec: Math.round(totalTokens / elapsedSec),
    };
    this.history.push(metric);
    return metric;
  }

  private collectParamTuples(): [Float32Array, Float32Array, Float32Array, Float32Array, boolean][] {
    const pairs: [Float32Array, Float32Array, Float32Array, Float32Array, boolean][] = [
      [this.state.wte, this.grads.wte, this.adamM.wte, this.adamV.wte, false],
      [this.state.wpe, this.grads.wpe, this.adamM.wpe, this.adamV.wpe, false],
      [this.state.lnFGamma, this.grads.lnFGamma, this.adamM.lnFGamma, this.adamV.lnFGamma, false],
      [this.state.lnFBeta, this.grads.lnFBeta, this.adamM.lnFBeta, this.adamV.lnFBeta, false],
      [this.state.lmHead, this.grads.lmHead, this.adamM.lmHead, this.adamV.lmHead, true],
      [this.state.lmBias, this.grads.lmBias, this.adamM.lmBias, this.adamV.lmBias, false],
    ];
    for (let l = 0; l < this.state.blocks.length; l++) {
      const s = this.state.blocks[l];
      const g = this.grads.blocks[l];
      const m = this.adamM.blocks[l];
      const v = this.adamV.blocks[l];
      pairs.push(
        [s.ln1Gamma, g.ln1Gamma, m.ln1Gamma, v.ln1Gamma, false],
        [s.ln1Beta, g.ln1Beta, m.ln1Beta, v.ln1Beta, false],
        [s.wq, g.wq, m.wq, v.wq, true],
        [s.wk, g.wk, m.wk, v.wk, true],
        [s.wv, g.wv, m.wv, v.wv, true],
        [s.wo, g.wo, m.wo, v.wo, true],
        [s.ln2Gamma, g.ln2Gamma, m.ln2Gamma, v.ln2Gamma, false],
        [s.ln2Beta, g.ln2Beta, m.ln2Beta, v.ln2Beta, false],
        [s.w1, g.w1, m.w1, v.w1, true],
        [s.b1, g.b1, m.b1, v.b1, false],
        [s.w2, g.w2, m.w2, v.w2, true],
        [s.b2, g.b2, m.b2, v.b2, false],
      );
    }
    return pairs;
  }

  private computeAndClipGradNorm(maxNorm: number): number {
    const tuples = this.paramTuples;
    let sqSum = 0;
    for (const [, g] of tuples) {
      for (let i = 0; i < g.length; i++) {
        sqSum += g[i] * g[i];
      }
    }
    const norm = Math.sqrt(sqSum);
    if (norm > maxNorm && norm > 0) {
      const coeff = maxNorm / norm;
      for (const [, g] of tuples) {
        for (let i = 0; i < g.length; i++) {
          g[i] *= coeff;
        }
      }
    }
    return norm;
  }

  private applyAdamW(lr: number): void {
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;
    const weightDecay = 0.005;
    const bc1 = 1 - Math.pow(beta1, this.step);
    const bc2 = 1 - Math.pow(beta2, this.step);

    for (const [w, g, m, v, decay] of this.paramTuples) {
      for (let i = 0; i < w.length; i++) {
        const grad = g[i];
        m[i] = beta1 * m[i] + (1 - beta1) * grad;
        v[i] = beta2 * v[i] + (1 - beta2) * grad * grad;
        const mHat = m[i] / bc1;
        const vHat = v[i] / bc2;
        if (decay) {
          w[i] -= lr * weightDecay * w[i];
        }
        w[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
      }
    }
  }

  public generate(
    prompt: string,
    options: { maxNewTokens?: number; temperature?: number; topK?: number } = {},
  ): GenerationResult {
    const maxNewTokens = options.maxNewTokens ?? 48;
    const temperature = Math.max(0.05, options.temperature ?? 0.35);
    const topK = Math.max(1, options.topK ?? 4);

    let tokens = encode(this.tokenizer, prompt);
    if (tokens.length === 0) {
      tokens = [this.tokenizer.specialTokens.user];
    }

    const generated: number[] = [];
    let lastCandidates: TokenCandidate[] = [];
    const V = this.state.vocabSize;

    for (let step = 0; step < maxNewTokens; step++) {
      const ctx = tokens.slice(-this.hyperparams.contextLength);
      const { logits } = this.forwardSequence(ctx);
      const lastOff = (ctx.length - 1) * V;

      const scaled: { id: number; logit: number }[] = [];
      for (let v = 0; v < V; v++) {
        if (v === this.tokenizer.specialTokens.pad) continue;
        scaled.push({ id: v, logit: logits[lastOff + v] / temperature });
      }
      scaled.sort((a, b) => b.logit - a.logit);
      const top = scaled.slice(0, Math.min(topK, scaled.length));
      const maxL = top[0].logit;

      let expSum = 0;
      const probs = top.map((item) => {
        const e = Math.exp(item.logit - maxL);
        expSum += e;
        return { id: item.id, prob: e };
      });
      for (const p of probs) p.prob /= expSum;

      lastCandidates = probs.slice(0, 5).map((p) => ({
        id: p.id,
        token: this.tokenizer.vocab[p.id] ?? '',
        probability: p.prob,
      }));

      const r = Math.random();
      let cdf = 0;
      let nextId = probs[0].id;
      for (const p of probs) {
        cdf += p.prob;
        if (r <= cdf) {
          nextId = p.id;
          break;
        }
      }

      if (
        nextId === this.tokenizer.specialTokens.endoftext ||
        nextId === this.tokenizer.specialTokens.user
      ) {
        break;
      }
      generated.push(nextId);
      tokens.push(nextId);
    }

    return {
      text: decode(this.tokenizer, generated),
      generatedTokens: generated.length,
      topCandidatesAtLastStep: lastCandidates,
    };
  }

  public exportCheckpoint(): QoweCheckpoint {
    const activeParamCount = countKernelParameters(this.state.vocabSize, this.hyperparams);
    return {
      version: 'qowe-1.0-kernel',
      createdAt: new Date().toISOString(),
      step: this.step,
      loss: this.lastLoss,
      perplexity: Math.exp(Math.min(12, this.lastLoss)),
      hyperparams: { ...this.hyperparams },
      vocabSize: this.state.vocabSize,
      activeParamCount,
      tokenizer: this.tokenizer,
      weights: {
        wte: Array.from(this.state.wte),
        wpe: Array.from(this.state.wpe),
        lnFGamma: Array.from(this.state.lnFGamma),
        lnFBeta: Array.from(this.state.lnFBeta),
        lmHead: Array.from(this.state.lmHead),
        lmBias: Array.from(this.state.lmBias),
        blocks: this.state.blocks.map((b) => ({
          ln1Gamma: Array.from(b.ln1Gamma),
          ln1Beta: Array.from(b.ln1Beta),
          wq: Array.from(b.wq),
          wk: Array.from(b.wk),
          wv: Array.from(b.wv),
          wo: Array.from(b.wo),
          ln2Gamma: Array.from(b.ln2Gamma),
          ln2Beta: Array.from(b.ln2Beta),
          w1: Array.from(b.w1),
          b1: Array.from(b.b1),
          w2: Array.from(b.w2),
          b2: Array.from(b.b2),
        })),
      },
      history: [...this.history],
    };
  }

  public static fromCheckpoint(cp: QoweCheckpoint, corpus: string): QoweTrainerEngine {
    const engine = new QoweTrainerEngine(cp.tokenizer, corpus, cp.hyperparams);
    engine.step = cp.step;
    engine.lastLoss = cp.loss;
    engine.history = [...cp.history];
    engine.state.wte.set(cp.weights.wte);
    engine.state.wpe.set(cp.weights.wpe);
    engine.state.lnFGamma.set(cp.weights.lnFGamma);
    engine.state.lnFBeta.set(cp.weights.lnFBeta);
    if (cp.weights.lmHead) engine.state.lmHead.set(cp.weights.lmHead);
    if (cp.weights.lmBias) engine.state.lmBias.set(cp.weights.lmBias);
    for (let l = 0; l < cp.weights.blocks.length; l++) {
      const src = cp.weights.blocks[l];
      const dst = engine.state.blocks[l];
      dst.ln1Gamma.set(src.ln1Gamma);
      dst.ln1Beta.set(src.ln1Beta);
      dst.wq.set(src.wq);
      dst.wk.set(src.wk);
      dst.wv.set(src.wv);
      dst.wo.set(src.wo);
      dst.ln2Gamma.set(src.ln2Gamma);
      dst.ln2Beta.set(src.ln2Beta);
      dst.w1.set(src.w1);
      dst.b1.set(src.b1);
      dst.w2.set(src.w2);
      dst.b2.set(src.b2);
    }
    return engine;
  }
}
