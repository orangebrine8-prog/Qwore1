export interface QoweArchitectureConfig {
  id: string;
  name: string;
  description: string;
  vocabSize: number;
  contextLength: number;
  dModel: number;
  nLayers: number;
  nHeads: number;
  dFF: number;
  tieEmbeddings: boolean;
}

export interface QoweArchitectureSpecs {
  tokenEmbeddingParams: number;
  positionEmbeddingParams: number;
  attentionParamsPerLayer: number;
  mlpParamsPerLayer: number;
  layerNormParamsPerLayer: number;
  totalBlockParams: number;
  finalLayerNormParams: number;
  lmHeadParams: number;
  totalParams: number;
  isOver100M: boolean;
  headDim: number;
  fp32WeightsMB: number;
  bf16WeightsMB: number;
  adamWOptimizerMB: number;
  totalTrainingVramGB: number;
  flopsPerToken: number;
}

export const QOWE_PRESETS: QoweArchitectureConfig[] = [
  {
    id: 'qowe-1.0-ultra-800b',
    name: 'Qowe 1.0 Ultra 800B',
    description: 'Frontier 158-layer 800B-class multimodal transformer with 32K context and 20,480 hidden width.',
    vocabSize: 131072,
    contextLength: 32768,
    dModel: 20480,
    nLayers: 158,
    nHeads: 160,
    dFF: 81920,
    tieEmbeddings: false,
  },
  {
    id: 'qowe-1.0-base',
    name: 'Qowe 1.0 Base',
    description: 'Standard 12-layer causal decoder-only transformer exceeding 100M parameters.',
    vocabSize: 50257,
    contextLength: 1024,
    dModel: 768,
    nLayers: 12,
    nHeads: 12,
    dFF: 3072,
    tieEmbeddings: true,
  },
  {
    id: 'qowe-1.0-large',
    name: 'Qowe 1.0 Pro',
    description: 'High-capacity 24-layer transformer with 1,024 hidden width and 2,048-token context.',
    vocabSize: 50257,
    contextLength: 2048,
    dModel: 1024,
    nLayers: 24,
    nHeads: 16,
    dFF: 4096,
    tieEmbeddings: true,
  },
];

export function calculateArchitectureSpecs(config: QoweArchitectureConfig): QoweArchitectureSpecs {
  const { vocabSize, contextLength, dModel, nLayers, nHeads, dFF, tieEmbeddings } = config;

  const tokenEmbeddingParams = vocabSize * dModel;
  const positionEmbeddingParams = contextLength * dModel;

  const attentionParamsPerLayer = 4 * dModel * dModel + 4 * dModel;
  const mlpParamsPerLayer = dModel * dFF + dFF + dFF * dModel + dModel;
  const layerNormParamsPerLayer = 4 * dModel;

  const totalBlockParams = nLayers * (attentionParamsPerLayer + mlpParamsPerLayer + layerNormParamsPerLayer);
  const finalLayerNormParams = 2 * dModel;
  const lmHeadParams = tieEmbeddings ? 0 : vocabSize * dModel;

  const totalParams =
    tokenEmbeddingParams +
    positionEmbeddingParams +
    totalBlockParams +
    finalLayerNormParams +
    lmHeadParams;

  const headDim = Math.floor(dModel / Math.max(1, nHeads));

  const fp32WeightsMB = (totalParams * 4) / (1024 * 1024);
  const bf16WeightsMB = (totalParams * 2) / (1024 * 1024);
  const adamWOptimizerMB = (totalParams * 16) / (1024 * 1024);
  const activationOverheadMB = (nLayers * contextLength * dModel * 12 * 2) / (1024 * 1024);
  const totalTrainingVramGB = (adamWOptimizerMB + activationOverheadMB) / 1024;

  const flopsPerToken = 2 * totalParams;

  return {
    tokenEmbeddingParams,
    positionEmbeddingParams,
    attentionParamsPerLayer,
    mlpParamsPerLayer,
    layerNormParamsPerLayer,
    totalBlockParams,
    finalLayerNormParams,
    lmHeadParams,
    totalParams,
    isOver100M: totalParams >= 100_000_000,
    headDim,
    fp32WeightsMB,
    bf16WeightsMB,
    adamWOptimizerMB,
    totalTrainingVramGB,
    flopsPerToken,
  };
}

export function formatParamCount(params: number): string {
  if (params >= 1_000_000_000) {
    return `${(params / 1_000_000_000).toFixed(2)}B`;
  }
  if (params >= 1_000_000) {
    return `${(params / 1_000_000).toFixed(2)}M`;
  }
  if (params >= 1_000) {
    return `${(params / 1_000).toFixed(1)}K`;
  }
  return String(params);
}
