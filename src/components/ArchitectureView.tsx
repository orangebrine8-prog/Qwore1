import React from 'react';
import { CheckCircle2 } from 'lucide-react';
import {
  calculateArchitectureSpecs,
  formatParamCount,
  QOWE_PRESETS,
  QoweArchitectureConfig,
} from '../engine/architecture';

interface ArchitectureViewProps {
  config: QoweArchitectureConfig;
  onChangeConfig: (next: QoweArchitectureConfig) => void;
  onOpenMobileMenu: () => void;
}

function formatMemoryMB(mb: number): string {
  const gb = mb / 1024;
  if (gb >= 1024) return `${(gb / 1024).toFixed(2)} TB`;
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  return `${mb.toFixed(1)} MB`;
}

function formatFlops(flops: number): string {
  if (flops >= 1e12) return `${(flops / 1e12).toFixed(2)} TFLOPs`;
  if (flops >= 1e9) return `${(flops / 1e9).toFixed(1)} GFLOPs`;
  return `${(flops / 1e6).toFixed(1)} MFLOPs`;
}

export const ArchitectureView: React.FC<ArchitectureViewProps> = ({
  config,
  onChangeConfig,
  onOpenMobileMenu,
}) => {
  const specs = calculateArchitectureSpecs(config);

  const breakdownRows = [
    {
      component: 'Token Embeddings (wte)',
      formula: `${config.vocabSize.toLocaleString()} × ${config.dModel.toLocaleString()}`,
      params: specs.tokenEmbeddingParams,
    },
    {
      component: 'Positional Embeddings (wpe)',
      formula: `${config.contextLength.toLocaleString()} × ${config.dModel.toLocaleString()}`,
      params: specs.positionEmbeddingParams,
    },
    {
      component: `Causal Multi-Head Attention (×${config.nLayers})`,
      formula: `${config.nLayers} × (4 × ${config.dModel.toLocaleString()}² + 4 × ${config.dModel.toLocaleString()})`,
      params: config.nLayers * specs.attentionParamsPerLayer,
    },
    {
      component: `Feed-Forward MLP (×${config.nLayers})`,
      formula: `${config.nLayers} × (2 × ${config.dModel.toLocaleString()} × ${config.dFF.toLocaleString()} + biases)`,
      params: config.nLayers * specs.mlpParamsPerLayer,
    },
    {
      component: `LayerNorms (Pre-Attn, Pre-MLP, Final)`,
      formula: `${config.nLayers} × 4d + 2d`,
      params: config.nLayers * specs.layerNormParamsPerLayer + specs.finalLayerNormParams,
    },
    {
      component: 'LM Output Head',
      formula: config.tieEmbeddings
        ? 'Weight-tied to wte (0 extra)'
        : `${config.vocabSize.toLocaleString()} × ${config.dModel.toLocaleString()}`,
      params: specs.lmHeadParams,
    },
  ];

  return (
    <div className="flex flex-col flex-1 h-full min-h-0 min-w-0 bg-[#f5f5f7] dark:bg-[#0b0b0d]">
      <header className="flex items-center justify-between gap-4 px-4 sm:px-6 py-2.5 sm:py-3 bg-white/95 dark:bg-[#161618]/95 border-b border-black/[0.06] dark:border-white/[0.08] shrink-0">
        <div className="flex items-center gap-3 min-w-0">
          <div className="min-w-0">
            <h1 className="text-sm sm:text-base font-semibold text-neutral-900 dark:text-white truncate">
              Qowe 1.0 Architecture &amp; 800B Scale Blueprint
            </h1>
            <p className="text-xs text-neutral-500 dark:text-neutral-400 truncate tabular-nums">
              {specs.totalParams.toLocaleString()} exact parameters
              <span aria-hidden="true"> · </span>
              {formatParamCount(specs.totalParams)} scale
            </p>
          </div>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-6">
        <div className="max-w-6xl mx-auto space-y-6">
          {/* Preset selector */}
          <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {QOWE_PRESETS.map((preset) => {
              const pSpecs = calculateArchitectureSpecs(preset);
              const isSelected = config.id === preset.id;
              return (
                <button
                  key={preset.id}
                  type="button"
                  onClick={() => onChangeConfig({ ...preset })}
                  className={`text-left p-5 rounded-[22px] border shadow-[0_1px_2px_rgba(0,0,0,0.03),0_8px_24px_rgba(0,0,0,0.03)] transition-colors ${
                    isSelected
                      ? 'bg-white dark:bg-[#1c1c1e] border-[#007aff] dark:border-[#0a84ff] ring-1 ring-[#007aff] dark:ring-[#0a84ff]'
                      : 'bg-white/95 dark:bg-[#1c1c1e]/95 border-black/[0.06] dark:border-white/[0.08] hover:border-black/[0.15] dark:hover:border-white/[0.2]'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-sm font-semibold text-neutral-900 dark:text-white">
                      {preset.name}
                    </span>
                    <span className="text-sm font-mono font-semibold text-[#007aff] dark:text-[#0a84ff] tabular-nums">
                      {formatParamCount(pSpecs.totalParams)}
                    </span>
                  </div>
                  <p className="mt-1.5 text-xs text-neutral-500 dark:text-neutral-400 leading-relaxed">
                    {preset.description}
                  </p>
                  <div className="mt-3 text-xs text-neutral-500 dark:text-neutral-400 font-mono tabular-nums">
                    {preset.nLayers}L · {preset.nHeads}H · d={preset.dModel.toLocaleString()} · ctx={preset.contextLength.toLocaleString()}
                  </div>
                </button>
              );
            })}
          </section>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Custom sliders */}
            <section className="lg:col-span-5 p-5 rounded-[22px] bg-white/95 dark:bg-[#1c1c1e]/95 border border-black/[0.06] dark:border-white/[0.08] shadow-[0_1px_2px_rgba(0,0,0,0.03),0_8px_24px_rgba(0,0,0,0.03)] space-y-4">
              <div>
                <h2 className="text-base font-semibold text-neutral-900 dark:text-white">
                  Hyperparameter Controls
                </h2>
                <p className="text-xs text-neutral-500 dark:text-neutral-400">
                  Scale transformer dimensions from 100M+ up to 800B+ parameters.
                </p>
              </div>

              <label className="block text-xs">
                <div className="flex justify-between mb-1">
                  <span className="font-medium text-neutral-700 dark:text-neutral-300">
                    Transformer Layers (n_layers)
                  </span>
                  <span className="font-mono tabular-nums text-neutral-900 dark:text-white">
                    {config.nLayers}
                  </span>
                </div>
                <input
                  type="range"
                  min="6"
                  max="160"
                  step="2"
                  value={config.nLayers}
                  onChange={(e) =>
                    onChangeConfig({ ...config, id: 'custom', name: 'Qowe 1.0 Custom', nLayers: Number(e.target.value) })
                  }
                  className="w-full accent-[#007aff]"
                />
              </label>

              <label className="block text-xs">
                <div className="flex justify-between mb-1">
                  <span className="font-medium text-neutral-700 dark:text-neutral-300">
                    Hidden Dimension (d_model)
                  </span>
                  <span className="font-mono tabular-nums text-neutral-900 dark:text-white">
                    {config.dModel.toLocaleString()}
                  </span>
                </div>
                <input
                  type="range"
                  min="512"
                  max="20480"
                  step="512"
                  value={config.dModel}
                  onChange={(e) => {
                    const dModel = Number(e.target.value);
                    onChangeConfig({
                      ...config,
                      id: 'custom',
                      name: 'Qowe 1.0 Custom',
                      dModel,
                      dFF: dModel * 4,
                      nHeads: Math.max(8, Math.floor(dModel / 128)),
                    });
                  }}
                  className="w-full accent-[#007aff]"
                />
              </label>

              <label className="block text-xs">
                <div className="flex justify-between mb-1">
                  <span className="font-medium text-neutral-700 dark:text-neutral-300">
                    Vocabulary Size (vocab_size)
                  </span>
                  <span className="font-mono tabular-nums text-neutral-900 dark:text-white">
                    {config.vocabSize.toLocaleString()}
                  </span>
                </div>
                <input
                  type="range"
                  min="32768"
                  max="131072"
                  step="4096"
                  value={config.vocabSize}
                  onChange={(e) =>
                    onChangeConfig({
                      ...config,
                      id: 'custom',
                      name: 'Qowe 1.0 Custom',
                      vocabSize: Number(e.target.value),
                    })
                  }
                  className="w-full accent-[#007aff]"
                />
              </label>

              <label className="block text-xs">
                <div className="flex justify-between mb-1">
                  <span className="font-medium text-neutral-700 dark:text-neutral-300">
                    Context Length
                  </span>
                  <span className="font-mono tabular-nums text-neutral-900 dark:text-white">
                    {config.contextLength.toLocaleString()} tokens
                  </span>
                </div>
                <input
                  type="range"
                  min="512"
                  max="32768"
                  step="512"
                  value={config.contextLength}
                  onChange={(e) =>
                    onChangeConfig({
                      ...config,
                      id: 'custom',
                      name: 'Qowe 1.0 Custom',
                      contextLength: Number(e.target.value),
                    })
                  }
                  className="w-full accent-[#007aff]"
                />
              </label>

              <label className="flex items-center justify-between pt-2 text-xs text-neutral-700 dark:text-neutral-300">
                <span className="font-medium">Tie Input &amp; LM Head Embeddings</span>
                <input
                  type="checkbox"
                  checked={config.tieEmbeddings}
                  onChange={(e) =>
                    onChangeConfig({
                      ...config,
                      id: 'custom',
                      name: 'Qowe 1.0 Custom',
                      tieEmbeddings: e.target.checked,
                    })
                  }
                  className="w-4 h-4 accent-[#007aff]"
                />
              </label>

              {/* Compute & Memory Footprint */}
              <div className="pt-4 border-t border-black/[0.06] dark:border-white/[0.08] grid grid-cols-2 gap-3 text-xs">
                <div>
                  <div className="text-neutral-500 dark:text-neutral-400">BF16 Weights</div>
                  <div className="font-mono font-semibold text-neutral-900 dark:text-white tabular-nums">
                    {formatMemoryMB(specs.bf16WeightsMB)}
                  </div>
                </div>
                <div>
                  <div className="text-neutral-500 dark:text-neutral-400">FP32 Weights</div>
                  <div className="font-mono font-semibold text-neutral-900 dark:text-white tabular-nums">
                    {formatMemoryMB(specs.fp32WeightsMB)}
                  </div>
                </div>
                <div>
                  <div className="text-neutral-500 dark:text-neutral-400">AdamW States (FP32)</div>
                  <div className="font-mono font-semibold text-neutral-900 dark:text-white tabular-nums">
                    {formatMemoryMB(specs.adamWOptimizerMB)}
                  </div>
                </div>
                <div>
                  <div className="text-neutral-500 dark:text-neutral-400">Forward FLOPs / Token</div>
                  <div className="font-mono font-semibold text-neutral-900 dark:text-white tabular-nums">
                    {formatFlops(specs.flopsPerToken)}
                  </div>
                </div>
              </div>
            </section>

            {/* Exact Parameter Breakdown Table */}
            <section className="lg:col-span-7 p-5 rounded-[22px] bg-white/95 dark:bg-[#1c1c1e]/95 border border-black/[0.06] dark:border-white/[0.08] shadow-[0_1px_2px_rgba(0,0,0,0.03),0_8px_24px_rgba(0,0,0,0.03)] space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h2 className="text-base font-semibold text-neutral-900 dark:text-white">
                    Exact Parameter Breakdown
                  </h2>
                  <p className="text-xs text-neutral-500 dark:text-neutral-400">
                    Layer-by-layer trainable weight tensors for {config.name}.
                  </p>
                </div>
                <div className="inline-flex items-center gap-1.5 text-xs font-medium text-emerald-700 dark:text-emerald-400">
                  <CheckCircle2 className="w-4 h-4" />
                  <span className="tabular-nums">
                    {specs.totalParams >= 800_000_000_000
                      ? 'Verified 800B+ Frontier Scale'
                      : specs.isOver100M
                        ? 'Verified > 100M Parameters'
                        : 'Under 100M Parameters'}
                  </span>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-black/[0.06] dark:border-white/[0.08] text-neutral-500 dark:text-neutral-400">
                      <th className="py-2.5 pr-3 font-medium">Component</th>
                      <th className="py-2.5 px-3 font-medium">Tensor Formula</th>
                      <th className="py-2.5 pl-3 text-right font-medium">Parameters</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-black/[0.05] dark:divide-white/[0.06]">
                    {breakdownRows.map((row) => (
                      <tr key={row.component}>
                        <td className="py-2.5 pr-3 font-medium text-neutral-800 dark:text-neutral-200">
                          {row.component}
                        </td>
                        <td className="py-2.5 px-3 font-mono text-neutral-500 dark:text-neutral-400">
                          {row.formula}
                        </td>
                        <td className="py-2.5 pl-3 text-right font-mono tabular-nums text-neutral-900 dark:text-neutral-100">
                          {row.params.toLocaleString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                  <tfoot>
                    <tr className="border-t border-black/[0.1] dark:border-white/[0.12]">
                      <td className="py-3 pr-3 font-semibold text-neutral-900 dark:text-white">
                        Total Parameters
                      </td>
                      <td className="py-3 px-3 font-mono text-neutral-500 dark:text-neutral-400">
                        {formatParamCount(specs.totalParams)}
                      </td>
                      <td className="py-3 pl-3 text-right font-mono font-bold text-[#007aff] dark:text-[#0a84ff] tabular-nums">
                        {specs.totalParams.toLocaleString()}
                      </td>
                    </tr>
                  </tfoot>
                </table>
              </div>

              {/* Architecture Dataflow Diagram */}
              <div className="pt-3 border-t border-black/[0.06] dark:border-white/[0.08]">
                <div className="text-xs font-medium text-neutral-700 dark:text-neutral-300 mb-3">
                  Causal Decoder Forward Pass
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center font-mono text-[11px]">
                  <div className="p-3 rounded-2xl bg-[#f5f5f7] dark:bg-[#121214]">
                    <div className="font-semibold text-neutral-900 dark:text-white">Token + Pos</div>
                    <div className="text-neutral-500 mt-0.5">[B, T, {config.dModel.toLocaleString()}]</div>
                  </div>
                  <div className="p-3 rounded-2xl bg-[#f5f5f7] dark:bg-[#121214]">
                    <div className="font-semibold text-neutral-900 dark:text-white">
                      {config.nLayers}× Causal Attn
                    </div>
                    <div className="text-neutral-500 mt-0.5">
                      {config.nHeads} heads · d_h={specs.headDim}
                    </div>
                  </div>
                  <div className="p-3 rounded-2xl bg-[#f5f5f7] dark:bg-[#121214]">
                    <div className="font-semibold text-neutral-900 dark:text-white">GELU MLP</div>
                    <div className="text-neutral-500 mt-0.5">
                      {config.dModel.toLocaleString()} → {config.dFF.toLocaleString()}
                    </div>
                  </div>
                  <div className="p-3 rounded-2xl bg-[#f5f5f7] dark:bg-[#121214]">
                    <div className="font-semibold text-neutral-900 dark:text-white">LM Logits</div>
                    <div className="text-neutral-500 mt-0.5">
                      [B, T, {config.vocabSize.toLocaleString()}]
                    </div>
                  </div>
                </div>
              </div>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
};
