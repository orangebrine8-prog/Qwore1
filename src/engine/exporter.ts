import { calculateArchitectureSpecs, formatParamCount, QoweArchitectureConfig } from './architecture';
import type { MultimodalCheckpoint } from './multimodal';
import type { QoweCheckpoint } from './trainer';

export interface QoweFullModelBundle {
  format: 'qowe-1.0-bundle';
  exportedAt: string;
  architecture: QoweArchitectureConfig;
  architectureSpecs: ReturnType<typeof calculateArchitectureSpecs>;
  textCheckpoint: QoweCheckpoint | null;
  multimodalCheckpoint: MultimodalCheckpoint | null;
}

export function buildFullModelBundle(
  architecture: QoweArchitectureConfig,
  textCheckpoint: QoweCheckpoint | null,
  multimodalCheckpoint: MultimodalCheckpoint | null,
): QoweFullModelBundle {
  return {
    format: 'qowe-1.0-bundle',
    exportedAt: new Date().toISOString(),
    architecture,
    architectureSpecs: calculateArchitectureSpecs(architecture),
    textCheckpoint,
    multimodalCheckpoint,
  };
}

export function buildPytorchScript(config: QoweArchitectureConfig): string {
  const specs = calculateArchitectureSpecs(config);
  return `"""
Qowe 1.0 — Standalone ${formatParamCount(specs.totalParams)} Parameter Multimodal LLM & Video Model (PyTorch)
Architecture: ${config.name}
Exact Parameters: ${specs.totalParams.toLocaleString()} (${formatParamCount(specs.totalParams)})
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class QoweConfig:
    vocab_size: int = ${config.vocabSize}
    context_length: int = ${config.contextLength}
    d_model: int = ${config.dModel}
    n_layers: int = ${config.nLayers}
    n_heads: int = ${config.nHeads}
    d_ff: int = ${config.dFF}
    tie_embeddings: bool = ${config.tieEmbeddings ? 'True' : 'False'}
    image_patch_tokens: int = 256
    video_frames: int = 16


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: QoweConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class QoweBlock(nn.Module):
    def __init__(self, cfg: QoweConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Qowe1Model(nn.Module):
    """Unified Text + Spatiotemporal Image/Video Transformer trained from scratch."""

    def __init__(self, cfg: QoweConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.context_length, cfg.d_model)
        self.blocks = nn.ModuleList([QoweBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight

        # From-scratch Visual & Video Latent Patch Head (B, d_model -> B, F, 3, 64, 64)
        self.video_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.video_frames * 3 * 16 * 16),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_text(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def forward_video(self, prompt_idx: torch.Tensor) -> torch.Tensor:
        B, T = prompt_idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=prompt_idx.device)
        x = self.wte(prompt_idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        pooled = self.ln_f(x)[:, -1, :]
        latents = self.video_head(pooled).view(B, self.cfg.video_frames, 3, 16, 16)
        return torch.sigmoid(F.interpolate(latents.flatten(0, 1), size=(64, 64), mode="bilinear")).view(
            B, self.cfg.video_frames, 3, 64, 64
        )


if __name__ == "__main__":
    cfg = QoweConfig()
    model = Qowe1Model(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized Qowe 1.0 with {total_params:,} parameters")
`;
}
