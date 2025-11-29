
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import  functools
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Attention de canal (SE)
# =============================================================================
class SEBlock(nn.Module):
    def __init__(self, C: int, r: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(C, max(1, C // r), bias=False),
            nn.ReLU(True),
            nn.Linear(max(1, C // r), C, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).flatten(1)           # (B, C)
        w = self.fc(w).view(b, c, 1, 1)       # (B, C, 1, 1)
        return x * w


# =============================================================================
# Task-conditioned pooling (query attention) + Transformer option
# =============================================================================
class TaskAttentionPooling(nn.Module):
    """
    Pooling attentionné conditionné par la tâche.
    Entrée: tokens (B, N, d). Chaque tâche a un vecteur requête q_t (d).
    Sortie: (B, d) par tâche -> petite tête de classification.
    """
    def __init__(self, d_model: int, tasks: Dict[str, int]):
        super().__init__()
        self.tasks = list(tasks.keys())
        self.query = nn.ParameterDict({t: nn.Parameter(torch.randn(d_model)) for t in self.tasks})

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        # tokens: (B, N, d)
        B, N, D = tokens.shape
        outputs = {}
        for t in self.tasks:
            q = self.query[t].view(1, 1, D)           # (1,1,d)
            scores = (tokens * q).sum(dim=-1)         # (B, N)
            w = scores.softmax(dim=1).unsqueeze(-1)   # (B, N, 1)
            pooled = (w * tokens).sum(dim=1)          # (B, d)
            outputs[t] = pooled
        return outputs


class TransformerBlock(nn.Module):
    """ Un petit encoder Transformer (MHA + FFN) partagé. """
    def __init__(self, d_model=256, nhead=4, dim_feedforward=512, dropout=0.1, num_layers=1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x):  # (B, N, d)
        return self.encoder(x)

# =============================================================================
# Modèle: PatchGAN non-chevauchant + Gram par patch + attentions
# =============================================================================
class MultiTaskPatchGANGramModelNonOverlapV2(nn.Module):
    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        norm: str = "batch",
        patch_size: int = 70,
        patch_div: int = 4,
        num_classes_per_task: Dict[str, int] | None = None,
        use_channel_attention: bool = False,
        use_token_attention: bool = True,
        gram_channels: int = 64,
        d_model: int = 256,
        transformer_layers: int = 1,
        transformer_heads: int = 4,
    ):
        super().__init__()
        self.num_classes_per_task = num_classes_per_task or {}
        self.patch_div = patch_div
        self.use_channel_attention = use_channel_attention
        self.use_token_attention = use_token_attention
        self.gram_channels = gram_channels

        if norm == "instance":
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
        elif norm == "group":
            norm_layer = functools.partial(nn.GroupNorm, num_groups=32)
        else:
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)

        # Backbone non-chevauchant k=s=4
        layers = []
        in_nc = input_nc
        num_filters = ndf
        kernel, stride, pad = 4, 4, 0
        rf = patch_size
        while rf > 4 and num_filters <= 512:
            layers += [
                nn.Conv2d(in_nc, num_filters, kernel_size=kernel, stride=stride, padding=pad, bias=False),
                norm_layer(num_filters),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_nc = num_filters
            num_filters *= 2
            rf /= stride

        # couche finale 1x1
        layers += [
            nn.Conv2d(in_nc, num_filters, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(num_filters),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.feature_extractor = nn.Sequential(*layers)
        self.C = num_filters

        # Channel attention (SE) optionnelle
        self.se = SEBlock(self.C) if use_channel_attention else nn.Identity()

        # Réduction de canaux avant Gram
        self.chan_proj = nn.Conv2d(self.C, gram_channels, kernel_size=1, bias=False)
        self.Cr = gram_channels
        token_dim_in = self.Cr * self.Cr

        # Proj tokens (r*r) -> d_model
        self.token_proj = nn.Linear(token_dim_in, d_model)

        # Transformer partagé (ablation possible)
        self.transformer = TransformerBlock(
            d_model=d_model, nhead=transformer_heads,
            dim_feedforward=2 * d_model, dropout=0.1, num_layers=transformer_layers
        ) if use_token_attention and transformer_layers > 0 else nn.Identity()

        # Pooling conditionné par tâche
        self.task_pool = TaskAttentionPooling(d_model=d_model, tasks=self.num_classes_per_task)

        # Têtes par tâche
        self.classifiers = nn.ModuleDict({
            t: nn.Linear(d_model, ncls) for t, ncls in self.num_classes_per_task.items()
        })

    @staticmethod
    def _pad_to_divisible(x: torch.Tensor, div: int) -> torch.Tensor:
        B, C, H, W = x.shape
        pad_h = (-H) % div
        pad_w = (-W) % div
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        return x

    def _split_patches(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, C, H, W = x.shape
        x = self._pad_to_divisible(x, self.patch_div)
        H, W = x.shape[-2:]
        ph, pw = H // self.patch_div, W // self.patch_div
        patches = F.unfold(x, kernel_size=(ph, pw), stride=(ph, pw))  # (B, C*ph*pw, Np)
        Np = patches.shape[-1]
        patches = patches.view(B, C, ph * pw, Np).permute(0, 3, 1, 2).contiguous()
        return patches, ph * pw  # (B, Np, C, N), N=ph*pw

    def forward(self, x):
        # Backbone + SE
        x = self.feature_extractor(x)     # (B, C, H, W)
        x = self.se(x)

        # Réduction de canaux
        x = self.chan_proj(x)             # (B, Cr, H, W)

        # Patches disjoints
        patches, patch_area = self._split_patches(x)         # (B, Np, Cr, N)

        # Gram par patch: G = (F F^T) / N  -> (B, Np, Cr, Cr)
        G = torch.matmul(patches, patches.transpose(2, 3)) / float(patch_area)

        # Tokens (B, Np, d)
        B, Np, Cr, _ = G.shape
        tokens = G.view(B, Np, Cr * Cr)
        tokens = self.token_proj(tokens)
        tokens = self.transformer(tokens)  # (B, Np, d_model) ou Identity

        # Pooling conditionné par tâche -> logits
        pooled_by_task = self.task_pool(tokens)
        out = {t: self.classifiers[t](pooled_by_task[t]) for t in self.num_classes_per_task.keys()}
        return out
