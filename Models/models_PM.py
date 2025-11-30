# models_PM.py
import functools
from typing import Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- SE channel attention ----------
class SE(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        hid = max(c // r, 1)
        self.mlp = nn.Sequential(
            nn.Linear(c, hid), nn.ReLU(inplace=True),
            nn.Linear(hid, c), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        wgt = self.mlp(x.mean((2, 3))).view(n, c, 1, 1)
        return x * wgt


# ---------- Tête par tâche avec attention ----------
class TaskHeadImproved(nn.Module):
    """
    - (option) SE (channel attention)
    - Conv1x1 -> logits d'attention -> Softmax spatial(τ) -> A
    - Conv1x1 classes -> GWAP normalisé par A  => logits [N, K]
    - Ablation: A uniforme (GAP), pas d'attention apprise
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_se: bool = True,
        tau: float = 0.7,
        use_softmax_spatial: bool = True,
        ablate_attention: bool = False
    ):
        super().__init__()
        self.use_se = use_se
        self.tau = tau
        self.use_softmax_spatial = use_softmax_spatial
        self.ablate_attention = ablate_attention

        self.se = SE(in_channels) if use_se else nn.Identity()
        self.attn_conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        self.cls_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, feat: torch.Tensor):
        N, C, H, W = feat.shape
        x = self.se(feat) if self.use_se and not self.ablate_attention else feat

        # Ablation: attention uniforme
        if self.ablate_attention:
            A = torch.ones((N, 1, H, W), device=feat.device, dtype=feat.dtype) / float(H * W)
            M = self.cls_conv(x)             # [N, K, H, W]
            logits = (M * A).sum(dim=(2, 3)) # GAP équivalent
            return logits, A

        a = self.attn_conv(x).view(N, 1, H * W)
        if self.use_softmax_spatial:
            A = torch.softmax(a / self.tau, dim=-1).view(N, 1, H, W)
        else:
            A = torch.sigmoid(a).view(N, 1, H, W)
            A = A / (A.sum(dim=(2, 3), keepdim=True) + 1e-6)

        M = self.cls_conv(x)                     # [N, K, H, W]
        num = (M * A).sum(dim=(2, 3))            # [N, K]
        den = (A.sum(dim=(2, 3)) + 1e-6)         # [N, 1]
        logits = num / den
        return logits, A


class MultiTaskPatchGAN(nn.Module):
    """
    PatchGAN tronqué multi-tâches.
    - Tronc convolutionnel
    - Une tête par tâche avec attention
    - API:
        * model(x) -> {task: logits}
        * model(x, return_full=True) -> {task: {'logits', 'attn'}}
        * model(x, return_embeddings=True) -> embeddings globaux
        * model(x, return_task_embeddings=True) -> (outputs, task_embeddings)
    """

    def __init__(
        self,
        tasks_dict: Dict[str, int],
        input_nc: int = 3,
        ndf: int = 64,
        norm: str = "instance",
        patch_size: int = 70,
        device: torch.device | str = "cpu",
        attn_tau: float = 0.7,
        attn_use_se: bool = True,
        attn_softmax_spatial: bool = True,
        ablate_attention: bool = False
    ):
        super().__init__()
        self.device = torch.device(device)
        self.tasks_dict = tasks_dict
        self.ablate_attention = ablate_attention

        if norm == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
        else:
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)

        layers = []
        num_filters = ndf
        kernel_size = 4
        padding = 1
        stride = 2
        receptive_field_size = float(patch_size)
        in_nc = input_nc

        while receptive_field_size > 4 and num_filters <= 512:
            layers += [
                nn.Conv2d(in_nc, num_filters, kernel_size, stride, padding),
                norm_layer(num_filters),
                nn.LeakyReLU(0.2, True)
            ]
            in_nc = num_filters
            num_filters *= 2
            receptive_field_size /= stride

        final_c = num_filters
        layers += [
            nn.Conv2d(in_nc, final_c, kernel_size, 1, padding),
            norm_layer(final_c),
            nn.LeakyReLU(0.2, True)
        ]

        self.trunk = nn.Sequential(*layers).to(self.device)

        self.task_heads = nn.ModuleDict()
        for task_name, nb_cls in tasks_dict.items():
            self.task_heads[task_name] = TaskHeadImproved(
                in_channels=final_c,
                out_channels=nb_cls,
                use_se=attn_use_se,
                tau=attn_tau,
                use_softmax_spatial=attn_softmax_spatial,
                ablate_attention=ablate_attention
            ).to(self.device)

    @torch.no_grad()
    def _embeddings_from_feats(self, feats: torch.Tensor, flatten: bool = True) -> torch.Tensor:
        if flatten:
            N = feats.size(0)
            return feats.view(N, -1).cpu()
        return feats.mean(dim=[2, 3]).cpu()  # global feature

    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False,
        return_task_embeddings: bool = False,
        return_full: bool = False
    ) -> Any:
        x = x.to(self.device)
        feats = self.trunk(x)  # [N, C, H, W]

        if return_task_embeddings:
            outputs: Dict[str, torch.Tensor] = {}
            task_embeddings: Dict[str, torch.Tensor] = {}
            for task_name, head in self.task_heads.items():
                logits, _A = head(feats)
                outputs[task_name] = logits
                task_embeddings[task_name] = feats.mean(dim=[2, 3]).cpu()
            return outputs, task_embeddings

        if return_embeddings:
            N, C, H, W = feats.shape
            feats_flat = feats.view(N, -1).cpu()
            return feats_flat

        if return_full:
            outputs = {}
            for t, head in self.task_heads.items():
                logits, A = head(feats)
                outputs[t] = {'logits': logits, 'attn': A}
            return outputs
        else:
            outputs = {}
            for t, head in self.task_heads.items():
                logits, _A = head(feats)
                outputs[t] = logits
            return outputs


class TaskSpecificModel(nn.Module):
    """
    Wrap pour extraire les logits d'une tâche spécifique (pour Grad-CAM / IG).
    """

    def __init__(self, model: MultiTaskPatchGAN, task_name: str):
        super().__init__()
        self.model = model
        self.task_name = task_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = self.model(x, return_full=False)
        return outs[self.task_name]


# ---------- Chargement de poids & infos ----------
def load_model_weights(model: nn.Module, path: str, device: torch.device, strict: bool = True):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    else:
        state = ckpt

    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith('module.') else k  # retire DataParallel
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=strict)
    if missing:
        print(f"[load] Missing keys ({len(missing)}): {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[load] Unexpected keys ({len(unexpected)}): {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    print(f"[load] strict={strict} -> OK (si pas d'exception)")


def checkpoint_has_se(path: str, device: torch.device | str = 'cpu') -> bool:
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict):
        sd = sd.get('model', sd.get('state_dict', sd))
    return any('.se.mlp.' in k for k in sd.keys())


def print_model_parameters(model: MultiTaskPatchGAN):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trunk_params = sum(p.numel() for p in model.trunk.parameters() if p.requires_grad)
    print("==== Comptage des paramètres ====")
    print(f"Paramètres totaux du modèle : {total_params}")
    print(f"Paramètres du tronc : {trunk_params}")
    print("Paramètres par tête de tâche :")
    for task, head in model.task_heads.items():
        head_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
        in_channels = head.cls_conv.in_channels
        out_channels = head.cls_conv.out_channels
        print(f"  - Tâche '{task}' : {head_params} paramètres "
              f"(in_channels={in_channels}, out_channels={out_channels})")
    print("=================================")
