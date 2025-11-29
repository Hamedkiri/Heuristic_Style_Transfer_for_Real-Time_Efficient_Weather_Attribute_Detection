# model_utils.py
import os
import json
from typing import Dict, Any

import torch
import torch.nn as nn

from Models.models_PMG  import (
    MultiTaskPatchGANGramModelNonOverlapV2
)


def print_model_parameters(model: nn.Module):
    """
    Décompte détaillé des paramètres pour MultiTaskPatchGANGramModelNonOverlapV2.
    (Version extraite telle quelle de ton script, avec léger nettoyage.)
    """
    def count_params(mod, trainable_only=True):
        if mod is None:
            return 0
        if trainable_only:
            return sum(p.numel() for p in mod.parameters() if p.requires_grad)
        return sum(p.numel() for p in mod.parameters())

    total_trainable = count_params(model, True)
    total_params = count_params(model, False)

    blocks = {
        'feature_extractor': getattr(model, 'feature_extractor', None),
        'se':                getattr(model, 'se', None),
        'chan_proj':         getattr(model, 'chan_proj', None),
        'token_proj':        getattr(model, 'token_proj', None),
        'transformer':       getattr(model, 'transformer', None),
    }

    block_stats = {}
    for name, mod in blocks.items():
        if mod is None:
            block_stats[name] = {'trainable': 0, 'total': 0, 'identity': False}
        else:
            is_identity = isinstance(mod, nn.Identity)
            block_stats[name] = {
                'trainable': 0 if is_identity else count_params(mod, True),
                'total':     0 if is_identity else count_params(mod, False),
                'identity':  is_identity
            }

    task_pool = getattr(model, 'task_pool', None)
    per_task_attn = {}
    query_params_total = 0
    if task_pool is not None and hasattr(task_pool, 'query'):
        for t, q in task_pool.query.items():
            n = q.numel()
            per_task_attn[t] = n
            query_params_total += n

    classifiers = getattr(model, 'classifiers', None)
    per_task_cls = {}
    if isinstance(classifiers, nn.ModuleDict):
        for t, head in classifiers.items():
            per_task_cls[t] = {
                'trainable': count_params(head, True),
                'total':     count_params(head, False)
            }

    se_params_train = 0 if block_stats['se']['identity'] else block_stats['se']['trainable']
    transf_params_train = 0 if block_stats['transformer']['identity'] else block_stats['transformer']['trainable']
    total_attention_trainable = se_params_train + transf_params_train + query_params_total

    print("==== Paramètres du modèle ====")
    print(f"Total params (all):           {total_params:,}")
    print(f"Total trainable params:       {total_trainable:,}")
    print("---- Blocs principaux ----")
    for name in ['feature_extractor', 'se', 'chan_proj', 'token_proj', 'transformer']:
        s = block_stats[name]
        if s['identity']:
            print(f"{name:>18}: Identity (0)")
        else:
            print(f"{name:>18}: trainable={s['trainable']:,} | total={s['total']:,}")

    print("---- Attention (global) ----")
    print(f"{'SE (channel)':>18}: {se_params_train:,} param. (trainables)")
    print(f"{'Transformer':>18}: {transf_params_train:,} param. (trainables)")
    print(f"{'Queries (tasks)':>18}: {query_params_total:,} param. (somme des d_model par tâche)")
    print(f"{'TOTAL ATTENTION':>18}: {total_attention_trainable:,} param. (trainables)")

    if per_task_attn or per_task_cls:
        print("-- Par tâche --")
        all_tasks = sorted(set(list(per_task_attn.keys()) + list(per_task_cls.keys())))
        for t in all_tasks:
            attn_n = per_task_attn.get(t, 0)
            cls_tr = per_task_cls.get(t, {}).get('trainable', 0)
            cls_to = per_task_cls.get(t, {}).get('total', 0)
            print(f"Task '{t}': attention(query)={attn_n:,} | "
                  f"classifier trainable={cls_tr:,} total={cls_to:,}")

    if hasattr(model, 'use_channel_attention'):
        print(f"use_channel_attention: {model.use_channel_attention}")
    if hasattr(model, 'use_token_attention'):
        print(f"use_token_attention:   {model.use_token_attention}")
    if hasattr(model, 'gram_channels'):
        print(f"gram_channels:         {model.gram_channels}")
    if hasattr(model, 'patch_div'):
        print(f"patch_div:             {model.patch_div}")
    print("=================================")


def find_sidecar_hparams(model_path: str) -> Dict[str, Any] | None:
    """
    Cherche un JSON d'hparams à côté du checkpoint :
      - best_overall_hyperparameters.json
      - best_hyperparameters.json
      - <basename>.json
    """
    base_dir = os.path.dirname(model_path)
    candidates = [
        os.path.join(base_dir, "best_overall_hyperparameters.json"),
        os.path.join(base_dir, "best_hyperparameters.json"),
        os.path.splitext(model_path)[0] + ".json",
    ]
    for p in candidates:
        if os.path.isfile(p):
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                if "hparams" in data and isinstance(data["hparams"], dict):
                    return data["hparams"]
                return data
            except Exception:
                pass
    return None


def build_model_from_hparams(tasks: Dict[str, list],
                             hparams: Dict[str, Any],
                             device: torch.device) -> nn.Module:
    """
    Construit MultiTaskPatchGANGramModelNonOverlapV2 à partir d'un dict d'hparams.
    """
    def _get(k, default):
        return hparams[k] if k in hparams else default

    ndf                = int(_get("ndf", 64))
    patch_size         = int(_get("patch_size", 70))
    patch_div          = int(_get("patch_div", 4))
    gram_channels      = int(_get("gram_channels", 64))
    d_model            = int(_get("d_model", 256))
    transformer_layers = int(_get("transformer_layers", 1))
    transformer_heads  = int(_get("transformer_heads", 4))
    use_token_attention   = bool(_get("use_token_attention", True))
    use_channel_attention = bool(_get("use_channel_attention", False))
    norm              = _get("norm", "batch")

    num_classes_per_task = {t: len(tasks[t]) for t in tasks.keys()}

    model = MultiTaskPatchGANGramModelNonOverlapV2(
        input_nc=3,
        ndf=ndf,
        norm=norm,
        patch_size=patch_size,
        patch_div=patch_div,
        num_classes_per_task=num_classes_per_task,
        use_channel_attention=use_channel_attention,
        use_token_attention=use_token_attention,
        gram_channels=gram_channels,
        d_model=d_model,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
    ).to(device)

    return model


def load_best_model(model: nn.Module,
                    model_path: str,
                    device: torch.device,
                    strict: bool = True) -> nn.Module:
    """
    Chargement robuste du checkpoint (strict=True puis fallback strict=False).
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=strict)
        print(f"Model loaded (strict={strict}) from {model_path}")
    except RuntimeError:
        print("[WARN] strict=True a échoué, tentative avec strict=False…")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(" - Missing keys:", missing)
        if unexpected:
            print(" - Unexpected keys:", unexpected)
        print(f"Model loaded (strict=False) from {model_path}")

    model.to(device)
    model.eval()
    return model
