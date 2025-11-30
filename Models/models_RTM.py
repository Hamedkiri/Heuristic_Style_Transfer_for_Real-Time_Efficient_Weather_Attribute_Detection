# model_multihead.py
import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn


class TaskAttentionHead(nn.Module):
    """Attention 'query par tâche' sur tokens spatiaux. Entrée: [B,HW,C] → Sortie: [B,C]."""
    def __init__(self, dim: int, token_dim: Optional[int] = None):
        super().__init__()
        d = token_dim or dim
        self.q = nn.Parameter(torch.randn(1, 1, d))   # requête apprise
        self.proj = nn.Linear(dim, d, bias=False)     # projette tokens -> d
        self.out  = nn.Linear(d, dim, bias=False)     # d -> C

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, HW, C]
        T = self.proj(tokens)                                    # [B, HW, d]
        q = self.q.expand(T.size(0), -1, -1)                     # [B, 1, d]
        attn = torch.softmax((q @ T.transpose(1, 2)) / math.sqrt(T.size(-1)), dim=-1)  # [B,1,HW]
        h = (attn @ T).squeeze(1)                                # [B, d]
        return self.out(h)                                       # [B, C]


class MultiHeadAttentionPerTaskModel(nn.Module):
    """
    ResNet tronqué (sans avgpool/fc) + (optionnel) attention par tâche.
    - use_attention=True  : tokens [B,HW,C] -> TaskAttentionHead -> MLP par tâche
    - use_attention=False : ablation, GAP [B,C] -> MLP par tâche
    Supporte le retour des embeddings (par tâche et/ou partagés) pour t-SNE.
    """
    def __init__(
        self,
        base_encoder: nn.Module,
        truncate_after_layer: int,
        tasks: Dict[str, Union[List, int]],
        device: Union[str, torch.device] = "cpu",
        use_attention: bool = True,
        attn_token_dim: Optional[int] = None,
        cls_hidden_dims: Optional[List[int]] = None,
        cls_num_layers: int = 0,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.tasks = {
            t: (len(v) if isinstance(v, (list, tuple)) else int(v))
            for t, v in tasks.items()
        }
        self.use_attention = use_attention

        # 1) Tronque le backbone sans avgpool/fc
        enc_layers = list(base_encoder.children())[:-2]  # conv1..layer4
        truncate_after_layer = max(1, min(truncate_after_layer, len(enc_layers)))
        self.truncated_encoder = nn.Sequential(*enc_layers[:truncate_after_layer]).to(self.device)

        # 2) Infère C (nb de canaux)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224).to(self.device)
            feat  = self.truncated_encoder(dummy)  # [1,C,H,W]
            C = feat.shape[1]
        self.num_features = C

        # 3) Heads attentionnelles + classifieurs MLP
        self.attentions  = nn.ModuleDict()
        self.classifiers = nn.ModuleDict()
        cls_hidden_dims = cls_hidden_dims or []

        for task, n_cls in self.tasks.items():
            key = task.replace(' ', '_')
            if self.use_attention:
                self.attentions[f"attention_{key}"] = TaskAttentionHead(C, attn_token_dim)
            # MLP: C -> hidden_dims[:cls_num_layers] -> n_cls
            hds = cls_hidden_dims[:cls_num_layers]
            dims = [C] + hds
            layers = []
            for i in range(len(dims) - 1):
                layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(inplace=True)]
            layers.append(nn.Linear(dims[-1], n_cls))
            self.classifiers[f"classifier_{key}"] = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_task_embeddings: bool = False,
        return_shared_embedding: bool = False
    ):
        x = x.to(self.device)
        feat = self.truncated_encoder(x)                # [B,C,H,W]
        B, C, H, W = feat.shape

        # Embedding 'partagé' pour t-SNE global
        shared = feat.mean(dim=(2, 3))                  # [B,C]

        logits_dict, task_embeds = {}, {}

        if self.use_attention:
            tokens = feat.flatten(2).transpose(1, 2)    # [B,HW,C]
            for attn_name, attn in self.attentions.items():
                task = attn_name.replace("attention_", "").replace('_', ' ')
                cls_name = f"classifier_{task.replace(' ', '_')}"
                h = attn(tokens)                        # [B,C] embedding par tâche
                logits_dict[task] = self.classifiers[cls_name](h)
                task_embeds[task] = h
        else:
            for task in self.tasks:
                cls_name = f"classifier_{task.replace(' ', '_')}"
                logits_dict[task] = self.classifiers[cls_name](shared)
                task_embeds[task] = shared

        if return_task_embeddings and return_shared_embedding:
            return logits_dict, task_embeds, shared
        if return_task_embeddings:
            return logits_dict, task_embeds
        if return_shared_embedding:
            return logits_dict, shared
        return logits_dict


class TaskSpecificModel(nn.Module):
    """Wrapper pour n’inférer que sur une tâche donnée (conserve le backbone partagé)."""
    def __init__(self, model: MultiHeadAttentionPerTaskModel, task_name: str):
        super().__init__()
        self.model = model
        self.task_name = task_name

    def forward(self, x):
        outputs = self.model(x)
        return outputs[self.task_name]


def print_model_parameters(model: MultiHeadAttentionPerTaskModel):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.truncated_encoder.parameters() if p.requires_grad)

    if hasattr(model, "pool") and isinstance(getattr(model, "pool"), nn.Module):
        pool_params = sum(p.numel() for p in model.pool.parameters() if p.requires_grad)
    else:
        pool_params = 0

    truncated_layers = list(model.truncated_encoder.children())
    num_truncated_layers = len(truncated_layers)

    print("==== Paramètres du modèle ====")
    print(f"Paramètres totaux du modèle : {total_params}")
    print(f"Nombre de blocs dans truncated_encoder : {num_truncated_layers}")
    print(f"Paramètres de l'encodeur tronqué : {encoder_params}")
    print(f"Paramètres de la couche de pooling : {pool_params} (0 si GAP implicite)")

    if hasattr(model, "attentions") and isinstance(model.attentions, nn.ModuleDict):
        print("Modules d'attention par tâche :")
        for key, attn in model.attentions.items():
            count = sum(p.numel() for p in attn.parameters() if p.requires_grad)
            theoretical = 3 * (model.num_features ** 2)  # approx. 3 proj. Q/K/V
            print(f"  {key}: {count} paramètres (théorie≈{theoretical})")
    else:
        print("Modules d'attention par tâche : (aucun)")

    if hasattr(model, "classifiers") and isinstance(model.classifiers, nn.ModuleDict):
        print("Modules classifieurs par tâche :")
        for key, classifier in model.classifiers.items():
            count = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
            if isinstance(classifier, nn.Linear):
                num_classes = classifier.out_features
                in_feats = classifier.in_features
                theoretical = in_feats * num_classes + num_classes
            elif isinstance(classifier, nn.Sequential):
                last_linear = None
                for m in reversed(classifier):
                    if isinstance(m, nn.Linear):
                        last_linear = m
                        break
                if last_linear is not None:
                    num_classes = last_linear.out_features
                    in_feats = last_linear.in_features
                    theoretical = in_feats * num_classes + num_classes
                else:
                    num_classes, theoretical = "?", "?"
            else:
                num_classes, theoretical = "?", "?"
            print(f"  {key}: {count} paramètres (théorie≈{theoretical})")
    else:
        print("Modules classifieurs par tâche : (aucun)")

    print("=================================")
