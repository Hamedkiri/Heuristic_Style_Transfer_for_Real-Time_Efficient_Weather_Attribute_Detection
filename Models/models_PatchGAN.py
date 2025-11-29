
import torch.nn as nn
import functools
from torch.utils.tensorboard import SummaryWriter

# -----------------------------------------------------------------------
# MODULE D'ATTENTION SPATIALE & TÊTES
# -----------------------------------------------------------------------
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mask = self.conv_mask(x)  # [N, 1, H, W]
        mask = self.sigmoid(mask)  # [N, 1, H, W]
        return x * mask


class TaskHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.attention = SpatialAttention(in_channels)
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=1, padding=1)

    def forward(self, feat_map):
        attn_feat = self.attention(feat_map)
        out = self.final_conv(attn_feat)
        out = out.mean(dim=[2, 3])
        return out


# -----------------------------------------------------------------------
# MODELE PATCHGAN MULTI-TACHES
# -----------------------------------------------------------------------
class MultiTaskPatchGAN(nn.Module):
    def __init__(self, tasks_dict, input_nc=3, ndf=64, norm="instance", patch_size=70, tensorboard_logdir=None):
        super().__init__()
        self.tasks_dict = tasks_dict
        self.tensorboard_logdir = tensorboard_logdir
        self.writer = SummaryWriter(log_dir=tensorboard_logdir) if tensorboard_logdir else None

        # Choix de la normalisation
        if norm == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.BatchNorm2d

        layers = []
        num_filters = ndf
        kernel_size = 4
        padding = 1
        stride = 2
        receptive_field_size = patch_size
        in_nc = input_nc

        # Construction dynamique de la trunk (noyau)
        while receptive_field_size > 4 and num_filters <= 512:
            layers.append(nn.Conv2d(in_nc, num_filters, kernel_size, stride, padding))
            layers.append(norm_layer(num_filters))
            layers.append(nn.LeakyReLU(0.2, True))
            in_nc = num_filters
            num_filters *= 2
            receptive_field_size /= stride

        final_conv = nn.Conv2d(in_nc, num_filters, kernel_size, 1, padding)
        layers.append(final_conv)
        layers.append(norm_layer(num_filters))
        layers.append(nn.LeakyReLU(0.2, True))

        self.trunk = nn.Sequential(*layers)

        # Création des têtes pour chaque tâche
        self.task_heads = nn.ModuleDict()
        for task_name, nb_cls in tasks_dict.items():
            self.task_heads[task_name] = TaskHead(
                in_channels=num_filters,
                out_channels=nb_cls
            )

    def forward(self, x):
        feats = self.trunk(x)
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(feats)
        return outputs

    def close_writer(self):
        if self.writer:
            self.writer.close()





# -------------------------------------------------------------------
# 3) PATCHGAN TRONQUÉ MULTI-TÂCHES
# -------------------------------------------------------------------
class MultiTaskPatchGANTest(nn.Module):
    """
    - Tronçon principal (inspiré d'un PatchGAN) : montée en filtre, kernel=4, stride=2...
    - tasks_dict = { "weather": 3, "time": 2, ... } => T têtes
    """
    def __init__(self, tasks_dict, input_nc=3, ndf=64, norm="instance", patch_size=70, device='cpu'):
        super().__init__()
        self.device = device
        self.tasks_dict = tasks_dict  # {task_name: nb_classes}
        if norm == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        else:
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)

        layers = []
        num_filters = ndf
        kernel_size = 4
        padding = 1
        stride = 2
        receptive_field_size = patch_size
        in_nc = input_nc

        # Construction du tronc
        while receptive_field_size > 4 and num_filters <= 512:
            layers.append(nn.Conv2d(in_nc, num_filters, kernel_size, stride, padding))
            layers.append(norm_layer(num_filters))
            layers.append(nn.LeakyReLU(0.2, True))
            in_nc = num_filters
            num_filters *= 2
            receptive_field_size /= stride

        # Couche finale "intermédiaire" (avant classification)
        final_conv = nn.Conv2d(in_nc, num_filters, kernel_size, 1, padding)
        layers.append(final_conv)
        layers.append(norm_layer(num_filters))
        layers.append(nn.LeakyReLU(0.2, True))

        self.trunk = nn.Sequential(*layers).to(self.device)

        # Têtes par tâche
        # Têtes par tâche : on crée une tête uniquement pour chaque tâche présente dans tasks_dict
        self.task_heads = nn.ModuleDict()
        for task_name, nb_cls in tasks_dict.items():
            self.task_heads[task_name] = TaskHead(num_filters, nb_cls).to(self.device)

    def forward(self, x, return_embeddings=False, return_task_embeddings=False):
        """
        - return_embeddings=True : renvoie la feature map aplatie => [N, C*H*W] ou [N,C,H,W] ?
        - return_task_embeddings=True : renvoie un dict {task_name: embeddings}, dans un patchGAN plus classique
          ce serait la sortie avant la conv finale par tâche.
        """
        x = x.to(self.device)
        feats = self.trunk(x)  # [N, num_filters, H, W]

        outputs = {}
        if return_task_embeddings:
            task_embeddings = {}
            for task_name, head in self.task_heads.items():
                out = head(feats)      # => [N, nb_cls]
                outputs[task_name] = out
                # Pour simplifier, on utilise la moyenne spatiale des features partagées comme embedding
                embed = feats.mean(dim=[2,3])  # [N, C]
                task_embeddings[task_name] = embed.cpu()
            return outputs, task_embeddings
        else:
            for task_name, head in self.task_heads.items():
                out = head(feats)
                outputs[task_name] = out
            if return_embeddings:
                N, C, H, W = feats.shape
                feats_flat = feats.view(N, -1).cpu()
                return feats_flat
            else:
                return outputs