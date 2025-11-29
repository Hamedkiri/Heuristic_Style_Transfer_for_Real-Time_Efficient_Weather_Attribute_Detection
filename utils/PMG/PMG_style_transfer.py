# style_transfer.py
from typing import List

import torch
import torch.optim as optim
from torchvision import transforms
from PIL import Image

import torch.nn.functional as F  # si besoin

def extract_patch_grams_for_image(model, img_tensor: torch.Tensor, detach: bool = True) -> List[torch.Tensor]:
    """
    Calcule la matrice de Gram pour chaque patch disjoint de l'image.
    On s'assure que le batch est de taille 1.
    """
    if img_tensor.shape[0] != 1:
        img_tensor = img_tensor[0:1]

    if detach:
        with torch.no_grad():
            feats = model.feature_extractor(img_tensor)
    else:
        feats = model.feature_extractor(img_tensor)

    B, C, H, W = feats.shape
    assert B == 1, "batch=1 attendu pour le transfert de style"

    patch_div = model.patch_div
    patch_h = H // patch_div
    patch_w = W // patch_div

    patches = feats.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
    newH = patches.size(2)
    newW = patches.size(3)
    nb_patches = newH * newW

    patches = patches.permute(0, 1, 2, 4, 3, 5).reshape(1, C, nb_patches, patch_h, patch_w)
    patches = patches.permute(0, 2, 1, 3, 4).contiguous()     # (1, nb_patches, C, ph, pw)
    patches = patches.reshape(1, nb_patches, C, patch_h * patch_w)

    grams_list = []
    for p_i in range(nb_patches):
        Fp = patches[0, p_i, :, :]  # (C, N)
        N = patch_h * patch_w
        G = torch.matmul(Fp, Fp.t()) / float(N)
        grams_list.append(G)

    return grams_list


def patch_gram_style_loss(grams_gen, grams_style):
    """
    Moyenne des erreurs L2 entre les matrices de Gram générées et celles du style.
    """
    assert len(grams_gen) == len(grams_style), "Le nombre de patchs doit être identique"
    loss_val = 0.0
    for G_gen, G_style in zip(grams_gen, grams_style):
        loss_val += F.mse_loss(G_gen, G_style)
    return loss_val / len(grams_gen)


def run_patch_gram_style_transfer(model,
                                  style_img_tensor: torch.Tensor,
                                  num_iterations: int = 300,
                                  lr: float = 0.05,
                                  init_type: str = "noise",
                                  device: torch.device | str = 'cpu',
                                  target_loss: float = 0.001) -> Image.Image:
    """
    Optimise une image générée pour avoir la même signature de style par patch
    que l'image de style (pas de fusion avec une image de contenu).
    """
    if style_img_tensor.shape[0] != 1:
        style_img_tensor = style_img_tensor[0:1]

    style_grams = extract_patch_grams_for_image(model, style_img_tensor, detach=True)

    B, C, H, W = style_img_tensor.shape
    assert B == 1

    if init_type == "noise":
        gen_data = torch.rand((1, C, H, W), device=device)
    else:
        gen_data = torch.full((1, C, H, W), 0.5, device=device)
    gen_data.requires_grad_(True)

    optimizer = optim.Adam([gen_data], lr=lr)

    for it in range(num_iterations):
        optimizer.zero_grad()
        grams_gen = extract_patch_grams_for_image(model, gen_data, detach=False)
        style_loss = patch_gram_style_loss(grams_gen, style_grams)
        style_loss.backward(retain_graph=True)
        optimizer.step()
        print(f"[{it + 1}/{num_iterations}] style loss = {style_loss.item():.6f}")
        if style_loss.item() < target_loss:
            print(f"Target style loss reached: {style_loss.item():.6f} < {target_loss}")
            break

    with torch.no_grad():
        gen_data.clamp_(0, 1)
        gen_img = gen_data.detach().cpu()[0]
        to_pil = transforms.ToPILImage()
        gen_img_pil = to_pil(gen_img)

    return gen_img_pil
