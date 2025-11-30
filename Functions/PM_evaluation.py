# eval_utils.py
import os
import json
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from Models.models_PM import TaskSpecificModel
from utils.datasets_utils import map_folder_to_class, _get_loader_paths


@torch.no_grad()
def compute_attn_embeddings_per_task_with_paths(model, loader, device, tasks_json):
    """
    Extrait des embeddings 'par tâche' pondérés par l'attention:
      e_t = sum_{h,w} (feat * A_t) / sum_{h,w} A_t  -> vecteur [C]
    Renvoie:
      - embeddings_data: dict(task -> np.array [N_t, C])
      - labels_data:     dict(task -> np.array [N_t])
      - img_paths_data:  dict(task -> List[str]) (aligné avec embeddings/labels de cette tâche)
    Seuls les échantillons avec un label défini pour la tâche sont inclus.
    """
    model.eval()
    embeddings_data = {t: [] for t in tasks_json.keys()}
    labels_data     = {t: [] for t in tasks_json.keys()}
    paths_data      = {t: [] for t in tasks_json.keys()}

    # hook pour récupérer la sortie du trunk
    feats_cache = []
    def hook_fn(_m, _inp, out):
        feats_cache.append(out.detach())
    h = model.trunk.register_forward_hook(hook_fn)

    # chemins d'images dans l'ordre du loader (shuffle=False)
    all_paths = _get_loader_paths(loader)
    ptr = 0

    for imgs, labels in loader:
        bsz = imgs.size(0)
        imgs = imgs.to(device, non_blocking=True)

        # forward complet pour récupérer les cartes d'attention
        outs = model(imgs, return_full=True)  # dict[task] -> {'logits':..., 'attn':...}
        feats = feats_cache.pop(0)            # [N,C,H,W] sortie du trunk
        N, C, H, W = feats.shape

        # par tâche: projeter via A_t
        for task in tasks_json.keys():
            A = outs[task]['attn']            # [N,1,H,W]
            lbl_t = labels[task]              # liste de labels (tenseurs ou None)
            # on ne garde que les échantillons avec label défini
            for i in range(N):
                # indice global → chemin image
                img_path = all_paths[ptr + i]
                lab = lbl_t[i]
                if lab is None:
                    continue
                lab = int(lab) if not torch.is_tensor(lab) else int(lab.item())
                # embedding attention-pondéré
                Ai = A[i]                     # [1,H,W]
                Fi = feats[i]                 # [C,H,W]
                num = (Fi * Ai).sum(dim=(1,2))         # [C]
                den = Ai.sum(dim=(1,2)).clamp_min(1e-6)  # [1]
                emb = (num / den).cpu().numpy()        # [C]
                embeddings_data[task].append(emb)
                labels_data[task].append(lab)
                paths_data[task].append(img_path)

        ptr += bsz

    # convert lists -> arrays
    for t in tasks_json.keys():
        if len(embeddings_data[t]) > 0:
            embeddings_data[t] = np.stack(embeddings_data[t], axis=0)
            labels_data[t]     = np.array(labels_data[t])
        else:
            embeddings_data[t] = np.empty((0, 0), dtype=np.float32)
            labels_data[t]     = np.array([], dtype=np.int64)

    h.remove()
    return embeddings_data, labels_data, paths_data

# -------------------------------------------------------------------
# Colormaps OpenCV
# -------------------------------------------------------------------
colormap_dict = {
    'autumn': cv2.COLORMAP_AUTUMN,
    'bone': cv2.COLORMAP_BONE,
    'hot': cv2.COLORMAP_HOT,
    'afmhot': cv2.COLORMAP_TURBO,
    'inferno': cv2.COLORMAP_INFERNO,
    'jet': cv2.COLORMAP_JET,
    'turbo': cv2.COLORMAP_TURBO,
    'viridis': cv2.COLORMAP_VIRIDIS,
    'magma': cv2.COLORMAP_MAGMA,
}


# -------------------------------------------------------------------
# Encadré texte + overlay
# -------------------------------------------------------------------
def annotate_and_save(
    img: Image.Image,
    text_lines: List[str],
    out_path: str,
    max_width_ratio: float = 0.35,
    min_font_px: int = 18,
    max_font_px: int = 32,
    pad_px: int = 10,
    bg_alpha: int = 220
):
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    W, H = img.size

    # Police
    font_path = None
    for name in ["DejaVuSans-Bold.ttf", "arialbd.ttf", "arial.ttf"]:
        try:
            font_path = name
            break
        except IOError:
            font_path = None

    if font_path is None:
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(font_path, size=min_font_px)

    draw_tmp = ImageDraw.Draw(img)
    target_width = W * max_width_ratio

    fsize = min_font_px
    while fsize < max_font_px and font_path is not None:
        cand = ImageFont.truetype(font_path, fsize + 2)
        test_w = max(draw_tmp.textsize(t, cand)[0] for t in text_lines)
        if test_w > target_width:
            break
        fsize += 2
        font = cand

    line_h = draw_tmp.textsize("Ag", font=font)[1]
    block_w = min(
        target_width,
        max(draw_tmp.textsize(t, font=font)[0] for t in text_lines)
    ) + 2 * pad_px
    block_h = line_h * len(text_lines) + 2 * pad_px
    x0, y0 = 10, 10

    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(
        [x0, y0, x0 + block_w, y0 + block_h],
        fill=(255, 255, 255, bg_alpha)
    )

    for i, line in enumerate(text_lines):
        tx, ty = x0 + pad_px, y0 + pad_px + i * line_h
        # petit contour noir
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((tx + dx, ty + dy), line, font=font, fill=(0, 0, 0, 255))
        draw.text((tx, ty), line, font=font, fill=(34, 139, 34, 255))

    result = Image.alpha_composite(img, overlay).convert("RGB")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.save(out_path)


# -------------------------------------------------------------------
# run_inference  (copie quasi inchangée de ton code)
# -------------------------------------------------------------------
def run_inference(
    model,
    image_folder: str,
    transform,
    device: torch.device,
    classes: Dict[str, list] | list,
    num_samples: Optional[int] = None,
    save_dir: Optional[str] = None,
    save_test_images: bool = False,
    visualize_gradcam: bool = False,
    save_gradcam_images: bool = False,
    gradcam_task: Optional[str] = None,
    colormap: str = "hot",
):
    """
    Version modulaire de run_inference (reprend ton implémentation).
    """

    from utils.datasets_utils import collect_image_paths  # pour éviter cycle d'import

    is_multi = isinstance(classes, dict)
    classes_ci = {k.lower(): v for k, v in classes.items()} if is_multi else {}
    paths = collect_image_paths(image_folder)
    if not paths:
        raise RuntimeError(f"Aucune image trouvée dans « {image_folder} »")
    import random
    if num_samples and len(paths) > num_samples:
        paths = random.sample(paths, num_samples)

    model = model.to(device).eval()
    results: Dict[str, Any] = {}

    # --- Grad-CAM prep ---
    grad_cam = None
    if visualize_gradcam or save_gradcam_images:
        if is_multi:
            if gradcam_task is None:
                gradcam_task = list(classes.keys())[0]
            if gradcam_task not in classes:
                raise ValueError(f"Tâche Grad-CAM inconnue : {gradcam_task}")
            gradcam_model = TaskSpecificModel(model, gradcam_task).to(device)
        else:
            gradcam_model = model
        gradcam_model.eval()

        # dernière conv du tronc
        for layer in reversed(list(gradcam_model.model.trunk)):
            if isinstance(layer, torch.nn.Conv2d):
                target_layer = layer
                break
        else:
            raise ValueError("Aucun nn.Conv2d trouvé pour Grad-CAM.")

        grad_cam = GradCAM(model=gradcam_model, target_layers=[target_layer])
        cmap_code = {
            "hot": cv2.COLORMAP_HOT,
            "jet": cv2.COLORMAP_JET,
            "turbo": cv2.COLORMAP_TURBO,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "inferno": cv2.COLORMAP_INFERNO,
        }.get(colormap, cv2.COLORMAP_HOT)

    for pth in paths:
        img = Image.open(pth).convert("RGB")
        img_np = np.array(img)
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)

        lines: List[str] = []
        pred_gc_idx = None
        prob_gc = None

        if is_multi:
            preds = {}
            for task, logits in out.items():
                probs = F.softmax(logits, 1)[0]
                prob, idx = probs.max(0)
                name = classes_ci[task.lower()][idx] if idx < len(classes_ci[task.lower()]) else str(idx.item())
                preds[task] = {"predicted_class": name, "probability": prob.item()}
                lines.append(f"{task}: {name} ({prob:.2f})")
                if task == gradcam_task:
                    pred_gc_idx, prob_gc = int(idx), float(prob)
        else:
            probs = F.softmax(out, 1)[0]
            prob, idx = probs.max(0)
            name = classes[idx] if idx < len(classes) else str(idx.item())
            preds = {"predicted_class": name, "probability": prob.item()}
            lines = [f"{name} ({prob:.2f})"]
            pred_gc_idx, prob_gc = int(idx), float(prob)

        results[pth] = preds

        if save_dir and save_test_images:
            rel = os.path.relpath(pth, image_folder)
            annotate_and_save(img, lines, os.path.join(save_dir, rel))

        if grad_cam is not None:
            x_gc = x.clone().requires_grad_(True)
            cam = grad_cam(input_tensor=x_gc, targets=[ClassifierOutputTarget(pred_gc_idx)])[0]
            cam = (cam - cam.min()) / (cam.ptp() + 1e-8)
            heat = cv2.applyColorMap(
                np.uint8(255 * cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))),
                cmap_code,
            )
            heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
            fusion = np.clip(0.5 * img_np + 0.5 * heat, 0, 255).astype(np.uint8)

            task_lbl = gradcam_task if is_multi else "Task"
            class_lbl = (
                classes_ci[gradcam_task.lower()][pred_gc_idx] if is_multi
                else classes[pred_gc_idx]
            )
            txt_line = f"{task_lbl}: {class_lbl} ({prob_gc:.2f})"
            fusion_pil = Image.fromarray(fusion)
            fname = os.path.splitext(os.path.basename(pth))[0]
            out_dir = os.path.join(save_dir, "GradCAM", class_lbl)
            os.makedirs(out_dir, exist_ok=True)
            annotate_and_save(
                fusion_pil, [txt_line],
                os.path.join(out_dir, f"{fname}_fusion.jpg")
            )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "inference_results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results



# -------------------------------------------------------------------
# 5) FONCTION DE TEST AVEC OPTIONS Grad-CAM ET Integrated Gradients
# -------------------------------------------------------------------
def test_classifier(model, test_loader, criterions, writer, save_dir, device, tasks_json,
                    prob_threshold=0.5, visualize_gradcam=False, save_gradcam_images=False,
                    measure_time=False, save_test_images=False, gradcam_task=None, colormap='hot',
                    integrated_gradients=False, integrated_gradients_task=None):

    model.eval()
    total_loss = 0.0
    total_samples = 0
    times = []

    all_preds = {t: [] for t in tasks_json.keys()}
    all_labels = {t: [] for t in tasks_json.keys()}

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Préparation de Grad-CAM si activé
    if visualize_gradcam:
        if gradcam_task is None:
            gradcam_task = list(tasks_json.keys())[0]
        if gradcam_task not in tasks_json:
            raise ValueError(f"La tâche '{gradcam_task}' n'existe pas.")
        gradcam_model = TaskSpecificModel(model, gradcam_task).to(device)
        gradcam_model.eval()
        target_layer = None
        for layer in reversed(list(gradcam_model.model.trunk)):
            if isinstance(layer, nn.Conv2d):
                target_layer = layer
                break
        if target_layer is None:
            raise ValueError("No Conv2d layer found for Grad-CAM.")
        grad_cam = GradCAM(model=gradcam_model, target_layers=[target_layer])

    # Préparation d'Integrated Gradients
    if integrated_gradients:
        from captum.attr import IntegratedGradients
        ig_models = {}
        ig = {}

        # Tâches concernées par IG
        if integrated_gradients_task is not None:
            tasks_to_compute = [integrated_gradients_task]
        else:
            tasks_to_compute = list(tasks_json.keys())

        for t in tasks_to_compute:
            ig_models[t] = TaskSpecificModel(model, t).to(device)
            ig_models[t].eval()
            ig[t] = IntegratedGradients(ig_models[t])

    for batch_idx, (inputs, labels) in enumerate(test_loader):
        start_time = time.time()

        inputs = inputs.to(device)
        inputs.requires_grad = True

        with torch.no_grad():
            outputs = model(inputs)

        loss = 0.0
        batch_size = inputs.size(0)
        preds_dict = {}
        max_probs_dict = {}
        labels_dict = {}

        # Calcul de la prédiction / CrossEntropy
        for t, criterion in criterions.items():
            t_labels = labels[t]
            if t_labels is not None:
                t_labels = t_labels.to(device)
                t_out = outputs[t]
                t_loss = criterion(t_out, t_labels)
                loss += t_loss

                probabilities = torch.softmax(t_out, dim=1)
                max_probs, preds = torch.max(probabilities, dim=1)

                unknown_mask = max_probs < prob_threshold
                preds[unknown_mask] = -1

                all_preds[t].extend(preds.cpu().numpy())
                all_labels[t].extend(t_labels.cpu().numpy())

                preds_dict[t] = preds.cpu().numpy()
                max_probs_dict[t] = max_probs.cpu().numpy()
                labels_dict[t] = t_labels.cpu().numpy()
            else:
                preds_dict[t] = np.array([-1]*batch_size)
                max_probs_dict[t] = np.array([-1]*batch_size)
                labels_dict[t] = np.array([-1]*batch_size)

        end_time = time.time()
        times.append(end_time - start_time)

        if integrated_gradients:
            for i in range(batch_size):
                # 1) Lire l'image avec PIL en RGB
                if isinstance(test_loader.dataset, Subset):
                    img_path = test_loader.dataset.dataset.samples[
                        test_loader.dataset.indices[batch_idx * test_loader.batch_size + i]
                    ][0]
                else:
                    img_path = test_loader.dataset.samples[batch_idx * test_loader.batch_size + i][0]

                # On récupère l'image en numpy [H,W,3] en RGB
                orig_img_rgb = np.array(Image.open(img_path).convert('RGB'))

                # 2) Convertir cette image en BGR, car OpenCV utilise BGR
                orig_img_bgr = cv2.cvtColor(orig_img_rgb, cv2.COLOR_RGB2BGR)

                # Sélection des tâches concernées par Integrated Gradients
                if integrated_gradients_task is not None:
                    tasks_ig = [integrated_gradients_task]
                else:
                    tasks_ig = list(tasks_json.keys())

                for t in tasks_ig:
                    input_tensor = inputs[i].unsqueeze(0)
                    baseline = torch.zeros_like(input_tensor)
                    target = int(labels_dict[t][i]) if labels_dict[t][i] >= 0 else 0

                    # Calcul des attributions IG
                    attributions = ig[t].attribute(input_tensor, baseline, target=target)
                    attr_np = attributions.squeeze().cpu().detach().numpy()

                    # Moyenne sur les canaux si >2D
                    if attr_np.ndim > 2:
                        attr_np = np.mean(attr_np, axis=0)

                    # Normaliser [0..1]
                    attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)

                    # Générer la heatmap (BGR direct depuis cv2.applyColorMap)
                    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * attr_np), cv2.COLORMAP_JET)

                    # Optionnel : si vous voulez tout faire en BGR, on ne convertit pas en RGB
                    # => On reste cohérent pour addWeighted
                    # heatmap_bgr est donc [H,W,3] BGR

                    # Redimensionner la heatmap pour correspondre à la taille d'origine
                    h, w = orig_img_bgr.shape[:2]  # (H,W,3)
                    heatmap_bgr = cv2.resize(heatmap_bgr, (w, h))

                    # 3) Faire le blending (addWeighted) en BGR
                    #   => alpha=0.2 => 80% image d'origine, 20% heatmap
                    overlay_bgr = cv2.addWeighted(orig_img_bgr, 0.8, heatmap_bgr, 0.2, 0)

                    # 4) Sauvegarder en BGR (cv2.imwrite attend BGR) => couleurs cohérentes
                    true_label = "Unknown" if labels_dict[t][i] == -1 else tasks_json[t][labels_dict[t][i]]
                    ig_folder = os.path.join(save_dir, "IntegratedGradients", t, true_label)
                    if not os.path.exists(ig_folder):
                        os.makedirs(ig_folder)

                    ig_save_path = os.path.join(
                        ig_folder,
                        f"IntegratedGrad_{t}_{batch_idx * test_loader.batch_size + i}.jpg"
                    )
                    cv2.imwrite(ig_save_path, overlay_bgr)

        # (2) Sauvegarde des images annotées et Grad-CAM
        for i in range(batch_size):
            idx = batch_idx * test_loader.batch_size + i
            if isinstance(test_loader.dataset, Subset):
                img_path = test_loader.dataset.dataset.samples[
                    test_loader.dataset.indices[idx]
                ][0]
            else:
                img_path = test_loader.dataset.samples[idx][0]

            img = Image.open(img_path)
            img_np = np.array(img.convert('RGB'))

            if save_test_images or (visualize_gradcam and save_gradcam_images):
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Sauvegarde d'une version annotée
            if save_test_images:
                annotated_img = img_cv.copy()
                y_start = 30
                y_step = 30
                for j, (t, clist) in enumerate(tasks_json.items()):
                    label_idx = labels_dict[t][i]
                    pred_idx = preds_dict[t][i]
                    prob = max_probs_dict[t][i]

                    true_label = "Unknown" if label_idx == -1 else clist[label_idx]
                    pred_label = "Unknown" if pred_idx == -1 else clist[pred_idx]
                    text = f"{t} - True: {true_label}, Pred: {pred_label}, Prob: {prob:.2f}"

                    y_pos = y_start + j * y_step
                    cv2.putText(
                        annotated_img,
                        text,
                        (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2
                    )

                main_label = "Unknown"
                for t in tasks_json.keys():
                    if t.lower() == "weather type":
                        if labels_dict[t][i] == -1:
                            main_label = "Unknown"
                        else:
                            main_label = tasks_json[t][labels_dict[t][i]]
                        break

                save_folder = os.path.join(save_dir, main_label)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                out_img_path = os.path.join(save_folder, f"test_image_{idx}.jpg")
                cv2.imwrite(out_img_path, annotated_img)

            # Grad-CAM
            if visualize_gradcam and save_gradcam_images:
                input_tensor = inputs[i].unsqueeze(0)
                t_idx = labels_dict[gradcam_task][i]
                pred_val = preds_dict[gradcam_task][i]
                prob = max_probs_dict[gradcam_task][i]

                true_label = "Unknown" if t_idx == -1 else tasks_json[gradcam_task][t_idx]
                pred_label = "Unknown" if pred_val == -1 else tasks_json[gradcam_task][pred_val]

                text = f"{gradcam_task} - True: {true_label}, Pred: {pred_label}, Prob: {prob:.2f}"

                target = [ClassifierOutputTarget(t_idx)]
                grayscale_cam = grad_cam(input_tensor=input_tensor, targets=target)[0]

                grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
                h, w = img_np.shape[:2]
                cam_resized = cv2.resize(grayscale_cam, (w, h))

                if colormap not in colormap_dict:
                    cmap_code = cv2.COLORMAP_HOT
                else:
                    cmap_code = colormap_dict[colormap]

                cmap_code = colormap_dict.get(colormap, cv2.COLORMAP_HOT)
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cmap_code)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                visualization = 0.5 * heatmap + 0.5 * img_np
                visualization = np.clip(visualization, 0, 255).astype(np.uint8)
                cv2.putText(visualization, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                gradcam_folder = os.path.join(save_dir, "GradCAM", true_label)
                os.makedirs(gradcam_folder, exist_ok=True)
                gradcam_save_path = os.path.join(gradcam_folder, f"GradCAM_{idx}.jpg")
                cv2.imwrite(gradcam_save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    average_loss = total_loss / total_samples

    metrics = {}
    for t in tasks_json.keys():
        if len(all_preds[t]) > 0:
            preds_np = np.array(all_preds[t])
            labels_np = np.array(all_labels[t])
            valid = preds_np != -1
            if valid.sum() > 0:
                acc = np.mean(preds_np[valid] == labels_np[valid])
                prec = precision_score(labels_np[valid], preds_np[valid], average='weighted', zero_division=0)
                rec = recall_score(labels_np[valid], preds_np[valid], average='weighted', zero_division=0)
                f1 = f1_score(labels_np[valid], preds_np[valid], average='weighted', zero_division=0)
                conf = confusion_matrix(labels_np[valid], preds_np[valid], labels=list(range(len(tasks_json[t]))))
            else:
                acc = prec = rec = f1 = 0.0
                conf = np.zeros((len(tasks_json[t]), len(tasks_json[t])))
            metrics[t] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "confusion_matrix": conf.tolist()
            }
            print(f"Tâche {t} - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
            print(f"Matrice de confusion pour {t}:\n{conf}\n")
        else:
            metrics[t] = {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "confusion_matrix": None
            }

    valid_accs = [m["accuracy"] for m in metrics.values() if m["accuracy"] is not None]
    avg_accuracy = float(np.mean(valid_accs)) if valid_accs else 0.0

    print(f"Performance moyenne - Acc: {avg_accuracy:.4f}")
    metrics["average"] = {"accuracy": avg_accuracy}
    metrics_path = os.path.join(save_dir, "test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Métriques enregistrées dans {metrics_path}")

    if writer:
        writer.add_scalar("Test/Loss", average_loss)
        writer.add_scalar("Test/Average_Accuracy", avg_accuracy)
        for t, m in metrics.items():
            if m["accuracy"] is not None and t != "average":
                writer.add_scalar(f"Test/{t}_Accuracy", m["accuracy"])
                writer.add_scalar(f"Test/{t}_Precision", m["precision"])
                writer.add_scalar(f"Test/{t}_Recall", m["recall"])
                writer.add_scalar(f"Test/{t}_F1_Score", m["f1_score"])

    if measure_time:
        times_path = os.path.join(save_dir, "times_test.json")
        with open(times_path, 'w') as f:
            json.dump(times, f, indent=4)
        print(f"Temps moyen par batch: {np.mean(times):.4f}s, total: {np.sum(times):.4f}s")



def test_folder_predictions(model, tasks, test_folder, transform, device, save_dir,
                            save_test_images=False, target_task=None):
    """
    Parcourt récursivement le dossier test_folder et effectue les prédictions.

    - Si target_task est précisé, on traite uniquement cette tâche pour :
         • l'annotation (images sauvegardées dans un sous-dossier portant le nom de la classe prédite),
         • le calcul des scores F1 (par classe et global) basé sur la ground truth extraite de la structure du dossier.
    - Sinon, le modèle effectue des prédictions pour toutes les tâches.
         • Les images sont rangées selon la tâche par défaut (la première tâche),
         • L'annotation affiche les prédictions pour toutes les tâches,
         • Le JSON final "folder_predictions.json" contient, pour chaque tâche, le nombre d'images par classe et
           les scores F1 (global et par classe) basés sur la ground truth extraite.
         • De plus, un second fichier "all_predictions.json" est généré, donnant pour chaque image
           (identifiée par son chemin relatif par rapport au dossier de test) l'ensemble des prédictions.

    Args:
        model (torch.nn.Module): Modèle multi-tâches chargé.
        tasks (dict): Dictionnaire associant chaque tâche à la liste de ses classes.
        test_folder (str): Chemin vers le dossier contenant les images de test.
        transform: Transformation à appliquer aux images.
        device: Appareil utilisé (CPU ou GPU).
        save_dir (str): Répertoire de sauvegarde des résultats.
        save_test_images (bool): Si True, enregistre les images annotées.
        target_task (str, optional): Nom de la tâche sur laquelle réaliser le test.
    """
    # Définir la tâche utilisée pour le classement et l'évaluation
    if target_task is not None:
        tasks_to_evaluate = {target_task: tasks[target_task]}
        folder_task = target_task
    else:
        tasks_to_evaluate = tasks  # toutes les tâches
        folder_task = list(tasks.keys())[0]  # on organise les images par la première tâche

    # Initialisation des dictionnaires pour chaque tâche évaluée
    predictions_by_task = {t: {} for t in tasks_to_evaluate.keys()}  # comptage par classe
    gt_by_task = {t: [] for t in tasks_to_evaluate.keys()}  # ground truth extraites
    pred_gt_by_task = {t: [] for t in tasks_to_evaluate.keys()}  # prédictions associées aux GT

    results = {}  # résultats complets par image (pour annotation)

    # Dossier de sauvegarde des images annotées
    if save_test_images:
        annotated_base_dir = os.path.join(save_dir, "annotated_images")
        os.makedirs(annotated_base_dir, exist_ok=True)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    # Parcours récursif des fichiers dans test_folder
    for root, dirs, files in os.walk(test_folder):
        for file in files:
            if not file.lower().endswith(valid_extensions):
                continue
            img_path = os.path.join(root, file)
            # On calcule le chemin relatif pour identifier l'image dans all_predictions.json
            rel_path = os.path.relpath(img_path, test_folder)
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Erreur lors du chargement de {img_path}: {e}")
                continue

            input_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)

            # Calcul des prédictions
            if target_task is not None:
                output = outputs[target_task]
                probabilities = torch.softmax(output, dim=1)
                max_prob, pred_idx = torch.max(probabilities, dim=1)
                pred_idx = pred_idx.item()
                max_prob = max_prob.item()
                predicted_class = tasks[target_task][pred_idx] if pred_idx < len(tasks[target_task]) else "Unknown"
                results[rel_path] = {target_task: {"predicted_class": predicted_class, "probability": max_prob}}
            else:
                image_preds = {}
                for t, output in outputs.items():
                    probabilities = torch.softmax(output, dim=1)
                    max_prob, pred_idx = torch.max(probabilities, dim=1)
                    pred_idx = pred_idx.item()
                    max_prob = max_prob.item()
                    predicted_class = tasks[t][pred_idx] if pred_idx < len(tasks[t]) else "Unknown"
                    image_preds[t] = {"predicted_class": predicted_class, "probability": max_prob}
                results[rel_path] = image_preds

            # Pour le classement, on utilise la prédiction pour folder_task
            if target_task is not None:
                key = target_task
                pred_for_folder = predicted_class
            else:
                key = folder_task
                pred_for_folder = results[rel_path][folder_task]["predicted_class"]
            predictions_by_task[key].setdefault(pred_for_folder, []).append(rel_path)

            # Extraction de la ground truth depuis la structure du dossier (si image dans un sous-dossier)
            if os.path.abspath(root) != os.path.abspath(test_folder):
                folder_name = os.path.basename(root)
                for t, class_list in tasks_to_evaluate.items():
                    gt_class = map_folder_to_class(folder_name, class_list)
                    if gt_class is not None:
                        gt_by_task[t].append(gt_class)
                        if target_task is not None:
                            pred_val = predicted_class
                        else:
                            pred_val = results[rel_path][t]["predicted_class"]
                        pred_gt_by_task[t].append(pred_val)

            # Annotation et sauvegarde de l'image
            if save_test_images:
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                y0, dy = 30, 30
                if target_task is not None:
                    annotation = f"{target_task}: {results[rel_path][target_task]['predicted_class']} ({results[rel_path][target_task]['probability']:.2f})"
                else:
                    annotation = ""
                    for t, pred in results[rel_path].items():
                        annotation += f"{t}: {pred['predicted_class']} ({pred['probability']:.2f})\n"
                cv2.putText(img_cv, annotation, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if folder_task in results[rel_path]:
                    folder_label = results[rel_path][folder_task]['predicted_class']
                else:
                    folder_label = list(results[rel_path].values())[0]['predicted_class']
                dest_folder = os.path.join(annotated_base_dir, folder_label)
                os.makedirs(dest_folder, exist_ok=True)
                annotated_path = os.path.join(dest_folder, file)
                cv2.imwrite(annotated_path, img_cv)
                cv2.imshow("Prédiction", img_cv)
                cv2.waitKey(100)
    if save_test_images:
        cv2.destroyAllWindows()

    # Calcul des scores F1 pour chaque tâche évaluée (sur les images avec ground truth)
    final_results = {}
    for t in tasks_to_evaluate.keys():
        f1_dict = {}
        global_f1 = None
        if len(gt_by_task[t]) > 0 and len(pred_gt_by_task[t]) > 0:
            unique_classes = list(set(gt_by_task[t]))
            f1_scores = f1_score(gt_by_task[t], pred_gt_by_task[t], labels=unique_classes, average=None)
            f1_dict = dict(zip(unique_classes, f1_scores))
            global_f1 = f1_score(gt_by_task[t], pred_gt_by_task[t], average='weighted')
        counts = {}
        for cls in tasks_to_evaluate[t]:
            counts[cls] = len(predictions_by_task[t].get(cls, []))
        final_results[t] = {"by_class": counts, "f1_score": f1_dict, "global_f1": global_f1}

    json_path = os.path.join(save_dir, "folder_predictions.json")
    with open(json_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"Résultats des prédictions sauvegardés dans {json_path}")

    # Si aucune tâche cible n'est spécifiée, enregistrer un second JSON contenant les prédictions complètes pour toutes les tâches,
    # en utilisant le chemin relatif des images.
    if target_task is None:
        all_pred_json_path = os.path.join(save_dir, "all_predictions.json")
        with open(all_pred_json_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Prédictions complètes sauvegardées dans {all_pred_json_path}")




def watch_folders_predictions(
    model,
    tasks,
    watch_folders,
    poll_intervals,
    transform,
    device,
    save_dir,
    save_dir_to_canon=None,
    eval_annotations=False,
    annotations_folders=None,
    truth_mapping_path=None,
    metrics_every: int = 50
):
    """
    Lance un thread process_watch_folder par dossier.
    Accepte indifféremment des listes *ou* des chaînes séparées par virgules.
    """
    # Normalisation
    watch_folders       = _to_list(watch_folders)
    poll_intervals      = [int(x) for x in _to_list(poll_intervals)]
    annotations_folders = _to_list(annotations_folders) if annotations_folders is not None else None

    # Contrôles
    if len(watch_folders) != len(poll_intervals):
        raise ValueError("Le nombre de dossiers à surveiller doit correspondre au nombre d’intervalles.")
    if eval_annotations:
        if annotations_folders is None or len(annotations_folders) != len(watch_folders):
            raise ValueError("`annotations_folders` doit contenir un chemin par dossier surveillé "
                             "(si --eval_annotations est activé).")

    # Lancement des threads
    threads = []
    for idx, folder in enumerate(watch_folders):
        sub_save_dir = os.path.join(save_dir, os.path.basename(os.path.normpath(folder)))
        is_first     = (idx == 0)
        ann_folder   = annotations_folders[idx] if eval_annotations else None

        t = threading.Thread(
            target=process_watch_folder,
            args=(
                model,
                tasks,
                folder,
                transform,
                device,
                sub_save_dir,
                poll_intervals[idx],
                save_dir_to_canon,
                is_first,
                eval_annotations,
                ann_folder,
                truth_mapping_path,
                metrics_every
            )
        )
        t.daemon = True
        t.start()
        threads.append(t)
        print(f"🔎  Surveillance lancée pour « {folder} » (intervalle : {poll_intervals[idx]} s).")

    for t in threads:
        t.join()


def process_watch_folder(
    model,
    tasks,
    watch_folder,
    transform,
    device,
    sub_save_dir,
    poll_interval,
    save_dir_to_canon=None,
    is_first=False,
    eval_annotations=False,
    annotations_folder=None,
    truth_mapping_path=None,
    metrics_every: int = 50
):
    """
    Surveille `watch_folder`, prédit en continu, historise et —
    si eval_annotations=True — déduit la vérité-terrain via
    `truth_mapping_path` et calcule précision/recall/F1 toutes les `metrics_every` images.
    """
    import os
    import re
    import time
    import json
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from sklearn.metrics import precision_score, recall_score, f1_score

    # 1) Charger les règles de vérité
    if eval_annotations and truth_mapping_path:
        with open(truth_mapping_path, 'r') as f:
            truth_rules = json.load(f)
    else:
        truth_rules = {}

    # 2) Préparer dossiers / fichiers
    os.makedirs(sub_save_dir, exist_ok=True)
    history_file   = os.path.join(sub_save_dir, "prediction_history.csv")
    perf_file      = os.path.join(sub_save_dir, "performance.json")
    perf_hist_file = os.path.join(sub_save_dir, "performance_history.csv")
    last_pred_file = os.path.join(sub_save_dir, "last_prediction.json")

    # Colonnes CSV prédictions (gt, pred, match, prob)
    pred_cols = ["timestamp", "image"]
    for t in tasks:
        pred_cols += [f"{t}_gt", f"{t}_pred", f"{t}_match", f"{t}_prob"]

    # Colonnes CSV métriques (precision, recall, f1, plus global_f1)
    metric_cols = ["timestamp"]
    for t in tasks:
        metric_cols += [f"{t}_precision", f"{t}_recall", f"{t}_f1"]
    metric_cols.append("global_f1")

    # Charger ou initier DataFrames
    history_df    = pd.read_csv(history_file   ) if os.path.exists(history_file)   else pd.DataFrame(columns=pred_cols)
    perf_hist_df  = pd.read_csv(perf_hist_file ) if os.path.exists(perf_hist_file) else pd.DataFrame(columns=metric_cols)

    # Préparer accumulateurs
    if eval_annotations:
        y_true = {t: [] for t in tasks}
        y_pred = {t: [] for t in tasks}
        n_eval = 0

    ts_re = re.compile(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')
    last_processed = None

    def eval_rule(rule, sensors):
        for cond in rule["when"]:
            val = sensors.get(cond["sensor"])
            if val is None: return False
            op = cond["op"]
            if   op == "eq":  ok = val == cond["value"]
            elif op == "neq": ok = val != cond["value"]
            elif op == "gt":  ok = val >  cond["value"]
            elif op == "lt":  ok = val <  cond["value"]
            elif op == "gte": ok = val >= cond["value"]
            elif op == "lte": ok = val <= cond["value"]
            elif op == "in":  ok = val in  cond["list"]
            else: ok = False
            if not ok: return False
        return True

    while True:
        # 3) Lister nouvelles images
        imgs = [
            f for f in os.listdir(watch_folder)
            if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))
            and ts_re.match(os.path.splitext(f)[0])
        ]
        if not imgs:
            time.sleep(poll_interval); continue
        imgs.sort()
        last_file = imgs[-1]
        if last_file == last_processed:
            time.sleep(poll_interval); continue
        last_processed = last_file

        # 4) Charger + prédire
        img = Image.open(os.path.join(watch_folder, last_file)).convert('RGB')
        inp = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outs = model(inp)

        # 5) Charger capteurs si dispo
        sensors = {}
        if eval_annotations and annotations_folder:
            ann_path = os.path.join(annotations_folder, os.path.splitext(last_file)[0] + ".json")
            if os.path.isfile(ann_path):
                ann = json.load(open(ann_path))
                for sv in ann.get("sensorValues", []):
                    sensors[sv["name"]] = sv.get("value")

        # 6) Construire row_pred
        ts = os.path.splitext(last_file)[0]
        row_pred = [ts, last_file]
        prediction = {}
        for t, out in outs.items():
            probs    = torch.softmax(out, dim=1)[0].cpu().numpy()
            idx      = int(probs.argmax())
            pred_cls = tasks[t][idx]
            prob     = float(probs[idx])
            # ground-truth
            gt = truth_rules.get(t, {}).get("default", "Unknown")
            for rule in truth_rules.get(t, {}).get("rules", []):
                if eval_rule(rule, sensors):
                    gt = rule["class"]
                    break
            match = int(gt == pred_cls)
            # stocker
            prediction[t] = {"predicted_class": pred_cls, "probability": prob}
            row_pred += [gt, pred_cls, match, prob]

        # 7) Historiser prédiction
        history_df.loc[len(history_df)] = row_pred
        history_df.to_csv(history_file, index=False)

        # 8) Écrire last_prediction.json
        with open(last_pred_file, 'w') as f:
            json.dump({"timestamp": ts, "image": last_file, "prediction": prediction}, f, indent=2)

        #print(eval_annotations)
        #print(sensors)
        #print(truth_rules)
        # 9) Évaluer périodiquement
        if eval_annotations and sensors and truth_rules:
            n_eval += 1
            # remplir y_true/y_pred
            for t in tasks:
                y_true[t].append(row_pred[pred_cols.index(f"{t}_gt")])
                y_pred[t].append(row_pred[pred_cols.index(f"{t}_pred")])

            if n_eval % metrics_every == 0:
                perf_line = {"timestamp": ts}
                for t in tasks:
                    prec = precision_score(y_true[t], y_pred[t], average='weighted', zero_division=0)
                    rec  = recall_score(   y_true[t], y_pred[t], average='weighted', zero_division=0)
                    f1   = f1_score(       y_true[t], y_pred[t], average='weighted', zero_division=0)
                    perf_line[f"{t}_precision"] = prec
                    perf_line[f"{t}_recall"]    = rec
                    perf_line[f"{t}_f1"]        = f1
                perf_line["global_f1"] = np.mean([perf_line[f"{t}_f1"] for t in tasks])

                # a) performance.json
                with open(perf_file, 'w') as pf:
                    json.dump(perf_line, pf, indent=2)
                # b) performance_history.csv
                perf_hist_df.loc[len(perf_hist_df)] = [perf_line[col] for col in metric_cols]
                perf_hist_df.to_csv(perf_hist_file, index=False)

                print(f"[{watch_folder}] ⏱  metrics update ({n_eval} images) → {perf_line}")

        # 10) Canon JSON pour premier dossier
        if is_first and save_dir_to_canon:
            canon = {"timestamp": ts, "image": last_file, "prediction": prediction}
            with open(os.path.join(save_dir_to_canon, "WeatherInfos.json"), 'w') as cf:
                json.dump(canon, cf, indent=2)

        time.sleep(poll_interval)

