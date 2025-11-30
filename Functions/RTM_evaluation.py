
from torch.utils.data import  Subset

import time

import threading
import datetime
import pandas as pd

from PIL import Image, ImageDraw, ImageFont

import os
import json

import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from captum.attr import IntegratedGradients

import re
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from Models.models_RTM import TaskSpecificModel

# Colormaps pour Grad-CAM
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


def map_folder_to_class(folder_name, class_list):
    """
    Essaie de faire correspondre le nom du dossier (ground truth)
    à l'une des classes en vérifiant si le nom du dossier est contenu
    dans le nom de la classe (sans tenir compte de la casse).
    """
    folder_lower = folder_name.lower()
    for cls in class_list:
        if folder_lower in cls.lower():
            return cls
    return None


def run_inference(
    model,
    image_folder,
    transform,
    device,
    classes,
    num_samples=None,
    save_dir=None,
    save_test_images=False
):
    """
    Boucle d'inférence générique :
    - model : n'importe quel modèle PyTorch retournant soit un Tensor de logits,
              soit un dict{task: logits}.
    - classes : list de noms (mono-tâche) ou dict{task: [noms]} (multi-tâche)
    - save_test_images : si True, on génère les images annotées dans save_dir
    """
    # Prétraitement du mapping mono- vs multi-tâche
    is_multi = isinstance(classes, dict)
    classes_ci = {}
    if is_multi:
        # On normalise les clés en minuscules
        for task, lst in classes.items():
            classes_ci[task.lower()] = lst

    # Collecte
    print(f"image_folder={image_folder}")
    img_paths = collect_image_paths(image_folder)
    if not img_paths:
        raise RuntimeError(f"Aucune image trouvée dans « {image_folder} »")
    if num_samples and num_samples < len(img_paths):
        img_paths = random.sample(img_paths, num_samples)

    model = model.to(device).eval()
    results = {}

    with torch.no_grad():
        for path in img_paths:
            img = Image.open(path).convert("RGB")
            inp = transform(img).unsqueeze(0).to(device)
            out = model(inp)

            text_lines = []
            preds = {}

            # Multi-tâche ?
            if isinstance(out, dict):
                for task, logits in out.items():
                    probs = F.softmax(logits, dim=1)[0]
                    prob, idx = probs.max(0)
                    # Récupère la liste de classes (insensible à la casse)
                    cls_list = classes_ci.get(task.lower(), None)
                    name = cls_list[idx] if cls_list and idx < len(cls_list) else str(idx.item())
                    preds[task] = {'predicted_class': name, 'probability': prob.item()}
                    text_lines.append(f"{task}: {name} ({prob:.2f})")

            # Mono-tâche
            else:
                probs = F.softmax(out, dim=1)[0]
                prob, idx = probs.max(0)
                if isinstance(classes, list) and idx < len(classes):
                    name = classes[idx]
                else:
                    name = str(idx.item())
                preds = {'predicted_class': name, 'probability': prob.item()}
                text_lines = [f"{name} ({prob:.2f})"]

            results[path] = preds

            # Sauvegarde annotée
            if save_dir and save_test_images:
                rel = os.path.relpath(path, image_folder)
                out_path = os.path.join(save_dir, rel)
                annotate_and_save(img, text_lines, out_path)

    # JSON global
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'inference_results.json'), 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def test(model, test_loader, criterions, writer, save_dir, device, tasks, prob_threshold,
         visualize_gradcam, save_gradcam_images, measure_time, save_test_images,
         gradcam_task=None, colormap='hot', integrated_gradients=False, show_gt_labels=True):
    model.eval()
    total_loss = 0.0
    all_preds = {task: [] for task in tasks.keys()}
    all_labels = {task: [] for task in tasks.keys()}
    total_samples = 0
    times = []

    os.makedirs(save_dir, exist_ok=True)

    # Chercher la tâche "weather type"
    weather_task_name = None
    for task_name in tasks.keys():
        if task_name.lower() == "weather type":
            weather_task_name = task_name
            break
    weather_task_available = weather_task_name is not None

    # Grad-CAM
    if visualize_gradcam:
        if gradcam_task is None:
            gradcam_task = list(tasks.keys())[0]
        if gradcam_task not in tasks:
            raise ValueError(f"La tâche '{gradcam_task}' n'existe pas dans le modèle.")

        gradcam_model = TaskSpecificModel(model, gradcam_task).to(device)
        gradcam_model.eval()

        target_layer = None
        for layer in reversed(list(gradcam_model.model.truncated_encoder)):
            if isinstance(layer, nn.Conv2d):
                target_layer = layer
                break
        if target_layer is None:
            raise ValueError("Aucune couche Conv2d trouvée dans truncated_encoder pour Grad-CAM.")

        grad_cam = GradCAM(model=gradcam_model, target_layers=[target_layer])
    else:
        grad_cam = None

    # Integrated Gradients
    if integrated_gradients:
        ig_models = {}
        ig = {}
        for t in tasks.keys():
            ig_models[t] = TaskSpecificModel(model, t).to(device)
            ig_models[t].eval()
            ig[t] = IntegratedGradients(ig_models[t])
    else:
        ig = None

    for batch_idx, (inputs, labels) in enumerate(test_loader):
        import time as _time
        start_time = _time.time()
        inputs = inputs.to(device)
        inputs.requires_grad = True

        with torch.no_grad():
            outputs = model(inputs)

        loss = 0.0
        batch_size = inputs.size(0)
        max_probs_dict = {}
        preds_dict = {}
        task_labels_dict = {}

        for task_name, criterion in criterions.items():
            task_labels = labels[task_name]
            if task_labels is not None:
                task_labels = task_labels.to(device)
                task_outputs = outputs[task_name]
                task_loss = criterion(task_outputs, task_labels)
                loss += task_loss

                probabilities = torch.softmax(task_outputs, dim=1)
                max_probs, preds = torch.max(probabilities, 1)
                unknown_mask = max_probs < prob_threshold
                preds[unknown_mask] = -1

                all_preds[task_name].extend(preds.cpu().numpy())
                all_labels[task_name].extend(task_labels.cpu().numpy())

                max_probs_dict[task_name] = max_probs.cpu().numpy()
                preds_dict[task_name] = preds.cpu().numpy()
                task_labels_dict[task_name] = task_labels.cpu().numpy()
            else:
                max_probs_dict[task_name] = np.array([-1]*batch_size)
                preds_dict[task_name] = np.array([-1]*batch_size)
                task_labels_dict[task_name] = np.array([-1]*batch_size)

        end_time = _time.time()
        times.append(end_time - start_time)

        # Integrated Gradients
        if integrated_gradients and ig is not None:
            for i in range(batch_size):
                for t in tasks.keys():
                    input_tensor = inputs[i].unsqueeze(0)
                    baseline = torch.zeros_like(input_tensor)
                    target = int(task_labels_dict[t][i]) if task_labels_dict[t][i] >= 0 else 0
                    attr = ig[t].attribute(input_tensor, baseline, target=target)
                    attr_np = attr.squeeze().cpu().detach().numpy()
                    attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)
                    ig_save_path = os.path.join(
                        save_dir, f"IntegratedGrad_{t}_{batch_idx * test_loader.batch_size + i}.jpg"
                    )
                    heatmap = cv2.applyColorMap(np.uint8(255 * attr_np), cv2.COLORMAP_JET)
                    cv2.imwrite(ig_save_path, heatmap)

        # Images, overlays, Grad-CAM
        for i in range(batch_size):
            idx = batch_idx * test_loader.batch_size + i
            if hasattr(test_loader.dataset, "dataset"):  # Subset
                img_path = test_loader.dataset.dataset.samples[test_loader.dataset.indices[idx]][0]
            else:
                img_path = test_loader.dataset.samples[idx][0]

            img = Image.open(img_path)
            img_np = np.array(img.convert('RGB'))
            img_cv = img_np.copy()

            if weather_task_available:
                weather_label_idx = task_labels_dict[weather_task_name][i]
                if weather_label_idx == -1:
                    weather_true_label = "Unknown"
                else:
                    weather_true_label = tasks[weather_task_name][weather_label_idx]
                label_dir = os.path.join(save_dir, weather_true_label)
            else:
                label_dir = save_dir

            os.makedirs(label_dir, exist_ok=True)

            if save_test_images:
                annotated_img = img_cv.copy()

                font = cv2.FONT_HERSHEY_SIMPLEX
                base_scale = 0.48
                min_scale = 0.34
                thickness = 1
                text_color = (0, 150, 0)

                pad_x, pad_y = 8, 6
                alpha = 0.45
                margin = 8
                gap_x = 8
                H, W = annotated_img.shape[:2]
                max_bar_h = int(0.20 * H)
                max_total_w = int(0.90 * W)
                max_cols_cap = 6

                lines = []
                for task_name, class_list in tasks.items():
                    label_idx = task_labels_dict[task_name][i]
                    pred_idx = preds_dict[task_name][i]
                    prob = max_probs_dict[task_name][i]

                    true_label = "Unknown" if label_idx == -1 else class_list[label_idx]
                    pred_label = "Unknown" if pred_idx == -1 else class_list[pred_idx]

                    if show_gt_labels:
                        lines.append(f"{task_name} - True: {true_label}, Pred: {pred_label}, Prob: {prob:.2f} ")
                    else:
                        lines.append("")
                        lines.append(f"{task_name} - Pred: {pred_label} ({prob:.2f})")

                if not lines:
                    img_filename = f"test_image_{idx}.jpg"
                    save_path = os.path.join(label_dir, img_filename)
                    cv2.imwrite(save_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
                else:
                    sizes_base = [cv2.getTextSize(t, font, base_scale, thickness)[0] for t in lines]
                    max_w_base = max(w for (w, _) in sizes_base)
                    col_w_base = max_w_base + 2 * pad_x
                    max_cols_by_w = max(1, (max_total_w + gap_x) // (col_w_base + gap_x))
                    max_cols_possible = int(min(max_cols_by_w, max_cols_cap))

                    best = None
                    for cols in range(max_cols_possible, 0, -1):
                        scale = base_scale
                        while scale >= min_scale:
                            sizes = [cv2.getTextSize(t, font, scale, thickness)[0] for t in lines]
                            max_w = max(w for (w, _) in sizes)
                            line_h = max(h for (_, h) in sizes)
                            y_step = max(int(line_h * 1.15), line_h)
                            rows = int(np.ceil(len(lines) / cols))
                            col_w = max_w + 2 * pad_x
                            col_h = line_h + (rows - 1) * y_step + 2 * pad_y
                            total_w = cols * col_w + (cols - 1) * gap_x
                            total_h = col_h
                            if total_w <= max_total_w and total_h <= max_bar_h:
                                best = (cols, scale, line_h, y_step)
                                break
                            scale -= 0.03
                        if best:
                            break

                    if not best:
                        cols, scale = 1, min_scale
                        sizes = [cv2.getTextSize(t, font, scale, thickness)[0] for t in lines]
                        line_h = max(h for (_, h) in sizes)
                        y_step = max(int(line_h * 1.15), line_h)
                    else:
                        cols, scale, line_h, y_step = best
                        sizes = [cv2.getTextSize(t, font, scale, thickness)[0] for t in lines]

                    rows = int(np.ceil(len(lines) / cols))
                    col_widths = []
                    for c in range(cols):
                        start = c * rows
                        end = min((c + 1) * rows, len(lines))
                        if start >= end:
                            col_widths.append(0)
                            continue
                        max_w_c = max(sizes[k][0] for k in range(start, end))
                        col_widths.append(max_w_c + 2 * pad_x)

                    total_w = sum(col_widths) + (cols - 1) * gap_x
                    total_w = min(total_w, W - 2 * margin)
                    _, baseline = cv2.getTextSize("Ag", font, scale, thickness)
                    col_h = line_h + (rows - 1) * y_step + 2 * pad_y + baseline

                    top_y = margin
                    left_x = margin
                    col_xs = []
                    x_acc = left_x
                    for cw in col_widths:
                        col_xs.append(x_acc)
                        x_acc += cw + gap_x

                    overlay = annotated_img.copy()
                    bg_left = left_x
                    bg_top = top_y
                    bg_right = min(W - margin, left_x + total_w)
                    bg_bottom = min(H - margin, top_y + col_h)
                    cv2.rectangle(
                        overlay, (bg_left, bg_top), (bg_right, bg_bottom),
                        (255, 255, 255), thickness=-1
                    )
                    cv2.addWeighted(overlay, alpha, annotated_img, 1 - alpha, 0, annotated_img)

                    for c in range(cols):
                        start = c * rows
                        end = min((c + 1) * rows, len(lines))
                        x_txt = col_xs[c] + pad_x
                        y0 = top_y + pad_y + line_h
                        for j in range(start, end):
                            y_txt = y0 + (j - start) * y_step
                            cv2.putText(
                                annotated_img, lines[j], (x_txt, y_txt),
                                font, scale, text_color, thickness, cv2.LINE_AA
                            )

                    img_filename = f"test_image_{idx}.jpg"
                    save_path = os.path.join(label_dir, img_filename)
                    img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path, img_bgr)

            # Grad-CAM
            if visualize_gradcam and save_gradcam_images and grad_cam is not None:
                input_tensor = inputs[i].unsqueeze(0)
                label_idx = task_labels_dict[gradcam_task][i]
                pred_idx = preds_dict[gradcam_task][i]
                prob = max_probs_dict[gradcam_task][i]

                true_label = "Unknown" if label_idx == -1 else tasks[gradcam_task][label_idx]
                pred_label = "Unknown" if pred_idx == -1 else tasks[gradcam_task][pred_idx]
                text = f"{gradcam_task} - True: {true_label}, Pred: {pred_label}, Prob: {prob:.2f}"

                target = [ClassifierOutputTarget(label_idx)]
                grayscale_cam = grad_cam(input_tensor=input_tensor, targets=target)[0]
                grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
                grayscale_cam_resized = cv2.resize(grayscale_cam, (img_np.shape[1], img_np.shape[0]))

                if colormap not in colormap_dict:
                    print(f"Colormap '{colormap}' non reconnu. Utilisation de 'hot'.")
                    colormap_code = cv2.COLORMAP_HOT
                else:
                    colormap_code = colormap_dict[colormap]

                heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam_resized), colormap_code)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                visualization = heatmap.astype(np.float32) / 255.0
                visualization = visualization * 0.5 + img_np.astype(np.float32)/255.0 * 0.5
                visualization = np.clip(visualization, 0, 1)
                visualization_cv = (visualization * 255).astype(np.uint8)

                cv2.putText(visualization_cv, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                combined_image = np.hstack((img_cv, visualization_cv))
                gradcam_save_path = os.path.join(label_dir, f"GradCAM_{idx}.jpg")
                combined_image_bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(gradcam_save_path, combined_image_bgr)

                if writer:
                    combined_image_tensor = torch.from_numpy(combined_image).permute(2, 0, 1)
                    writer.add_image(f'GradCAM/Images/{idx}', combined_image_tensor, global_step=idx)

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    average_loss = total_loss / max(1, total_samples)
    metrics = {}

    for task_name in tasks.keys():
        if all_preds[task_name]:
            preds = np.array(all_preds[task_name])
            labels_np = np.array(all_labels[task_name])
            valid_indices = preds != -1
            if valid_indices.sum() > 0:
                accuracy = np.mean(preds[valid_indices] == labels_np[valid_indices])
                precision = precision_score(labels_np[valid_indices], preds[valid_indices],
                                            average='weighted', zero_division=0)
                recall = recall_score(labels_np[valid_indices], preds[valid_indices],
                                      average='weighted', zero_division=0)
                f1 = f1_score(labels_np[valid_indices], preds[valid_indices],
                              average='weighted', zero_division=0)
                conf_matrix = confusion_matrix(
                    labels_np[valid_indices], preds[valid_indices],
                    labels=list(range(len(tasks[task_name])))
                )
            else:
                accuracy = precision = recall = f1 = 0.0
                conf_matrix = np.zeros(
                    (len(tasks[task_name]), len(tasks[task_name]))
                )
            metrics[task_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': conf_matrix.tolist()
            }
            print(f'Tâche {task_name} - Acc: {accuracy:.4f}, Prec: {precision:.4f}, '
                  f'Rec: {recall:.4f}, F1: {f1:.4f}')
            print(f'Matrice de confusion pour {task_name}:\n{conf_matrix}\n')
        else:
            metrics[task_name] = {
                'accuracy': None,
                'precision': None,
                'recall': None,
                'f1_score': None,
                'confusion_matrix': None
            }

    accuracy_scores = [m['accuracy'] for m in metrics.values() if m['accuracy'] is not None]
    precision_scores = [m['precision'] for m in metrics.values() if m['precision'] is not None]
    recall_scores = [m['recall'] for m in metrics.values() if m['recall'] is not None]
    f1_scores = [m['f1_score'] for m in metrics.values() if m['f1_score'] is not None]

    if f1_scores:
        average_accuracy = float(np.mean(accuracy_scores))
        average_precision = float(np.mean(precision_scores))
        average_recall = float(np.mean(recall_scores))
        average_f1 = float(np.mean(f1_scores))
    else:
        average_accuracy = average_precision = average_recall = average_f1 = 0.0

    print(f'Perf moyennes - Acc: {average_accuracy:.4f}, Prec: {average_precision:.4f}, '
          f'Rec: {average_recall:.4f}, F1: {average_f1:.4f}')

    metrics['average'] = {
        'accuracy': average_accuracy,
        'precision': average_precision,
        'recall': average_recall,
        'f1_score': average_f1
    }

    metrics_path = os.path.join(save_dir, "test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Métriques du test enregistrées dans {metrics_path}")

    if writer:
        writer.add_scalar("Test/Loss", average_loss)
        writer.add_scalar("Test/Average_Accuracy", average_accuracy)
        writer.add_scalar("Test/Average_Precision", average_precision)
        writer.add_scalar("Test/Average_Recall", average_recall)
        writer.add_scalar("Test/Average_F1_Score", average_f1)
        for task_name, task_metrics in metrics.items():
            if task_name != 'average' and task_metrics['accuracy'] is not None:
                writer.add_scalar(f"Test/{task_name}_Accuracy", task_metrics['accuracy'])
                writer.add_scalar(f"Test/{task_name}_Precision", task_metrics['precision'])
                writer.add_scalar(f"Test/{task_name}_Recall", task_metrics['recall'])
                writer.add_scalar(f"Test/{task_name}_F1_Score", task_metrics['f1_score'])

    if measure_time and times:
        times_path = os.path.join(save_dir, "times_test.json")
        with open(times_path, 'w') as f:
            json.dump(times, f, indent=4)
        print(f"Temps moyen par lot: {np.mean(times):.4f}s | total: {np.sum(times):.4f}s")






def load_best_model(model, filepath, strict_backbone: bool = True, verbose: bool = True):
    """
    Chargement partiel robuste avec remap:
      - supporte 'module.' ; 'backbone.'/ 'truncated_encoder.' / ResNet brut
      - remap des classifieurs: 'classifiers.classifier_X.weight/bias'
        → dernière couche linéaire du MLP: 'classifiers.classifier_X.{last_idx}.weight/bias'
      - si shapes diffèrent, copie partielle
    """
    ckpt = torch.load(filepath, map_location=model.device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    ckpt = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}

    has_backbone   = any(k.startswith("backbone.")          for k in ckpt)
    has_truncated  = any(k.startswith("truncated_encoder.") for k in ckpt)
    if has_backbone:
        feat_prefix_ckpt = "backbone."
    elif has_truncated:
        feat_prefix_ckpt = "truncated_encoder."
    else:
        feat_prefix_ckpt = None

    feat_prefix_model = (
        "truncated_encoder."
        if any(k.startswith("truncated_encoder.") for k in model.state_dict())
        else "backbone."
    )

    remapped: Dict[str, torch.Tensor] = {}

    # 2) remap features
    if feat_prefix_ckpt is None:
        root2idx = {
            "conv1": 0, "bn1": 1, "relu": 2, "maxpool": 3,
            "layer1": 4, "layer2": 5, "layer3": 6, "layer4": 7,
        }
        cur_children = list(model.truncated_encoder.children())
        for k, v in ckpt.items():
            root = k.split('.')[0]
            if root not in root2idx:
                continue
            idx = root2idx[root]
            if idx >= len(cur_children):
                continue
            new_k = f"{feat_prefix_model}{idx}{k[len(root):]}"
            remapped[new_k] = v
    else:
        if feat_prefix_ckpt == feat_prefix_model:
            remapped = dict(ckpt)
        else:
            cut = len(feat_prefix_ckpt)
            for k, v in ckpt.items():
                if k.startswith(feat_prefix_ckpt):
                    new_k = f"{feat_prefix_model}{k[cut:]}"
                    remapped[new_k] = v
                else:
                    remapped[k] = v

    # 3) remap classifieurs
    cls_last_linear_idx = {}
    for cls_name, mod in model.classifiers.items():
        last_idx = None
        if isinstance(mod, nn.Sequential):
            for i, m in enumerate(mod):
                if isinstance(m, nn.Linear):
                    last_idx = i
        elif isinstance(mod, nn.Linear):
            last_idx = None
        cls_last_linear_idx[cls_name] = last_idx

    converted = dict(remapped)
    pat_simple = re.compile(r'^classifiers\.(classifier_[^\.]+)\.(weight|bias)$')
    for k, v in list(remapped.items()):
        m = pat_simple.match(k)
        if m:
            cls_name, wb = m.group(1), m.group(2)
            last_idx = cls_last_linear_idx.get(cls_name, None)
            if last_idx is None:
                new_k = f"classifiers.{cls_name}.{wb}"
            else:
                new_k = f"classifiers.{cls_name}.{last_idx}.{wb}"
            converted[new_k] = v
            if verbose and new_k != k:
                print(f"[remap] {k}  →  {new_k}")

    remapped = converted

    new_state = model.state_dict()
    to_load = {}
    for k, v in remapped.items():
        if k not in new_state:
            if verbose and (k.startswith("classifiers.") or k.startswith("attentions.")):
                print(f"[skip] {k} absent du modèle courant")
            continue
        if v.shape == new_state[k].shape:
            to_load[k] = v
        else:
            tgt = new_state[k].clone()
            slices = tuple(slice(0, min(a, b)) for a, b in zip(v.shape, tgt.shape))
            tgt[slices] = v[slices]
            to_load[k] = tgt
            if verbose:
                print(f"[resize] {k}: {v.shape} → {tgt.shape}")

    if strict_backbone:
        missing = [
            k for k in new_state
            if k.startswith(feat_prefix_model) and k not in to_load
        ]
        if missing:
            raise RuntimeError(
                f"Backbone keys manquantes ({len(missing)}). Ex: {missing[:10]}"
            )

    msg = model.load_state_dict(to_load, strict=False)
    if verbose:
        print(f"✔ {len(to_load)} tenseurs chargés (missing={len(msg.missing_keys)}, "
              f"unexpected={len(msg.unexpected_keys)})")
    model.to(model.device)



def compute_embeddings_with_paths(model, loader, device, per_task_tsne=False):
    """
    Retourne embeddings + labels + chemins d’images.
    - per_task_tsne=True  → dicts par tâche avec embeddings [N_t, C]
    - per_task_tsne=False → embeddings 'partagés' (GAP) [N, C]
    """
    model.eval()

    # Pré-collecte des chemins pour ne pas dépendre de start/end approximatifs
    if isinstance(loader.dataset, Subset):
        base_ds = loader.dataset.dataset
        idx_list = list(loader.dataset.indices)
        full_paths = [base_ds.samples[i][0] for i in idx_list]
    else:
        base_ds = loader.dataset
        full_paths = [base_ds.samples[i][0] for i in range(len(base_ds))]

    ptr = 0  # pointeur dans full_paths

    if per_task_tsne:
        task_embeddings = {t: [] for t in model.tasks.keys()}
        task_labels     = {t: [] for t in model.tasks.keys()}
        task_img_paths  = {t: [] for t in model.tasks.keys()}
    else:
        all_embeddings = []
        all_labels     = []
        img_paths      = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            cur_bs = inputs.size(0)
            batch_paths = full_paths[ptr:ptr+cur_bs]
            ptr += cur_bs

            if per_task_tsne:
                # on récupère les embeddings par tâche
                _, embeddings = model(inputs, return_task_embeddings=True)
                for task_name, emb_tensor in embeddings.items():
                    # labels pour cette tâche
                    y = labels[task_name]
                    m = y.ne(-1)
                    if m.any():
                        emb_sel = emb_tensor[m].cpu().numpy()
                        lbl_sel = y[m].cpu().numpy()
                        paths_sel = [p for p, keep in zip(batch_paths, m.tolist()) if keep]
                        task_embeddings[task_name].extend(emb_sel)
                        task_labels[task_name].extend(lbl_sel.tolist())
                        task_img_paths[task_name].extend(paths_sel)
            else:
                # Embedding 'partagé' (GAP) pour un t-SNE global
                _, shared = model(inputs, return_shared_embedding=True)
                # on prend les labels de la première tâche (pour colorer)
                first_task = next(iter(labels.keys()))
                y = labels[first_task]
                m = y.ne(-1)
                if m.any():
                    emb_sel = shared[m].cpu().numpy()
                    lbl_sel = y[m].cpu().numpy()
                    paths_sel = [p for p, keep in zip(batch_paths, m.tolist()) if keep]
                    all_embeddings.append(emb_sel)
                    all_labels.extend(lbl_sel.tolist())
                    img_paths.extend(paths_sel)

    if per_task_tsne:
        out_emb, out_lbl, out_paths = {}, {}, {}
        for t in task_embeddings:
            if len(task_embeddings[t]) > 0:
                out_emb[t]  = np.stack(task_embeddings[t], axis=0)
                out_lbl[t]  = np.array(task_labels[t], dtype=int)
                out_paths[t]= list(task_img_paths[t])
            else:
                out_emb[t]  = np.zeros((0, model.num_features), dtype=np.float32)
                out_lbl[t]  = np.array([], dtype=int)
                out_paths[t]= []
        return out_emb, out_lbl, out_paths
    else:
        if len(all_embeddings) == 0:
            return np.zeros((0, model.num_features), dtype=np.float32), np.array([], dtype=int), []
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels     = np.array(all_labels, dtype=int)
        return all_embeddings, all_labels, img_paths



def test_folder_predictions(model, tasks, test_folder, transform, device, save_dir,
                            save_test_images=False, target_task=None):
    """
    Parcourt récursivement le dossier test_folder et effectue les prédictions.

    - Si target_task est précisé, la fonction évalue uniquement cette tâche :
         • Les images annotées sont sauvegardées dans un sous-dossier portant le nom de la classe prédite.
         • Des scores F1 (par classe et global) sont calculés en comparant la ground truth extraite de la structure du dossier.
    - Sinon, le modèle effectue des prédictions pour toutes les tâches.
         • Les images sont rangées selon la tâche par défaut (la première tâche).
         • Le JSON final 'folder_predictions.json' contient, pour chaque tâche, le nombre d'images par classe et
           les scores F1.
         • Un second fichier 'all_predictions.json' est généré, contenant pour chaque image l'ensemble des prédictions.
    """
    # Choix de la ou des tâches à évaluer
    if target_task is not None:
        tasks_to_evaluate = {target_task: tasks[target_task]}
        folder_task = target_task
    else:
        tasks_to_evaluate = tasks
        folder_task = list(tasks.keys())[0]

    # Initialisation des dictionnaires pour le comptage des prédictions et pour la ground truth
    predictions_by_task = {t: {} for t in tasks_to_evaluate.keys()}
    gt_by_task = {t: [] for t in tasks_to_evaluate.keys()}
    pred_gt_by_task = {t: [] for t in tasks_to_evaluate.keys()}
    results = {}  # Pour stocker les prédictions complètes par image

    # Dossier pour sauvegarder les images annotées
    if save_test_images:
        annotated_base_dir = os.path.join(save_dir, "annotated_images")
        os.makedirs(annotated_base_dir, exist_ok=True)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    for root, dirs, files in os.walk(test_folder):
        for file in files:
            if not file.lower().endswith(valid_extensions):
                continue
            img_path = os.path.join(root, file)
            rel_path = os.path.relpath(img_path, test_folder)
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Erreur lors du chargement de {img_path}: {e}")
                continue

            input_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)

            # Calcul des prédictions pour target_task ou pour toutes les tâches
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

            # Extraction de la ground truth à partir de la structure du dossier
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
                    annotation = "\n".join([f"{t}: {pred['predicted_class']} ({pred['probability']:.2f})" for t, pred in
                                            results[rel_path].items()])
                cv2.putText(img_cv, annotation, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                dest_folder = os.path.join(annotated_base_dir, results[rel_path][folder_task]['predicted_class'])
                os.makedirs(dest_folder, exist_ok=True)
                annotated_path = os.path.join(dest_folder, file)
                cv2.imwrite(annotated_path, img_cv)
                #cv2.imshow("Prédiction", img_cv)
                #cv2.waitKey(100)
    if save_test_images:
        cv2.destroyAllWindows()

    # Calcul des scores F1 si la ground truth est présente
    final_results = {}
    for t in tasks_to_evaluate.keys():
        f1_dict = {}
        global_f1 = None
        if len(gt_by_task[t]) > 0 and len(pred_gt_by_task[t]) > 0:
            unique_classes = list(set(gt_by_task[t]))
            f1_scores = f1_score(gt_by_task[t], pred_gt_by_task[t], labels=unique_classes, average=None)
            f1_dict = dict(zip(unique_classes, f1_scores))
            global_f1 = f1_score(gt_by_task[t], pred_gt_by_task[t], average='weighted')
        counts = {cls: len(predictions_by_task[t].get(cls, [])) for cls in tasks_to_evaluate[t]}
        final_results[t] = {"by_class": counts, "f1_score": f1_dict, "global_f1": global_f1}

    json_path = os.path.join(save_dir, "folder_predictions.json")
    with open(json_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"Résultats des prédictions sauvegardés dans {json_path}")

    # Si aucune tâche cible n'est spécifiée, on sauvegarde aussi l'ensemble des prédictions
    if target_task is None:
        all_pred_json_path = os.path.join(save_dir, "all_predictions.json")
        with open(all_pred_json_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Prédictions complètes sauvegardées dans {all_pred_json_path}")


def process_watch_folder(model, tasks, watch_folder, transform, device, sub_save_dir, poll_interval,
                         save_dir_to_canon=None, is_first=False):
    """
    Surveille en continu un dossier watch_folder contenant des images nommées avec un timestamp.
    Seuls les fichiers dont le nom (sans extension) correspond au format "YYYY-MM-DD_HH-MM-SS" sont considérés.
    Pour chaque nouvelle image détectée, le modèle est appliqué et :
      - Le fichier "last_prediction.json" est mis à jour avec le nom de l'image, le timestamp et la prédiction.
      - L'historique est mis à jour dans "prediction_history.csv".
      - Si save_dir_to_canon est spécifié et is_first True, la prédiction est aussi enregistrée dans save_dir_to_canon/WeatherInfos.json.
    """
    os.makedirs(sub_save_dir, exist_ok=True)
    history_file = os.path.join(sub_save_dir, "prediction_history.csv")
    columns = ["timestamp", "image"]
    for t, class_list in tasks.items():
        columns.extend([f"{t}_predicted_class", f"{t}_probability"])
    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)
    else:
        history_df = pd.DataFrame(columns=columns)

    last_processed = None
    # Expression régulière pour vérifier le format timestamp (sans extension)
    timestamp_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')
    print(f"[{watch_folder}] Surveillance toutes les {poll_interval} secondes dans {sub_save_dir}...")
    while True:
        files = [f for f in os.listdir(watch_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        # Filtrer uniquement les fichiers dont le nom correspond au pattern
        valid_files = []
        for f in files:
            name_no_ext = os.path.splitext(f)[0]
            if timestamp_pattern.match(name_no_ext):
                valid_files.append(f)
        if not valid_files:
            time.sleep(poll_interval)
            continue

        valid_files.sort()
        last_file = valid_files[-1]
        if last_file == last_processed:
            time.sleep(poll_interval)
            continue
        last_processed = last_file

        full_path = os.path.join(watch_folder, last_file)
        try:
            img = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"[{watch_folder}] Erreur lors du chargement de {full_path}: {e}")
            time.sleep(poll_interval)
            continue

        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)

        prediction = {}
        for t, output in outputs.items():
            probabilities = torch.softmax(output, dim=1)
            max_prob, pred_idx = torch.max(probabilities, dim=1)
            pred_idx = pred_idx.item()
            max_prob = max_prob.item()
            predicted_class = tasks[t][pred_idx] if pred_idx < len(tasks[t]) else "Unknown"
            prediction[t] = {"predicted_class": predicted_class, "probability": max_prob}

        # Le nom du fichier (sans extension) est utilisé comme timestamp si possible
        timestamp_str = os.path.splitext(last_file)[0]
        try:
            datetime.datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
        except Exception as e:
            print(f"[{watch_folder}] Erreur de parsing du timestamp pour {last_file}: {e}")
            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Enregistrement du JSON de la dernière prédiction dans sub_save_dir
        last_pred_json = os.path.join(sub_save_dir, "last_prediction.json")
        with open(last_pred_json, "w") as f:
            json.dump({"timestamp": timestamp_str, "image": last_file, "prediction": prediction}, f, indent=4)
        print(f"[{watch_folder}] Prédiction de {last_file} enregistrée dans {last_pred_json}")

        if save_dir_to_canon is not None and is_first:
            os.makedirs(save_dir_to_canon, exist_ok=True)
            canon_json = os.path.join(save_dir_to_canon, "WeatherInfos.json")
            with open(canon_json, "w") as f:
                json.dump({"timestamp": timestamp_str, "image": last_file, "prediction": prediction}, f, indent=4)
            print(f"[{watch_folder}] Prédiction de {last_file} enregistrée dans {canon_json}")

        # Mise à jour de l'historique
        row = {"timestamp": timestamp_str, "image": last_file}
        for t, pred in prediction.items():
            row[f"{t}_predicted_class"] = pred["predicted_class"]
            row[f"{t}_probability"] = pred["probability"]
        history_df = pd.concat([history_df, pd.DataFrame([row])], ignore_index=True)
        history_df.to_csv(history_file, index=False)
        print(f"[{watch_folder}] Historique mis à jour dans {history_file}")

        time.sleep(poll_interval)


def watch_folders_predictions(model, tasks, watch_folders, poll_intervals, transform, device, save_dir,
                              save_dir_to_canon=None):
    """
    Surveille plusieurs dossiers simultanément.
    Pour chaque dossier de watch_folders, les sorties (last_prediction.json et prediction_history.csv)
    sont enregistrées dans un sous-dossier de save_dir portant le même nom que le dossier surveillé.

    Si save_dir_to_canon est spécifié, pour le premier dossier de la liste, la prédiction est aussi enregistrée
    dans save_dir_to_canon/WeatherInfos.json.
    """
    if len(watch_folders) != len(poll_intervals):
        raise ValueError("Le nombre de dossiers et d'intervalles doit être identique.")

    threads = []
    for idx, folder in enumerate(watch_folders):
        folder_name = os.path.basename(os.path.normpath(folder))
        sub_save_dir = os.path.join(save_dir, folder_name)
        is_first = (idx == 0)
        t = threading.Thread(target=process_watch_folder, args=(
            model, tasks, folder, transform, device, sub_save_dir, poll_intervals[idx], save_dir_to_canon, is_first))
        t.daemon = True
        threads.append(t)
        t.start()
        print(f"Lancement de la surveillance pour {folder} avec un intervalle de {poll_intervals[idx]} secondes.")

    for t in threads:
        t.join()


def test_benchmark_folder(
    model: torch.nn.Module,
    device: torch.device,
    benchmark_folder: str,
    mapping_path: str,
    tasks_json: dict,
    transform,
    save_dir: str,
    roc_dir: str,
    auto_mapping: bool = False,
    num_samples: int = None,
    optimize_threshold: bool = True,
    # --- options d'overlay (seul GT/Pred est affiché) ---
    save_pred_images: bool = False,
    pred_images_dir: str | None = None,
    overlay_topk: int = 1,                 # ignoré pour l’overlay simplifié
    draw_prob_threshold: float | None = None,  # ignoré pour l’overlay simplifié
    max_width: int = 1280,
    font_scale: float = 0.6,
    thickness: int = 2,
):
    """
    Évalue le modèle sur le benchmark et, si demandé, sauvegarde les images annotées.
    L’annotation affiche UNIQUEMENT, par tâche: "GT=<classe_benchmark> | Pred=<classe_benchmark_prédite>",
    en vert si correct, rouge sinon.
    """
    import os, json, itertools
    import numpy as np
    import cv2
    from PIL import Image
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

    # 1) Mapping initial
    with open(mapping_path, 'r') as f:
        initial_mapping = json.load(f)

    bench_classes = {task: list(initial_mapping[task].keys()) for task in initial_mapping}

    # 2) Parcours récursif des images (top dossier = classe benchmark)
    images = []  # [(path, top_cls)]
    for root, _, files in os.walk(benchmark_folder):
        rel = os.path.relpath(root, benchmark_folder)
        if rel == ".":
            continue
        top_cls = rel.split(os.sep)[0]
        if all(top_cls not in bench_classes[task] for task in bench_classes):
            continue
        for fn in files:
            if not fn.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                continue
            images.append((os.path.join(root, fn), top_cls))

    # Échantillonnage si demandé
    if num_samples is not None and num_samples < len(images):
        import random
        images = random.sample(images, num_samples)

    # GT indexés (espace benchmark)
    gt = {task: [] for task in initial_mapping}
    for _, bench_cls in images:
        for task in initial_mapping:
            lowers = [b.lower() for b in bench_classes[task]]
            idx = lowers.index(bench_cls.lower()) if bench_cls.lower() in lowers else len(lowers) - 1
            gt[task].append(idx)

    # 3) Prédictions (espace modèle)
    model.to(device).eval()
    model_preds = {t: [] for t in initial_mapping}
    model_probs = {t: [] for t in initial_mapping}
    with torch.no_grad():
        for img_path, _ in images:
            img = Image.open(img_path).convert('RGB')
            x = transform(img).unsqueeze(0).to(device)
            outputs = model(x)
            for task in initial_mapping:
                p = torch.softmax(outputs[task][0], dim=0).cpu().numpy()
                model_probs[task].append(p)
                model_preds[task].append(int(p.argmax()))

    # 4) Confusion (model_class × bench_class) → pour chercher le mapping
    confusion = {}
    for task in initial_mapping:
        M = len(tasks_json[task])
        B = len(bench_classes[task])
        C = np.zeros((M, B), dtype=int)
        for mc, bc in zip(model_preds[task], gt[task]):
            C[mc, bc] += 1
        confusion[task] = C

    # 5) Mapping optimal (ou inverse du mapping fourni)
    inverted = {}
    if auto_mapping:
        print("\n=== Recherche exhaustive des mappings ===")
        for task, C in confusion.items():
            M, B = C.shape
            best_score, best_vec = -1.0, None
            for vec in itertools.product(range(B), repeat=M):
                A = np.zeros((B, B), dtype=int)
                for mc in range(M):
                    A[vec[mc]] += C[mc]
                f1s = []
                for b in range(B):
                    tp = A[b, b]
                    p_sum = A[b].sum()
                    t_sum = A[:, b].sum()
                    p = tp / p_sum if p_sum else 0.0
                    r = tp / t_sum if t_sum else 0.0
                    f1s.append(2*p*r/(p+r) if (p+r) else 0.0)
                score = float(np.mean(f1s))
                if score > best_score:
                    best_score, best_vec = score, vec
            inverted[task] = {tasks_json[task][mc].lower(): best_vec[mc] for mc in range(len(best_vec))}
            print(f"✅  Meilleur F1-macro « {task} » = {best_score:.4f}")
    else:
        for task, mp in initial_mapping.items():
            inv = {}
            for bidx, bench_cls in enumerate(bench_classes[task]):
                for mc_name in mp[bench_cls]:
                    inv[mc_name.lower()] = bidx
            inverted[task] = inv

    # 6) Mapping final (sauvegarde)
    final_mapping = {}
    for task, bench_list in bench_classes.items():
        mp = {b: [] for b in bench_list}
        for mc_name in tasks_json[task]:
            bidx = inverted[task].get(mc_name.lower(), len(bench_list) - 1)
            mp[bench_list[bidx]].append(mc_name)
        final_mapping[task] = mp

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "best_mapping.json"), 'w') as f:
        json.dump(final_mapping, f, indent=2)

    # 7) Remapping des probas → espace benchmark
    preds_b, probs_b = {}, {}
    for task in initial_mapping:
        B = len(bench_classes[task])
        preds_b[task], probs_b[task] = [], []
        for p_model in model_probs[task]:
            p_b = np.zeros(B, dtype=float)
            for mc_idx, mc_name in enumerate(tasks_json[task]):
                b = inverted[task].get(mc_name.lower(), B - 1)
                p_b[b] += p_model[mc_idx]
            probs_b[task].append(p_b)
            preds_b[task].append(int(p_b.argmax()))

    # 8) Métriques + ROC (+ seuils si besoin, même s’ils ne servent pas à l’overlay simplifié)
    os.makedirs(roc_dir, exist_ok=True)
    summary = {}
    for task in initial_mapping:
        y_true = np.array(gt[task], dtype=int)
        y_pred = np.array(preds_b[task], dtype=int)
        if not probs_b[task]:
            print(f"[Warning] pas de probabilités pour la tâche '{task}', métriques ignorées.")
            continue
        y_prob = np.vstack(probs_b[task])
        B = len(bench_classes[task])
        labels = list(range(B))

        prec_pc = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        rec_pc  = recall_score(   y_true, y_pred, labels=labels, average=None, zero_division=0)
        f1_pc   = f1_score(       y_true, y_pred, labels=labels, average=None, zero_division=0)
        prec_m  = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec_m   = recall_score(   y_true, y_pred, average='macro', zero_division=0)
        f1_m    = f1_score(       y_true, y_pred, average='macro', zero_division=0)

        # AUC + tracé ROC
        auc_pc = []
        for i in range(B):
            try:
                auc_pc.append(roc_auc_score((y_true == i).astype(int), y_prob[:, i]))
            except ValueError:
                auc_pc.append(None)
        auc_global = float(np.nanmean([a for a in auc_pc if a is not None])) if any(auc_pc) else None

        plt.figure()
        for i, color in zip(range(B), itertools.cycle(['aqua','darkorange','cornflowerblue','green','red','purple','brown','olive'])):
            if auc_pc[i] is None:
                continue
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
            plt.plot(fpr, tpr, color=color, label=f"{bench_classes[task][i]} (AUC={auc_pc[i]:.2f})")
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC – {task}")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(roc_dir, f"roc_{task.replace(' ','_')}.png"))
        plt.close()

        summary[task] = {
            "n_samples": int(len(y_true)),
            "per_class": {
                "precision": {bench_classes[task][i]: float(prec_pc[i]) for i in labels},
                "recall":    {bench_classes[task][i]: float(rec_pc[i])  for i in labels},
                "f1_score":  {bench_classes[task][i]: float(f1_pc[i])   for i in labels},
                "auc":       {bench_classes[task][i]: (None if auc_pc[i] is None else float(auc_pc[i])) for i in labels},
            },
            "global": {
                "precision_macro": float(prec_m),
                "recall_macro":    float(rec_m),
                "f1_macro":        float(f1_m),
                "auc_macro":       (None if auc_global is None else float(auc_global))
            }
        }

    # 9) Sauvegarde du résumé
    with open(os.path.join(save_dir, "benchmark_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    # 10) Images annotées — n’afficher que GT & Pred par tâche
    if save_pred_images:
        out_dir = pred_images_dir or os.path.join(save_dir, "pred_images")
        os.makedirs(out_dir, exist_ok=True)
        print(f"[overlay] Sauvegarde des images annotées → {out_dir}")

        def _maybe_resize(bgr_img, max_w):
            h, w = bgr_img.shape[:2]
            if w <= max_w:
                return bgr_img
            scale = max_w / float(w)
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        def _put_text_with_bg(img, text, org, txt_color, font, font_scale, thickness):
            ((tw, th), baseline) = cv2.getTextSize(text, font, font_scale, thickness)
            x, y = org
            pad = 4
            # fond noir semi-opaque
            bg = img[y - th - pad:y + baseline + pad, x - pad:x + tw + pad]
            if bg.size != 0:
                overlay = img.copy()
                cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), (0, 0, 0), -1)
                # alpha blending
                alpha = 0.5
                img[y - th - pad:y + baseline + pad, x - pad:x + tw + pad] = cv2.addWeighted(
                    overlay[y - th - pad:y + baseline + pad, x - pad:x + tw + pad],
                    alpha,
                    img[y - th - pad:y + baseline + pad, x - pad:x + tw + pad],
                    1 - alpha,
                    0.0
                )
            cv2.putText(img, text, org, font, font_scale, txt_color, thickness, lineType=cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        y0, step = 30, 28

        for idx, (img_path, _) in enumerate(images):
            bgr = cv2.imread(img_path)
            if bgr is None:
                pil = Image.open(img_path).convert('RGB')
                bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            bgr = _maybe_resize(bgr, max_width)

            line = 0
            for task in initial_mapping:
                true_idx = int(gt[task][idx])
                pred_idx = int(preds_b[task][idx])
                # noms sécurisés
                gt_name   = bench_classes[task][true_idx] if 0 <= true_idx < len(bench_classes[task]) else "Unknown"
                pred_name = bench_classes[task][pred_idx] if 0 <= pred_idx < len(bench_classes[task]) else "Unknown"

                ok = (true_idx == pred_idx)
                color = (0, 200, 0) if ok else (0, 0, 255)  # vert si correct, sinon rouge
                text = f"{task}: GT={gt_name} | Pred={pred_name}"

                y = y0 + line * step
                _put_text_with_bg(bgr, text, (10, y), color, font, font_scale, thickness)
                line += 1

            base = os.path.basename(img_path)
            out_name = f"{idx:06d}__{base}"
            cv2.imwrite(os.path.join(out_dir, out_name), bgr)

    print(f"\n✅  Résumé sauvé dans {os.path.join(save_dir,'benchmark_summary.json')}")




VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

def collect_image_paths(folder):
    """
    Retourne la liste de tous les fichiers images dans `folder` et ses sous-dossiers.
    """
    paths = []
    for root, _, files in os.walk(folder):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in VALID_EXTS:
                paths.append(os.path.join(root, fn))
    return paths


def annotate_and_save(
    img: Image.Image,
    text_lines,
    out_path,
    max_width_ratio: float = 0.35,   # ≤ 35 % de la largeur
    min_font_px: int = 18,
    max_font_px: int = 32,
    pad_px: int = 10,
    bg_alpha: int = 220             # fond blanc quasi-opaque
):
    """
    Incruste un encart blanc + texte vert contour noir, ergonomique et lisible.
    - L'encart n'excède jamais max_width_ratio * largeur de l'image.
    - La taille de police est ajustée dans [min_font_px, max_font_px].
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    W, H = img.size

    # ----- police : on tente DejaVuSans-Bold puis fallback défaut -----
    for name in ["DejaVuSans-Bold.ttf", "arialbd.ttf", "arial.ttf"]:
        try:
            font_path = name  # Pillow cherche dans ses fonts internes
            break
        except IOError:
            font_path = None
    if font_path is None:
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(font_path, size=min_font_px)

    draw_tmp = ImageDraw.Draw(img)
    target_width = W * max_width_ratio

    # ----- ajuste dynamiquement la taille de police -----
    fsize = min_font_px
    while fsize < max_font_px:
        cand = ImageFont.truetype(font_path, fsize + 2) if font_path else font
        test_w = max(draw_tmp.textsize(t, cand)[0] for t in text_lines)
        if test_w > target_width:          # on dépasse la limite
            break
        fsize += 2
        font = cand

    # ----- dimensions de la box -----
    line_h = draw_tmp.textsize("Ag", font=font)[1]
    block_w = min(target_width, max(draw_tmp.textsize(t, font=font)[0] for t in text_lines)) + 2 * pad_px
    block_h = line_h * len(text_lines) + 2 * pad_px
    x0, y0 = 10, 10                        # coin supérieur-gauche constant

    # ----- overlay -----
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([x0, y0, x0 + block_w, y0 + block_h], fill=(255, 255, 255, bg_alpha))

    # ----- texte avec contour noir fin -----
    for i, line in enumerate(text_lines):
        tx, ty = x0 + pad_px, y0 + pad_px + i * line_h
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:   # contour
            draw.text((tx+dx, ty+dy), line, font=font, fill=(0,0,0,255))
        draw.text((tx, ty), line, font=font, fill=(34,139,34,255))

    result = Image.alpha_composite(img, overlay).convert("RGB")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.save(out_path)
