# evaluation.py
import os
import json
import time
import csv

import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc
)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


IGNORE_INDEX = -100


def test_model_optimized(model,
                         test_loader,
                         criterions: dict,
                         writer,
                         save_dir: str,
                         device: torch.device,
                         tasks: dict,
                         prob_threshold: float,
                         visualize_gradcam: bool = False,
                         save_gradcam_images: bool = False,
                         gradcam_task: str | None = None,
                         colormap: str = 'hot'):
    """
    Fonction de test + ROC + Grad-CAM, extraite telle quelle de ton script.
    """
    os.makedirs(save_dir, exist_ok=True)
    roc_dir = os.path.join(save_dir, "roc")
    os.makedirs(roc_dir, exist_ok=True)
    gradcam_dir = os.path.join(save_dir, "gradcam")
    if visualize_gradcam and save_gradcam_images:
        os.makedirs(gradcam_dir, exist_ok=True)

    def _find_last_conv2d(sequential_module: nn.Module):
        for layer in reversed(list(sequential_module)):
            if isinstance(layer, nn.Conv2d):
                return layer
        return None

    def _denormalize_imagenet(t: torch.Tensor):
        t = t.detach().cpu().float()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        return img

    def _apply_colormap(gray_cam: np.ndarray, cmap_name='hot'):
        cm = plt.get_cmap(cmap_name)
        heat = cm(np.clip(gray_cam, 0, 1))[:, :, :3]
        return heat

    def _overlay_cam_on_image(img01: np.ndarray, heat01: np.ndarray, alpha=0.45):
        over = (1 - alpha) * img01 + alpha * heat01
        return np.clip(over, 0, 1)

    def _hstack_and_save(left01: np.ndarray, right01: np.ndarray, out_path: str):
        cat = np.hstack([
            (left01 * 255).astype(np.uint8),
            (right01 * 255).astype(np.uint8)
        ])
        cv2.imwrite(out_path, cv2.cvtColor(cat, cv2.COLOR_RGB2BGR))

    class TaskSpecificModelForGradCAM(nn.Module):
        def __init__(self, base_model, gradcam_task):
            super().__init__()
            self.base = base_model
            self.gradcam_task = gradcam_task

        def forward(self, x):
            out = self.base(x)
            return out[self.gradcam_task]

    model.eval()
    total_loss = 0.0
    total_samples = 0
    times = []

    all_preds = {t: [] for t in tasks.keys()}
    all_labels = {t: [] for t in tasks.keys()}
    all_probs_list = {t: [] for t in tasks.keys()}

    # Préparation Grad-CAM
    if visualize_gradcam:
        cam_task = gradcam_task if gradcam_task is not None else next(iter(tasks.keys()))
        if cam_task not in tasks:
            print(f"[GradCAM] Task '{cam_task}' absente de tasks -> CAM désactivé.")
            visualize_gradcam = False
        else:
            gradcam_model = TaskSpecificModelForGradCAM(model, gradcam_task=cam_task).to(device)
            gradcam_model.eval()
            last_conv_layer = _find_last_conv2d(model.feature_extractor)
            if last_conv_layer is None:
                print("[GradCAM] Aucune Conv2d trouvée -> CAM désactivé.")
                visualize_gradcam = False
            else:
                cam = GradCAM(model=gradcam_model, target_layers=[last_conv_layer])

    # ---------- Boucle test ----------
    for batch_idx, (inputs, labels) in enumerate(test_loader):
        t0 = time.time()
        inputs = inputs.to(device, non_blocking=True)

        with torch.no_grad():
            outputs = model(inputs)

        batch_loss = 0.0
        batch_size = inputs.size(0)

        for task, criterion in criterions.items():
            y = labels[task].to(device, non_blocking=True)
            m = (y != IGNORE_INDEX)
            if not m.any():
                continue
            logits = outputs[task][m]
            target = y[m]
            loss_val = criterion(logits, target)
            batch_loss += loss_val

            probs = torch.softmax(logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            preds[max_probs < prob_threshold] = -1

            all_preds[task].extend(preds.detach().cpu().numpy().tolist())
            all_labels[task].extend(target.detach().cpu().numpy().tolist())
            all_probs_list[task].extend(probs.detach().cpu().numpy().tolist())

        total_loss += batch_loss.item() * batch_size
        total_samples += batch_size
        times.append(time.time() - t0)

        if visualize_gradcam:
            with torch.enable_grad():
                logits_cam_task = model(inputs)[cam_task]
            class_names_cam = list(tasks[cam_task])
            for i in range(batch_size):
                gt = labels[cam_task][i].item()
                if gt is not None and gt != IGNORE_INDEX and gt >= 0:
                    cls_idx = int(gt)
                else:
                    cls_idx = int(torch.argmax(logits_cam_task[i]).item())
                targets = [ClassifierOutputTarget(cls_idx)]
                cls_name = class_names_cam[cls_idx] if 0 <= cls_idx < len(class_names_cam) else str(cls_idx)

                grayscale_cam = cam(input_tensor=inputs[i].unsqueeze(0), targets=targets)[0]
                H, W = inputs[i].shape[-2:]
                grayscale_cam = cv2.resize(grayscale_cam, (W, H), interpolation=cv2.INTER_LINEAR)

                orig_rgb01 = _denormalize_imagenet(inputs[i])
                heat_rgb01 = _apply_colormap(grayscale_cam, colormap)
                overlay = _overlay_cam_on_image(orig_rgb01, heat_rgb01, alpha=0.45)

                if save_gradcam_images:
                    out_name = (f"gradcam_b{batch_idx:04d}_i{i:03d}_task-"
                                f"{cam_task}_class-{cls_idx}-{cls_name}.png")
                    out_path = os.path.join(gradcam_dir, out_name)
                    _hstack_and_save(orig_rgb01, overlay, out_path)

    average_loss = (total_loss / total_samples) if total_samples else 0.0
    metrics = {}
    summary_rows = []

    for task, class_list in tasks.items():
        class_names = list(class_list)
        preds_arr = np.array(all_preds[task], dtype=int) if len(all_preds[task]) else np.array([], dtype=int)
        labels_arr = np.array(all_labels[task], dtype=int) if len(all_labels[task]) else np.array([], dtype=int)
        probs_arr = (np.array(all_probs_list[task], dtype=float)
                     if len(all_probs_list[task]) else np.zeros((0, len(class_names)), dtype=float))

        valid_mask = (preds_arr != -1)
        if valid_mask.sum() > 0:
            acc = float(np.mean(preds_arr[valid_mask] == labels_arr[valid_mask]))
            prec = float(precision_score(labels_arr[valid_mask], preds_arr[valid_mask],
                                         average='weighted', zero_division=0))
            rec = float(recall_score(labels_arr[valid_mask], preds_arr[valid_mask],
                                     average='weighted', zero_division=0))
            f1 = float(f1_score(labels_arr[valid_mask], preds_arr[valid_mask],
                                average='weighted', zero_division=0))
            conf = confusion_matrix(labels_arr[valid_mask], preds_arr[valid_mask]).tolist()
        else:
            acc = prec = rec = f1 = 0.0
            conf = []

        auc_macro = None
        auc_micro = None
        auc_per_class = {}
        roc_fig_path = None

        roc_csv_path = os.path.join(roc_dir, f"{task}_roc_data.csv")
        with open(roc_csv_path, "w", newline="") as fcsv:
            writer_csv = csv.writer(fcsv)
            writer_csv.writerow(["curve_type", "class_name", "fpr", "tpr"])

            if probs_arr.shape[0] > 0:
                true_valid = (labels_arr != IGNORE_INDEX)
                y_true = labels_arr[true_valid]
                probs_valid = probs_arr[true_valid]
                n_classes = len(class_names)

                present_classes = np.unique(y_true)
                if len(present_classes) >= 2 and probs_valid.shape[1] == n_classes:
                    if n_classes == 2:
                        if set([0, 1]).issubset(set(present_classes)) and probs_valid.shape[1] >= 2:
                            pos_name = class_names[1] if len(class_names) > 1 else "class_1"
                            y_bin = (y_true == 1).astype(int)
                            scores_pos = probs_valid[:, 1]
                            fpr, tpr, _ = roc_curve(y_bin, scores_pos)
                            auc_val = auc(fpr, tpr)
                            auc_per_class[pos_name] = float(auc_val)
                            auc_micro = float(auc_val)
                            auc_macro = float(auc_val)

                            for x, yv in zip(fpr, tpr):
                                writer_csv.writerow(["binary", pos_name, float(x), float(yv)])

                            plt.figure(figsize=(7, 6))
                            plt.plot(fpr, tpr, lw=2, label=f"{pos_name} (AUC={auc_val:.3f})")
                            plt.plot([0, 1], [0, 1], '--', color='gray', lw=1)
                            plt.xlim([0, 1])
                            plt.ylim([0, 1.05])
                            plt.xlabel("False Positive Rate")
                            plt.ylabel("True Positive Rate")
                            plt.title(f"ROC - Task: {task} (binary)")
                            plt.legend(loc="lower right", fontsize=9)
                            roc_fig_path = os.path.join(roc_dir, f"{task}_roc.png")
                            plt.tight_layout()
                            plt.savefig(roc_fig_path, dpi=150)
                            plt.close()
                    else:
                        y_bin_full = label_binarize(y_true, classes=list(range(n_classes)))
                        fpr_dict, tpr_dict, auc_dict = {}, {}, {}
                        valid_class_indices = []
                        for c in range(n_classes):
                            y_c = y_bin_full[:, c]
                            if y_c.sum() > 0 and y_c.sum() < y_c.shape[0]:
                                fpr_c, tpr_c, _ = roc_curve(y_c, probs_valid[:, c])
                                auc_c = auc(fpr_c, tpr_c)
                                fpr_dict[c], tpr_dict[c], auc_dict[c] = fpr_c, tpr_c, auc_c
                                cls_name = class_names[c] if 0 <= c < len(class_names) else f"class_{c}"
                                auc_per_class[cls_name] = float(auc_c)
                                valid_class_indices.append(c)
                                for x, yv in zip(fpr_c, tpr_c):
                                    writer_csv.writerow(["ovr", cls_name, float(x), float(yv)])

                        if len(valid_class_indices) > 0:
                            auc_macro = float(np.mean([auc_dict[c] for c in valid_class_indices]))
                            y_micro = y_bin_full[:, valid_class_indices].ravel()
                            p_micro = probs_valid[:, valid_class_indices].ravel()
                            fpr_mi, tpr_mi, _ = roc_curve(y_micro, p_micro)
                            auc_micro = float(auc(y_micro, p_micro))
                            for x, yv in zip(fpr_mi, tpr_mi):
                                writer_csv.writerow(["micro", "micro", float(x), float(yv)])

                            plt.figure(figsize=(9, 7))
                            for c in valid_class_indices:
                                cls_name = class_names[c] if 0 <= c < len(class_names) else f"class_{c}"
                                plt.plot(fpr_dict[c], tpr_dict[c], lw=1.2, alpha=0.8,
                                         label=f"{cls_name} (AUC={auc_dict[c]:.3f})")
                            plt.plot(fpr_mi, tpr_mi, lw=2.0, color='black',
                                     label=f"micro-avg (AUC={auc_micro:.3f})")
                            plt.plot([0, 1], [0, 1], '--', color='gray', lw=1)
                            plt.xlim([0, 1])
                            plt.ylim([0, 1.05])
                            plt.xlabel("False Positive Rate")
                            plt.ylabel("True Positive Rate")
                            plt.title(f"ROC - Task: {task} (multiclass)")
                            plt.legend(loc="lower right", fontsize=8)
                            roc_fig_path = os.path.join(roc_dir, f"{task}_roc.png")
                            plt.tight_layout()
                            plt.savefig(roc_fig_path, dpi=150)
                            plt.close()

        metrics[task] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'confusion_matrix': conf,
            'auc_macro': auc_macro,
            'auc_micro': auc_micro,
            'auc_per_class': auc_per_class if len(auc_per_class) > 0 else None,
            'roc_png': roc_fig_path,
            'class_names': class_names,
        }

        summary_rows.append({
            "task": task,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "auc_macro": auc_macro if auc_macro is not None else "",
            "auc_micro": auc_micro if auc_micro is not None else "",
            "roc_png": roc_fig_path if roc_fig_path else "",
        })

        msg = f"[Task {task}] Acc={acc:.4f}, Prec={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}"
        if auc_macro is not None:
            msg += f", AUC_macro={auc_macro:.4f}"
        if auc_micro is not None:
            msg += f", AUC_micro={auc_micro:.4f}"
        print(msg)

    f1_list = [metrics[t]['f1_score'] for t in metrics if metrics[t]['f1_score'] is not None]
    overall_f1 = float(np.mean(f1_list)) if len(f1_list) else 0.0
    print(f"Overall F1: {overall_f1:.4f}")

    if writer:
        writer.add_scalar("Test/Loss", average_loss)
        writer.add_scalar("Test/Overall_F1", overall_f1)

    metrics_out = {
        "test_loss": float(average_loss),
        "overall_f1": float(overall_f1),
        "tasks": metrics
    }
    with open(os.path.join(save_dir, "metrics_test.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    csv_path = os.path.join(save_dir, "metrics_test_summary.csv")
    with open(csv_path, "w", newline="") as fcsv:
        fieldnames = ["task", "accuracy", "precision", "recall", "f1_score",
                      "auc_macro", "auc_micro", "roc_png"]
        writer_csv = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer_csv.writeheader()
        for row in summary_rows:
            writer_csv.writerow(row)

    return average_loss, metrics, overall_f1, times
