# benchmark_utils.py
import os
import json
import itertools

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt


def test_benchmark_folder(
    model: torch.nn.Module,
    device: torch.device,
    benchmark_folder: str,
    mapping_path: str,
    tasks_json: dict,
    transform,
    save_dir: str,
    roc_dir: str,
    auto_mapping: bool = False
):
    """
    Version extraite de ton script.
    """
    with open(mapping_path, 'r') as f:
        initial_mapping = json.load(f)

    bench_classes = {
        task: list(initial_mapping[task].keys())
        for task in initial_mapping
    }

    images = []
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

    gt = {task: [] for task in initial_mapping}
    for _, bench_cls in images:
        for task in initial_mapping:
            lowers = [b.lower() for b in bench_classes[task]]
            low = bench_cls.lower()
            if low in lowers:
                idx = lowers.index(low)
            else:
                idx = len(lowers) - 1
            gt[task].append(idx)

    model.to(device)
    model.eval()
    model_preds = {t: [] for t in initial_mapping}
    model_probs = {t: [] for t in initial_mapping}

    with torch.no_grad():
        for img_path, _ in images:
            img = Image.open(img_path).convert('RGB')
            x = transform(img).unsqueeze(0).to(device)
            outputs = model(x)
            for task in initial_mapping:
                probs = torch.softmax(outputs[task][0], dim=0).cpu().numpy()
                model_probs[task].append(probs)
                model_preds[task].append(int(probs.argmax()))

    confusion = {}
    for task in initial_mapping:
        M = len(tasks_json[task])
        B = len(bench_classes[task])
        C = np.zeros((M, B), dtype=int)
        for mc, bc in zip(model_preds[task], gt[task]):
            C[mc, bc] += 1
        confusion[task] = C

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
                    f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
                score = np.mean(f1s)
                if score > best_score:
                    best_score, best_vec = score, vec
            inverted[task] = {
                tasks_json[task][mc].lower(): best_vec[mc]
                for mc in range(len(best_vec))
            }
            print(f"✅  Meilleur F1-macro « {task} » = {best_score:.4f}")
    else:
        for task, mp in initial_mapping.items():
            inv = {}
            for bidx, bench_cls in enumerate(bench_classes[task]):
                for mc_name in mp[bench_cls]:
                    inv[mc_name.lower()] = bidx
            inverted[task] = inv

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

    preds_b, probs_b = {}, {}
    for task in initial_mapping:
        B = len(bench_classes[task])
        preds_b[task], probs_b[task] = [], []
        for p_model in model_probs[task]:
            p_bench = np.zeros(B, dtype=float)
            for idx_mc, mc_name in enumerate(tasks_json[task]):
                b = inverted[task].get(mc_name.lower(), B - 1)
                p_bench[b] += p_model[idx_mc]
            probs_b[task].append(p_bench)
            preds_b[task].append(int(p_bench.argmax()))

    os.makedirs(roc_dir, exist_ok=True)
    summary = {}
    for task in initial_mapping:
        y_true = np.array(gt[task])
        y_pred = np.array(preds_b[task])
        if not probs_b[task]:
            print(f"[Warning] pas de probabilités pour la tâche '{task}', métriques ignorées.")
            continue
        y_prob = np.vstack(probs_b[task])
        B = len(bench_classes[task])
        labels = list(range(B))

        prec_pc = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        rec_pc = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        f1_pc = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

        prec_m = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec_m = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_m = f1_score(y_true, y_pred, average='macro', zero_division=0)

        auc_pc = []
        for i in range(B):
            try:
                auc_pc.append(roc_auc_score((y_true == i).astype(int), y_prob[:, i]))
            except ValueError:
                auc_pc.append(None)
        auc_global = np.mean([a for a in auc_pc if a is not None]) if any(auc_pc) else None

        plt.figure()
        colors = ['aqua', 'darkorange', 'cornflowerblue', 'green',
                  'red', 'purple', 'brown', 'olive']
        for i, color in zip(range(B), itertools.cycle(colors)):
            if auc_pc[i] is None:
                continue
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
            plt.plot(fpr, tpr, color=color,
                     label=f"{bench_classes[task][i]} (AUC={auc_pc[i]:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC – {task}")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(roc_dir, f"roc_{task.replace(' ', '_')}.png"))
        plt.close()

        summary[task] = {
            "n_samples": int(len(y_true)),
            "per_class": {
                "precision": {bench_classes[task][i]: float(prec_pc[i]) for i in labels},
                "recall": {bench_classes[task][i]: float(rec_pc[i]) for i in labels},
                "f1_score": {bench_classes[task][i]: float(f1_pc[i]) for i in labels},
                "auc": {bench_classes[task][i]: auc_pc[i] for i in labels}
            },
            "global": {
                "precision_macro": prec_m,
                "recall_macro": rec_m,
                "f1_macro": f1_m,
                "auc_macro": auc_global
            }
        }

    with open(os.path.join(save_dir, "benchmark_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅  Résumé sauvé dans {os.path.join(save_dir, 'benchmark_summary.json')}")
