# main_patchgan.py
import argparse
import os
import json
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np

from utils.datasets_utils import MultiTaskDataset
from Models.models_PM import (
    MultiTaskPatchGAN,
    load_model_weights,
    checkpoint_has_se,
    print_model_parameters,
)
from Functions.PM_evaluation import (
    test_classifier,   # à coller dans eval_utils.py
    run_inference,
    compute_attn_embeddings_per_task_with_paths,
    test_folder_predictions,
    watch_folders_predictions
)
from utils.tsne_utils import (

    compute_embeddings_with_paths,
    perform_tsne,
    plot_tsne_interactive,
)
from utils.camera_utils import run_camera
from utils.benchmark_utils import test_benchmark_folder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test d'un PatchGAN Multi-tâches avec divers modes"
    )

    # chemins de base
    parser.add_argument('--data', type=str,
                        help='JSON du dataset (obligatoire pour classifier/clustering/tsne/inference)')
    parser.add_argument('--build_classifier', type=str, required=True,
                        help='JSON de description des tâches/classes')
    parser.add_argument('--config_path', type=str, required=True,
                        help="JSON d'hyperparamètres du modèle")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Fichier .pth du modèle entraîné')
    parser.add_argument('--save_dir', default='results', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--tensorboard', action='store_true')

    parser.add_argument(
        '--mode',
        choices=[
            'classifier', 'tsne', 'tsne_interactive', 'camera', 'clustering',
            'folder', 'benchmark_patchGAN_Gram', 'watch_folder', 'inference'
        ],
        default='classifier'
    )

    # Explainability
    parser.add_argument('--visualize_gradcam', action='store_true')
    parser.add_argument('--save_gradcam_images', action='store_true')
    parser.add_argument('--gradcam_task', type=str, default=None)
    parser.add_argument('--colormap', type=str, default='hot')
    parser.add_argument('--integrated_gradients', action='store_true')
    parser.add_argument('--integrated_gradients_task', type=str, default=None)

    # Inference / mesure
    parser.add_argument('--prob_threshold', default=0.5, type=float)
    parser.add_argument('--measure_time', action='store_true')
    parser.add_argument('--save_test_images', action='store_true')
    parser.add_argument('--count_params', action='store_true')

    # Données / dossiers
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--test_images_folder', type=str)
    parser.add_argument('--test_following_task', type=str, default=None)
    parser.add_argument('--image_folder', type=str)
    parser.add_argument('--search_folder', type=str, default=None)
    parser.add_argument('--find_images_by_sub_folder', type=str, default=None)

    # t-SNE / clustering
    parser.add_argument('--colors', nargs='+', default=None, metavar='COLORS')
    parser.add_argument('--per_task_tsne', action='store_true')
    parser.add_argument('--per_task', action='store_true')
    parser.add_argument('--clustering_class', type=str)
    parser.add_argument('--min_cluster_size', type=int, nargs='+', default=[10, 15, 20])
    parser.add_argument('--min_samples', type=int, nargs='+', default=[5, 10])

    # Caméra
    parser.add_argument('--kalman_filter', action='store_true')
    parser.add_argument('--camera_index', type=int, default=0)
    parser.add_argument('--save_camera_video', action='store_true')

    # Benchmark
    parser.add_argument('--benchmark_folder', type=str)
    parser.add_argument('--benchmark_mapping', type=str)
    parser.add_argument('--roc_output', type=str, default='roc_curves')
    parser.add_argument('--auto_mapping', action='store_true')

    # Watch folders
    parser.add_argument('--watch_folders', type=str, default=None)
    parser.add_argument('--poll_intervals', type=str, default=None)
    parser.add_argument('--save_dir_to_canon', default=None, type=str)
    parser.add_argument('--eval_annotations', action='store_true')
    parser.add_argument('--annotations_folders', type=str, default=None)
    parser.add_argument('--truth_mapping', type=str, default=None)
    parser.add_argument('--metry_every', default=50, type=int)

    # Attention
    parser.add_argument('--ablate_attention', action='store_true')
    parser.add_argument('--attn_use_se', action='store_true')
    parser.add_argument('--attn_tau', type=float, default=0.7)
    parser.add_argument('--attn_no_softmax', action='store_true')

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    writer = (SummaryWriter(log_dir=os.path.join(args.save_dir, 'TensorBoard'))
              if args.tensorboard else None)

    # hyperparams & tâches
    with open(args.config_path, 'r') as f:
        best_config = json.load(f)
    with open(args.build_classifier, 'r') as f:
        tasks_json = json.load(f)

    tasks_dict = {t: len(cls_list) for t, cls_list in tasks_json.items()}
    print(f"Nombre de tâches: {len(tasks_dict)}")
    for t, n in tasks_dict.items():
        print(f"  Tâche '{t}': {n} classes")

    patch_size = best_config.get('patch_size', 70)
    attn_tau = float(best_config.get('attn_tau', args.attn_tau))
    attn_softmax_spatial = bool(best_config.get('attn_softmax_spatial', not args.attn_no_softmax))

    ckpt_has_se = checkpoint_has_se(args.model_path, device)
    attn_use_se = True if ckpt_has_se else bool(best_config.get('attn_use_se', args.attn_use_se))
    print(f"[build] ckpt_has_se={ckpt_has_se} | attn_use_se(model)={attn_use_se} | ablate={args.ablate_attention}")

    model = MultiTaskPatchGAN(
        tasks_dict=tasks_dict,
        input_nc=3,
        ndf=64,
        norm="instance",
        patch_size=patch_size,
        device=device,
        attn_tau=attn_tau,
        attn_use_se=attn_use_se,
        attn_softmax_spatial=attn_softmax_spatial,
        ablate_attention=args.ablate_attention
    ).to(device)

    load_model_weights(model, args.model_path, device, strict=True)

    if args.count_params:
        print_model_parameters(model)

    std_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # -------------- Modes --------------

    if args.mode == 'classifier':
        if not args.data:
            raise ValueError("Spécifiez --data pour classifier.")
        dataset = MultiTaskDataset(args.data, args.build_classifier, std_transform,
                                   args.search_folder, args.find_images_by_sub_folder)
        if args.num_samples is not None:
            idx = list(range(len(dataset)))
            random.shuffle(idx)
            dataset = Subset(dataset, idx[:args.num_samples])
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=4, pin_memory=(device.type == 'cuda'))

        criterions = {t: nn.CrossEntropyLoss().to(device) for t in tasks_json.keys()}

        test_classifier(
            model, test_loader, criterions, writer, args.save_dir, device, tasks_json,
            prob_threshold=args.prob_threshold,
            visualize_gradcam=args.visualize_gradcam,
            save_gradcam_images=args.save_gradcam_images,
            measure_time=args.measure_time,
            save_test_images=args.save_test_images,
            gradcam_task=args.gradcam_task,
            colormap=args.colormap,
            integrated_gradients=args.integrated_gradients,
            integrated_gradients_task=args.integrated_gradients_task
        )

    elif args.mode == 'folder':
        if not args.test_images_folder:
            raise ValueError("Spécifiez --test_images_folder pour folder.")
        test_folder_predictions(
            model, tasks_json, args.test_images_folder, std_transform,
            device, args.save_dir,
            save_test_images=args.save_test_images,
            target_task=args.test_following_task
        )

    elif args.mode == 'inference':
        if not args.image_folder:
            raise ValueError("Spécifiez --image_folder pour inference.")
        run_inference(
            model, args.image_folder, std_transform, device,
            classes=tasks_json,
            num_samples=args.num_samples,
            save_dir=args.save_dir,
            save_test_images=args.save_test_images
        )

    elif args.mode == 'tsne':
        if not args.data:
            raise ValueError("Spécifiez --data pour tsne.")
        dataset = MultiTaskDataset(args.data, args.build_classifier, std_transform,
                                   args.search_folder, args.find_images_by_sub_folder)
        if args.num_samples is not None:
            idx = list(range(len(dataset)))
            random.shuffle(idx)
            dataset = Subset(dataset, idx[:args.num_samples])
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=4, pin_memory=(device.type == 'cuda'))

        if args.per_task:
            emb_dict, lbl_dict, _paths_dict = compute_attn_embeddings_per_task_with_paths(
                model, test_loader, device, tasks_json
            )
            for task_name, emb in emb_dict.items():
                lbl = lbl_dict[task_name]
                if emb.size == 0 or lbl.size == 0:
                    print(f"[t-SNE] Task '{task_name}': aucun échantillon labellisé.")
                    continue
                perform_tsne(
                    emb, lbl, {task_name: tasks_json[task_name]},
                    args.colors, args.save_dir, task_name=task_name
                )
        else:
            embeddings_data, labels_data, img_paths = compute_embeddings_with_paths(
                model, test_loader, device, tasks_json, per_task_tsne=args.per_task_tsne
            )
            if args.per_task_tsne:
                for task_name in embeddings_data.keys():
                    emb = embeddings_data[task_name]
                    lbl = labels_data[task_name]
                    perform_tsne(
                        emb, lbl, {task_name: tasks_json[task_name]},
                        args.colors, args.save_dir, task_name=task_name
                    )
            else:
                perform_tsne(
                    embeddings_data, labels_data, tasks_json,
                    args.colors, args.save_dir
                )

    elif args.mode == 'tsne_interactive':
        if not args.data:
            raise ValueError("Spécifiez --data pour tsne_interactive.")
        dataset = MultiTaskDataset(args.data, args.build_classifier, std_transform,
                                   args.search_folder, args.find_images_by_sub_folder)
        if args.num_samples is not None:
            idx = list(range(len(dataset)))
            random.shuffle(idx)
            dataset = Subset(dataset, idx[:args.num_samples])
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=4, pin_memory=(device.type == 'cuda'))

        if args.per_task:
            emb_dict, lbl_dict, paths_dict = compute_attn_embeddings_per_task_with_paths(
                model, test_loader, device, tasks_json
            )
            plot_tsne_interactive(
                emb_dict, lbl_dict, tasks_json, paths_dict,
                colors=args.colors, save_dir=args.save_dir
            )
        else:
            embeddings_data, labels_data, img_paths_data = compute_embeddings_with_paths(
                model, test_loader, device, tasks_json, per_task_tsne=args.per_task_tsne
            )
            plot_tsne_interactive(
                embeddings_data, labels_data, tasks_json, img_paths_data,
                colors=args.colors, save_dir=args.save_dir
            )

    elif args.mode == 'clustering':
        import hdbscan
        if not args.data:
            raise ValueError("Spécifiez --data pour clustering.")
        if not args.clustering_class:
            raise ValueError("Spécifiez --clustering_class pour clustering.")

        dataset = MultiTaskDataset(args.data, args.build_classifier, std_transform,
                                   args.search_folder, args.find_images_by_sub_folder)
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=4, pin_memory=(device.type == 'cuda'))

        embeddings, labels, img_paths = compute_embeddings_with_paths(
            model, test_loader, device, tasks_json, per_task_tsne=False
        )

        class_index, target_task_name = None, None
        for tname, clist in tasks_json.items():
            if args.clustering_class in clist:
                class_index = clist.index(args.clustering_class)
                target_task_name = tname
                break
        if class_index is None:
            raise ValueError(f"Classe '{args.clustering_class}' non trouvée.")

        selected = (labels == class_index)
        class_embeddings = embeddings[selected]
        class_img_paths = [img_paths[i] for i in range(len(labels)) if selected[i]]

        best_num_clusters, best_cluster_labels, best_params = 0, None, {}
        for mcs in args.min_cluster_size:
            for ms in args.min_samples:
                clustering = hdbscan.HDBSCAN(
                    min_cluster_size=mcs, min_samples=ms
                ).fit(class_embeddings)
                labels_c = clustering.labels_
                num_clusters = len(set(labels_c)) - (1 if -1 in labels_c else 0)
                if num_clusters > best_num_clusters:
                    best_num_clusters = num_clusters
                    best_cluster_labels = labels_c
                    best_params = {'min_cluster_size': mcs, 'min_samples': ms}

        if best_cluster_labels is None:
            print("Aucun cluster trouvé.")
        else:
            cluster_info = {}
            for lbl in set(best_cluster_labels):
                idxs = [i for i, c in enumerate(best_cluster_labels) if c == lbl]
                cluster_info[str(lbl)] = {
                    'num_images': len(idxs),
                    'img_paths': [class_img_paths[i] for i in idxs]
                }
            out_cluster_path = os.path.join(
                args.save_dir,
                f"{args.clustering_class}_clustering_results.json"
            )
            with open(out_cluster_path, 'w') as f:
                json.dump(
                    {
                        'num_clusters': best_num_clusters,
                        'clusters': cluster_info,
                        'best_params': best_params
                    },
                    f, indent=4
                )
            print(f"Clustering results saved to {out_cluster_path}")

    elif args.mode == 'camera':
        run_camera(
            model, std_transform, tasks_json, args.save_dir,
            args.prob_threshold, args.measure_time,
            args.camera_index, args.kalman_filter,
            args.save_camera_video
        )

    elif args.mode == 'benchmark_patchGAN_Gram':
        if not args.benchmark_folder or not args.benchmark_mapping:
            raise ValueError(
                "Pour 'benchmark_patchGAN_Gram', précisez --benchmark_folder et --benchmark_mapping"
            )
        test_benchmark_folder(
            model, device, args.benchmark_folder, args.benchmark_mapping,
            tasks_json, std_transform, args.save_dir,
            args.roc_output, args.auto_mapping
        )

    elif args.mode == 'watch_folder':
        if args.watch_folders is None:
            raise ValueError("--watch_folders doit être spécifié en mode watch_folder")

        watch_folders_predictions(
            model,
            tasks_json,
            args.watch_folders,
            args.poll_intervals,
            std_transform,
            device,
            args.save_dir,
            save_dir_to_canon=args.save_dir_to_canon,
            eval_annotations=args.eval_annotations,
            annotations_folders=args.annotations_folders,
            truth_mapping_path=args.truth_mapping,
            metrics_every=args.metry_every
        )

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
