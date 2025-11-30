# main_multitask_eval.py
import argparse
import os
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models

from utils.datasets_utils import MultiTaskDataset, collate_multitask
from Models.models_RTM import MultiHeadAttentionPerTaskModel, print_model_parameters
from utils.tsne_utils import  perform_tsne, plot_tsne_interactive
from Functions.RTM_evaluation import test, load_best_model, test_benchmark_folder, test_folder_predictions, watch_folders_predictions, run_inference
from utils.camera_utils import run_camera
import numpy as np


def build_argparser():
    parser = argparse.ArgumentParser(description='Test du modèle multi-tâches avec différents modes')
    parser.add_argument('--data', type=str, help='Chemin vers le fichier JSON du dataset')
    parser.add_argument('--config_path', type=str, required=True, help='Chemin vers le fichier JSON des hyperparamètres')
    parser.add_argument('--model_path', type=str, required=True, help='Chemin vers le modèle entraîné')
    parser.add_argument('--batch_size', default=32, type=int, help='Taille de lot pour le test')
    parser.add_argument('--save_dir', default='results', type=str, help='Répertoire pour enregistrer les résultats')
    parser.add_argument('--tensorboard', action='store_true', help='Activer TensorBoard')
    parser.add_argument('--build_classifier', type=str, required=True, help='Chemin vers le fichier JSON des classes')
    parser.add_argument(
        '--mode',
        choices=['classifier', 'tsne', 'tsne_interactive', 'camera', 'clustering',
                 'folder', 'watch_folder', 'inference', 'benchmark'],
        default='classifier'
    )
    parser.add_argument('--prob_threshold', default=0.5, type=float)
    parser.add_argument('--visualize_gradcam', action='store_true')
    parser.add_argument('--save_gradcam_images', action='store_true')
    parser.add_argument('--measure_time', action='store_true')
    parser.add_argument('--save_test_images', action='store_true')
    parser.add_argument('--colors', nargs='+', default=None)
    parser.add_argument('--clustering_class', type=str)
    parser.add_argument('--min_cluster_size', type=int, nargs='+', default=[10, 15, 20])
    parser.add_argument('--min_samples', type=int, nargs='+', default=[5, 10])
    parser.add_argument('--kalman_filter', action='store_true')
    parser.add_argument('--camera_index', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--save_camera_video', action='store_true')
    parser.add_argument('--gradcam_task', type=str, default=None)
    parser.add_argument('--colormap', type=str, default='hot')
    parser.add_argument('--per_task_tsne', action='store_true')
    parser.add_argument('--count_params', action='store_true')
    parser.add_argument('--integrated_gradients', action='store_true')
    parser.add_argument('--test_images_folder', type=str)
    parser.add_argument('--target_task', type=str, default=None)
    parser.add_argument('--search_folder', type=str, default=None)
    parser.add_argument('--watch_folders', type=str, default=None)
    parser.add_argument('--poll_intervals', type=str, default=None)
    parser.add_argument('--save_dir_to_canon', default=None, type=str)
    parser.add_argument('--image_folder', type=str)
    parser.add_argument('--benchmark_folder', type=str)
    parser.add_argument('--benchmark_mapping', type=str)
    parser.add_argument('--roc_output', type=str, default='roc_curves')
    parser.add_argument('--auto_mapping', action='store_true')

    parser.add_argument('--no_attention', action='store_true')
    parser.add_argument('--attn_token_dim', type=int, default=None)
    parser.add_argument('--cls_hidden_dims', type=int, nargs='*', default=[])
    parser.add_argument('--cls_num_layers', type=int, default=0)
    parser.add_argument('--find_images_by_sub_folder', default=None)
    parser.add_argument('--no_gt_labels', action='store_true')

    parser.add_argument('--save_pred_images', action='store_true')
    parser.add_argument('--pred_images_dir', type=str, default=None)
    parser.add_argument('--overlay_topk', type=int, default=1)
    parser.add_argument('--draw_prob_threshold', type=float, default=None)
    parser.add_argument('--overlay_max_width', type=int, default=1280)
    parser.add_argument('--overlay_font_scale', type=float, default=0.6)
    parser.add_argument('--overlay_thickness', type=int, default=2)

    return parser


def main():
    from Functions.RTM_evaluation import compute_embeddings_with_paths
    parser = build_argparser()
    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'TensorBoard')) if args.tensorboard else None

    # Hyperparamètres
    with open(args.config_path, 'r') as f:
        best_config = json.load(f)
    truncate_layer = int(best_config.get('truncate_layer', 10))

    # Tâches / classes
    with open(args.build_classifier, 'r') as f:
        tasks = json.load(f)
    print(f"Nombre de classifieurs (tâches) : {len(tasks)}")
    for task_name, class_list in tasks.items():
        print(f" - {task_name}: {len(class_list)} classes")

    # Transformations
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Modèle
    base_encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = MultiHeadAttentionPerTaskModel(
        base_encoder=base_encoder,
        truncate_after_layer=truncate_layer,
        tasks=tasks,
        device=device,
        use_attention=(not args.no_attention),
        attn_token_dim=args.attn_token_dim,
        cls_hidden_dims=args.cls_hidden_dims,
        cls_num_layers=args.cls_num_layers
    ).to(device)
    print(f"[Model] use_attention={not args.no_attention} | attn_token_dim={args.attn_token_dim} | "
          f"C={model.num_features}")

    # Checkpoint
    load_best_model(model, args.model_path, strict_backbone=True)

    # Dataset / DataLoader si nécessaire
    needs_data = args.mode in ['classifier', 'clustering', 'tsne', 'tsne_interactive',
                               'folder', 'watch_folder']
    if needs_data:
        if args.data is None:
            raise ValueError("Le chemin '--data' doit être spécifié pour ce mode.")
        dataset = MultiTaskDataset(
            data_json=args.data,
            classes_json=args.build_classifier,
            transform=transform,
            search_folder=args.search_folder if args.find_images_by_sub_folder is None else None,
        )

        if args.find_images_by_sub_folder:
            new_samples = []
            for img_path, lab in dataset.samples:
                sub = os.path.basename(os.path.dirname(img_path))
                base = os.path.basename(img_path)
                new_path = os.path.join(args.find_images_by_sub_folder, sub, base)
                new_samples.append((new_path, lab))
            dataset.samples = new_samples

        if args.num_samples is not None:
            import random as _rnd
            indices = list(range(len(dataset)))
            _rnd.shuffle(indices)
            indices = indices[:min(args.num_samples, len(indices))]
            from torch.utils.data import Subset
            dataset = Subset(dataset, indices)

        test_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
            collate_fn=collate_multitask
        )
    else:
        test_loader = None

    if args.count_params:
        print_model_parameters(model)

    # ---- MODES ----
    if args.mode == 'classifier':
        criterions = {task: nn.CrossEntropyLoss().to(device) for task in tasks.keys()}
        test(
            model, test_loader, criterions, writer, args.save_dir, device, tasks,
            args.prob_threshold, args.visualize_gradcam, args.save_gradcam_images,
            args.measure_time, args.save_test_images,
            args.gradcam_task, args.colormap,
            integrated_gradients=args.integrated_gradients,
            show_gt_labels=(not args.no_gt_labels)
        )

    elif args.mode == 'tsne':
        embeddings_data, labels_data, img_paths = compute_embeddings_with_paths(
            model, test_loader, device, per_task_tsne=args.per_task_tsne
        )
        if args.per_task_tsne:
            for task_name in tasks.keys():
                emb = embeddings_data[task_name]
                lbl = np.array(labels_data[task_name])
                results = {'embeddings': emb.tolist(), 'labels': lbl.tolist()}
                out = os.path.join(args.save_dir, f'embeddings_{task_name.replace(" ", "_")}.json')
                with open(out, 'w') as f:
                    json.dump(results, f)
                perform_tsne(emb, lbl, {task_name: tasks[task_name]}, args.colors, args.save_dir, task_name=task_name)
        else:
            all_emb = embeddings_data
            all_lbl = labels_data
            with open(os.path.join(args.save_dir, 'embeddings.json'), 'w') as f:
                json.dump({'embeddings': all_emb.tolist(), 'labels': all_lbl.tolist()}, f)
            perform_tsne(all_emb, all_lbl, tasks, args.colors, args.save_dir)

    elif args.mode == 'tsne_interactive':
        embeddings_data, labels_data, img_paths_data = compute_embeddings_with_paths(
            model, test_loader, device, per_task_tsne=args.per_task_tsne
        )
        if args.per_task_tsne:
            plot_tsne_interactive(embeddings_data, labels_data, tasks, img_paths_data,
                                  args.colors, save_dir=args.save_dir)
        else:
            embeddings_dict = {'All Tasks': embeddings_data}
            labels_dict     = {'All Tasks': labels_data}
            img_paths_dict  = {'All Tasks': img_paths_data}
            tasks_dict      = {'All Tasks': tasks[list(tasks.keys())[0]]}
            plot_tsne_interactive(embeddings_dict, labels_dict, tasks_dict, img_paths_dict,
                                  args.colors, save_dir=args.save_dir)

    elif args.mode == 'camera':
        run_camera(
            model, transform, tasks, args.save_dir, args.prob_threshold,
            args.measure_time, args.camera_index, args.kalman_filter, args.save_camera_video
        )

    elif args.mode == 'folder':
        if not args.test_images_folder:
            raise ValueError("En mode 'folder', --test_images_folder est requis.")
        print(f"Prédictions sur le dossier: {args.test_images_folder}")
        test_folder_predictions(
            model, tasks, args.test_images_folder, transform, device, args.save_dir,
            save_test_images=args.save_test_images, target_task=args.target_task
        )

    elif args.mode == 'watch_folder':
        if args.watch_folders is None:
            raise ValueError("--watch_folders doit être spécifié en mode watch_folder")
        watch_folders = [s.strip() for s in args.watch_folders.split(',')]
        if args.poll_intervals is None:
            poll_intervals = [5] * len(watch_folders)
        else:
            poll_intervals = [int(s.strip()) for s in args.poll_intervals.split(',')]
        watch_folders_predictions(
            model, tasks, watch_folders, poll_intervals, transform, device,
            args.save_dir, args.save_dir_to_canon
        )

    elif args.mode == 'inference':
        inf_tfm = transform  # tu peux en définir un spécifique si besoin
        run_inference(
            model=model, image_folder=args.image_folder, transform=inf_tfm, device=device,
            classes=tasks, num_samples=args.num_samples,
            save_dir=args.save_dir, save_test_images=args.save_test_images
        )

    elif args.mode == 'benchmark':
        if not args.benchmark_folder or not args.benchmark_mapping:
            raise ValueError("Pour 'benchmark', précisez --benchmark_folder et --benchmark_mapping")
        test_benchmark_folder(
            model=model, device=device,
            benchmark_folder=args.benchmark_folder,
            mapping_path=args.benchmark_mapping,
            tasks_json=tasks, transform=transform,
            save_dir=args.save_dir, roc_dir=args.roc_output,
            auto_mapping=args.auto_mapping, num_samples=args.num_samples,
            save_pred_images=args.save_pred_images,
            pred_images_dir=args.pred_images_dir,
            overlay_topk=args.overlay_topk,
            draw_prob_threshold=args.draw_prob_threshold,
        )

    if writer:
        writer.close()


if __name__ == '__main__':
    main()
