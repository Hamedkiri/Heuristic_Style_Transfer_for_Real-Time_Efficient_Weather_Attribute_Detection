# main.py
import os
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from utils.datasets_utils import (
    MultiTaskDataset,
    create_dataloader,
    build_default_transform,
    subsample_dataset,
)
from utils.PMG.PMG_model_utils import (
    build_model_from_hparams,
    load_best_model,
    print_model_parameters,
    find_sidecar_hparams,
)
from Functions.PMG_evaluation import test_model_optimized
from utils.tsne_utils import (
    compute_embeddings_with_paths,
    perform_tsne,  # à importer si tu l’as collée dans tsne_utils.py
    plot_tsne_interactive
)
from utils.camera_utils import run_camera
from utils.benchmark_utils import test_benchmark_folder
from utils.PMG.PMG_style_transfer import run_patch_gram_style_transfer


def main():
    parser = argparse.ArgumentParser(
        description="Test Multi-Task PatchGAN Gram Model with Options (modulaire)"
    )
    parser.add_argument('--data', type=str, help="Path to dataset JSON")
    parser.add_argument('--build_classifier', type=str, required=True,
                        help="Path to tasks/classes JSON")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to trained .pth model")
    parser.add_argument('--config_path', type=str, default=None,
                        help="Path to JSON config for hyperparams")

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--save_dir', default='results', type=str)
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--mode', choices=[
        'classifier', 'tsne', 'tsne_interactive',
        'clustering', 'camera', 'benchmark'
    ], default='classifier')
    parser.add_argument('--prob_threshold', default=0.5, type=float)
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
    parser.add_argument('--visualize_gradcam', action='store_true')
    parser.add_argument('--save_gradcam_images', action='store_true')
    parser.add_argument('--gradcam_task', type=str, default=None)
    parser.add_argument('--colormap', type=str, default='hot')
    parser.add_argument('--per_task_tsne', action='store_true')
    parser.add_argument('--count_params', action='store_true')
    parser.add_argument('--style_transfer', action='store_true')
    parser.add_argument('--target_loss', type=float, default=1e-18)
    parser.add_argument('--style_iterations', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--init_type', type=str, default="noise")
    parser.add_argument('--search_folder', type=str, default=None)

    parser.add_argument('--benchmark_folder', type=str)
    parser.add_argument('--benchmark_mapping', type=str)
    parser.add_argument('--roc_output', type=str, default='roc_curves')
    parser.add_argument('--auto_mapping', action='store_true')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.save_dir, exist_ok=True)
    writer = (SummaryWriter(log_dir=os.path.join(args.save_dir, 'TensorBoard'))
              if args.tensorboard else None)

    with open(args.build_classifier, 'r') as f:
        tasks = json.load(f)
    print("Tasks:", tasks)

    # ----- Hyperparams -----
    if args.config_path is not None:
        if not os.path.isfile(args.config_path):
            raise FileNotFoundError(f"No config file at {args.config_path}")
        with open(args.config_path, 'r') as f:
            cfg = json.load(f)
        hparams = cfg.get("hparams", cfg)
        print("Loaded config from --config_path:", hparams)
    else:
        hparams = find_sidecar_hparams(args.model_path) or {}
        if hparams:
            print("Loaded hparams from sidecar.")
        else:
            print("[WARN] No hparams found (no --config_path and no sidecar). "
                  "Fallback to defaults; risque d’incompatibilité avec le checkpoint.")

    # ----- Modèle -----
    model = build_model_from_hparams(tasks, hparams, device)
    model.device = device
    load_best_model(model, args.model_path, device)

    if args.count_params:
        print_model_parameters(model)

    # ----- Données (si nécessaire) -----
    if args.mode in ['classifier', 'tsne', 'tsne_interactive', 'clustering']:
        if not args.data:
            raise ValueError("--data is required for this mode.")
        transform = build_default_transform(img_size=224)
        dataset = MultiTaskDataset(args.data, args.build_classifier,
                                   transform=transform,
                                   search_folder=args.search_folder)
        dataset = subsample_dataset(dataset, args.num_samples)

        task_names = list(tasks.keys())
        test_loader = create_dataloader(dataset, task_names,
                                        batch_size=args.batch_size,
                                        num_workers=4, shuffle=False)
    else:
        test_loader = None

    # ===================== MODES =====================

    if args.mode == 'classifier':
        criterions = {t: nn.CrossEntropyLoss().to(device) for t in tasks.keys()}
        avg_loss, metrics, overall_f1, times = test_model_optimized(
            model, test_loader, criterions, writer, args.save_dir, device, tasks,
            prob_threshold=args.prob_threshold,
            visualize_gradcam=args.visualize_gradcam,
            save_gradcam_images=args.save_gradcam_images,
            gradcam_task=args.gradcam_task,
            colormap=args.colormap
        )

        if args.measure_time and times:
            with open(os.path.join(args.save_dir, "times_classifier.json"), "w") as f:
                json.dump(times, f, indent=2)

        if args.style_transfer:
            print("Running patch-based style transfer on the dataset...")
            base_dataset = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
            style_out_dir = os.path.join(args.save_dir, "StyleTransfer")
            os.makedirs(style_out_dir, exist_ok=True)
            num_imgs = args.num_samples if args.num_samples is not None else len(base_dataset)
            for idx in range(num_imgs):
                img_path = base_dataset.samples[idx][0]
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                print(f"Processing style transfer for image {idx + 1}/{num_imgs}: {img_path}")
                style_img = Image.open(img_path).convert('RGB')
                style_transform = build_default_transform(img_size=224)
                style_img_tensor = style_transform(style_img).unsqueeze(0).to(device)
                gen_img_pil = run_patch_gram_style_transfer(
                    model,
                    style_img_tensor,
                    num_iterations=args.style_iterations,
                    lr=args.lr,
                    init_type=args.init_type,
                    device=device,
                    target_loss=args.target_loss
                )
                orig_img_resized = style_img.resize((256, 256), Image.LANCZOS)
                width, height = orig_img_resized.size
                combined = Image.new("RGB", (width * 2, height))
                combined.paste(orig_img_resized, (0, 0))
                combined.paste(gen_img_pil, (width, 0))
                out_path = os.path.join(style_out_dir, f"{base_name}_styled.png")
                combined.save(out_path)
                print(f"Saved styled image: {out_path}")

    elif args.mode == 'tsne':
        embeddings, labels, img_paths = compute_embeddings_with_paths(
            model, test_loader, device, per_task_tsne=args.per_task_tsne
        )
        if args.per_task_tsne:
            for t in embeddings:
                perform_tsne(embeddings[t], labels[t], tasks[t],
                             args.colors, args.save_dir, t)
        else:
            first_task = list(tasks.keys())[0]
            perform_tsne(embeddings, labels, tasks[first_task],
                         args.colors, args.save_dir, "AllTasks")

    elif args.mode == 'tsne_interactive':

        embeddings_data, labels_data, img_paths_data = compute_embeddings_with_paths(
            model, test_loader, device, per_task_tsne=True
        )
        plot_tsne_interactive(embeddings_data, labels_data, tasks, img_paths_data,
                              colors=args.colors, num_clusters=None, save_dir=args.save_dir)



    elif args.mode == 'camera':
        transform_cam = build_default_transform(img_size=224)
        run_camera(model, transform_cam, tasks, args.save_dir, args.prob_threshold,
                   args.measure_time, args.camera_index, args.kalman_filter,
                   args.save_camera_video)

    elif args.mode == 'benchmark':
        tasks_json = tasks
        if not args.benchmark_folder or not args.benchmark_mapping:
            raise ValueError("Pour le mode 'benchmark20042025', précisez "
                             "--benchmark_folder et --benchmark_mapping")
        transform = build_default_transform(img_size=224)
        test_benchmark_folder(
            model=model,
            device=device,
            benchmark_folder=args.benchmark_folder,
            mapping_path=args.benchmark_mapping,
            tasks_json=tasks_json,
            transform=transform,
            save_dir=args.save_dir,
            roc_dir=args.roc_output,
            auto_mapping=args.auto_mapping
        )

    if writer:
        writer.close()


if __name__ == '__main__':
    main()
