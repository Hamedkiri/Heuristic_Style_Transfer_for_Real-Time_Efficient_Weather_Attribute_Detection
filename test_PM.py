import argparse
import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.tensorboard import SummaryWriter
import random
import cv2
import hdbscan
from Functions.function_PatchGAN import run_camera, load_model_weights, print_model_parameters, test_classifier, test_folder_predictions, perform_tsne, plot_tsne_interactive, compute_embeddings_with_paths, watch_folders_predictions, test_benchmark_folder, run_inference
from datas import MultiTaskDataset
from Models.models_PatchGAN import MultiTaskPatchGANTest
# -------------------------------------------------------------------
# Le dictionnaire de colormaps OpenCV pour Grad-CAM
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
    # Vous pouvez en ajouter d'autres si nécessaire
}




# -------------------------------------------------------------------
# 4) MODELE POUR GRADCAM (sélectionner la tâche)


# -------------------------------------------------------------------
# 5) TOUTES LES FONCTIONS DE TEST
#    (classif, tsne, clustering, camera, etc.)
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Test d\'un PatchGAN Multi-tâches avec divers modes')
    parser.add_argument('--data', type=str, help='Chemin vers le fichier JSON du dataset (obligatoire pour classifier/clustering/tsne)')
    parser.add_argument('--build_classifier', type=str, required=True, help='Chemin vers le JSON de description des tâches/classes')
    parser.add_argument('--config_path', type=str, required=True, help='Chemin vers le JSON d\'hyperparamètres du modèle')
    parser.add_argument('--model_path', type=str, required=True, help='Chemin vers le fichier .pth du modèle entraîné')
    parser.add_argument('--batch_size', default=32, type=int, help='Taille de lot pour le test')
    parser.add_argument('--save_dir', default='results', type=str, help='Répertoire de sortie pour les résultats')
    parser.add_argument('--tensorboard', action='store_true', help='Activer TensorBoard')
    parser.add_argument('--mode', choices=['classifier', 'tsne', 'tsne_interactive', 'camera', 'clustering', 'folder', 'watch_folder', 'benchmark'],
                        default='classifier', help='Mode d\'opération')
    parser.add_argument('--prob_threshold', default=0.5, type=float, help='Seuil de probabilité pour considérer Inconnu')
    parser.add_argument('--visualize_gradcam', action='store_true', help='Activer Grad-CAM')
    parser.add_argument('--save_gradcam_images', action='store_true', help='Enregistrer les images Grad-CAM')
    parser.add_argument('--measure_time', action='store_true', help='Mesurer et enregistrer le temps moyen de traitement')
    parser.add_argument('--save_test_images', action='store_true', help='Sauvegarder les images de test annotées')
    parser.add_argument('--gradcam_task', type=str, default=None, help='Tâche sur laquelle calculer Grad-CAM')
    parser.add_argument('--colormap', type=str, default='hot', help='Colormap Grad-CAM (ex: hot, inferno, turbo...)')
    parser.add_argument('--colors', nargs='+', default=None, metavar='COLORS', help='Couleurs pour t-SNE ou clustering')
    parser.add_argument('--clustering_class', type=str, help='Nom de la classe pour clustering HDBSCAN')
    parser.add_argument('--min_cluster_size', type=int, nargs='+', default=[10, 15, 20],
                        metavar='MIN_CLUSTER_SIZE', help='Liste de min_cluster_size pour HDBSCAN')
    parser.add_argument('--min_samples', type=int, nargs='+', default=[5, 10],
                        metavar='MIN_SAMPLES', help='Liste de min_samples pour HDBSCAN')
    parser.add_argument('--kalman_filter', action='store_true', help='Appliquer un Kalman Filter sur les prédictions cam')
    parser.add_argument('--camera_index', type=int, default=0, help='Index de la caméra')
    parser.add_argument('--num_samples', type=int, default=None, help='Nombre d\'images à tester (subset aléatoire)')
    parser.add_argument('--save_camera_video', action='store_true', help='Enregistrer le flux caméra dans un fichier')
    parser.add_argument('--per_task_tsne', action='store_true', help='Effectuer le t-SNE par tâche (embeddings)')
    # Option pour afficher le nombre de paramètres du modèle
    parser.add_argument('--count_params', action='store_true', help='Afficher le nombre de paramètres du modèle')
    parser.add_argument('--integrated_gradients', action='store_true', help='Calculer Integrated Gradients pour chaque tâche')
    parser.add_argument('--integrated_gradients_task', type=str, default=None,
                        help='Tâche pour laquelle calculer Integrated Gradients. Si non spécifiée, le calcul sera effectué pour toutes les tâches.')
    parser.add_argument('--test_images_folder', type=str, help='Chemin vers le dossier contenant les images de test (utilisé en mode folder)')
    parser.add_argument('--test_following_task', type=str, default=None,
                        help='indique la tâche pour le test')
    parser.add_argument('--search_folder',  type=str, default=None, help='Search file')

    parser.add_argument('--watch_folders', type=str, default=None,
                        help='Liste (virgule séparée) de dossiers à surveiller pour les images de test')
    parser.add_argument('--poll_intervals', type=str, default=None,
                        help='Liste (virgule séparée) des intervalles (en secondes) de sondage pour chaque dossier')
    parser.add_argument('--save_dir_to_canon', default=None, type=str,
                        help='Répertoire de sortie pour le fichier WeatherInfos.json (pour le premier dossier)')

    parser.add_argument('--benchmark_folder', type=str,
                        help="Dossier racine du benchmark_patchGAN_Gram (sous-dossiers = classes)")
    parser.add_argument('--benchmark_mapping', type=str,
                        help="JSON de mapping benchmark_patchGAN_Gram→modèle")
    parser.add_argument('--roc_output', type=str, default='roc_curves',
                        help="Répertoire où enregistrer les courbes ROC")

    parser.add_argument('--auto_mapping',
                        action='store_true',
                        help='Cherche automatiquement la meilleure correspondance classes-modèle→benchmark_patchGAN_Gram')


    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # TensorBoard
    if args.tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'TensorBoard'))
    else:
        writer = None

    # Charger l'hyperparam JSON
    with open(args.config_path, 'r') as f:
        best_config = json.load(f)

    # Charger la description des tâches
    with open(args.build_classifier, 'r') as f:
        tasks_json = json.load(f)
    # tasks_dict : { "weather": 3, "time_of_day": 2, ... }
    tasks_dict = {task_name: len(class_list) for task_name, class_list in tasks_json.items()}
    print(f"Nombre de tâches: {len(tasks_dict)}")
    for t, n in tasks_dict.items():
        print(f"  Tâche '{t}': {n} classes")

    # Récupération de patch_size, etc.
    patch_size = best_config.get('patch_size', 70)

    # Construire le PatchGAN multi-tâches
    model = MultiTaskPatchGANTest(
        tasks_dict=tasks_dict,
        input_nc=3,
        ndf=64,
        norm="instance",
        patch_size=patch_size,
        device=device
    ).to(device)

    # Charger le state_dict
    load_model_weights(model, args.model_path, device)

    # Si l'option --count_params est spécifiée, afficher les informations sur les paramètres
    if args.count_params:
        print_model_parameters(model)

    # En fonction du mode, on charge le dataset (sauf en mode camera)
    if args.mode in ['classifier', 'clustering', 'tsne', 'tsne_interactive', 'folder', 'watch_folder']:
        if not args.data:
            raise ValueError("Vous devez spécifier --data pour le mode classifier/clustering/tsne.")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        dataset = MultiTaskDataset(data_json=args.data, classes_json=args.build_classifier, transform=transform,
                                   search_folder=args.search_folder,
                                   find_images_by_sub_folder=args.find_images_by_sub_folder)

        # Subset si args.num_samples
        if args.num_samples is not None:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            indices = indices[:args.num_samples]
            dataset = Subset(dataset, indices)

        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # On exécute en fonction du mode
    if args.mode == 'classifier':
        criterions = {task_name: nn.CrossEntropyLoss().to(device) for task_name in tasks_json.keys()}
        test_classifier(model, test_loader, criterions, writer, args.save_dir, device, tasks_json,
                        prob_threshold=args.prob_threshold,
                        visualize_gradcam=args.visualize_gradcam,
                        save_gradcam_images=args.save_gradcam_images,
                        measure_time=args.measure_time,
                        save_test_images=args.save_test_images,
                        gradcam_task=args.gradcam_task,
                        colormap=args.colormap, integrated_gradients=args.integrated_gradients)

    elif args.mode == 'folder':
        if not args.test_images_folder:
            raise ValueError("Vous devez spécifier --test_images_folder pour le mode folder.")
        test_folder_predictions(model, tasks_json, args.test_images_folder, transform, device, args.save_dir,
                                save_test_images=args.save_test_images, target_task=args.test_following_task)

    elif args.mode == 'tsne':
        embeddings_data, labels_data, img_paths = compute_embeddings_with_paths(
            model, test_loader, device, tasks_json, per_task_tsne=args.per_task_tsne
        )
        if args.per_task_tsne:
            for task_name in embeddings_data.keys():
                emb = embeddings_data[task_name]
                lbl = labels_data[task_name]
                perform_tsne(emb, lbl, {task_name: tasks_json[task_name]}, args.colors, args.save_dir, task_name=task_name)
        else:
            perform_tsne(embeddings_data, labels_data, tasks_json, args.colors, args.save_dir)
    elif args.mode == 'tsne_interactive':
        embeddings_data, labels_data, img_paths_data = compute_embeddings_with_paths(
            model, test_loader, device, tasks_json, per_task_tsne=args.per_task_tsne
        )
        plot_tsne_interactive(embeddings_data, labels_data, tasks_json, img_paths_data, colors=args.colors, save_dir=args.save_dir)
    elif args.mode == 'clustering':
        if not args.clustering_class:
            raise ValueError("Vous devez spécifier --clustering_class pour le mode clustering.")
        embeddings, labels, img_paths = compute_embeddings_with_paths(
            model, test_loader, device, tasks_json, per_task_tsne=False
        )
        class_index = None
        target_task_name = None
        for tname, clist in tasks_json.items():
            if args.clustering_class in clist:
                class_index = clist.index(args.clustering_class)
                target_task_name = tname
                break
        if class_index is None:
            raise ValueError(f"Classe '{args.clustering_class}' non trouvée dans les tâches.")
        selected_indices = (labels == class_index)
        class_embeddings = embeddings[selected_indices]
        class_img_paths = [img_paths[i] for i in range(len(labels)) if selected_indices[i]]
        best_num_clusters = 0
        best_cluster_labels = None
        best_params = {}
        for min_cluster_size in args.min_cluster_size:
            for min_samples in args.min_samples:
                clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                             min_samples=min_samples).fit(class_embeddings)
                cluster_labels = clustering.labels_
                num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                if num_clusters > best_num_clusters:
                    best_num_clusters = num_clusters
                    best_cluster_labels = cluster_labels
                    best_params = {'min_cluster_size': min_cluster_size, 'min_samples': min_samples}
        if best_cluster_labels is None:
            print("Aucun cluster trouvé.")
            return
        cluster_labels = best_cluster_labels
        cluster_info = {}
        unique_labels = set(cluster_labels)
        for lbl in unique_labels:
            indices_lbl = [i for i, c in enumerate(cluster_labels) if c == lbl]
            cluster_info[str(lbl)] = {
                'num_images': len(indices_lbl),
                'img_paths': [class_img_paths[i] for i in indices_lbl]
            }
        clustering_results = {
            'num_clusters': best_num_clusters,
            'clusters': cluster_info,
            'best_params': best_params
        }
        out_cluster_path = os.path.join(args.save_dir, f"{args.clustering_class}_clustering_results.json")
        with open(out_cluster_path, 'w') as f:
            json.dump(clustering_results, f, indent=4)
        print(f"Clustering results saved to {out_cluster_path}")


    elif args.mode == 'camera':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        run_camera(
            model, transform, tasks_json, args.save_dir,
            args.prob_threshold, args.measure_time,
            args.camera_index, args.kalman_filter,
            args.save_camera_video
        )

    elif args.mode == 'benchmark':
        if not args.benchmark_folder or not args.benchmark_mapping:
            raise ValueError("Pour le mode 'benchmark_patchGAN_Gram', précisez --benchmark_folder et --benchmark_mapping")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
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


    elif args.mode =="inference":

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        run_inference(
            model=model,
            image_folder=args.image_folder,
            transform=transform,
            device=device,
            num_samples=args.num_samples,
            save_dir=args.save_dir,
            save_test_images=args.save_test_images,
            classes=tasks_json,
            visualize_gradcam=args.visualize_gradcam,
            save_gradcam_images=args.save_gradcam_images,
            gradcam_task=args.gradcam_task,

        )

    elif args.mode == "watch_folder":

        if args.watch_folders is None:
            raise ValueError("--watch_folders doit être spécifié en mode watch_folder")

        # Convertir les listes séparées par des virgules en listes Python

        watch_folders = [s.strip() for s in args.watch_folders.split(',')]

        if args.poll_intervals is None:

            # Si aucun intervalle n'est spécifié, utiliser 5 secondes par défaut pour tous

            poll_intervals = [5] * len(watch_folders)


        else:

            poll_intervals = [int(s.strip()) for s in args.poll_intervals.split(',')]

        watch_folders_predictions(model, tasks_json, watch_folders, poll_intervals, transform, device, args.save_dir,
                                  args.save_dir_to_canon)

    if writer:
        writer.close()


if __name__=="__main__":
    main()
