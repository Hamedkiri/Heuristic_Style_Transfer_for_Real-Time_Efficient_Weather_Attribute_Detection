import argparse
import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import hdbscan
import cv2
from Functions.PM_evaluation import load_best_model, print_model_parameters, perform_tsne, plot_tsne_interactive, test, test_folder_predictions, watch_folders_predictions, compute_embeddings_with_paths, run_camera
from Models.models_RTM import MultiHeadAttentionPerTaskModel, TaskSpecificModel
from datas import MultiTaskDataset
# Définition du dictionnaire des colormaps
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
    # Ajoutez d'autres colormaps si nécessaire
}










def main():
    parser = argparse.ArgumentParser(description='Test du modèle multi-tâches avec différents modes')
    parser.add_argument('--data', type=str, help='Chemin vers le fichier JSON du dataset')
    parser.add_argument('--config_path', type=str, required=True, help='Chemin vers le fichier JSON des hyperparamètres')
    parser.add_argument('--model_path', type=str, required=True, help='Chemin vers le modèle entraîné')
    parser.add_argument('--batch_size', default=32, type=int, help='Taille de lot pour le test')
    parser.add_argument('--save_dir', default='results', type=str, help='Répertoire pour enregistrer les résultats')
    parser.add_argument('--tensorboard', action='store_true', help='Activer TensorBoard')
    parser.add_argument('--build_classifier', type=str, required=True, help='Chemin vers le fichier JSON des classes')
    parser.add_argument('--mode', choices=['classifier', 'tsne', 'tsne_interactive', 'camera', 'clustering', 'folder',
                                           'watch_folder'],
                        default='classifier', help='Mode d\'opération')
    parser.add_argument('--prob_threshold', default=0.5, type=float, help='Seuil de probabilité pour considérer une classe comme inconnue')
    parser.add_argument('--visualize_gradcam', action='store_true', help='Visualiser Grad-CAM')
    parser.add_argument('--save_gradcam_images', action='store_true', help='Enregistrer les images Grad-CAM')
    parser.add_argument('--measure_time', action='store_true', help='Mesurer et enregistrer le temps moyen de traitement par image')
    parser.add_argument('--save_test_images', action='store_true', help='Sauvegarder les images de test avec prédictions et probabilités')
    parser.add_argument('--colors', nargs='+', default=None, metavar='COLORS', help='Liste des couleurs pour t-SNE ou clustering')
    parser.add_argument('--clustering_class', type=str, help='Nom de la classe pour le clustering HDBSCAN')
    parser.add_argument('--min_cluster_size', type=int, nargs='+', default=[10, 15, 20], metavar='MIN_CLUSTER_SIZE', help='Liste des valeurs min_cluster_size pour HDBSCAN')
    parser.add_argument('--min_samples', type=int, nargs='+', default=[5, 10], metavar='MIN_SAMPLES', help='Liste des valeurs min_samples pour HDBSCAN')
    parser.add_argument('--kalman_filter', action='store_true', help='Appliquer un filtre de Kalman pour lisser les prédictions de la caméra')
    parser.add_argument('--camera_index', type=int, default=0, help='Index de la caméra à utiliser')
    parser.add_argument('--num_samples', type=int, default=None, help='Nombre d\'images à tester')
    parser.add_argument('--save_camera_video', action='store_true', help='Sauvegarder le flux vidéo de la caméra')  # Nouvelle option
    parser.add_argument('--gradcam_task', type=str, default=None, help='Nom de la tâche sur laquelle calculer Grad-CAM')  # Nouvelle option
    parser.add_argument('--colormap', type=str, default='hot',
                        help='Colormap pour les visualisations Grad-CAM (par exemple, hot, autumn, afmhot)')
    parser.add_argument('--per_task_tsne', action='store_true', help='Effectuer le t-SNE par tâche sur les embeddings du mécanisme d\'attention lié à la tâche')
    parser.add_argument('--count_params', action='store_true', help='Afficher le nombre de paramètres du modèle')
    parser.add_argument('--integrated_gradients', action='store_true', help='Calculer IntegratedGradients pour chaque tâche')
    parser.add_argument('--test_images_folder', type=str,
                        help="Chemin vers un dossier contenant des images pour les prédictions")
    parser.add_argument('--target_task', type=str, default=None,
                        help="Nom de la tâche à tester (si non spécifié, toutes les tâches seront évaluées)")
    parser.add_argument('--search_folder',  type=str, default=None, help='Search file')
    parser.add_argument('--watch_folders', type=str, default=None,
                        help='Liste (virgule séparée) de dossiers à surveiller pour les images de test')
    parser.add_argument('--poll_intervals', type=str, default=None,
                        help='Liste (virgule séparée) des intervalles (en secondes) de sondage pour chaque dossier')
    parser.add_argument('--save_dir_to_canon', default=None, type=str,
                        help='Répertoire de sortie pour le fichier WeatherInfos.json (pour le premier dossier)')


    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'TensorBoard'))
    else:
        writer = None

    # Chargement des hyperparamètres
    with open(args.config_path, 'r') as f:
        best_config = json.load(f)

    # Récupération des hyperparamètres nécessaires
    truncate_layer = best_config.get('truncate_layer', 10)  # Valeur par défaut si non présente




    # Chargement des tâches et des classes à partir de --build_classifier
    with open(args.build_classifier, 'r') as f:
        tasks = json.load(f)

    # Affichage des tâches
    print(f"Nombre de classifieurs (tâches) : {len(tasks)}")
    for task_name, class_list in tasks.items():
        print(f"Classifieur pour la tâche '{task_name}' traite {len(class_list)} classes : {class_list}")

    # Définition des transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # Normalisation avec les mêmes moyennes et écarts-types que lors de l'entraînement
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Création du modèle
    base_encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = MultiHeadAttentionPerTaskModel(base_encoder, truncate_layer, tasks, device=device).to(device)

    # Chargement du modèle entraîné
    load_best_model(model, args.model_path)

    # Mode 'classifier' ou 'clustering' nécessite les données
    if args.mode in ['classifier', 'clustering', 'tsne', 'tsne_interactive', 'prediction', 'watch_folder']:
        if args.data is None:
            raise ValueError("Le chemin vers les données '--data' doit être spécifié pour ce mode.")
        # Chargement du dataset
        dataset = MultiTaskDataset(data_json=args.data, classes_json=args.build_classifier, transform=transform,
                                   search_folder=args.search_folder,
                                   find_images_by_sub_folder=args.find_images_by_sub_folder)

        # Application de num_samples si spécifié
        if args.num_samples is not None:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            indices = indices[:args.num_samples]
            dataset = Subset(dataset, indices)

        # Création du DataLoader
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Si l'option --count_params est activée, afficher les informations sur les paramètres
    if args.count_params:
            print_model_parameters(model)

    if args.mode == 'classifier':
        # Définition des critères
        criterions = {task: nn.CrossEntropyLoss().to(device) for task in tasks.keys()}

        # Exécution du test
        test(model, test_loader, criterions, writer, args.save_dir, device, tasks, args.prob_threshold,
             args.visualize_gradcam, args.save_gradcam_images, args.measure_time, args.save_test_images,
             args.gradcam_task, args.colormap, integrated_gradients=args.integrated_gradients)

    elif args.mode == 'tsne':
        embeddings_data, labels_data, img_paths = compute_embeddings_with_paths(model, test_loader, device, per_task_tsne=args.per_task_tsne)

        if args.per_task_tsne:
            for task_name in tasks.keys():
                embeddings = embeddings_data[task_name]
                labels = np.array(labels_data[task_name])

                # Sauvegarder les embeddings et labels par tâche
                results = {'embeddings': embeddings.tolist(), 'labels': labels.tolist()}

                output_path = os.path.join(args.save_dir, f'embeddings_{task_name.replace(" ", "_")}.json')

                with open(output_path, 'w') as f:
                    json.dump(results, f)

                perform_tsne(embeddings, labels, {task_name: tasks[task_name]}, args.colors, args.save_dir, task_name=task_name)
        else:
            all_embeddings = embeddings_data
            all_labels = labels_data
            # Sauvegarder les embeddings et labels

            results = {'embeddings': all_embeddings.tolist(), 'labels': all_labels.tolist()}

            output_path = os.path.join(args.save_dir, 'embeddings.json')

            with open(output_path, 'w') as f:
                json.dump(results, f)

            perform_tsne(all_embeddings, all_labels, tasks, args.colors, args.save_dir)


    elif args.mode == 'tsne_interactive':

        embeddings_data, labels_data, img_paths_data = compute_embeddings_with_paths(model, test_loader, device,
                                                                                     per_task_tsne=args.per_task_tsne)

        if args.per_task_tsne:

            plot_tsne_interactive(embeddings_data, labels_data, tasks, img_paths_data, args.colors,
                                  save_dir=args.save_dir)

        else:

            # Encapsuler les données dans des dictionnaires avec une clé générique

            embeddings_dict = {'All Tasks': embeddings_data}

            labels_dict = {'All Tasks': labels_data}

            img_paths_dict = {'All Tasks': img_paths_data}

            tasks_dict = {'All Tasks': tasks[list(tasks.keys())[0]]}  # Utiliser les classes de la première tâche

            plot_tsne_interactive(embeddings_dict, labels_dict, tasks_dict, img_paths_dict, args.colors,
                                  save_dir=args.save_dir)


    elif args.mode == 'clustering':
        if not args.clustering_class:
            raise ValueError("L'option --clustering_class doit être spécifiée pour le mode clustering")
        # Obtenir les embeddings de la classe spécifiée
        all_embeddings, all_labels, img_paths = compute_embeddings_with_paths(model, test_loader, device)
        class_index = None
        for idx, (task_name, class_list) in enumerate(tasks.items()):
            if args.clustering_class in class_list:
                class_index = class_list.index(args.clustering_class)
                break
        if class_index is None:
            raise ValueError(f"Classe '{args.clustering_class}' non trouvée dans les tâches.")
        # Filtrer les embeddings de la classe
        class_embeddings = all_embeddings[all_labels == class_index]
        class_img_paths = [img_paths[i] for i in range(len(all_labels)) if all_labels[i] == class_index]

        # Appliquer HDBSCAN
        best_num_clusters = 0
        best_cluster_labels = None
        best_params = {}

        for min_cluster_size in args.min_cluster_size:
            for min_samples in args.min_samples:
                print(f"Testing HDBSCAN with min_cluster_size={min_cluster_size}, min_samples={min_samples}")
                clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit(class_embeddings)
                cluster_labels = clustering.labels_

                num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                print(f"Number of clusters found: {num_clusters}")

                if num_clusters > best_num_clusters:
                    best_num_clusters = num_clusters
                    best_cluster_labels = cluster_labels
                    best_params = {'min_cluster_size': min_cluster_size, 'min_samples': min_samples}

        if best_cluster_labels is None:
            raise ValueError("No clusters found with the provided HDBSCAN parameters.")

        cluster_labels = best_cluster_labels

        # Préparer les résultats du clustering
        cluster_info = {}
        unique_labels = set(cluster_labels)
        for label in unique_labels:
            indices = [i for i, lbl in enumerate(cluster_labels) if lbl == label]
            cluster_info[str(label)] = {
                'num_images': len(indices),
                'img_paths': [class_img_paths[i] for i in indices]
            }

        clustering_results = {
            'num_clusters': best_num_clusters,
            'clusters': cluster_info,
            'best_params': best_params
        }

        clustering_output_path = os.path.join(args.save_dir, f'{args.clustering_class}_clustering_results.json')
        with open(clustering_output_path, 'w') as f:
            json.dump(clustering_results, f)

        print(f"Clustering results saved in '{clustering_output_path}' with parameters {best_params}")

        plot_tsne_interactive(class_embeddings, cluster_labels, [f'Cluster {i}' for i in range(best_num_clusters)] + ['Noise'], class_img_paths, colors=args.colors, num_clusters=best_num_clusters, save_dir=args.save_dir)

    elif args.mode == 'camera':

        run_camera(model, transform, tasks, args.save_dir, args.prob_threshold, args.measure_time, args.camera_index,
                   args.kalman_filter, args.save_camera_video)  # Passez l'argument ici

    elif args.test_images_folder and args.mode == 'prediction' :
        # Ici, on suppose que vous avez déjà créé le modèle, chargé les transformations et les tâches.

        print(f"Lancement des prédictions sur le dossier {args.test_images_folder}...")

        test_folder_predictions(model, tasks, args.test_images_folder, transform, device, args.save_dir,

                                save_test_images=args.save_test_images, target_task=args.target_task)

        return  # Sortir du main après ce mode de test

    elif args.mode == "watch_folder":
        if args.watch_folders is None:
            raise ValueError("--watch_folders doit être spécifié en mode watch_folder")
        # Convertir la liste de dossiers et la liste d'intervalles (séparées par des virgules)
        watch_folders = [s.strip() for s in args.watch_folders.split(',')]
        if args.poll_intervals is None:
            # Intervalle par défaut de 5 secondes pour chaque dossier
            poll_intervals = [5] * len(watch_folders)
        else:
            poll_intervals = [int(s.strip()) for s in args.poll_intervals.split(',')]
        watch_folders_predictions(model, tasks, watch_folders, poll_intervals, transform, device, args.save_dir,
                                  args.save_dir_to_canon)

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()