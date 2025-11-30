import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import KFold
from datas import MultiTaskDataset
from torch.utils.tensorboard import SummaryWriter
from Models.models_PM import MultiTaskPatchGAN
from Functions.function_PatchGAN import train_model, evaluate_model

# -----------------------------------------------------------------------
# MAIN: ENTRAINEMENT MULTI-FOLD AVEC CHARGEMENT DES HYPERPARAMÈTRES VIA config_path
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Entraînement du Multi-Task PatchGAN avec hyperparamètres fixes")
    parser.add_argument('--data', type=str, required=True, help='Chemin vers le fichier JSON des données')
    parser.add_argument('--build_classifier', type=str, required=True, help='Chemin vers le fichier JSON des classes')
    parser.add_argument('--epochs', default=25, type=int, help='Nombre d’époques par fold')
    parser.add_argument('--save_dir', default='saved_models', type=str, help='Répertoire de sauvegarde des résultats')
    parser.add_argument('--tensorboard', action='store_true', help='Utiliser TensorBoard')
    parser.add_argument('--k_folds', default=2, type=int, help='Nombre de folds pour la validation croisée')

    # Hyperparamètres par défaut (seront ignorés si --config_path est fourni)
    parser.add_argument('--batch_size', default=32, type=int, help='Taille du batch')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--patch_size', default=70, type=int, help='Taille du patch pour le PatchGAN')

    parser.add_argument('--freeze_encoder', action='store_true', help='Geler les couches du trunk')
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help='Chemin vers les poids préentraînés (optionnel)')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Chemin vers le fichier JSON contenant les hyperparamètres')
    parser.add_argument('--search_folder',  type=str, default=None, help='Search file')


    args = parser.parse_args()

    # Si un fichier de config est fourni, on charge les hyperparamètres
    if args.config_path is not None:
        if os.path.exists(args.config_path):
            with open(args.config_path, 'r') as f:
                config = json.load(f)
            args.batch_size = config.get("batch_size", args.batch_size)
            args.lr = config.get("lr", args.lr)
            args.patch_size = config.get("patch_size", args.patch_size)
            print("Hyperparamètres chargés depuis", args.config_path)
            print(f"  batch_size: {args.batch_size}, lr: {args.lr}, patch_size: {args.patch_size}")
        else:
            raise FileNotFoundError(f"Le fichier de configuration {args.config_path} n'existe pas.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Transformation des images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Chargement du dataset et des tâches
    dataset = MultiTaskDataset(data_json=args.data, classes_json=args.build_classifier, transform=transform,
                               search_folder=args.search_folder,
                               find_images_by_sub_folder=args.find_images_by_sub_folder)
    tasks_dict = {task_name: len(class_list) for task_name, class_list in dataset.classes.items()}
    print("Tâches chargées :")
    for t, n in tasks_dict.items():
        print(f"  Tâche '{t}' -> {n} classes")

    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'tensorboard')) if args.tensorboard else None

    # Définition des fonctions de perte par tâche
    criterions_dict = {task: nn.CrossEntropyLoss().to(device) for task in tasks_dict.keys()}

    # Validation croisée avec KFold
    kf = KFold(n_splits=args.k_folds, shuffle=True)
    fold = 0
    for train_idx, val_idx in kf.split(dataset):
        print(f"=== Fold {fold} ===")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # Création du modèle
        model = MultiTaskPatchGAN(
            tasks_dict=tasks_dict,
            input_nc=3,
            ndf=64,
            norm="instance",
            patch_size=args.patch_size,
            tensorboard_logdir=None
        ).to(device)

        # Chargement des poids préentraînés si spécifié
        if args.pretrained_weights is not None:
            model.load_state_dict(torch.load(args.pretrained_weights))
            print(f"Poids préentraînés chargés depuis {args.pretrained_weights}")

        # Optionnel : geler les paramètres du trunk
        if args.freeze_encoder:
            for name, param in model.trunk.named_parameters():
                param.requires_grad = False

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9)

        # Entraînement
        model = train_model(model, train_loader, criterions_dict, optimizer, num_epochs=args.epochs, writer=writer,
                            fold=fold)
        # Évaluation
        val_loss, val_acc, metrics_dict = evaluate_model(model, val_loader, criterions_dict, writer=writer, fold=fold)

        # Sauvegarde des poids et de la config pour ce fold
        fold_model_path = os.path.join(args.save_dir, f"best_model_fold_{fold}.pth")
        torch.save(model.state_dict(), fold_model_path)
        fold_config = {
            'batch_size': args.batch_size,
            'lr': args.lr,
            'patch_size': args.patch_size,
            'freeze_encoder': args.freeze_encoder,
            'epochs': args.epochs,
            'model_path': fold_model_path
        }
        fold_config_path = os.path.join(args.save_dir, f"best_hyperparams_fold_{fold}.json")
        with open(fold_config_path, 'w') as f:
            json.dump(fold_config, f, indent=4)
        print(f"Fold {fold} - Modèle sauvegardé dans {fold_model_path}")
        print(f"Fold {fold} - Hyperparamètres sauvegardés dans {fold_config_path}")

        fold += 1

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
