import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import threading
import re
import datetime
import pandas as pd
from PIL import Image, ImageTk





def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.truncated_encoder.parameters() if p.requires_grad)
    pool_params = sum(p.numel() for p in model.pool.parameters() if p.requires_grad)

    # Vérification du nombre exact de couches tronquées
    truncated_layers = list(model.truncated_encoder.children())
    num_truncated_layers = len(truncated_layers)

    print("==== Paramètres du modèle ====")
    print(f"Paramètres totaux du modèle : {total_params}")
    print(f"Nombre de couches tronquées : {num_truncated_layers}")
    print(f"Paramètres de l'encodeur tronqué : {encoder_params}")
    print(f"Paramètres de la couche de pooling : {pool_params}")

    print("Modules d'attention par tâche :")
    for key, attn in model.attentions.items():
        count = sum(p.numel() for p in attn.parameters() if p.requires_grad)
        theoretical = 3 * (model.num_features ** 2)  # Théorie : 3 matrices de projection
        print(f"  {key}: {count} paramètres (théoriques: {theoretical})")

    print("Modules classifieurs par tâche :")
    for key, classifier in model.classifiers.items():
        count = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        num_classes = classifier.out_features
        theoretical = model.num_features * num_classes + num_classes  # Théorie : W * classes + biais
        print(f"  {key}: {count} paramètres (théoriques: {theoretical})")

    print("=================================")









def load_best_model(model, filepath):
    checkpoint = torch.load(filepath, map_location=model.device)
    model.load_state_dict(checkpoint)
    model.to(model.device)
    print(f"Model loaded from {filepath}")







def compute_embeddings_with_paths(model, loader, device, per_task_tsne=False):
    model.eval()
    if per_task_tsne:
        task_embeddings = {task_name: [] for task_name in model.tasks.keys()}
        task_labels = {task_name: [] for task_name in model.tasks.keys()}
        task_img_paths = {task_name: [] for task_name in model.tasks.keys()}
    else:
        all_embeddings = []
        all_labels = []
        img_paths = []
    dataset_length = len(loader.dataset)
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            if per_task_tsne:
                outputs, embeddings = model(inputs, return_task_embeddings=True)
                batch_size = inputs.size(0)
                # Obtenir les chemins d'images pour le batch
                if isinstance(loader.dataset, Subset):
                    indices = loader.dataset.indices[batch_idx * loader.batch_size : batch_idx * loader.batch_size + batch_size]
                    batch_img_paths = [loader.dataset.dataset.samples[idx][0] for idx in indices]
                else:
                    batch_img_paths = [loader.dataset.samples[idx][0] for idx in range(batch_idx * loader.batch_size, batch_idx * loader.batch_size + batch_size)]
                for i in range(batch_size):
                    for task_name in model.tasks.keys():
                        label = labels[task_name][i]
                        if label >= 0:
                            emb = embeddings[task_name][i].cpu().numpy()
                            task_embeddings[task_name].append(emb)
                            task_labels[task_name].append(label.item())
                            task_img_paths[task_name].append(batch_img_paths[i])
            else:
                embeddings = model(inputs, return_embeddings=True)
                embeddings = embeddings.cpu().numpy()
                all_embeddings.append(embeddings)
                # Nous prenons les labels de la première tâche
                task_name = list(labels.keys())[0]
                task_labels_batch = labels[task_name]
                all_labels.extend(task_labels_batch.numpy())
                # Obtenir les chemins d'images pour le batch
                batch_size = inputs.size(0)
                if isinstance(loader.dataset, Subset):
                    indices = loader.dataset.indices[batch_idx * loader.batch_size : batch_idx * loader.batch_size + batch_size]
                    batch_img_paths = [loader.dataset.dataset.samples[idx][0] for idx in indices]
                else:
                    batch_img_paths = [loader.dataset.samples[idx][0] for idx in range(batch_idx * loader.batch_size, batch_idx * loader.batch_size + batch_size)]
                img_paths.extend(batch_img_paths)
    if per_task_tsne:
        for task_name in task_embeddings.keys():
            task_embeddings[task_name] = np.stack(task_embeddings[task_name], axis=0)
            task_labels[task_name] = np.array(task_labels[task_name])
        return task_embeddings, task_labels, task_img_paths
    else:
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels = np.array(all_labels)
        return all_embeddings, all_labels, img_paths


def perform_tsne(embeddings, labels, tasks, colors=None, results_dir='results', task_name=None):
    print("Réalisation de t-SNE...")
    embeddings_flat = embeddings.reshape(embeddings.shape[0], -1)
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings_flat)

    plt.figure(figsize=(10, 10))
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    if colors and len(colors) >= num_classes:
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    else:
        color_map = {label: plt.cm.tab20(i / num_classes) for i, label in enumerate(unique_labels)}

    if task_name:
        class_names = tasks[task_name]
    else:
        class_names = tasks[list(tasks.keys())[0]]

    for label in unique_labels:
        indices = labels == label
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1],
                    label=class_names[label], color=color_map[label])
    plt.legend()
    if task_name:
        tsne_plot_path = os.path.join(results_dir, f'tsne_plot_{task_name.replace(" ", "_")}.png')
    else:
        tsne_plot_path = os.path.join(results_dir, 'tsne_plot.png')
    plt.savefig(tsne_plot_path)
    plt.show()
    print(f"t-SNE plot saved to '{tsne_plot_path}'")

def plot_tsne_interactive(embeddings_data, labels_data, tasks, img_paths_data, colors=None, num_clusters=None, save_dir='results'):
    import matplotlib
    matplotlib.use('TkAgg')  # Utiliser TkAgg pour l'interface Tkinter

    from PIL import Image, ImageTk
    import tkinter as tk
    from tkinter import ttk, colorchooser
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.path import Path
    from matplotlib.widgets import PolygonSelector
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import os

    # Création de la fenêtre Tkinter
    root = tk.Tk()
    root.title("Interactive t-SNE with Images")

    # Création de la figure matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))

    # Vérifier s'il y a plusieurs tâches ou une seule
    if len(embeddings_data) == 1:
        # Une seule tâche, pas besoin de sélection de tâche
        single_task_mode = True
        current_task_name = list(embeddings_data.keys())[0]
    else:
        single_task_mode = False
        current_task_name = None

    tsne_results = None
    labels = None
    class_names = None
    unique_labels = None
    scatter = None
    color_map = None
    img_paths = None
    filename_to_path = None

    # Création des frames
    left_frame = tk.Frame(root)
    left_frame.grid(row=0, column=0, sticky='nsew')

    right_frame = tk.Frame(root)
    right_frame.grid(row=0, column=1, sticky='nsew')

    # Ajout de la figure matplotlib au frame de gauche
    canvas = FigureCanvasTkAgg(fig, master=left_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

    # Variables pour le polygone
    polygon = []
    polygon_selector = None
    polygon_cleared = True  # Variable pour savoir si le polygone a été effacé

    # Affichage des images
    img_label = tk.Label(right_frame)
    img_label.pack(pady=10)

    label_text = tk.StringVar()
    label_label = tk.Label(right_frame, textvariable=label_text, justify='left')
    label_label.pack()

    inside_points_label = tk.StringVar()
    inside_points_count_label = tk.Label(right_frame, textvariable=inside_points_label)
    inside_points_count_label.pack()

    # Dropdown pour afficher les images à l'intérieur du polygone
    dropdown_points = []
    dropdown = ttk.Combobox(right_frame, state="readonly")
    dropdown.pack(fill='x', pady=5)
    dropdown.bind("<<ComboboxSelected>>", lambda event: on_dropdown_select())

    # Fonction pour changer la couleur d'une classe
    def change_class_color():
        selected = class_selector.get()
        if selected:
            label_str = selected.split(':')[0]
            label = int(label_str)
            color_code = colorchooser.askcolor(title="Choisir une couleur")[1]
            if color_code:
                color_map[label] = color_code
                # Mettre à jour le graphique avec les nouvelles couleurs
                scatter.set_color([color_map[int(lbl)] for lbl in labels])
                # Mettre à jour la légende
                ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=class_names[int(lbl)],
                                              markerfacecolor=color_map[int(lbl)], markersize=10) for lbl in unique_labels])
                canvas.draw()

    # Sélecteur de classe
    class_selector_label = tk.Label(right_frame, text="Sélectionnez une classe :")
    class_selector_label.pack(pady=5)

    class_selector = ttk.Combobox(right_frame, state="readonly")
    class_selector.pack(pady=5)

    change_color_button = tk.Button(right_frame, text="Changer la couleur de la classe", command=change_class_color)
    change_color_button.pack(pady=5)

    # Boutons pour la sélection de polygone
    button_frame = tk.Frame(right_frame)
    button_frame.pack(pady=10)

    close_button = tk.Button(button_frame, text="Fermer le polygone", command=lambda: analyze_polygon())
    close_button.pack(side='left', padx=5)

    clear_button = tk.Button(button_frame, text="Effacer le polygone", command=lambda: clear_polygon())
    clear_button.pack(side='left', padx=5)

    # Fonction pour effacer le polygone
    def clear_polygon():
        nonlocal polygon_selector
        nonlocal polygon_cleared
        polygon.clear()
        if polygon_selector:
            polygon_selector.disconnect_events()  # Déconnecter les événements du sélecteur de polygone
            polygon_selector.set_visible(False)  # Masquer la ligne du sélecteur
            del polygon_selector  # Supprimer le sélecteur de polygone
            polygon_selector = None
        while ax.patches:
            ax.patches.pop().remove()  # Supprimer tous les patches de l'axe
        fig.canvas.draw()
        inside_points_label.set("")  # Effacer le label des points à l'intérieur
        label_text.set("")  # Effacer le texte du label
        img_label.config(image='')  # Effacer l'image affichée
        dropdown.set('')  # Réinitialiser le dropdown
        dropdown['values'] = []
        polygon_cleared = True  # Indiquer que le polygone a été effacé

    # Fonction pour mettre à jour le graphique lors du changement de tâche
    def update_plot(task_name):
        nonlocal tsne_results, labels, class_names, unique_labels, scatter, color_map, img_paths, filename_to_path, current_task_name

        current_task_name = task_name

        # Effacer l'axe
        ax.clear()

        # Récupérer les embeddings, labels et chemins d'images pour la tâche sélectionnée
        embeddings = embeddings_data[task_name]
        labels = labels_data[task_name]
        img_paths = img_paths_data[task_name]

        # Préparer un mapping des noms de fichiers vers les chemins
        filename_to_path = {os.path.basename(path): path for path in img_paths}

        # Calculer le t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_flat = embeddings.reshape(embeddings.shape[0], -1)
        tsne_results = tsne.fit_transform(embeddings_flat)

        # Obtenir les noms de classes
        class_names = tasks[task_name]

        # Obtenir les labels uniques
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)

        # Préparer le color_map
        if colors and len(colors) >= num_classes:
            color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        else:
            color_palette = plt.cm.get_cmap("tab20", num_classes)
            color_map = {label: color_palette(i / num_classes) for i, label in enumerate(unique_labels)}

        # Tracer le scatter plot
        scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=[color_map[int(label)] for label in labels], picker=True)

        # Créer la légende
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=class_names[int(label)],
                                      markerfacecolor=color_map[int(label)], markersize=10) for label in unique_labels]
        ax.legend(handles=legend_elements)

        ax.set_title(f"t-SNE pour la tâche : {task_name}")

        # Redessiner le canvas
        canvas.draw()

        # Mettre à jour le sélecteur de classes
        class_selector['values'] = [f"{label}: {class_names[label]}" for label in unique_labels]
        if unique_labels.size > 0:
            class_selector.current(0)

        # Réinitialiser les éléments liés au polygone
        clear_polygon()

    # Gestionnaire d'événement pour la sélection de tâche
    def on_task_select(event):
        selected_task = task_selector.get()
        update_plot(selected_task)

    # Gestionnaires d'événements et fonctions ajustées pour utiliser les données de la tâche actuelle

    # Gestionnaire pour le clic sur les points
    def onpick(event):
        ind = event.ind[0]
        img_path = img_paths[ind]
        display_image(img_path, class_names[int(labels[ind])])

    fig.canvas.mpl_connect('pick_event', onpick)

    # Gestion du polygone
    def enable_polygon_selector(event):
        nonlocal polygon_selector
        nonlocal polygon_cleared
        if event.button == 3:  # Clic droit
            if polygon_selector is None or polygon_cleared:
                polygon_selector = PolygonSelector(ax, onselect=onselect, useblit=True)
                polygon_cleared = False
                print("Sélecteur de polygone activé.")

    def onselect(verts):
        polygon.clear()
        polygon.extend(verts)
        print("Sommets du polygone:", verts)

    def analyze_polygon():
        if len(polygon) < 3:
            print("Polygone non fermé. Sélectionnez au moins 3 points.")
            return

        inside_points = []
        outside_points = []
        polygon_path = Path(polygon)  # Utiliser Path de matplotlib.path

        for i, (x, y) in enumerate(tsne_results):
            point = (x, y)
            if polygon_path.contains_point(point):
                inside_points.append({"path": img_paths[i], "class": class_names[int(labels[i])], "position": point})
            else:
                outside_points.append({"path": img_paths[i], "class": class_names[int(labels[i])], "position": point})

        # Enregistrer seulement les noms de fichiers au lieu des chemins complets
        for point in inside_points:
            point['filename'] = os.path.basename(point['path'])
            del point['path']  # Supprimer le chemin complet

        for point in outside_points:
            point['filename'] = os.path.basename(point['path'])
            del point['path']  # Supprimer le chemin complet

        filename_suffix = current_task_name.replace(' ', '_') if current_task_name else 'task'
        save_json(inside_points, os.path.join(save_dir, f"inside_polygon_{filename_suffix}.json"))
        save_json(outside_points, os.path.join(save_dir, f"outside_polygon_{filename_suffix}.json"))

        inside_points_label.set(f"Points à l'intérieur du polygone: {len(inside_points)}")
        # Afficher les points à l'intérieur du polygone
        update_dropdown(inside_points)

    def update_dropdown(inside_points):
        dropdown_values = [f"{point['filename']} ({point['class']})" for point in inside_points]
        dropdown['values'] = dropdown_values
        dropdown_points.clear()
        dropdown_points.extend(inside_points)
        if dropdown_values:
            dropdown.current(0)  # Sélectionner le premier élément
            on_dropdown_select()

    def save_json(data, filename):
        with open(filename, "w") as f:
            json.dump(data, f, default=convert_to_serializable)

    def convert_to_serializable(obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return obj

    def on_dropdown_select():
        selection = dropdown.current()
        if selection >= 0:
            point = dropdown_points[selection]
            # Récupérer le chemin complet de l'image en utilisant un mapping
            img_path = filename_to_path[point['filename']]
            display_image(img_path, point['class'])

    def display_image(img_path, label):
        img = Image.open(img_path)
        img = img.resize((400, 400), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        img_label.configure(image=img_tk)
        img_label.image = img_tk
        label_text.set(f"Label: {label}\nFichier: {os.path.basename(img_path)}")

    fig.canvas.mpl_connect('button_press_event', enable_polygon_selector)

    # Fonctionnalités de zoom et dézoom
    def on_key_press(event):
        if event.key == '+':
            zoom(1.2)
        elif event.key == '-':
            zoom(0.8)

    def on_scroll(event):
        # Delta dépend du backend utilisé, on le normalise
        if event.button == 'up':
            zoom(1.1)
        elif event.button == 'down':
            zoom(0.9)

    def zoom(factor):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xdata = np.mean(xlim)
        ydata = np.mean(ylim)
        width = (xlim[1] - xlim[0]) * factor
        height = (ylim[1] - ylim[0]) * factor
        ax.set_xlim([xdata - width / 2, xdata + width / 2])
        ax.set_ylim([ydata - height / 2, ydata + height / 2])
        canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # Configuration des poids des colonnes et des lignes
    root.grid_columnconfigure(0, weight=3)
    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(0, weight=1)

    left_frame.grid_rowconfigure(0, weight=1)
    left_frame.grid_columnconfigure(0, weight=1)

    # Si plusieurs tâches, ajouter le sélecteur de tâche
    if not single_task_mode:
        task_selector_label = tk.Label(right_frame, text="Sélectionnez une tâche :")
        task_selector_label.pack(pady=5)

        task_selector = ttk.Combobox(right_frame, state="readonly", values=list(tasks.keys()))
        task_selector.pack(pady=5)
        task_selector.bind("<<ComboboxSelected>>", on_task_select)

    # Initialisation avec la première tâche
    if single_task_mode:
        update_plot(current_task_name)
    else:
        initial_task = list(tasks.keys())[0]
        task_selector.set(initial_task)
        update_plot(initial_task)

    # Démarrer la boucle principale Tkinter
    root.mainloop()







def test(model, test_loader, criterions, writer, save_dir, device, tasks, prob_threshold,
         visualize_gradcam, save_gradcam_images, measure_time, save_test_images,
         gradcam_task=None, colormap='hot', integrated_gradients=False):
    model.eval()
    total_loss = 0.0
    all_preds = {task: [] for task in tasks.keys()}
    all_labels = {task: [] for task in tasks.keys()}
    total_samples = 0
    times = []

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Vérifier si la tâche "weather type" existe
    weather_task_name = None
    for task_name in tasks.keys():
        if task_name.lower() == "weather type":
            weather_task_name = task_name
            break
    if weather_task_name is None:
        print("La tâche 'weather type' n'est pas présente dans les tâches.")
        weather_task_available = False
    else:
        weather_task_available = True

    # Préparer le modèle Grad-CAM si visualize_gradcam est True
    if visualize_gradcam:
        if gradcam_task is None:
            gradcam_task = list(tasks.keys())[0]  # Choisir la première tâche par défaut
        if gradcam_task not in tasks:
            raise ValueError(f"La tâche '{gradcam_task}' n'existe pas dans le modèle.")

        # Créer le modèle spécifique à la tâche pour Grad-CAM
        gradcam_model = TaskSpecificModel(model, gradcam_task)
        gradcam_model.to(device)
        gradcam_model.eval()

        # Sélectionner la couche cible appropriée pour Grad-CAM
        # Adapter cet accès en fonction de la structure de votre modèle
        try:
            # Essayons d'accéder à la dernière couche convolutionnelle
            target_layer = None
            for layer in reversed(list(gradcam_model.model.truncated_encoder)):
                if isinstance(layer, nn.Conv2d):
                    target_layer = layer
                    break
            if target_layer is None:
                raise ValueError("Aucune couche Conv2d trouvée dans truncated_encoder pour Grad-CAM.")
        except AttributeError:
            raise AttributeError("Le modèle n'a pas l'attribut 'truncated_encoder'.")

        grad_cam = GradCAM(model=gradcam_model, target_layers=[target_layer])

    if integrated_gradients:
        from captum.attr import IntegratedGradients
        ig_models = {}
        ig = {}
        for t in tasks.keys():
            ig_models[t] = TaskSpecificModel(model, t).to(device)
            ig_models[t].eval()
            ig[t] = IntegratedGradients(ig_models[t])

    for batch_idx, (inputs, labels) in enumerate(test_loader):
        start_time = time.time()
        inputs = inputs.to(device)
        inputs.requires_grad = True  # Activer requires_grad pour les entrées

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

                probabilities = torch.nn.functional.softmax(task_outputs, dim=1)
                max_probs, preds = torch.max(probabilities, 1)
                # Application du seuil de probabilité
                unknown_mask = max_probs < prob_threshold
                preds[unknown_mask] = -1  # On peut utiliser -1 pour représenter "Inconnu"

                all_preds[task_name].extend(preds.cpu().numpy())
                all_labels[task_name].extend(task_labels.cpu().numpy())

                # Stocker les prédictions et probabilités pour chaque tâche
                max_probs_dict[task_name] = max_probs.cpu().numpy()
                preds_dict[task_name] = preds.cpu().numpy()
                task_labels_dict[task_name] = task_labels.cpu().numpy()
            else:
                # Si les labels sont None, on remplit avec des valeurs par défaut
                max_probs_dict[task_name] = np.array([-1]*batch_size)
                preds_dict[task_name] = np.array([-1]*batch_size)
                task_labels_dict[task_name] = np.array([-1]*batch_size)

        end_time = time.time()
        times.append(end_time - start_time)

        # Calcul d'IntegratedGradients si activé
        if integrated_gradients:
            for i in range(batch_size):
                for t in tasks.keys():
                    input_tensor = inputs[i].unsqueeze(0)
                    baseline = torch.zeros_like(input_tensor)
                    target = int(task_labels_dict[t][i]) if task_labels_dict[t][i] >= 0 else 0
                    attr = ig[t].attribute(input_tensor, baseline, target=target)
                    attr_np = attr.squeeze().cpu().detach().numpy()
                    attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)
                    ig_save_path = os.path.join(save_dir,
                                                f"IntegratedGrad_{t}_{batch_idx * test_loader.batch_size + i}.jpg")
                    heatmap = cv2.applyColorMap(np.uint8(255 * attr_np), cv2.COLORMAP_JET)
                    cv2.imwrite(ig_save_path, heatmap)



        # Traitement des images pour la sauvegarde et la visualisation Grad-CAM
        for i in range(batch_size):
            idx = batch_idx * test_loader.batch_size + i
            if isinstance(test_loader.dataset, Subset):
                img_path = test_loader.dataset.dataset.samples[test_loader.dataset.indices[idx]][0]
            else:
                img_path = test_loader.dataset.samples[idx][0]
            img = Image.open(img_path)
            img_np = np.array(img.convert('RGB'))
            img_cv = img_np.copy()  # Utiliser une copie pour OpenCV (format RGB)

            # Déterminer le répertoire de sauvegarde des images
            if weather_task_available:
                # Obtenir le label de vérité terrain pour "weather type"
                weather_label_idx = task_labels_dict[weather_task_name][i]
                if weather_label_idx == -1:
                    weather_true_label = "Unknown"
                else:
                    weather_true_label = tasks[weather_task_name][weather_label_idx]
                # Créer le sous-dossier pour le label de vérité terrain
                label_dir = os.path.join(save_dir, weather_true_label)
            else:
                label_dir = save_dir  # Si la tâche "weather type" n'est pas disponible

            if not os.path.exists(label_dir):
                os.makedirs(label_dir)

            # Sauvegarder les images annotées pour toutes les tâches si --save_test_images est activé
            if save_test_images:
                annotated_img = img_cv.copy()
                y_start = 30
                y_step = 30  # Espace vertical entre les lignes de texte

                for j, (task_name, class_list) in enumerate(tasks.items()):
                    label_idx = task_labels_dict[task_name][i]
                    pred_idx = preds_dict[task_name][i]
                    prob = max_probs_dict[task_name][i]

                    if label_idx == -1:
                        true_label = "Unknown"
                    else:
                        true_label = class_list[label_idx]

                    if pred_idx == -1:
                        pred_label = "Unknown"
                    else:
                        pred_label = class_list[pred_idx]

                    text = f"{task_name} - True: {true_label}, Pred: {pred_label}, Prob: {prob:.2f}"
                    y_pos = y_start + j * y_step
                    cv2.putText(annotated_img, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Nom du fichier image
                img_filename = f"test_image_{idx}.jpg"
                save_path = os.path.join(label_dir, img_filename)
                # Convertir l'image en BGR pour OpenCV
                img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, img_bgr)

            # Sauvegarder les visualisations Grad-CAM si les options sont activées
            if visualize_gradcam and save_gradcam_images:
                input_tensor = inputs[i].unsqueeze(0)
                label_idx = task_labels_dict[gradcam_task][i]
                pred_idx = preds_dict[gradcam_task][i]
                prob = max_probs_dict[gradcam_task][i]

                if label_idx == -1:
                    true_label = "Unknown"
                else:
                    true_label = tasks[gradcam_task][label_idx]

                if pred_idx == -1:
                    pred_label = "Unknown"
                else:
                    pred_label = tasks[gradcam_task][pred_idx]

                text = f"{gradcam_task} - True: {true_label}, Pred: {pred_label}, Prob: {prob:.2f}"

                target = [ClassifierOutputTarget(label_idx)]

                # Calculer Grad-CAM
                grayscale_cam = grad_cam(input_tensor=input_tensor, targets=target)[0]

                # Normaliser grayscale_cam entre 0 et 1
                grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)

                # Redimensionner grayscale_cam pour correspondre à img_np
                grayscale_cam_resized = cv2.resize(grayscale_cam, (img_np.shape[1], img_np.shape[0]))

                # Vérifier si le colormap spécifié est valide
                if colormap not in colormap_dict:
                    print(f"Colormap '{colormap}' non reconnu. Utilisation du colormap par défaut 'hot'.")
                    colormap_code = cv2.COLORMAP_HOT
                else:
                    colormap_code = colormap_dict[colormap]

                # Appliquer le colormap manuellement
                heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam_resized), colormap_code)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                visualization = heatmap.astype(np.float32) / 255.0
                visualization = visualization * 0.5 + img_np.astype(np.float32)/255.0 * 0.5
                visualization = np.clip(visualization, 0, 1)

                # Convertir la visualisation en format uint8
                visualization_cv = (visualization * 255).astype(np.uint8)

                # Annoter uniquement la tâche spécifiée
                cv2.putText(visualization_cv, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Créer l'image combinée avec l'image originale annotée et la visualisation Grad-CAM annotée
                combined_image = np.hstack((img_cv, visualization_cv))

                # Nom du fichier image Grad-CAM
                gradcam_save_path = os.path.join(label_dir, f"GradCAM_{idx}.jpg")

                # Convertir en BGR pour OpenCV
                combined_image_bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(gradcam_save_path, combined_image_bgr)

                # Optionnel : ajouter l'image combinée à TensorBoard
                if writer:
                    combined_image_tensor = torch.from_numpy(combined_image).permute(2, 0, 1)
                    writer.add_image(f'GradCAM/Images/{idx}', combined_image_tensor, global_step=idx)

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    # Calcul des métriques
    average_loss = total_loss / total_samples
    metrics = {}
    for task_name in tasks.keys():
        if all_preds[task_name]:
            preds = np.array(all_preds[task_name])
            labels_np = np.array(all_labels[task_name])
            valid_indices = preds != -1  # Ignorer les prédictions inconnues
            if valid_indices.sum() > 0:
                accuracy = np.mean(preds[valid_indices] == labels_np[valid_indices])
                precision = precision_score(labels_np[valid_indices], preds[valid_indices], average='weighted', zero_division=0)
                recall = recall_score(labels_np[valid_indices], preds[valid_indices], average='weighted', zero_division=0)
                f1 = f1_score(labels_np[valid_indices], preds[valid_indices], average='weighted', zero_division=0)
                conf_matrix = confusion_matrix(labels_np[valid_indices], preds[valid_indices], labels=list(range(len(tasks[task_name]))))
            else:
                accuracy = precision = recall = f1 = 0.0
                conf_matrix = np.zeros((len(tasks[task_name]), len(tasks[task_name])))
            metrics[task_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': conf_matrix.tolist()
            }
            print(f'Tâche {task_name} - Exactitude: {accuracy:.4f}, Précision: {precision:.4f}, Rappel: {recall:.4f}, Score F1: {f1:.4f}')
            print(f'Matrice de confusion pour {task_name}:\n{conf_matrix}\n')
        else:
            metrics[task_name] = {
                'accuracy': None,
                'precision': None,
                'recall': None,
                'f1_score': None,
                'confusion_matrix': None
            }

    # Calcul des performances moyennes sur toutes les tâches
    accuracy_scores = [m['accuracy'] for m in metrics.values() if m['accuracy'] is not None]
    precision_scores = [m['precision'] for m in metrics.values() if m['precision'] is not None]
    recall_scores = [m['recall'] for m in metrics.values() if m['recall'] is not None]
    f1_scores = [m['f1_score'] for m in metrics.values() if m['f1_score'] is not None]

    if f1_scores:
        average_accuracy = np.mean(accuracy_scores)
        average_precision = np.mean(precision_scores)
        average_recall = np.mean(recall_scores)
        average_f1 = np.mean(f1_scores)
    else:
        average_accuracy = average_precision = average_recall = average_f1 = 0.0

    print(f'Performances moyennes - Exactitude: {average_accuracy:.4f}, Précision: {average_precision:.4f}, Rappel: {average_recall:.4f}, Score F1: {average_f1:.4f}')

    # Ajouter les performances moyennes au dictionnaire des métriques
    metrics['average'] = {
        'accuracy': average_accuracy,
        'precision': average_precision,
        'recall': average_recall,
        'f1_score': average_f1
    }

    # Sauvegarde des métriques dans un fichier
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
            if task_metrics['accuracy'] is not None and task_name != 'average':
                writer.add_scalar(f"Test/{task_name}_Accuracy", task_metrics['accuracy'])
                writer.add_scalar(f"Test/{task_name}_Precision", task_metrics['precision'])
                writer.add_scalar(f"Test/{task_name}_Recall", task_metrics['recall'])
                writer.add_scalar(f"Test/{task_name}_F1_Score", task_metrics['f1_score'])

    if measure_time:
        times_path = os.path.join(save_dir, "times_test.json")
        with open(times_path, 'w') as f:
            json.dump(times, f, indent=4)
        print(f"Temps moyen de traitement par lot: {np.mean(times):.4f} secondes")
        print(f"Temps total de traitement: {np.sum(times):.4f} secondes")






def run_camera(model, transform, tasks, save_dir, prob_threshold, measure_time, camera_index, kalman_filter, save_camera_video):
    import tkinter as tk
    from tkinter import ttk
    import time
    import json
    from screeninfo import get_monitors

    device = model.device
    model.eval()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra")
        return

    # Obtenir les dimensions de l'écran
    monitors = get_monitors()
    screen = monitors[0]
    screen_width = screen.width
    screen_height = screen.height

    # Créer la fenêtre d'affichage en mode normal (pour permettre le basculement)
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    # Variable pour suivre l'état plein écran
    full_screen_state = False

    # Variables de contrôle pour l'enregistrement
    recording = False
    video_writer = None

    # Création de l'interface de contrôle avec Tkinter
    control_window = tk.Tk()
    control_window.title("Contrôle Enregistrement")

    # Variables Tkinter
    rec_var = tk.BooleanVar(value=False)
    video_name_var = tk.StringVar()

    def toggle_recording():
        nonlocal recording, video_writer
        recording = not recording
        rec_var.set(recording)
        if recording:
            btn_toggle.config(text="Arrêter l'enregistrement")
        else:
            btn_toggle.config(text="Démarrer l'enregistrement")
            if video_writer is not None:
                video_writer.release()
                video_writer = None
                print("Enregistrement arrêté.")

    def toggle_fullscreen():
        nonlocal full_screen_state
        if not full_screen_state:
            cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            btn_fullscreen.config(text="Quitter le plein écran")
        else:
            cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            btn_fullscreen.config(text="Plein écran")
        full_screen_state = not full_screen_state

    # Interface de contrôle Tkinter
    lbl = ttk.Label(control_window, text="Nom de la vidéo (optionnel) :")
    lbl.pack(padx=10, pady=5)
    entry_video = ttk.Entry(control_window, textvariable=video_name_var, width=30)
    entry_video.pack(padx=10, pady=5)
    btn_toggle = ttk.Button(control_window, text="Démarrer l'enregistrement", command=toggle_recording)
    btn_toggle.pack(padx=10, pady=5)
    btn_fullscreen = ttk.Button(control_window, text="Plein écran", command=toggle_fullscreen)
    btn_fullscreen.pack(padx=10, pady=5)
    # Positionnement de la fenêtre de contrôle
    control_window.geometry("300x200+50+50")

    times = []

    # Préparation du Kalman Filter si activé
    if kalman_filter:
        from pykalman import KalmanFilter
        state_means = {}
        state_covariances = {}
        kf = {}
        for task_name, class_list in tasks.items():
            num_classes = len(class_list)
            kf[task_name] = KalmanFilter(initial_state_mean=np.zeros(num_classes),
                                         initial_state_covariance=np.eye(num_classes),
                                         n_dim_obs=num_classes)
            state_means[task_name] = np.zeros(num_classes)
            state_covariances[task_name] = np.eye(num_classes)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur: Impossible de lire l'image de la caméra")
            break

        start_time = time.time()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        img_tensor = transform(pil_img).unsqueeze(0).to(device)
        outputs = model(img_tensor)

        text_lines = []
        for task_name, task_output in outputs.items():
            probabilities = torch.nn.functional.softmax(task_output, dim=1)[0].detach().cpu().numpy()
            if kalman_filter:
                state_means[task_name], state_covariances[task_name] = kf[task_name].filter_update(
                    state_mean=state_means[task_name],
                    state_covariance=state_covariances[task_name],
                    observation=probabilities
                )
                probabilities = state_means[task_name]
            pred_idx = np.argmax(probabilities)
            pred_prob = probabilities[pred_idx]
            if pred_prob < prob_threshold:
                pred_label = "Unknown"
            else:
                pred_label = tasks[task_name][pred_idx]
            text_lines.append(f"{task_name}: {pred_label} ({pred_prob:.2f})")

        end_time = time.time()
        times.append(end_time - start_time)

        # Redimensionner le cadre à la taille de l'écran
        frame_resized = cv2.resize(frame, (screen_width, screen_height))
        y0 = 30
        y_step = 40
        for i, line in enumerate(text_lines):
            y_pos = y0 + i * y_step
            cv2.putText(frame_resized, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        # Gestion de l'enregistrement vidéo via le bouton de contrôle
        if save_camera_video:
            if recording and video_writer is None:
                # Si aucun nom n'a été saisi, générer un nom basé sur le timestamp
                vname = video_name_var.get().strip()
                if vname == "":
                    vname = f"video_{int(time.time())}"
                output_video_path = os.path.join(save_dir, f"{vname}.avi")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(output_video_path, fourcc, 20.0, (screen_width, screen_height))
                print(f"Enregistrement démarré: {output_video_path}")
            elif not recording and video_writer is not None:
                video_writer.release()
                video_writer = None
            if video_writer is not None:
                video_writer.write(frame_resized)

        cv2.imshow("Camera", frame_resized)
        # Mise à jour de l'interface Tkinter
        control_window.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print("Enregistrement vidéo terminé.")
    cv2.destroyAllWindows()

    if measure_time and len(times) > 0:
        times_path = os.path.join(save_dir, "times_camera.json")
        with open(times_path, 'w') as f:
            json.dump(times, f, indent=4)
        print(f"Temps moyen de traitement par image: {np.mean(times):.4f}s, total: {np.sum(times):.4f}s")

    control_window.destroy()



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