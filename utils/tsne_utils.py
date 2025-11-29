# tsne_utils.py
import os
from typing import Dict, Tuple, List

import numpy as np
import torch
from torch.utils.data import Subset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector

import matplotlib

matplotlib.use('TkAgg')
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, colorchooser
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os, json, numpy as np

# (plot_tsne_interactive : version inchangée, juste isolée)


def compute_embeddings_with_paths(model,
                                  loader,
                                  device: torch.device,
                                  per_task_tsne: bool = False):
    """
    Même logique que dans ton script :
      - per_task_tsne=True -> dict par tâche
      - sinon -> un seul ensemble.
    """
    model.eval()

    if per_task_tsne:
        embeddings_data = {t: [] for t in model.num_classes_per_task.keys()}
        labels_data = {t: [] for t in model.num_classes_per_task.keys()}
        img_paths_data = {t: [] for t in model.num_classes_per_task.keys()}
    else:
        embeddings_data, labels_data, img_paths_data = [], [], []

    with torch.no_grad():
        for batch_idx, (inputs, label_dict) in enumerate(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            B = inputs.size(0)

            if per_task_tsne:
                for t in model.num_classes_per_task.keys():
                    out_t = outputs[t].detach().cpu().numpy()
                    embeddings_data[t].append(out_t)
                    labels_data[t].extend(label_dict[t].detach().cpu().numpy().tolist())

                if isinstance(loader.dataset, Subset):
                    ds_ = loader.dataset.dataset
                    idxs = loader.dataset.indices[batch_idx * B: batch_idx * B + B]
                    for t in model.num_classes_per_task.keys():
                        img_paths_data[t].extend([ds_.samples[i][0] for i in idxs])
                else:
                    ds_ = loader.dataset
                    start_i = batch_idx * B
                    end_i = start_i + B
                    for t in model.num_classes_per_task.keys():
                        img_paths_data[t].extend([ds_.samples[i][0] for i in range(start_i, end_i)])
            else:
                first_t = next(iter(outputs.keys()))
                out_ = outputs[first_t].detach().cpu().numpy()
                embeddings_data.append(out_)
                labels_data.extend(label_dict[first_t].detach().cpu().numpy().tolist())
                if isinstance(loader.dataset, Subset):
                    ds_ = loader.dataset.dataset
                    idxs = loader.dataset.indices[batch_idx * B: batch_idx * B + B]
                    img_paths_data.extend([ds_.samples[i][0] for i in idxs])
                else:
                    ds_ = loader.dataset
                    start_i = batch_idx * B
                    end_i = start_i + B
                    img_paths_data.extend([ds_.samples[i][0] for i in range(start_i, end_i)])

    if per_task_tsne:
        for t in embeddings_data:
            embeddings_data[t] = (np.concatenate(embeddings_data[t], axis=0)
                                  if len(embeddings_data[t]) else np.empty((0, 0)))
            labels_data[t] = np.array(labels_data[t]) if len(labels_data[t]) else np.array([])
        return embeddings_data, labels_data, img_paths_data
    else:
        embeddings_data = (np.concatenate(embeddings_data, axis=0)
                           if len(embeddings_data) else np.empty((0, 0)))
        labels_data = np.array(labels_data) if len(labels_data) else np.array([])
        return embeddings_data, labels_data, img_paths_data


def perform_tsne(embeddings,
                 labels,
                 class_list,
                 colors,
                 results_dir: str,
                 task_name: str):
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 10))
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    if colors and len(colors) >= num_classes:
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    else:
        cmap = plt.cm.get_cmap("tab20", num_classes)
        color_map = {label: cmap(i / num_classes) for i, label in enumerate(unique_labels)}

    for lbl in unique_labels:
        mask = (labels == lbl)
        if lbl < 0 or lbl >= len(class_list):
            lbl_name = "Unknown"
        else:
            lbl_name = class_list[lbl]
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                    label=lbl_name, color=color_map[lbl])
    plt.legend()
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"tsne_{task_name.replace(' ', '_')}.png")
    plt.savefig(out_path)
    plt.show()
    print(f"t-SNE figure saved to {out_path}")

def plot_tsne_interactive(attentive_embeddings_data, labels_data, tasks, img_paths_data, colors=None, num_clusters=None,
                          save_dir='results'):
    """
    Ouvre une interface interactive Tkinter pour explorer un t-SNE calculé sur les attentive embeddings.

    L'interface permet :
      - de choisir une tâche (si plusieurs sont présentes),
      - de recalculer le t-SNE pour la tâche sélectionnée,
      - de zoomer/dézoomer,
      - de tracer un polygone sur le plot pour sélectionner des points,
      - d'afficher la ou les images associées à chaque point sélectionné.

    Args:
        attentive_embeddings_data (dict): Dictionnaire associant chaque tâche à ses attentive embeddings (numpy array de forme (N, C, H, W)).
        labels_data (dict): Dictionnaire associant chaque tâche à ses labels (numpy array).
        tasks (dict): Dictionnaire associant chaque tâche à la liste de ses classes.
        img_paths_data (dict): Dictionnaire associant chaque tâche à la liste des chemins d'images.
        colors (list, optional): Liste de couleurs à utiliser pour le plot.
        num_clusters (int, optional): (Non utilisé ici, mais peut être étendu pour le clustering interactif).
        save_dir (str): Répertoire où sauvegarder d'éventuelles sorties (ex. fichiers JSON des points sélectionnés).
    """


    # Déterminer si l'on travaille avec un dictionnaire (plusieurs tâches) ou un seul tableau (tâche unique)
    if isinstance(attentive_embeddings_data, dict):
        single_task_mode = (len(attentive_embeddings_data) == 1)
        if single_task_mode:
            current_task_name = list(attentive_embeddings_data.keys())[0]
        else:
            current_task_name = None
    else:
        single_task_mode = True
        current_task_name = None



    tsne_results = None
    labels = None
    class_names = None
    unique_labels = None
    scatter = None
    color_map = None
    img_paths = None
    filename_to_path = None
    polygon = []
    polygon_selector = None
    polygon_cleared = True

    # Création des frames pour l'interface
    root = tk.Tk()
    root.title("Interactive t-SNE with Images")
    left_frame = tk.Frame(root)
    left_frame.grid(row=0, column=0, sticky='nsew')
    right_frame = tk.Frame(root)
    right_frame.grid(row=0, column=1, sticky='nsew')

    # Intégration de la figure matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))
    canvas = FigureCanvasTkAgg(fig, master=left_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

    # Zone d'affichage d'image et informations
    img_label = tk.Label(right_frame)
    img_label.pack(pady=10)
    label_text = tk.StringVar()
    label_label = tk.Label(right_frame, textvariable=label_text, justify='left')
    label_label.pack()
    inside_points_label = tk.StringVar()
    inside_points_count_label = tk.Label(right_frame, textvariable=inside_points_label)
    inside_points_count_label.pack()

    dropdown_points = []
    dropdown = ttk.Combobox(right_frame, state="readonly")
    dropdown.pack(fill='x', pady=5)
    dropdown.bind("<<ComboboxSelected>>", lambda event: on_dropdown_select())

    def change_class_color():
        selected = class_selector.get()
        if selected:
            label_str = selected.split(':')[0]
            label_val = int(label_str)
            color_code = colorchooser.askcolor(title="Choisir une couleur")[1]
            if color_code:
                color_map[label_val] = color_code
                scatter.set_color([color_map[int(lbl)] for lbl in labels])
                ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=class_names[int(lbl)],
                                              markerfacecolor=color_map[int(lbl)], markersize=10) for lbl in
                                   unique_labels])
                canvas.draw()

    class_selector_label = tk.Label(right_frame, text="Sélectionnez une classe :")
    class_selector_label.pack(pady=5)
    class_selector = ttk.Combobox(right_frame, state="readonly")
    class_selector.pack(pady=5)
    change_color_button = tk.Button(right_frame, text="Changer la couleur de la classe", command=change_class_color)
    change_color_button.pack(pady=5)

    button_frame = tk.Frame(right_frame)
    button_frame.pack(pady=10)
    close_button = tk.Button(button_frame, text="Fermer le polygone", command=lambda: analyze_polygon())
    close_button.pack(side='left', padx=5)
    clear_button = tk.Button(button_frame, text="Effacer le polygone", command=lambda: clear_polygon())
    clear_button.pack(side='left', padx=5)

    def clear_polygon():
        nonlocal polygon_selector, polygon_cleared
        polygon.clear()
        if polygon_selector:
            polygon_selector.disconnect_events()
            polygon_selector.set_visible(False)
            del polygon_selector
            polygon_selector = None
        while ax.patches:
            ax.patches.pop().remove()
        fig.canvas.draw()
        inside_points_label.set("")
        label_text.set("")
        img_label.config(image='')
        dropdown.set('')
        dropdown['values'] = []
        polygon_cleared = True

    def update_plot(task_name):
        nonlocal tsne_results, labels, class_names, unique_labels, scatter, color_map, img_paths, filename_to_path, current_task_name
        current_task_name = task_name
        ax.clear()
        # Utiliser les attentive embeddings pour la tâche sélectionnée
        if isinstance(attentive_embeddings_data, dict):
            embeddings = attentive_embeddings_data[task_name]
            labels_local = labels_data[task_name]
            img_paths = img_paths_data[task_name]
            class_names = tasks[task_name]
        else:
            embeddings = attentive_embeddings_data
            labels_local = labels_data
            img_paths = img_paths_data
            class_names = tasks[list(tasks.keys())[0]]
        filename_to_path = {os.path.basename(path): path for path in img_paths}
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_flat = embeddings.reshape(embeddings.shape[0], -1)
        tsne_results = tsne.fit_transform(embeddings_flat)
        labels = labels_local
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        if colors and len(colors) >= num_classes:
            color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        else:
            color_palette = plt.cm.get_cmap("tab20", num_classes)
            color_map = {label: color_palette(i / num_classes) for i, label in enumerate(unique_labels)}
        scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=[color_map[int(label)] for label in labels],
                             picker=True)
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=class_names[int(label)],
                                      markerfacecolor=color_map[int(label)], markersize=10) for label in unique_labels]
        ax.legend(handles=legend_elements)
        ax.set_title(f"t-SNE pour la tâche : {task_name}" if task_name else "t-SNE")
        canvas.draw()
        class_selector['values'] = [f"{label}: {class_names[label]}" for label in unique_labels]
        if unique_labels.size > 0:
            class_selector.current(0)
        clear_polygon()

    def on_task_select(event):
        selected_task = task_selector.get()
        update_plot(selected_task)

    def onpick(event):
        ind = event.ind[0]
        img_path = img_paths[ind]
        display_image(img_path, class_names[int(labels[ind])])

    fig.canvas.mpl_connect('pick_event', onpick)

    def enable_polygon_selector(event):
        nonlocal polygon_selector, polygon_cleared
        if event.button == 3:  # clic droit
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
        polygon_path = Path(polygon)
        for i, (x, y) in enumerate(tsne_results):
            point = (x, y)
            if polygon_path.contains_point(point):
                inside_points.append({"path": img_paths[i], "class": class_names[int(labels[i])], "position": point})
            else:
                outside_points.append({"path": img_paths[i], "class": class_names[int(labels[i])], "position": point})
        for point in inside_points:
            point['filename'] = os.path.basename(point['path'])
            del point['path']
        for point in outside_points:
            point['filename'] = os.path.basename(point['path'])
            del point['path']
        filename_suffix = current_task_name.replace(' ', '_') if current_task_name else 'task'
        with open(os.path.join(save_dir, f"inside_polygon_{filename_suffix}.json"), "w") as f:
            json.dump(inside_points, f)
        with open(os.path.join(save_dir, f"outside_polygon_{filename_suffix}.json"), "w") as f:
            json.dump(outside_points, f)
        inside_points_label.set(f"Points à l'intérieur du polygone: {len(inside_points)}")
        update_dropdown(inside_points)

    def update_dropdown(inside_points):
        dropdown_values = [f"{point['filename']} ({point['class']})" for point in inside_points]
        dropdown['values'] = dropdown_values
        dropdown_points.clear()
        dropdown_points.extend(inside_points)
        if dropdown_values:
            dropdown.current(0)
            on_dropdown_select()

    def on_dropdown_select():
        selection = dropdown.current()
        if selection >= 0:
            point = dropdown_points[selection]
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

    # ------------------------------------------------------------------
    # 0) Variable globale supplémentaire
    # ------------------------------------------------------------------
    last_click = {'pos': None, 'marker': None}  # on stocke aussi le handle

    # ------------------------------------------------------------------
    # 1) Gestion du clic gauche
    # ------------------------------------------------------------------
    def on_mouse_click(event):
        if event.button == 1 and event.inaxes is not None and event.xdata is not None:
            # Effacer l’ancien marqueur s’il existe
            if last_click['marker'] is not None:
                last_click['marker'].remove()

            # Mémoriser la nouvelle position
            last_click['pos'] = (event.xdata, event.ydata)

            # Dessiner la nouvelle croix et conserver son handle
            last_click['marker'] = ax.scatter(*last_click['pos'],
                                              marker='x', c='k', s=30, zorder=3)
            canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_mouse_click)

    # ------------------------------------------------------------
    # 1) Remplacer complètement la fonction zoom
    # ------------------------------------------------------------
    def zoom(scale: float):
        """scale >1 : zoom avant ; scale <1 : zoom arrière, centré sur last_click"""
        if scale <= 0:
            return

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        # Point de référence : dernier clic ou centre courant
        if last_click['pos'] and None not in last_click['pos']:
            cx, cy = last_click['pos']
        else:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

        new_w = (x1 - x0) / scale
        new_h = (y1 - y0) / scale

        ax.set_xlim(cx - new_w / 2, cx + new_w / 2)
        ax.set_ylim(cy - new_h / 2, cy + new_h / 2)
        canvas.draw_idle()

    # ------------------------------------------------------------
    # 2) Adapter la molette
    # ------------------------------------------------------------
    # Molette : on ignore désormais event.xdata / ydata
    def on_scroll(event):
        direction = getattr(event, "step", 1 if event.button == "up" else -1)
        base = 1.2
        scale = base if direction > 0 else 1 / base
        zoom(scale)

    # Clavier
    def on_key_press(event):
        base = 1.2
        if event.key in ['+', '=']:
            zoom(base)
        elif event.key == '-':
            zoom(1 / base)

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key_press)


    root.grid_columnconfigure(0, weight=3)
    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(0, weight=1)
    left_frame.grid_rowconfigure(0, weight=1)
    left_frame.grid_columnconfigure(0, weight=1)

    if not single_task_mode:
        task_selector_label = tk.Label(right_frame, text="Sélectionnez une tâche :")
        task_selector_label.pack(pady=5)
        task_selector = ttk.Combobox(right_frame, state="readonly", values=list(tasks.keys()))
        # Attribuer la liste **après** la création
        task_names = list(tasks.keys())
        task_selector['values'] = task_names

        # Fixe l’élément affiché : préférez `.current(0)` à `.set(...)`
        if task_names:
            task_selector.current(0)  # sélectionne la première tâche visuellement

        task_selector.pack(pady=5)
        task_selector.bind("<<ComboboxSelected>>", on_task_select)

    if single_task_mode:
        update_plot(list(tasks.keys())[0])
    else:
        initial_task = list(tasks.keys())[0]
        task_selector.set(initial_task)
        update_plot(initial_task)

    root.mainloop()
