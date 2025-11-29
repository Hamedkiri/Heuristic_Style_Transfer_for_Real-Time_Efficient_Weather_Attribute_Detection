# camera_utils.py
import os
import time

import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
from screeninfo import get_monitors
from pykalman import KalmanFilter

import tkinter as tk
from tkinter import ttk


def run_camera(model,
               transform,
               tasks: dict,
               save_dir: str,
               prob_threshold: float,
               measure_time: bool,
               camera_index: int,
               kalman_filter: bool,
               save_camera_video: bool):
    """
    Version extraite de ton script, isolée dans un module.
    """
    device = model.device
    model.eval()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la caméra")
        return

    screen = get_monitors()[0]
    screen_width, screen_height = screen.width, screen.height
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    full_screen_state = False

    control_window = tk.Tk()
    control_window.title("Contrôle Enregistrement")
    rec_var = tk.BooleanVar(value=False)
    video_name_var = tk.StringVar()

    recording = False
    video_writer = None

    def toggle_recording():
        nonlocal recording, video_writer
        recording = not recording
        rec_var.set(recording)
        btn_toggle.config(text="Arrêter l'enregistrement" if recording else "Démarrer l'enregistrement")
        if not recording and video_writer:
            video_writer.release()
            video_writer = None
            print("Enregistrement arrêté.")

    def toggle_fullscreen():
        nonlocal full_screen_state
        prop = cv2.WND_PROP_FULLSCREEN
        cv2.setWindowProperty("Camera", prop,
                              cv2.WINDOW_FULLSCREEN if not full_screen_state else cv2.WINDOW_NORMAL)
        btn_fullscreen.config(text="Quitter le plein écran" if not full_screen_state else "Plein écran")
        full_screen_state = not full_screen_state

    ttk.Label(control_window, text="Nom de la vidéo (optionnel) :").pack(padx=10, pady=5)
    ttk.Entry(control_window, textvariable=video_name_var, width=30).pack(padx=10, pady=5)
    btn_toggle = ttk.Button(control_window, text="Démarrer l'enregistrement", command=toggle_recording)
    btn_toggle.pack(padx=10, pady=5)
    btn_fullscreen = ttk.Button(control_window, text="Plein écran", command=toggle_fullscreen)
    btn_fullscreen.pack(padx=10, pady=5)
    control_window.geometry("300x200+50+50")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    padding_x = 10
    padding_y = 10
    y0, y_step = 40, 40

    def longest_label(lst):
        return max(lst + ["Unknown"], key=len)

    sample_lines = [
        f"{task}: {longest_label(cls)} (1.00)"
        for task, cls in tasks.items()
    ]
    text_sizes = [cv2.getTextSize(l, font, font_scale, thickness)[0] for l in sample_lines]
    max_text_width = max(w for (w, h) in text_sizes)
    font_height = max(h for (w, h) in text_sizes)

    box_left = 0
    box_top = y0 - font_height - padding_y
    box_right = max_text_width + 2 * padding_x
    box_bottom = y0 + (len(tasks) - 1) * y_step + padding_y

    if kalman_filter:
        kf, state_means, state_cov = {}, {}, {}
        for t, cls in tasks.items():
            M = len(cls)
            kf[t] = KalmanFilter(initial_state_mean=np.zeros(M),
                                 initial_state_covariance=np.eye(M),
                                 n_dim_obs=M)
            state_means[t] = np.zeros(M)
            state_cov[t] = np.eye(M)

    times = []

    os.makedirs(save_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur : lecture caméra")
            break

        start = time.time()
        img_tensor = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
        times.append(time.time() - start)

        text_lines = []
        for task, out in outputs.items():
            probs = torch.softmax(out, 1)[0].detach().cpu().numpy()
            if kalman_filter:
                sm, sc = kf[task].filter_update(state_means[task], state_cov[task], probs)
                state_means[task], state_cov[task] = sm, sc
                probs = sm
            idx = int(probs.argmax())
            label = "Unknown" if probs[idx] < prob_threshold else tasks[task][idx]
            text_lines.append(f"{task}: {label} ({probs[idx]:.2f})")

        frame_res = cv2.resize(frame, (screen_width, screen_height))

        overlay = frame_res.copy()
        cv2.rectangle(overlay, (box_left, box_top), (box_right, box_bottom),
                      (255, 255, 255), thickness=-1)
        cv2.addWeighted(overlay, 0.4, frame_res, 0.6, 0, frame_res)

        for i, line in enumerate(text_lines):
            y = y0 + i * y_step
            cv2.putText(frame_res, line, (padding_x, y),
                        font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

        if save_camera_video:
            if recording and video_writer is None:
                name = video_name_var.get().strip() or f"video_{int(time.time())}"
                path = os.path.join(save_dir, f"{name}.avi")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(path, fourcc, 20.0,
                                               (screen_width, screen_height))
                print("Enregistrement démarré :", path)
            elif not recording and video_writer:
                video_writer.release()
                video_writer = None
            if video_writer:
                video_writer.write(frame_res)

        cv2.imshow("Camera", frame_res)
        control_window.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    control_window.destroy()

    if measure_time and times:
        with open(os.path.join(save_dir, "times_camera.json"), "w") as f:
            import json
            json.dump(times, f, indent=2)
        print(f"Temps moyen de traitement : {np.mean(times):.4f}s – total : {np.sum(times):.1f}s")
