
import argparse
import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
import random
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import hdbscan
from matplotlib.path import Path
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import threading
import re
import datetime
import pandas as pd
from PIL import Image, ImageTk
# Pour IntegratedGradients (Captum)
from captum.attr import IntegratedGradients




class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SingleHeadAttention, self).__init__()
        self.embed_dim = embed_dim

        # Projections for queries, keys, and values
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Attention scores
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.embed_dim)
        attn_weights = self.softmax(attn_scores)

        # Weighted attention output
        attn_output = torch.bmm(attn_weights, V)
        return attn_output

class MultiHeadAttentionPerTaskModel(nn.Module):
    def __init__(self, base_encoder, truncate_after_layer, tasks, device='cpu'):
        super(MultiHeadAttentionPerTaskModel, self).__init__()
        self.device = device
        self.tasks = tasks

        # Remove the last linear layer of ResNet50 before truncating
        layers = list(base_encoder.children())[:-1]  # Remove the last linear layer
        self.truncated_encoder = nn.Sequential(*layers[:truncate_after_layer]).to(self.device)

        # Adaptive pooling and flatten
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # Compute output features
        dummy_input = torch.zeros(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            output = self.truncated_encoder(dummy_input)
            output = self.pool(output)
            output = self.flatten(output)
            self.num_features = output.shape[1]

        # Independent attention and classifier for each task
        self.attentions = nn.ModuleDict()
        self.classifiers = nn.ModuleDict()

        for task_name, value in tasks.items():
            # Si value est une liste de classes, on prend sa longueur, sinon on suppose que c'est déjà un entier
            num_classes = len(value) if isinstance(value, list) else value
            attention_name = f"attention_{task_name.replace(' ', '_')}"
            classifier_name = f"classifier_{task_name.replace(' ', '_')}"
            self.attentions[attention_name] = SingleHeadAttention(self.num_features).to(self.device)
            self.classifiers[classifier_name] = nn.Linear(self.num_features, num_classes).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.truncated_encoder(x)
        x = self.pool(x)
        x = self.flatten(x)

        outputs = {}
        for attention_name, attention in self.attentions.items():
            attn_output = attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))  # [batch_size, 1, embed_dim]
            attn_output = attn_output.squeeze(1)  # [batch_size, embed_dim]

            # Extract the classifier name corresponding to the attention mechanism
            classifier_name = attention_name.replace("attention", "classifier")
            task_name = classifier_name.replace('classifier_', '').replace('_', ' ')
            outputs[task_name] = self.classifiers[classifier_name](attn_output)

        return outputs

class TaskSpecificModel(nn.Module):
    def __init__(self, model, task_name):
        super(TaskSpecificModel, self).__init__()
        self.model = model
        self.task_name = task_name

    def forward(self, x):
        outputs = self.model(x)
        return outputs[self.task_name]
