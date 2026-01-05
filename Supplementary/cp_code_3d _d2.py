# cp_fault_diagnosis_complete.py
# Author: UMAR | Complete CP Fault Diagnosis with Ensemble CNNs
# Processes multiple pressure datasets (3 bar, 3.5 bar, 4 bar) separately
# Generates: Confusion Matrix, ROC Curves, t-SNE (2D & 3D), Classification Report
# Modified to use exactly 300 samples per class
#
# Usage (PowerShell):
#   python cp_fault_diagnosis_complete.py --dataset_name "3bar" --data_dir "F:\CP Data\3 bar" --results_dir "F:\Faisal Work\CP Work\Results" --img_size 256 --epochs 40 --tta 8 --samples_per_class 300

import os, random, argparse, json, math, time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_plot_style():
    """Publication-quality plot settings"""
    sns.set_style("white")
    sns.set_context("talk")
    plt.rcParams.update({
        "savefig.dpi": 1000,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "font.weight": "bold",
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.grid": False
    })

def class_counts_from_indices(dataset, indices):
    counts = {}
    for i in indices:
        _, y = dataset[i]
        counts[y] = counts.get(y, 0) + 1
    return counts

def compute_class_weights_from_counts(counts, num_classes):
    total = sum(counts.get(i, 0) for i in range(num_classes))
    weights = []
    for i in range(num_classes):
        c = counts.get(i, 1)
        w = total / (num_classes * c)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)

def short_labels_from_full(names: List[str]) -> List[str]:
    """Convert full class names to short labels"""
    out = []
    for n in names:
        nl = n.lower()
        if "impeller" in nl:
            out.append("IF")
        elif "hole" in nl:
            out.append("MSH")
        elif "scratch" in nl:
            out.append("MSS")
        elif "normal" in nl:
            out.append("N")
        else:
            out.append(n)
    return out

# ---------------------------
# Modified Sampling Function
# ---------------------------
def sample_equal_per_class(dataset, samples_per_class=300, seed=42):
    """
    Sample exactly 'samples_per_class' samples from each class.
    Returns indices of sampled data.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Get all indices and their labels
    all_indices = list(range(len(dataset)))
    all_labels = [dataset[i][1] for i in all_indices]
    
    # Group indices by class
    class_indices = {}
    for idx, label in zip(all_indices, all_labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    num_classes = len(class_indices)
    print(f"\nOriginal class distribution:")
    for cls_id in sorted(class_indices.keys()):
        print(f"  Class {cls_id}: {len(class_indices[cls_id])} samples")
    
    # Sample exactly samples_per_class from each class
    sampled_indices = []
    for cls_id in sorted(class_indices.keys()):
        cls_idx = class_indices[cls_id]
        
        if len(cls_idx) < samples_per_class:
            print(f"\n⚠️  Warning: Class {cls_id} has only {len(cls_idx)} samples, less than requested {samples_per_class}")
            print(f"    Using all {len(cls_idx)} samples for this class")
            sampled_indices.extend(cls_idx)
        else:
            # Randomly sample samples_per_class indices
            selected = np.random.choice(cls_idx, size=samples_per_class, replace=False).tolist()
            sampled_indices.extend(selected)
    
    print(f"\nSampled class distribution:")
    sampled_labels = [dataset[i][1] for i in sampled_indices]
    for cls_id in sorted(class_indices.keys()):
        count = sampled_labels.count(cls_id)
        print(f"  Class {cls_id}: {count} samples")
    
    print(f"\nTotal sampled: {len(sampled_indices)} samples")
    
    return sampled_indices

def stratified_split_from_sampled(dataset, sampled_indices, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Create stratified train/val/test split from pre-sampled indices.
    """
    # Get labels for sampled indices
    targets = [dataset[i][1] for i in sampled_indices]
    indices_array = np.array(sampled_indices)
    
    # First split: train vs (val + test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1-train_ratio, random_state=seed)
    train_rel, temp_rel = next(sss1.split(indices_array, targets))
    
    train_idx = indices_array[train_rel]
    temp_idx = indices_array[temp_rel]
    y_temp = np.array(targets)[temp_rel]
    
    # Second split: val vs test
    val_size = int(val_ratio * len(sampled_indices))
    test_size = len(temp_idx) - val_size
    
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    val_rel, test_rel = next(sss2.split(temp_idx, y_temp))
    
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]
    
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

# ---------------------------
# Models
# ---------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, pool=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
    
    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.pool(x)
        return x

class CNN1_Local(nn.Module):
    """3x3 kernels, 3 conv blocks - Local features"""
    def __init__(self, num_classes=4, in_ch=3, drop=0.2):
        super().__init__()
        self.b1 = ConvBlock(in_ch, 32, k=3)
        self.b2 = ConvBlock(32, 64, k=3)
        self.b3 = ConvBlock(64, 128, k=3)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(drop),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.head(self.b3(self.b2(self.b1(x))))

class CNN2_Global(nn.Module):
    """5x5 kernels, 4 conv blocks - Global features"""
    def __init__(self, num_classes=4, in_ch=3, drop=0.3):
        super().__init__()
        self.b1 = ConvBlock(in_ch, 32, k=5)
        self.b2 = ConvBlock(32, 64, k=5)
        self.b3 = ConvBlock(64, 128, k=5)
        self.b4 = ConvBlock(128, 256, k=5)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(drop),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.head(self.b4(self.b3(self.b2(self.b1(x)))))

class DWSeparableConv(nn.Module):
    """Depthwise separable convolution"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x

class CNN3_Compact(nn.Module):
    """Depthwise separable (MobileNet-style) - Efficient"""
    def __init__(self, num_classes=4, in_ch=3, drop=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential(
            DWSeparableConv(32, 64, stride=1), nn.MaxPool2d(2),
            DWSeparableConv(64, 128, stride=1), nn.MaxPool2d(2),
            DWSeparableConv(128, 256, stride=1), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(drop),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))

class MetaMLP(nn.Module):
    """Meta-learner for ensemble fusion"""
    def __init__(self, num_models=3, num_classes=4, hidden=16, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_models * num_classes, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

# ---------------------------
# Data Loading & Augmentation
# ---------------------------
def build_transforms(img_size=256, grayscale=False, weak_aug=False):
    base = []
    if grayscale:
        base.append(transforms.Grayscale(num_output_channels=1))
    
    train_tf = transforms.Compose(base + [
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(10 if not weak_aug else 5),
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.05) if not weak_aug else transforms.ColorJitter(0.05, 0.05, 0.05, 0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*(1 if grayscale else 3), [0.5]*(1 if grayscale else 3)),
    ])
    
    test_tf = transforms.Compose(base + [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*(1 if grayscale else 3), [0.5]*(1 if grayscale else 3)),
    ])
    
    return train_tf, test_tf

def make_loaders(data_dir, img_size=256, batch_size=32, num_workers=0, seed=42, 
                 grayscale=False, weak_aug=False, samples_per_class=300):
    """
    Modified to sample exactly samples_per_class from each class before splitting.
    """
    train_tf, test_tf = build_transforms(img_size, grayscale=grayscale, weak_aug=weak_aug)
    
    # Load full dataset first
    base_full = datasets.ImageFolder(root=data_dir, transform=test_tf)
    
    # Sample equal number from each class
    sampled_indices = sample_equal_per_class(base_full, samples_per_class=samples_per_class, seed=seed)
    
    # Create stratified split from sampled indices
    train_idx, val_idx, test_idx = stratified_split_from_sampled(
        base_full, sampled_indices, train_ratio=0.7, val_ratio=0.15, seed=seed
    )
    
    # Create datasets with appropriate transforms
    base_train = datasets.ImageFolder(root=data_dir, transform=train_tf)
    base_eval = datasets.ImageFolder(root=data_dir, transform=test_tf)
    
    train_ds = Subset(base_train, train_idx)
    val_ds = Subset(base_eval, val_idx)
    test_ds = Subset(base_eval, test_idx)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    
    class_to_idx = base_eval.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    return train_loader, val_loader, test_loader, idx_to_class, (train_idx, val_idx, test_idx), base_eval

# ---------------------------
# Mixup / CutMix
# ---------------------------
def rand_bbox(W, H, lam):
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def apply_mixup_cutmix(x, y, mixup_alpha=0.0, cutmix_alpha=0.0):
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return x, y, None
    
    lam = 1.0
    if cutmix_alpha > 0 and np.random.rand() < 0.5:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        x2 = x.flip(0)
        W = x.size(3)
        H = x.size(2)
        x1_, y1_, x2_, y2_ = rand_bbox(W, H, lam)
        x[:, :, y1_:y2_, x1_:x2_] = x2[:, :, y1_:y2_, x1_:x2_]
        lam = 1 - ((x2_-x1_) * (y2_-y1_) / (W * H))
        y_a, y_b = y, y.flip(0)
    else:
        lam = np.random.beta(max(1e-8, mixup_alpha), max(1e-8, mixup_alpha))
        x2 = x.flip(0)
        x = lam * x + (1 - lam) * x2
        y_a, y_b = y, y.flip(0)
    
    return x, (y_a, y_b, lam), "mixed"

def loss_with_mixing(criterion, logits, y_or_tuple):
    if isinstance(y_or_tuple, tuple):
        y_a, y_b, lam = y_or_tuple
        return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
    return criterion(logits, y_or_tuple)

# ---------------------------
# Training
# ---------------------------
def train_one_model(model, train_loader, val_loader, epochs, lr, device, ckpt_path,
                    class_weights=None, label_smoothing=0.0, mixup_alpha=0.0, cutmix_alpha=0.0):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        label_smoothing=label_smoothing
    )
    
    best_val = float('inf')
    best_state = None
    patience = 10
    no_imp = 0
    
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0
        tr_correct = 0
        n = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x, y_mix, mixed_flag = apply_mixup_cutmix(x, y, mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha)
            
            opt.zero_grad()
            logits = model(x)
            loss = loss_with_mixing(criterion, logits, y_mix if mixed_flag else y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            
            tr_loss += loss.item() * x.size(0)
            tr_correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
        
        sched.step()
        
        # Validation
        model.eval()
        vl_loss = 0
        vl_correct = 0
        m = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                vl_loss += loss.item() * x.size(0)
                vl_correct += (logits.argmax(1) == y).sum().item()
                m += x.size(0)
        
        tr_loss /= max(1, n)
        tr_acc = tr_correct / max(1, n)
        vl_loss /= max(1, m)
        vl_acc = vl_correct / max(1, m)
        
        print(f"Epoch {ep:03d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {vl_loss:.4f} acc {vl_acc:.3f}")
        
        if vl_loss < best_val:
            best_val = vl_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print("Early stopping.")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model

# ---------------------------
# Inference & Evaluation
# ---------------------------
@torch.no_grad()
def predict_probs(model, loader, device, tta=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_probs = []
    all_labels = []
    
    for x, y in loader:
        x = x.to(device)
        
        if tta <= 1:
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        else:
            probs_accum = 0
            for i in range(tta):
                x_aug = x
                if i % 2 == 1:
                    x_aug = torch.flip(x_aug, dims=[3])  # horizontal flip
                if i % 4 == 2:
                    x_aug = torch.flip(x_aug, dims=[2])  # vertical flip
                logits = model(x_aug)
                probs_accum += F.softmax(logits, dim=1)
            probs = (probs_accum / tta).cpu().numpy()
            logits = model(x).cpu().numpy()
        
        all_logits.append(logits if tta <= 1 else model(x).cpu().numpy())
        all_probs.append(probs)
        all_labels.append(y.numpy())
    
    return np.vstack(all_logits), np.vstack(all_probs), np.concatenate(all_labels)

def soft_vote(probs_list: List[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(probs_list, axis=0), axis=0)

def train_meta_mlp(val_probs_list, val_labels, num_classes=4, epochs=100, lr=1e-3, hidden=16, device=None):
    X = np.concatenate(val_probs_list, axis=1)
    y = val_labels
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    
    model = MetaMLP(num_models=len(val_probs_list), num_classes=num_classes, hidden=hidden, drop=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    best_state = None
    patience = 15
    no_imp = 0
    
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        opt.step()
        
        with torch.no_grad():
            vl_loss = loss.item()
        
        if vl_loss < best_loss:
            best_loss = vl_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    model.eval()
    return model

# ---------------------------
# Visualization Functions
# ---------------------------
def save_confusion_matrix(figpath, cm, labels):
    set_plot_style()
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=labels, yticklabels=labels, cbar=False,
                     annot_kws={"size": 20, "fontweight": "bold"})
    ax.set_xlabel('Predicted Label', fontsize=16, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=16, fontweight='bold')
    plt.setp(ax.get_xticklabels(), fontweight='bold', fontsize=14, rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(figpath, dpi=1000, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {figpath}")

def save_roc_curve(figpath, y_true, probs, labels):
    set_plot_style()
    K = len(labels)
    y_bin = label_binarize(y_true, classes=list(range(K)))
    
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each class
    for i in range(K):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=3, label=f'{labels[i]} (AUC = {roc_auc:.2f})')
    
    # Diagonal reference line (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--', label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(figpath, dpi=1000, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {figpath}")

def save_tsne(figpath, features, y_true, labels):
    """2D t-SNE visualization"""
    set_plot_style()
    N = features.shape[0]
    perplexity = max(5, min(30, (N - 1) // 3))
    
    tsne = TSNE(n_components=2, init="pca", random_state=42,
                perplexity=perplexity, learning_rate="auto")
    xy = tsne.fit_transform(features)
    
    markers = ['o', 's', '^', 'v']
    colors = ['blue', 'red', 'green', 'purple']
    
    plt.figure(figsize=(12, 10))
    for i, cname in enumerate(labels):
        sel = (y_true == i)
        plt.scatter(xy[sel, 0], xy[sel, 1],
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    label=cname, alpha=0.7, s=30)
    
    plt.legend(title="Classes", loc='upper right',
               prop={'weight': 'bold', 'size': 12}, title_fontsize=13)
    plt.xlabel('t-SNE Component 1', fontsize=14, fontweight='bold')
    plt.ylabel('t-SNE Component 2', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figpath, dpi=1000, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {figpath}")

def save_3d_scatter(figpath, features, y_true, labels):
    """3D scatter plot using UMAP/PCA for dimensionality reduction"""
    set_plot_style()
    
    # Use PCA for 3D reduction
    pca = PCA(n_components=3, random_state=42)
    coords_3d = pca.fit_transform(features)
    
    # Color scheme matching your image
    colors = ['green', 'orange', 'blue', 'red']  # IF, MSH, MSS, Normal
    markers = ['s', 's', 's', 's']  # All squares like in the image
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each class
    for i, cname in enumerate(labels):
        sel = (y_true == i)
        ax.scatter(coords_3d[sel, 0], 
                  coords_3d[sel, 1], 
                  coords_3d[sel, 2],
                  c=colors[i % len(colors)],
                  marker=markers[i % len(markers)],
                  label=cname,
                  alpha=0.7,
                  s=50,
                  edgecolors='black',
                  linewidth=0.5)
    
    # Set labels with bold font
    ax.set_xlabel('UMAP1', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('UMAP2', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_zlabel('UMAP3', fontsize=14, fontweight='bold', labelpad=10)
    
    # Legend
    ax.legend(loc='upper right', fontsize=12, frameon=True, 
              prop={'weight': 'bold'})
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(figpath, dpi=1000, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {figpath}")

def save_classification_report(filepath, y_true, y_pred, labels):
    report = classification_report(y_true, y_pred, target_names=labels, digits=4, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"Overall Accuracy: {acc:.4f}\n\n")
        f.write("="*70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(report)
        f.write("\n" + "="*70 + "\n")
    
    print(f"  ✓ Saved: {filepath}")
    return acc

# ---------------------------
# Main Workflow
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="CP Fault Diagnosis - Complete Pipeline with Equal Sampling")
    ap.add_argument("--dataset_name", type=str, required=True, help="Dataset name (e.g., '3bar', '3.5bar', '4bar')")
    ap.add_argument("--data_dir", type=str, required=True, help="Path to dataset with class folders")
    ap.add_argument("--results_dir", type=str, required=True, help="Base results directory")
    ap.add_argument("--samples_per_class", type=int, default=300, help="Number of samples to use per class")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ensemble", type=str, default="soft", choices=["soft", "meta"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--grayscale", action="store_true")
    ap.add_argument("--weak_aug", action="store_true")
    ap.add_argument("--use_class_weights", action="store_true")
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--mixup_alpha", type=float, default=0.2)
    ap.add_argument("--cutmix_alpha", type=float, default=0.2)
    ap.add_argument("--tta", type=int, default=8)
    args = ap.parse_args()
    
    # Setup
    set_seed(args.seed)
    set_plot_style()
    device = device_auto()
    
    print("="*80)
    print(f"CP FAULT DIAGNOSIS - {args.dataset_name.upper()}")
    print(f"EQUAL SAMPLING: {args.samples_per_class} samples per class")
    print("="*80)
    print(f"Device: {device}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Results Directory: {args.results_dir}")
    print("="*80 + "\n")
    
    # Create output directories
    dataset_result_dir = Path(args.results_dir) / args.dataset_name
    checkpoints_dir = dataset_result_dir / "checkpoints"
    figures_dir = dataset_result_dir / "figures"
    reports_dir = dataset_result_dir / "reports"
    
    for d in [checkpoints_dir, figures_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Load data with equal sampling
    print("Loading data with equal sampling per class...")
    train_loader, val_loader, test_loader, idx_to_class, (train_idx, val_idx, test_idx), base_eval = make_loaders(
        args.data_dir, img_size=args.img_size, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed,
        grayscale=args.grayscale, weak_aug=args.weak_aug,
        samples_per_class=args.samples_per_class
    )
    
    num_classes = len(idx_to_class)
    in_ch = 1 if args.grayscale else 3
    class_names_full = [idx_to_class[i] for i in range(num_classes)]
    class_names_short = short_labels_from_full(class_names_full)
    
    print(f"\nClasses: {class_names_short}")
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    
    # Verify class distribution in splits
    print("\nClass distribution in splits:")
    for split_name, split_idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        counts = class_counts_from_indices(base_eval, split_idx)
        print(f"  {split_name}: {counts}")
    
    # Class weights
    class_w = None
    if args.use_class_weights:
        counts = class_counts_from_indices(base_eval, train_idx)
        class_w = compute_class_weights_from_counts(counts, num_classes)
        print(f"\nClass weights: {class_w.tolist()}\n")
    
    # Build models
    print("Building ensemble models...")
    m1 = CNN1_Local(num_classes=num_classes, in_ch=in_ch)
    m2 = CNN2_Global(num_classes=num_classes, in_ch=in_ch)
    m3 = CNN3_Compact(num_classes=num_classes, in_ch=in_ch)
    
    print(f"CNN1_Local params: {count_params(m1):,}")
    print(f"CNN2_Global params: {count_params(m2):,}")
    print(f"CNN3_Compact params: {count_params(m3):,}\n")
    
    # Train models
    print("="*80)
    print("TRAINING MODELS")
    print("="*80)
    
    print("\n[1/3] Training CNN1_Local...")
    m1 = train_one_model(m1, train_loader, val_loader, args.epochs, args.lr, device,
                         checkpoints_dir / "cnn1_local.pt",
                         class_weights=class_w, label_smoothing=args.label_smoothing,
                         mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha)
    
    print("\n[2/3] Training CNN2_Global...")
    m2 = train_one_model(m2, train_loader, val_loader, args.epochs, args.lr, device,
                         checkpoints_dir / "cnn2_global.pt",
                         class_weights=class_w, label_smoothing=args.label_smoothing,
                         mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha)
    
    print("\n[3/3] Training CNN3_Compact...")
    m3 = train_one_model(m3, train_loader, val_loader, args.epochs, args.lr, device,
                         checkpoints_dir / "cnn3_compact.pt",
                         class_weights=class_w, label_smoothing=args.label_smoothing,
                         mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha)
    
    # Predictions
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS")
    print("="*80)
    
    print(f"\nRunning inference with TTA={args.tta}...")
    v1_logits, v1, yv = predict_probs(m1.to(device), val_loader, device, tta=args.tta)
    v2_logits, v2, _ = predict_probs(m2.to(device), val_loader, device, tta=args.tta)
    v3_logits, v3, _ = predict_probs(m3.to(device), val_loader, device, tta=args.tta)
    
    t1_logits, t1, yt = predict_probs(m1.to(device), test_loader, device, tta=args.tta)
    t2_logits, t2, _ = predict_probs(m2.to(device), test_loader, device, tta=args.tta)
    t3_logits, t3, _ = predict_probs(m3.to(device), test_loader, device, tta=args.tta)
    
    # Ensemble
    print("\n" + "="*80)
    print("ENSEMBLE FUSION")
    print("="*80)
    
    if args.ensemble == "soft":
        print("\nUsing soft voting ensemble...")
        val_fused = soft_vote([v1, v2, v3])
        test_fused = soft_vote([t1, t2, t3])
        test_logits_fused = soft_vote([t1_logits, t2_logits, t3_logits])
    else:
        print("\nTraining meta-learner...")
        meta = train_meta_mlp([v1, v2, v3], yv, num_classes=num_classes, device=device)
        
        X_val = np.concatenate([v1, v2, v3], axis=1)
        with torch.no_grad():
            val_logits = meta(torch.tensor(X_val, dtype=torch.float32, device=device)).cpu().numpy()
        val_fused = F.softmax(torch.tensor(val_logits), dim=1).numpy()
        
        X_test = np.concatenate([t1, t2, t3], axis=1)
        with torch.no_grad():
            test_logits = meta(torch.tensor(X_test, dtype=torch.float32, device=device)).cpu().numpy()
        test_fused = F.softmax(torch.tensor(test_logits), dim=1).numpy()
        test_logits_fused = test_logits
        
        torch.save(meta.state_dict(), checkpoints_dir / "meta_mlp.pt")
    
    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION & VISUALIZATION")
    print("="*80)
    
    preds_ensemble = test_fused.argmax(1)
    cm_ensemble = confusion_matrix(yt, preds_ensemble)
    
    print("\nGenerating visualizations...")
    
    # Confusion Matrix
    save_confusion_matrix(
        figures_dir / f"{args.dataset_name}_confusion_matrix.png",
        cm_ensemble,
        class_names_short
    )
    
    # ROC Curve (with diagonal reference line)
    save_roc_curve(
        figures_dir / f"{args.dataset_name}_roc_curve.png",
        yt,
        test_fused,
        class_names_short
    )
    
    # 2D t-SNE
    save_tsne(
        figures_dir / f"{args.dataset_name}_tsne_2d.png",
        test_logits_fused,
        yt,
        class_names_short
    )
    
    # 3D Scatter Plot
    save_3d_scatter(
        figures_dir / f"{args.dataset_name}_3d_scatter.png",
        test_logits_fused,
        yt,
        class_names_short
    )
    
    # Classification Report
    acc = save_classification_report(
        reports_dir / f"{args.dataset_name}_classification_report.txt",
        yt,
        preds_ensemble,
        class_names_short
    )
    
    # Save metadata
    metadata = {
        "dataset_name": args.dataset_name,
        "data_dir": args.data_dir,
        "samples_per_class": args.samples_per_class,
        "num_classes": num_classes,
        "class_names": class_names_short,
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "img_size": args.img_size,
        "epochs": args.epochs,
        "ensemble_type": args.ensemble,
        "tta": args.tta,
        "test_accuracy": float(acc),
        "seed": args.seed
    }
    
    with open(reports_dir / f"{args.dataset_name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save split indices
    with open(checkpoints_dir / "split_indices.json", 'w') as f:
        json.dump({"train": train_idx, "val": val_idx, "test": test_idx}, f, indent=2)
    
    with open(checkpoints_dir / "idx_to_class.json", 'w', encoding='utf-8') as f:
        json.dump(idx_to_class, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\n✓ Test Accuracy: {acc:.4f}")
    print(f"✓ Samples per class: {args.samples_per_class}")
    print(f"✓ Results saved to: {dataset_result_dir}")
    print(f"  - Checkpoints: {checkpoints_dir}")
    print(f"  - Figures: {figures_dir}")
    print(f"  - Reports: {reports_dir}")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()