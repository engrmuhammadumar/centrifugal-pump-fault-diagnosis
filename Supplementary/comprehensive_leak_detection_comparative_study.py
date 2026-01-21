"""
Comprehensive Comparative Study: Pre-trained Models vs Physics-Informed DNN
for Multi-Modal Leak Detection in Water Distribution Systems

For submission to: MSSP, Reliability Engineering and System Safety, Engineering Failure Analysis

Author: Muhammad Umar
University of Ulsan, South Korea
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score, brier_score_loss, cohen_kappa_score,
    matthews_corrcoef, log_loss
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

from scipy import stats
from scipy.stats import ttest_rel, wilcoxon
import time
from thop import profile, clever_format
from collections import defaultdict
import itertools

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Data paths - Update this to match your actual path structure
    BASE_PATH = r"E:\Upwork Project\AI_Leak_Detection_Project\images\cwt_log"
    SENSORS = ['Accelerometer', 'Dynamic Pressure Sensor', 'Hydrophones']
    TOPOLOGY = 'Looped'
    CLASSES = ['Circumferential Crack', 'Gasket Leak', 'Longitudinal Crack', 'No-leak', 'Orifice Leak']
    
    # Results directory
    RESULTS_DIR = r"E:\5 Paper\Results"
    
    # Training configuration
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4
    IMG_SIZE = 224
    N_FOLDS = 5
    N_RUNS = 3  # Multiple runs for statistical significance
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model names
    MODEL_NAMES = [
        'ResNet18',
        'ResNet50',
        'DenseNet121',
        'EfficientNet_B0',
        'MobileNetV3_Large',
        'ShuffleNetV2',
        'ConvNeXt_Tiny',
        'ViT_B16',
        'Swin_T',
        'PI_DNN'  # Your Physics-Informed model
    ]

config = Config()

# Create results directory structure
def create_results_structure():
    """Create organized directory structure for results"""
    dirs = [
        config.RESULTS_DIR,
        f"{config.RESULTS_DIR}/confusion_matrices",
        f"{config.RESULTS_DIR}/roc_curves",
        f"{config.RESULTS_DIR}/tsne_plots",
        f"{config.RESULTS_DIR}/calibration_curves",
        f"{config.RESULTS_DIR}/statistical_analysis",
        f"{config.RESULTS_DIR}/models_checkpoints",
        f"{config.RESULTS_DIR}/classification_reports",
        f"{config.RESULTS_DIR}/performance_metrics",
        f"{config.RESULTS_DIR}/comparison_tables",
        f"{config.RESULTS_DIR}/sensor_ablation",
        f"{config.RESULTS_DIR}/figures_for_paper",
        f"{config.RESULTS_DIR}/latex_tables",
    ]
    
    for sensor_combo in ['single_sensor', 'two_sensor_fusion', 'three_sensor_fusion']:
        for d in dirs:
            Path(f"{d}/{sensor_combo}").mkdir(parents=True, exist_ok=True)
    
    print("✓ Results directory structure created")

create_results_structure()

# ============================================================================
# DATASET CLASS
# ============================================================================

class LeakDetectionDataset(Dataset):
    """Multi-modal dataset for leak detection"""
    
    def __init__(self, data_info, transform=None, sensors_to_use=['Accelerometer']):
        """
        Args:
            data_info: List of dictionaries with 'paths' (dict of sensor paths) and 'label'
            transform: Image transformations
            sensors_to_use: List of sensors to include
        """
        self.data_info = data_info
        self.transform = transform
        self.sensors_to_use = sensors_to_use
        
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        item = self.data_info[idx]
        images = []
        
        # Load images from each sensor
        for sensor in self.sensors_to_use:
            img_path = item['paths'][sensor]
            img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            
            images.append(img)
        
        # Stack images along channel dimension if multiple sensors
        if len(images) > 1:
            image = torch.cat(images, dim=0)  # Concatenate along channel dimension
        else:
            image = images[0]
        
        label = item['label']
        
        return image, label

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_dataset_paths(base_path, sensors, topology, classes):
    """Load all image paths with labels"""
    data_info = []
    
    print(f"\n🔍 Searching for images...")
    print(f"   Base path: {base_path}")
    print(f"   Topology: {topology}")
    print(f"   Sensors: {sensors}")
    
    # Check if base path exists
    if not os.path.exists(base_path):
        print(f"❌ ERROR: Base path does not exist: {base_path}")
        print(f"   Please check your path and update BASE_PATH in Config class")
        return data_info
    
    # Get all images from first sensor to establish the dataset
    first_sensor = sensors[0]
    sensor_path = os.path.join(base_path, first_sensor, topology)
    
    print(f"   First sensor path: {sensor_path}")
    
    if not os.path.exists(sensor_path):
        print(f"❌ ERROR: Sensor path does not exist: {sensor_path}")
        print(f"   Available items in base_path:")
        try:
            for item in os.listdir(base_path):
                print(f"     - {item}")
        except Exception as e:
            print(f"     Error listing directory: {e}")
        return data_info
    
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(sensor_path, class_name)
        
        if not os.path.exists(class_path):
            print(f"⚠️  Warning: Class path does not exist: {class_path}")
            print(f"   Available classes in sensor path:")
            try:
                for item in os.listdir(sensor_path):
                    print(f"     - {item}")
            except Exception as e:
                print(f"     Error listing directory: {e}")
            continue
        
        images = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"   Found {len(images)} images for class: {class_name}")
        
        for img_name in images:
            # Create paths dict for all sensors
            paths_dict = {}
            all_exist = True
            
            for sensor in sensors:
                sensor_img_path = os.path.join(base_path, sensor, topology, class_name, img_name)
                if os.path.exists(sensor_img_path):
                    paths_dict[sensor] = sensor_img_path
                else:
                    all_exist = False
                    break
            
            # Only add if all sensors have the image
            if all_exist and len(paths_dict) == len(sensors):
                data_info.append({
                    'paths': paths_dict,
                    'label': class_idx,
                    'class_name': class_name
                })
    
    print(f"\n✓ Total valid samples loaded: {len(data_info)}")
    
    # Show distribution
    if len(data_info) > 0:
        print(f"\n📊 Class distribution:")
        class_counts = {}
        for item in data_info:
            class_name = item['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in class_counts.items():
            print(f"   {class_name}: {count} samples")
    
    return data_info

def get_data_transforms():
    """Get training and validation transforms"""
    
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class MultiModalWrapper(nn.Module):
    """Wrapper for handling multi-modal inputs"""
    
    def __init__(self, base_model, num_sensors, num_classes):
        super().__init__()
        self.num_sensors = num_sensors
        self.base_model = base_model
        
        # Modify first conv layer to accept multi-modal input
        if num_sensors > 1:
            if hasattr(base_model, 'conv1'):
                # For ResNet, DenseNet
                original_conv = base_model.conv1
                self.base_model.conv1 = nn.Conv2d(
                    3 * num_sensors, 
                    original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=False
                )
            elif hasattr(base_model, 'features') and hasattr(base_model.features[0], '0'):
                # For EfficientNet, MobileNet
                original_conv = base_model.features[0][0]
                base_model.features[0][0] = nn.Conv2d(
                    3 * num_sensors,
                    original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=False
                )
            elif hasattr(base_model, 'stem'):
                # For ConvNeXt
                original_conv = base_model.stem[0]
                base_model.stem[0] = nn.Conv2d(
                    3 * num_sensors,
                    original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding
                )
            elif hasattr(base_model, 'patch_embed'):
                # For ViT and Swin
                if hasattr(base_model.patch_embed, 'proj'):
                    original_conv = base_model.patch_embed.proj
                    base_model.patch_embed.proj = nn.Conv2d(
                        3 * num_sensors,
                        original_conv.out_channels,
                        kernel_size=original_conv.kernel_size,
                        stride=original_conv.stride,
                        padding=original_conv.padding
                    )
    
    def forward(self, x):
        return self.base_model(x)

def get_model(model_name, num_classes, num_sensors=1, pretrained=True):
    """Get model by name"""
    
    if model_name == 'ResNet18':
        base_model = models.resnet18(pretrained=pretrained)
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
        
    elif model_name == 'ResNet50':
        base_model = models.resnet50(pretrained=pretrained)
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
        
    elif model_name == 'DenseNet121':
        base_model = models.densenet121(pretrained=pretrained)
        base_model.classifier = nn.Linear(base_model.classifier.in_features, num_classes)
        
    elif model_name == 'EfficientNet_B0':
        base_model = models.efficientnet_b0(pretrained=pretrained)
        base_model.classifier[1] = nn.Linear(base_model.classifier[1].in_features, num_classes)
        
    elif model_name == 'MobileNetV3_Large':
        base_model = models.mobilenet_v3_large(pretrained=pretrained)
        base_model.classifier[3] = nn.Linear(base_model.classifier[3].in_features, num_classes)
        
    elif model_name == 'ShuffleNetV2':
        base_model = models.shufflenet_v2_x1_0(pretrained=pretrained)
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
        
    elif model_name == 'ConvNeXt_Tiny':
        base_model = models.convnext_tiny(pretrained=pretrained)
        base_model.classifier[2] = nn.Linear(base_model.classifier[2].in_features, num_classes)
        
    elif model_name == 'ViT_B16':
        base_model = models.vit_b_16(pretrained=pretrained)
        base_model.heads.head = nn.Linear(base_model.heads.head.in_features, num_classes)
        
    elif model_name == 'Swin_T':
        base_model = models.swin_t(pretrained=pretrained)
        base_model.head = nn.Linear(base_model.head.in_features, num_classes)
        
    elif model_name == 'PI_DNN':
        # Physics-Informed Deep Neural Network (simplified version)
        base_model = PhysicsInformedDNN(num_classes, num_sensors)
        return base_model  # Don't wrap, already handles multi-modal
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Wrap with multi-modal handler
    model = MultiModalWrapper(base_model, num_sensors, num_classes)
    
    return model

class PhysicsInformedDNN(nn.Module):
    """Physics-Informed Deep Neural Network for Leak Detection"""
    
    def __init__(self, num_classes, num_sensors):
        super().__init__()
        
        # Sensor-specific feature extractors
        self.sensor_encoders = nn.ModuleList([
            self._make_encoder() for _ in range(num_sensors)
        ])
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(512 * num_sensors, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classifier = nn.Linear(256, num_classes)
        
        # Physics-aware attention (simplified)
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        
    def _make_encoder(self):
        """Create a CNN encoder for each sensor"""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            
            self._make_layer(64, 128, 2),
            self._make_layer(128, 256, 2),
            self._make_layer(256, 512, 2),
            
            nn.AdaptiveAvgPool2d(1)
        )
    
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, 2, 1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        
        for _ in range(blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Split input by sensors (3 channels per sensor)
        batch_size = x.size(0)
        num_sensors = x.size(1) // 3
        
        features = []
        for i in range(num_sensors):
            sensor_input = x[:, i*3:(i+1)*3, :, :]
            feat = self.sensor_encoders[i](sensor_input)
            feat = feat.view(batch_size, -1)
            features.append(feat)
        
        # Apply attention across sensors
        if len(features) > 1:
            stacked_features = torch.stack(features, dim=0)  # (num_sensors, batch, 512)
            attended_features, _ = self.attention(stacked_features, stacked_features, stacked_features)
            fused = torch.cat([attended_features[i] for i in range(num_sensors)], dim=1)
        else:
            fused = features[0]
        
        # Fusion and classification
        fused = self.fusion(fused)
        output = self.classifier(fused)
        
        return output

# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def evaluate_model(model, dataloader, criterion, device, return_predictions=False):
    """Evaluate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    if return_predictions:
        return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    return epoch_loss, epoch_acc

def calculate_flops_params(model, input_size, device):
    """Calculate FLOPs and parameters"""
    model.eval()
    dummy_input = torch.randn(1, *input_size).to(device)
    
    try:
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        return flops, params
    except:
        # Fallback: count parameters only
        params = sum(p.numel() for p in model.parameters())
        return "N/A", f"{params/1e6:.2f}M"

def measure_inference_time(model, dataloader, device, num_iterations=100):
    """Measure inference latency"""
    model.eval()
    times = []
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_iterations:
                break
            
            inputs = inputs.to(device)
            
            # Warm up
            if i < 10:
                _ = model(inputs)
                continue
            
            # Measure
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            _ = model(inputs)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.time()
            
            times.append((end - start) / inputs.size(0))  # Per sample
    
    return np.mean(times) * 1000  # Convert to ms

# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_all_metrics(y_true, y_pred, y_prob, class_names):
    """Calculate comprehensive metrics"""
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
    metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
    metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Additional metrics
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # Binarize labels for multi-class ROC-AUC and Brier Score
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    # ROC-AUC (One-vs-Rest)
    try:
        metrics['roc_auc_macro'] = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
        metrics['roc_auc_weighted'] = roc_auc_score(y_true_bin, y_prob, average='weighted', multi_class='ovr')
    except:
        metrics['roc_auc_macro'] = 0.0
        metrics['roc_auc_weighted'] = 0.0
    
    # Brier Score (multi-class)
    try:
        brier_scores = []
        for i in range(len(class_names)):
            bs = brier_score_loss(y_true_bin[:, i], y_prob[:, i])
            brier_scores.append(bs)
        metrics['brier_score_mean'] = np.mean(brier_scores)
        metrics['brier_score_per_class'] = brier_scores
    except:
        metrics['brier_score_mean'] = 0.0
        metrics['brier_score_per_class'] = [0.0] * len(class_names)
    
    # Log Loss
    try:
        metrics['log_loss'] = log_loss(y_true, y_prob)
    except:
        metrics['log_loss'] = 0.0
    
    # Expected Calibration Error (ECE)
    try:
        ece = calculate_ece(y_true, y_prob)
        metrics['ece'] = ece
    except:
        metrics['ece'] = 0.0
    
    # Confusion Matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics

def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error"""
    predictions = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    accuracies = (predictions == y_true)
    
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1
    
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            avg_confidence = np.mean(confidences[mask])
            avg_accuracy = np.mean(accuracies[mask])
            ece += np.abs(avg_confidence - avg_accuracy) * np.sum(mask)
    
    ece /= len(y_true)
    return ece

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_confusion_matrix(cm, class_names, save_path, title='Confusion Matrix'):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(y_true, y_prob, class_names, save_path, title='ROC Curves'):
    """Plot ROC curves for multi-class"""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    # Plot ROC curve for each class
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, lw=2, 
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_tsne(features, labels, class_names, save_path, title='t-SNE Visualization'):
    """Plot t-SNE visualization"""
    
    # Reduce to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        mask = labels == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[color], label=class_name, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    plt.xlabel('t-SNE Component 1', fontsize=14, fontweight='bold')
    plt.ylabel('t-SNE Component 2', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def extract_features_for_tsne(model, dataloader, device):
    """Extract features from penultimate layer for t-SNE"""
    model.eval()
    features = []
    labels = []
    
    def hook_fn(module, input, output):
        features.append(output.detach().cpu().numpy())
    
    # Register hook on the layer before classifier
    if hasattr(model, 'base_model'):
        if hasattr(model.base_model, 'fc'):
            handle = model.base_model.avgpool.register_forward_hook(hook_fn)
        elif hasattr(model.base_model, 'classifier'):
            if isinstance(model.base_model.classifier, nn.Sequential):
                handle = model.base_model.classifier[-2].register_forward_hook(hook_fn)
            else:
                handle = model.base_model.features.register_forward_hook(hook_fn)
        elif hasattr(model.base_model, 'head'):
            handle = model.base_model.norm.register_forward_hook(hook_fn)
        else:
            handle = model.base_model.register_forward_hook(hook_fn)
    else:
        handle = model.fusion.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for inputs, lbls in dataloader:
            inputs = inputs.to(device)
            _ = model(inputs)
            labels.extend(lbls.numpy())
    
    handle.remove()
    
    # Concatenate all features
    features = np.concatenate(features, axis=0)
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
    
    return features, np.array(labels)

def plot_calibration_curve(y_true, y_prob, n_bins, save_path, title='Calibration Curve'):
    """Plot calibration curve"""
    predictions = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(int)
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        accuracies, confidences, n_bins=n_bins, strategy='uniform'
    )
    
    plt.figure(figsize=(10, 8))
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model', linewidth=2, markersize=8)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    
    plt.xlabel('Mean Predicted Probability', fontsize=14, fontweight='bold')
    plt.ylabel('Fraction of Positives', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=12)
    plt.grid(alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# COMPARISON AND STATISTICAL ANALYSIS
# ============================================================================

def perform_statistical_tests(results_dict, metric_name='accuracy'):
    """Perform statistical significance tests between models"""
    
    model_names = list(results_dict.keys())
    n_models = len(model_names)
    
    # Get metric values for all models across runs
    metric_values = {model: results_dict[model][metric_name] for model in model_names}
    
    # Pairwise t-tests
    pairwise_pvalues = np.ones((n_models, n_models))
    
    for i in range(n_models):
        for j in range(i+1, n_models):
            model_i = model_names[i]
            model_j = model_names[j]
            
            values_i = metric_values[model_i]
            values_j = metric_values[model_j]
            
            # Paired t-test
            _, p_value = ttest_rel(values_i, values_j)
            pairwise_pvalues[i, j] = p_value
            pairwise_pvalues[j, i] = p_value
    
    return pairwise_pvalues, model_names

def create_comparison_table(all_results, sensor_config):
    """Create comprehensive comparison table"""
    
    rows = []
    
    for model_name in config.MODEL_NAMES:
        if model_name not in all_results:
            continue
        
        results = all_results[model_name]
        
        row = {
            'Model': model_name,
            'Accuracy (%)': f"{np.mean(results['accuracy'])*100:.2f} ± {np.std(results['accuracy'])*100:.2f}",
            'Precision (%)': f"{np.mean(results['precision_macro'])*100:.2f} ± {np.std(results['precision_macro'])*100:.2f}",
            'Recall (%)': f"{np.mean(results['recall_macro'])*100:.2f} ± {np.std(results['recall_macro'])*100:.2f}",
            'F1-Score (%)': f"{np.mean(results['f1_macro'])*100:.2f} ± {np.std(results['f1_macro'])*100:.2f}",
            'ROC-AUC': f"{np.mean(results['roc_auc_macro']):.4f} ± {np.std(results['roc_auc_macro']):.4f}",
            'Brier Score': f"{np.mean(results['brier_score_mean']):.4f} ± {np.std(results['brier_score_mean']):.4f}",
            'ECE': f"{np.mean(results['ece']):.4f} ± {np.std(results['ece']):.4f}",
            'Cohen Kappa': f"{np.mean(results['cohen_kappa']):.4f} ± {np.std(results['cohen_kappa']):.4f}",
            'MCC': f"{np.mean(results['mcc']):.4f} ± {np.std(results['mcc']):.4f}",
            'Latency (ms)': f"{np.mean(results['latency']):.2f} ± {np.std(results['latency']):.2f}",
            'Parameters': results['params'][0],
            'FLOPs': results['flops'][0]
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    return df

def plot_model_comparison_bar(all_results, metric_name, ylabel, save_path, title):
    """Plot bar chart comparing models"""
    
    model_names = []
    means = []
    stds = []
    
    for model_name in config.MODEL_NAMES:
        if model_name not in all_results:
            continue
        
        results = all_results[model_name]
        model_names.append(model_name)
        means.append(np.mean(results[metric_name]))
        stds.append(np.std(results[metric_name]))
    
    # Sort by mean value
    sorted_indices = np.argsort(means)[::-1]
    model_names = [model_names[i] for i in sorted_indices]
    means = [means[i] for i in sorted_indices]
    stds = [stds[i] for i in sorted_indices]
    
    # Plot
    plt.figure(figsize=(14, 8))
    colors = ['#d32f2f' if name == 'PI_DNN' else '#1976d2' for name in model_names]
    
    x = np.arange(len(model_names))
    plt.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    plt.xlabel('Model', fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_latex_table(df, caption, label, save_path):
    """Generate LaTeX table from dataframe"""
    
    latex_str = df.to_latex(index=False, escape=False, column_format='l' + 'c'*(len(df.columns)-1))
    
    # Add caption and label
    latex_str = latex_str.replace('\\begin{tabular}', 
                                  f'\\caption{{{caption}}}\\label{{{label}}}\n\\begin{{tabular}}')
    
    # Wrap in table environment
    latex_str = '\\begin{table}[htbp]\n\\centering\n' + latex_str + '\n\\end{table}'
    
    # Save
    with open(save_path, 'w') as f:
        f.write(latex_str)
    
    print(f"✓ LaTeX table saved: {save_path}")

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_and_evaluate_model(model_name, train_loader, val_loader, num_classes, 
                             num_sensors, device, sensor_config):
    """Complete training and evaluation pipeline for one model"""
    
    print(f"\n{'='*80}")
    print(f"Training {model_name} with {sensor_config}")
    print(f"{'='*80}")
    
    # Get model
    model = get_model(model_name, num_classes, num_sensors, pretrained=True)
    model = model.to(device)
    
    # Calculate FLOPs and parameters
    input_size = (3 * num_sensors, config.IMG_SIZE, config.IMG_SIZE)
    flops, params = calculate_flops_params(model, input_size, device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, 
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    
    # Training
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    
    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    val_loss, val_acc, y_true, y_pred, y_prob = evaluate_model(
        model, val_loader, criterion, device, return_predictions=True
    )
    
    # Calculate all metrics
    metrics = calculate_all_metrics(y_true, y_pred, y_prob, config.CLASSES)
    
    # Measure latency
    latency = measure_inference_time(model, val_loader, device)
    
    print(f"\n{model_name} Final Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Score: {metrics['f1_macro']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc_macro']:.4f}")
    print(f"  Latency: {latency:.2f} ms")
    print(f"  Parameters: {params}")
    print(f"  FLOPs: {flops}")
    
    return {
        'model': model,
        'metrics': metrics,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'latency': latency,
        'params': params,
        'flops': flops
    }

def run_complete_study():
    """Run complete comparative study"""
    
    print("="*80)
    print("COMPREHENSIVE COMPARATIVE STUDY FOR MULTI-MODAL LEAK DETECTION")
    print("="*80)
    
    # Load dataset
    print("\n📁 Loading dataset...")
    data_info = load_dataset_paths(config.BASE_PATH, config.SENSORS, config.TOPOLOGY, config.CLASSES)
    print(f"✓ Total samples: {len(data_info)}")
    print(f"✓ Classes: {config.CLASSES}")
    
    # Get transforms
    train_transform, val_transform = get_data_transforms()
    
    # Sensor configurations to test
    sensor_configs = {
        'single_sensor': [
            (['Accelerometer'], 'Accelerometer'),
            (['Dynamic Pressure Sensor'], 'Dynamic Pressure Sensor'),
            (['Hydrophones'], 'Hydrophones')
        ],
        'two_sensor_fusion': [
            (['Accelerometer', 'Dynamic Pressure Sensor'], 'Acc+Pressure'),
            (['Accelerometer', 'Hydrophones'], 'Acc+Hydrophone'),
            (['Dynamic Pressure Sensor', 'Hydrophones'], 'Pressure+Hydrophone')
        ],
        'three_sensor_fusion': [
            (['Accelerometer', 'Dynamic Pressure Sensor', 'Hydrophones'], 'All Sensors')
        ]
    }
    
    # Store all results
    all_results = {}
    
    # Run for each sensor configuration
    for config_type, sensor_list in sensor_configs.items():
        print(f"\n{'#'*80}")
        print(f"SENSOR CONFIGURATION: {config_type.upper().replace('_', ' ')}")
        print(f"{'#'*80}")
        
        for sensors, sensor_name in sensor_list:
            print(f"\n{'='*80}")
            print(f"Testing with sensors: {sensor_name}")
            print(f"{'='*80}")
            
            num_sensors = len(sensors)
            
            # Results storage for this config
            config_results = {model_name: defaultdict(list) for model_name in config.MODEL_NAMES}
            
            # Multiple runs for statistical significance
            for run in range(config.N_RUNS):
                print(f"\n🔄 Run {run+1}/{config.N_RUNS}")
                
                # Split data
                train_data, val_data = train_test_split(
                    data_info, test_size=0.2, stratify=[d['label'] for d in data_info], 
                    random_state=42 + run
                )
                
                # Create datasets
                train_dataset = LeakDetectionDataset(train_data, train_transform, sensors)
                val_dataset = LeakDetectionDataset(val_data, val_transform, sensors)
                
                # Create dataloaders
                train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                                        shuffle=True, num_workers=config.NUM_WORKERS, 
                                        pin_memory=True)
                val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                                       shuffle=False, num_workers=config.NUM_WORKERS, 
                                       pin_memory=True)
                
                # Train and evaluate each model
                for model_name in config.MODEL_NAMES:
                    try:
                        result = train_and_evaluate_model(
                            model_name, train_loader, val_loader, 
                            len(config.CLASSES), num_sensors, config.DEVICE, sensor_name
                        )
                        
                        # Store metrics
                        for key, value in result['metrics'].items():
                            if isinstance(value, np.ndarray):
                                continue  # Skip arrays like confusion matrix
                            config_results[model_name][key].append(value)
                        
                        config_results[model_name]['latency'].append(result['latency'])
                        config_results[model_name]['params'].append(result['params'])
                        config_results[model_name]['flops'].append(result['flops'])
                        
                        # Save visualizations (only for last run)
                        if run == config.N_RUNS - 1:
                            base_path = f"{config.RESULTS_DIR}/{config_type}/{sensor_name.replace('+', '_')}"
                            
                            # Confusion Matrix
                            cm_path = f"{base_path}/confusion_matrices/{model_name}_confusion_matrix.png"
                            os.makedirs(os.path.dirname(cm_path), exist_ok=True)
                            plot_confusion_matrix(result['metrics']['confusion_matrix'], 
                                                config.CLASSES, cm_path, 
                                                f'{model_name} - Confusion Matrix ({sensor_name})')
                            
                            # ROC Curves
                            roc_path = f"{base_path}/roc_curves/{model_name}_roc_curves.png"
                            os.makedirs(os.path.dirname(roc_path), exist_ok=True)
                            plot_roc_curves(result['y_true'], result['y_prob'], 
                                          config.CLASSES, roc_path, 
                                          f'{model_name} - ROC Curves ({sensor_name})')
                            
                            # t-SNE
                            features, labels = extract_features_for_tsne(result['model'], val_loader, config.DEVICE)
                            tsne_path = f"{base_path}/tsne_plots/{model_name}_tsne.png"
                            os.makedirs(os.path.dirname(tsne_path), exist_ok=True)
                            plot_tsne(features, labels, config.CLASSES, tsne_path, 
                                    f'{model_name} - t-SNE ({sensor_name})')
                            
                            # Calibration Curve
                            cal_path = f"{base_path}/calibration_curves/{model_name}_calibration.png"
                            os.makedirs(os.path.dirname(cal_path), exist_ok=True)
                            plot_calibration_curve(result['y_true'], result['y_prob'], 10, cal_path,
                                                 f'{model_name} - Calibration Curve ({sensor_name})')
                            
                            # Classification Report
                            report = classification_report(result['y_true'], result['y_pred'], 
                                                          target_names=config.CLASSES, digits=4)
                            report_path = f"{base_path}/classification_reports/{model_name}_report.txt"
                            os.makedirs(os.path.dirname(report_path), exist_ok=True)
                            with open(report_path, 'w') as f:
                                f.write(f"Classification Report - {model_name} ({sensor_name})\n")
                                f.write("="*80 + "\n\n")
                                f.write(report)
                            
                            print(f"  ✓ Saved visualizations for {model_name}")
                    
                    except Exception as e:
                        print(f"  ✗ Error training {model_name}: {str(e)}")
                        continue
            
            # Save comparison table
            comparison_df = create_comparison_table(config_results, sensor_name)
            table_path = f"{config.RESULTS_DIR}/{config_type}/comparison_tables/{sensor_name.replace('+', '_')}_comparison.csv"
            os.makedirs(os.path.dirname(table_path), exist_ok=True)
            comparison_df.to_csv(table_path, index=False)
            print(f"\n✓ Comparison table saved: {table_path}")
            
            # Generate LaTeX table
            latex_path = f"{config.RESULTS_DIR}/{config_type}/latex_tables/{sensor_name.replace('+', '_')}_table.tex"
            os.makedirs(os.path.dirname(latex_path), exist_ok=True)
            generate_latex_table(comparison_df, 
                               f"Performance Comparison - {sensor_name}",
                               f"tab:{sensor_name.lower().replace('+', '_').replace(' ', '_')}",
                               latex_path)
            
            # Statistical tests
            pvalues, model_names = perform_statistical_tests(config_results, 'accuracy')
            
            # Save p-values
            pvalue_df = pd.DataFrame(pvalues, index=model_names, columns=model_names)
            pvalue_path = f"{config.RESULTS_DIR}/{config_type}/statistical_analysis/{sensor_name.replace('+', '_')}_pvalues.csv"
            os.makedirs(os.path.dirname(pvalue_path), exist_ok=True)
            pvalue_df.to_csv(pvalue_path)
            print(f"✓ Statistical test results saved: {pvalue_path}")
            
            # Comparison plots
            metrics_to_plot = [
                ('accuracy', 'Accuracy', 'Accuracy'),
                ('f1_macro', 'F1-Score (Macro)', 'F1-Score'),
                ('roc_auc_macro', 'ROC-AUC (Macro)', 'ROC-AUC'),
                ('latency', 'Inference Latency (ms)', 'Latency')
            ]
            
            for metric, ylabel, name in metrics_to_plot:
                plot_path = f"{config.RESULTS_DIR}/{config_type}/figures_for_paper/{sensor_name.replace('+', '_')}_{name.lower()}_comparison.png"
                os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                plot_model_comparison_bar(config_results, metric, ylabel, plot_path,
                                        f'{name} Comparison - {sensor_name}')
            
            print(f"✓ Comparison plots saved")
            
            # Store for overall analysis
            all_results[f"{config_type}_{sensor_name}"] = config_results
    
    # Create overall summary
    print(f"\n{'='*80}")
    print("GENERATING OVERALL SUMMARY")
    print(f"{'='*80}")
    
    # Summary across all configurations
    summary_rows = []
    
    for config_key, results in all_results.items():
        config_type, sensor_name = config_key.rsplit('_', 1)
        
        for model_name, metrics in results.items():
            if not metrics:
                continue
            
            summary_rows.append({
                'Configuration': config_type.replace('_', ' ').title(),
                'Sensors': sensor_name,
                'Model': model_name,
                'Accuracy (%)': f"{np.mean(metrics['accuracy'])*100:.2f}",
                'F1-Score': f"{np.mean(metrics['f1_macro']):.4f}",
                'ROC-AUC': f"{np.mean(metrics['roc_auc_macro']):.4f}",
                'Latency (ms)': f"{np.mean(metrics['latency']):.2f}",
                'Parameters': metrics['params'][0]
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = f"{config.RESULTS_DIR}/OVERALL_SUMMARY.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Overall summary saved: {summary_path}")
    
    # Create Excel workbook with all results
    excel_path = f"{config.RESULTS_DIR}/COMPLETE_RESULTS.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Overall Summary', index=False)
        
        for config_key, results in all_results.items():
            sheet_name = config_key.replace('_', ' ')[:31]  # Excel sheet name limit
            df = create_comparison_table(results, config_key)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"✓ Complete results Excel file saved: {excel_path}")
    
    print(f"\n{'='*80}")
    print("✅ COMPARATIVE STUDY COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"\nAll results saved in: {config.RESULTS_DIR}")
    print(f"\nKey outputs:")
    print(f"  • Confusion matrices: {config.RESULTS_DIR}/*/confusion_matrices/")
    print(f"  • ROC curves: {config.RESULTS_DIR}/*/roc_curves/")
    print(f"  • t-SNE plots: {config.RESULTS_DIR}/*/tsne_plots/")
    print(f"  • Calibration curves: {config.RESULTS_DIR}/*/calibration_curves/")
    print(f"  • Classification reports: {config.RESULTS_DIR}/*/classification_reports/")
    print(f"  • Comparison tables: {config.RESULTS_DIR}/*/comparison_tables/")
    print(f"  • Statistical analysis: {config.RESULTS_DIR}/*/statistical_analysis/")
    print(f"  • LaTeX tables: {config.RESULTS_DIR}/*/latex_tables/")
    print(f"  • Figures for paper: {config.RESULTS_DIR}/*/figures_for_paper/")
    print(f"  • Overall summary: {summary_path}")
    print(f"  • Complete results: {excel_path}")

# ============================================================================
# RUN THE COMPLETE STUDY
# ============================================================================

if __name__ == "__main__":
    run_complete_study()