import os
import pickle
import shutil
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import onnx
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
# Path to the pickle file inside the LSWMD.pkl directory
DATA_PKL_PATH = r"c:\Users\hp\Downloads\IESA Hackathon\LSWMD.pkl\LSWMD.pkl" # Updated based on list_dir output
OUTPUT_ROOT = "output"
DATASET_DIR = os.path.join(OUTPUT_ROOT, "Dataset")
MODELS_DIR = os.path.join(OUTPUT_ROOT, "models")
RESULTS_DIR = os.path.join(OUTPUT_ROOT, "results")

# Requirements
IMG_SIZE = 224
BATCH_SIZE = 32
LR = 3e-4
EPOCHS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class Mapping
CLASS_MAPPING = {
    'Center': 'center',
    'Donut': 'donut',
    'Edge-Loc': 'edge_loc',
    'Edge-Ring': 'edge_ring',
    'Loc': 'loc',
    'Scratch': 'scratch',
    'Random': 'other',
    'Near-full': 'other',
    'none': 'clean'
}

TARGET_CLASSES = ['clean', 'other', 'center', 'donut', 'edge_loc', 'edge_ring', 'loc', 'scratch']
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(TARGET_CLASSES)}

def setup_directories():
    print("[INFO] Setting up output directories...")
    if os.path.exists(OUTPUT_ROOT):
        print(f"[WARN] Output directory {OUTPUT_ROOT} already exists. Attempting clean up...")
        try:
            shutil.rmtree(OUTPUT_ROOT)
        except OSError as e:
            print(f"[WARN] Could not fully delete output directory: {e}. Continuing with overwrite...")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    for split in ['Train', 'Validation', 'Test']:
        for cls_name in TARGET_CLASSES:
            os.makedirs(os.path.join(DATASET_DIR, split, cls_name), exist_ok=True)

def load_and_process_dataframe():
    print(f"[INFO] Loading {DATA_PKL_PATH}...")
    if not os.path.exists(DATA_PKL_PATH):
        raise FileNotFoundError(f"Could not find {DATA_PKL_PATH}")

    try:
        # Robust hack for legacy pandas pickles
        import sys
        import types
        from pandas import Index, RangeIndex
        
        # Helper to create/patch modules
        def patch_module(name):
            if name not in sys.modules:
                m = types.ModuleType(name)
                m.__path__ = [] # Mark as package so submodules can be imported
                sys.modules[name] = m
            return sys.modules[name]

        # Patch pandas.indexes
        p_indexes = patch_module('pandas.indexes')
        p_indexes_base = patch_module('pandas.indexes.base')
        p_indexes_numeric = patch_module('pandas.indexes.numeric')
        p_indexes_range = patch_module('pandas.indexes.range')

        # Helper for _new_Index
        def _new_Index(cls, d):
            return cls.__new__(cls, **d)

        # Map removed classes to valid ones
        p_indexes.Index = Index
        p_indexes.base = p_indexes_base
        p_indexes_base.Index = Index
        p_indexes_base._new_Index = _new_Index
        
        p_indexes_numeric.Int64Index = Index
        p_indexes_numeric.Float64Index = Index
        
        p_indexes_range.RangeIndex = RangeIndex

        # Also patch pandas.core.indexes.numeric if it's missing (it was removed in pandas 2.0)
        p_core_indexes = patch_module('pandas.core.indexes')
        p_core_indexes_base = patch_module('pandas.core.indexes.base')
        p_core_numeric = patch_module('pandas.core.indexes.numeric')
        p_core_range = patch_module('pandas.core.indexes.range')
        
        p_core_indexes_base._new_Index = _new_Index
        p_core_numeric.Int64Index = Index
        p_core_numeric.Float64Index = Index
        p_core_range.RangeIndex = RangeIndex

        # Use latin1 encoding for legacy python 2 pickle compatibility
        with open(DATA_PKL_PATH, 'rb') as f:
            df = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"[ERROR] Failed to load pickle: {e}")
        return None

    print(f"[INFO] Loaded dataframe with shape: {df.shape}")
    
    # Extract needed columns. Adjust column names if they differ slightly (LSWMD usually: waferMap, dieSize, lotName, waferIndex, trianTestLabel, failureType)
    # failureType is typically a numpy array of shape (1,1) or list. Need to extract string.
    
    print("[INFO] Processing labels...")
    
    def extract_failure_type(x):
        if isinstance(x, np.ndarray):
            if x.size > 0:
                val = x[0][0]
                return val if isinstance(val, str) else 'none'
        return 'none' # Default to none if empty or weird

    # Check extracted failure type column 'failureType'
    if 'failureType' in df.columns:
        # It's often nested.
        df['label_raw'] = df['failureType'].apply(extract_failure_type)
    else:
        print("[ERROR] 'failureType' column missing.")
        return None

    # Map labels
    print("[INFO] Mapping labels to 8 classes...")
    df['label'] = df['label_raw'].apply(lambda x: CLASS_MAPPING.get(x, 'other'))
    
    # Handle NaN or 'none' explicitly again if mapping failed or original was NaN
    # User said: Label "none" or NaN = clean
    # The mapping above 'none' -> 'clean', but we should catch any missed NaNs
    df['label'] = df['label'].fillna('clean')
    
    # Check class distribution
    print("Class distribution:\n", df['label'].value_counts())
    
    # Remove corrupted/empty wafer maps
    # Wafer map should be a 2D array
    print("[INFO] Filtering valid wafer maps...")
    
    def is_valid_wafer(x):
        return isinstance(x, np.ndarray) and x.ndim == 2 and x.size > 0

    df = df[df['waferMap'].apply(is_valid_wafer)].copy()
    
    return df

def generate_images(df):
    print("[INFO] Generating images and splitting dataset...")
    
    # Stratified Split
    # Since we need Train/Val/Test = 70/15/15
    # First split: Train (70%) vs Temp (30%)
    # Second split: Val (50% of Temp -> 15%) vs Test (50% of Temp -> 15%)
    
    X = df.index.values
    y = df['label'].values
    
    # Add index to dataframe to track easily
    df['original_index'] = df.index
    
    try:
        X_train_idx, X_temp_idx, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        X_val_idx, X_test_idx, y_val, y_test = train_test_split(
            X_temp_idx, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )
    except ValueError as e:
        print(f"[WARN] Stratified split failed (maybe too few samples for some class). Fallback to random split. Error: {e}")
        X_train_idx, X_temp_idx, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val_idx, X_test_idx, y_val, y_test = train_test_split(X_temp_idx, y_temp, test_size=0.5, random_state=42)

    split_map = {}
    for idx in X_train_idx: split_map[idx] = 'Train'
    for idx in X_val_idx: split_map[idx] = 'Validation'
    for idx in X_test_idx: split_map[idx] = 'Test'
    
    # Only process rows that are part of the split (effectively all, but good to be explicit)
    count = 0
    
    print(f"[INFO] Processing {len(df)} images. This may take a while...")
    
    # Use tqdm for progress
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if idx not in split_map: continue
        
        split_name = split_map[idx]
        label = row['label']
        wm = row['waferMap']
        
        # Preprocessing
        # 1. Resize to 224x224
        # Wafer maps are often categorical (0,1,2).
        # We want to treat as image.
        try:
            # Convert to float for resizing interpolation or keep int for nearest?
            # User wants "grayscale images", implied continuous or visible structure.
            # Using cubic or linear interpolation can smooth out the patterns which is good for CNNs trained on natural images.
            # However, nearest neighbor preserves the exact defect codes.
            # Let's use Linear/Area for resizing to get "grayscale" intensity levels.
            
            # First, normalize to 0-255 BEFORE resize? Or AFTER?
            # Typically: Resize then Normalize.
            # WaferMap values: 0, 1, 2.
            # Let's cast to uint8
            wm = wm.astype(np.uint8)
            
            # Resize
            # If map is smaller than 224, cv2.INTER_LINEAR or CUBIC.
            # If larger, INTER_AREA.
            resized = cv2.resize(wm, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            
            # Normalize 0-255
            # Values are likely small integers.
            # Min-Max normalization
            v_min, v_max = resized.min(), resized.max()
            if v_max - v_min > 0:
                normalized = ((resized - v_min) / (v_max - v_min) * 255.0).astype(np.uint8)
            else:
                normalized = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            
            # Save
            filename = f"{label}_{idx}.png"
            # Path
            save_path = os.path.join(DATASET_DIR, split_name, label, filename)
            
            # Save using PIL
            img_pil = Image.fromarray(normalized, mode='L') # L = 8-bit pixels, black and white
            img_pil.save(save_path)
            
            count += 1
            
        except Exception as e:
            print(f"[ERROR] processing image {idx}: {e}")
            continue

    print(f"[INFO] Generated {count} images.")

def get_dataloaders():
    print("[INFO] Creating DataLoaders...")
    
    # Transformation: Grayscale to 3 channel (expand), ToTensor (scales to 0-1)
    # User said: "Wafer maps are grayscale; convert to 3-channel by repeating the channel."
    
    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(), # dataset values 0-255 -> 0.0-1.0
        # Optional: Normalize with ImageNet mean/std if leveraging pretrained weights effectively
        # But user didn't strictly ask for ImageNet norm, just "Normalize values to 0–255" which implies input range.
        # MobileNet expects 0-1 or normalized. ToTensor does 0-1.
        # Let's add standard ImageNet normalization for better convergence with pretrained model.
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dir = os.path.join(DATASET_DIR, 'Train')
    val_dir = os.path.join(DATASET_DIR, 'Validation')
    test_dir = os.path.join(DATASET_DIR, 'Test')
    
    train_dataset = ImageFolder(train_dir, transform=data_transforms)
    val_dataset = ImageFolder(val_dir, transform=data_transforms)
    test_dataset = ImageFolder(test_dir, transform=data_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"[INFO] Classes: {train_dataset.classes}")
    return train_loader, val_loader, test_loader, train_dataset.classes

def train_model(train_loader, val_loader, num_classes):
    print(f"[INFO] Initializing MobileNetV3-Small on {DEVICE}...")
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    
    # Replace classifier
    # Structure: model.classifier is a Sequential
    # Last layer is typically Linear(in_features=1024, out_features=1000)
    # Creating a new head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"[INFO] Starting training for {EPOCHS} epochs...")
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Train Loop
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total
        
        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Save logs to text file per epoch
        with open(os.path.join(RESULTS_DIR, "training_logs.txt"), "a") as f:
            f.write(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}\n")

    # Save model
    save_path = os.path.join(MODELS_DIR, "mobilenetv3_lswmd.pth")
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Model saved to {save_path}")
    
    return model, history

def evaluate_model(model, test_loader, classes):
    print("[INFO] Evaluating on Test set...")
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Precision (Macro): {prec:.4f}")
    print(f"Test Recall (Macro): {rec:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification Report
    cr = classification_report(all_labels, all_preds, target_names=classes)
    
    # Save Results
    with open(os.path.join(RESULTS_DIR, "metrics_summary.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision (Macro): {prec:.4f}\n")
        f.write(f"Recall (Macro): {rec:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(cr)
        
    # Plot CM
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()
    
    print(f"[INFO] Metrics and plots saved to {RESULTS_DIR}")

def export_to_onnx(model):
    print("[INFO] Exporting to ONNX...")
    model.eval()
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    onnx_path = os.path.join(MODELS_DIR, "model.onnx")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"[INFO] ONNX model exported to {onnx_path}")
    return onnx_path

def validate_onnx_inference(onnx_path, test_loader, classes):
    print("[INFO] Validating ONNX Inference...")
    
    # Grab 10 random images from test_dataset (via loader)
    # We'll just iterate once and pick 10
    
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # Select first 10
    if images.size(0) > 10:
        images = images[:10]
        labels = labels[:10]
    
    # Start Inference Session
    ort_session = ort.InferenceSession(onnx_path)
    
    print("\n--- ONNX Inference Results ---")
    print(f"{'Index':<5} {'Actual':<15} {'Predicted':<15} {'Match':<5}")
    
    correct = 0
    total = len(labels)
    
    for i in range(total):
        img = images[i].unsqueeze(0).numpy() # Shape (1, 3, 224, 224)
        label_idx = labels[i].item()
        actual_class = classes[label_idx]
        
        ort_inputs = {ort_session.get_inputs()[0].name: img}
        ort_outs = ort_session.run(None, ort_inputs)
        
        pred_idx = np.argmax(ort_outs[0])
        pred_class = classes[pred_idx]
        
        match = "YES" if pred_idx == label_idx else "NO"
        if match == "YES": correct += 1
        
        print(f"{i:<5} {actual_class:<15} {pred_class:<15} {match:<5}")
        
    print(f"\nONNX Validation Accuracy (mini-batch): {correct}/{total}")

def main():
    # Check if dataset already exists to skip generation
    if os.path.exists(os.path.join(DATASET_DIR, "Train", "clean")):
        print(f"[INFO] Dataset found in {DATASET_DIR}. Skipping data generation...")
        skip_generation = True
    else:
        skip_generation = False

    if not skip_generation:
        if not os.path.exists(DATA_PKL_PATH):
            print(f"[ERROR] Data file not found at {DATA_PKL_PATH}")
            return
        setup_directories()
        df = load_and_process_dataframe()
        if df is None: return
        generate_images(df)
    
    # Now that images are on disk, create dataloaders
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    
    # Train
    model, history = train_model(train_loader, val_loader, len(classes))
    
    # Evaluate
    evaluate_model(model, test_loader, classes)
    
    # Export
    onnx_path = export_to_onnx(model)
    
    # Validate ONNX
    validate_onnx_inference(onnx_path, test_loader, classes)
    
    print("[INFO] Pipeline Completed Successfully.")
    print(f"[INFO] All outputs available in {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
