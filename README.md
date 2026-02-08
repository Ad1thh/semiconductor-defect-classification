# Semiconductor Defect Classification Pipeline

This repository contains an end-to-end pipeline for classifying semiconductor wafer defects using the `LSWMD.pkl` dataset. The pipeline preprocesses the data, trains a MobileNetV3-Small model, evaluates performance, and exports an ONNX model for inference.

## Prerequisites

Ensure you practice the following steps before running the script:

1.  **Python 3.7+** installed.
2.  **LSWMD.pkl** file located at the configured path (script expects it inside a folder named `LSWMD.pkl` in the current directory, or adjust `DATA_PKL_PATH` in the script).

## Installation

Install the required Python packages:

```bash
py -m pip install numpy pandas opencv-python torch torchvision scikit-learn matplotlib seaborn onnx onnxruntime tqdm pillow
```

## Usage

Run the single script to execute the entire pipeline:

```bash
py train_lswmd_mobilenetv3_onnx.py
```

## Pipeline Steps

1.  **Data Loading**: Reads `LSWMD.pkl`.
2.  **Preprocessing**:
    *   Extracts wafer maps and labels.
    *   Maps labels to 8 classes: `clean`, `other`, `center`, `donut`, `edge_loc`, `edge_ring`, `loc`, `scratch`.
    *   Filters invalid/corrupt maps.
    *   Converts maps to 224x224 grayscale images (saved as PNG).
    *   Splits data into Train (70%), Validation (15%), Test (15%).
3.  **Training**:
    *   Model: MobileNetV3-Small (pretrained backbone).
    *   Optimizer: Adam (lr=3e-4).
    *   Loss: CrossEntropyLoss.
    *   Epochs: 8.
4.  **Evaluation**:
    *   Calculates Accuracy, Precision, Recall on Test set.
    *   Generates Confusion Matrix and Classification Report.
5.  **Export**:
    *   Saves trained model as `.pth`.
    *   Exports model to ONNX format with dynamic batch size.
6.  **Validation**:
    *   Runs inference on 10 random test images using ONNX Runtime to verify correctness.

## Output

All artifacts are saved in the `output/` directory:

*   **output/Dataset/**: Generated images organized by split and class.
*   **output/models/**:
    *   `mobilenetv3_lswmd.pth`: PyTorch model weights.
    *   `model.onnx`: Exported ONNX model.
*   **output/results/**:
    *   `training_logs.txt`: Loss/Acc per epoch.
    *   `metrics_summary.txt`: Final test metrics.
    *   `confusion_matrix.png`: Heatmap of confusion matrix.
