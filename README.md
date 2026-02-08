# Semiconductor Defect Classification Pipeline

This repository contains an end-to-end deep learning pipeline for classifying semiconductor wafer defects using the **WM-811K (LSWMD)** dataset. The pipeline handles data preprocessing, model training using **MobileNetV3-Small**, performance evaluation, and exporting the trained model to **ONNX** format for efficient inference.

## Features

*   **Automated Data Pipeline**: Handles loading the legacy pickle dataset, preprocessing wafer maps, and generating dataset splits (Train/Val/Test).
*   **Deep Learning Model**: Utilizes a pre-trained **MobileNetV3-Small** architecture fine-tuned for 8-class defect classification.
*   **Performance Evaluation**: Generates detailed classification reports, confusion matrices, and loss/accuracy plots.
*   **ONNX Export**: Automatically exports the trained PyTorch model to ONNX for optimized deployment.
*   **Inference Script**: Includes a standalone script `run_inference.py` for testing the model on single images.

## Project Structure

```
├── LSWMD.pkl/             # Directory containing the dataset file
│   └── LSWMD.pkl          # The raw pickle dataset (WM-811K)
├── output/                # Generated artifacts (created automatically)
│   ├── Dataset/           # Processed images (Train/Validation/Test)
│   ├── models/            # Saved PyTorch (.pth) and ONNX (.onnx) models
│   └── results/           # Logs, metrics, and confusion matrix plots
├── train_lswmd_mobilenetv3_onnx.py  # Main pipeline script
├── run_inference.py       # Inference script for individual images
└── README.md              # Project documentation
```

## Prerequisites

*   **Python 3.7+**
*   **Dataset**: The `LSWMD.pkl` file must be present. The script expects it at `LSWMD.pkl/LSWMD.pkl` relative to the project root (or update `DATA_PKL_PATH` in `train_lswmd_mobilenetv3_onnx.py`).

## Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Install dependencies**:
    ```bash
    pip install numpy pandas opencv-python torch torchvision scikit-learn matplotlib seaborn onnx onnxruntime tqdm pillow
    ```

## Usage

### 1. Training the Model

Run the main script to execute the entire pipeline (Preprocessing -> Training -> Evaluation -> Export):

```bash
python train_lswmd_mobilenetv3_onnx.py
```

*   **Note**: The first run will process the raw `LSWMD.pkl` file and generate thousands of images in the `output/Dataset` directory. This may take some time. Subsequent runs will detect the existing dataset and skip this step.

### 2. Running Inference

To test the trained ONNX model on a single wafer map image:

```bash
python run_inference.py path/to/image.png
```

**Example:**
```bash
python run_inference.py output/Dataset/Test/scratch/scratch_12345.png
```

## Performance Results

The model achieves high accuracy on the test set. Typical metrics (may vary slightly based on random splits):

*   **Overall Accuracy**: ~98%
*   **Macro Precision**: ~0.69
*   **Macro Recall**: ~0.62

### Defect Classes
<p align="center">
  <img src="assets/wafer_defect_classes.png" width="600"/>
</p>
The model classifies wafers into one of the following 8 categories:
*   `Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`, `Random` (mapped to `other`), `Scratch`, `Near-full` (mapped to `other`), `None` (mapped to `clean`).

## Outputs

After training, check the `output/` directory for:
*   **Models**: `output/models/mobilenetv3_lswmd.pth` and `model.onnx`
*   **Metrics**: `output/results/metrics_summary.txt`
*   **Visualization**: `output/results/confusion_matrix.png`
