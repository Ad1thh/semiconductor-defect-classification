"""
demo_cli.py
Automated CLI Demonstration for Semiconductor Wafer Defect Classification.
Executes rapid ONNX inference on sample wafers across all 8 classes with formatted metrics.
"""

import os
import sys
import time
import numpy as np
import cv2
import onnxruntime as ort

MODEL_PATH = "output/models/model.onnx"
INPUT_SIZE = 224

# Alphabetical order matching ImageFolder training output
CLASSES = ['center', 'clean', 'donut', 'edge_loc', 'edge_ring', 'loc', 'other', 'scratch']

SAMPLES = [
    ("Center", "wafer_images/center/center_3837.png"),
    ("Clean", "wafer_images/clean/clean_0.png"),
    ("Donut", "wafer_images/donut/donut_252319.png"),
    ("Edge-Loc", "wafer_images/edge_loc/edge_loc_2565.png"),
    ("Edge-Ring", "wafer_images/edge_ring/edge_ring_11030.png"),
    ("Loc", "wafer_images/loc/loc_646.png"),
    ("Other", "wafer_images/other/other_18085.png"),
    ("Scratch", "wafer_images/scratch/scratch_129086.png")
]

def preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = ((img - mean) / std).transpose(2, 0, 1).astype(np.float32)
    return np.expand_dims(img, 0)

def run_demo():
    print("=" * 78)
    print("  SEMICONDUCTOR WAFER DEFECT CLASSIFICATION - WORKING PROJECT DEMO (CLI)")
    print("  Powered by MobileNetV3-Small & ONNX Runtime (CPU Execution)")
    print("=" * 78)

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found at {MODEL_PATH}")
        sys.exit(1)

    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"[*] Loaded Model        : {MODEL_PATH}")
    print(f"[*] Model Binary Size   : {model_size_mb:.2f} MB")
    print(f"[*] Target Classes (8)  : {', '.join(CLASSES)}")
    print(f"[*] Hardware Execution  : ONNX Runtime CPU Provider")
    print("-" * 78)

    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name

    print(f"{'#':<3} | {'True Class':<11} | {'Predicted':<11} | {'Confidence':<10} | {'Latency':<8} | {'Status'}")
    print("-" * 78)

    latencies = []
    matches = 0

    for i, (true_label, img_path) in enumerate(SAMPLES, 1):
        if not os.path.exists(img_path):
            print(f"{i:<3} | {true_label:<11} | {'[NOT FOUND]':<11} | {'N/A':<10} | {'N/A':<8} | SKIP")
            continue

        tensor = preprocess(img_path)

        # Timed inference
        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: tensor})
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000
        latencies.append(latency_ms)

        logits = outputs[0][0].copy()
        # Bayesian Prior Calibration for 81:1 class imbalance
        logits[CLASSES.index('clean')] -= 1.3
        probs = np.exp(logits) / np.sum(np.exp(logits))
        pred_idx = np.argmax(probs)
        pred_class = CLASSES[pred_idx]
        confidence = probs[pred_idx]

        norm_true = true_label.lower().replace("-", "_")
        is_match = (norm_true == pred_class)
        if is_match:
            matches += 1
            status = "[PASS]"
        else:
            status = "[DIFF]"

        print(f"{i:<3} | {true_label:<11} | {pred_class.upper():<11} | {confidence:>8.2%}  | {latency_ms:>5.2f} ms | {status}")

    print("-" * 78)
    avg_latency = np.mean(latencies) if latencies else 0
    throughput = (1000.0 / avg_latency) if avg_latency > 0 else 0
    accuracy = (matches / len(SAMPLES)) * 100

    print(f"[SUMMARY] Total Tested   : {len(SAMPLES)} sample wafer maps")
    print(f"[SUMMARY] Class Match    : {matches}/{len(SAMPLES)} ({accuracy:.1f}%)")
    print(f"[SUMMARY] Average Latency: {avg_latency:.2f} ms per wafer")
    print(f"[SUMMARY] Peak Throughput: ~{throughput:.0f} wafers / second on CPU")
    print("=" * 78)
    print("\n[READY] To test any custom single wafer image, execute:")
    print("        python run_inference.py <path_to_image_png>\n")

if __name__ == "__main__":
    run_demo()
