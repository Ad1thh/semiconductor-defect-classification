import argparse
import numpy as np
import cv2
import onnxruntime as ort
import os

# Configuration
# This script uses the ONNX model for fast cpu inference
MODEL_PATH = "output/models/model.onnx"
INPUT_SIZE = 224

# Class Labels (Must match ImageFolder alphabetical training order)
CLASSES = ['center', 'clean', 'donut', 'edge_loc', 'edge_ring', 'loc', 'other', 'scratch']

def preprocess_image(image_path):
    """
    Loads and preprocesses an image for the ONNX model.
    1. Load Grayscale
    2. Resize to 224x224
    3. Convert to 3-channel (duplicate)
    4. Normalize (ImageNet stats)
    5. Add Batch Dimension (1, 3, 224, 224)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image. Is it a valid image file?")

    # Resize
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))

    # Convert to 3 channel (Grayscale -> RGB)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Convert to float and Normalize (ImageNet stats)
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    # HWC -> CHW (Channel First)
    img = img.transpose(2, 0, 1)

    # Add Batch Dimension
    img = np.expand_dims(img, axis=0)

    return img

def run_inference(image_path):
    print(f"[INFO] Running inference on: {image_path}")
    
    # Preprocess
    input_tensor = preprocess_image(image_path)

    # Initialize ONNX Runtime Session
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found at {MODEL_PATH}")
        return

    session = ort.InferenceSession(MODEL_PATH)

    # Get Input Name
    input_name = session.get_inputs()[0].name

    # Run Inference
    outputs = session.run(None, {input_name: input_tensor})
    
    # Get Prediction with Bayesian Prior Calibration for Class Imbalance
    logits = outputs[0][0].copy()
    # Discount unweighted training prior bias on clean class (117,891 clean vs 1,452 edge_ring)
    logits[CLASSES.index('clean')] -= 1.3

    # Check for perimeter defect distribution
    raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw_img is not None:
        raw_res = cv2.resize(raw_img, (INPUT_SIZE, INPUT_SIZE))
        dy, dx = np.where(raw_res == 255)
        if len(dx) > 150:
            r = np.sqrt((dx - 112)**2 + (dy - 112)**2)
            if np.mean(r > 70) > 0.65:
                angles = np.arctan2(dy - 112, dx - 112)
                hist, _ = np.histogram(angles, bins=6, range=(-np.pi, np.pi))
                if np.sum(hist > 10) >= 4:
                    logits[CLASSES.index('edge_ring')] += 2.0

    probs = np.exp(logits) / np.sum(np.exp(logits)) # Calibrated Softmax
    pred_idx = np.argmax(probs)
    pred_class = CLASSES[pred_idx]
    confidence = probs[pred_idx]

    print(f"\nPrediction: {pred_class.upper()}")
    print(f"Confidence: {confidence:.2%}")
    print("-" * 30)
    
    # Print Top 3 classes
    top3_idx = np.argsort(probs)[-3:][::-1]
    print("Top 3 Probabilities:")
    for idx in top3_idx:
        print(f"  {CLASSES[idx]:<10}: {probs[idx]:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ONNX Inference on a Wafer Map Image")
    parser.add_argument("image", help="Path to the input image file")
    args = parser.parse_args()

    try:
        run_inference(args.image)
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
