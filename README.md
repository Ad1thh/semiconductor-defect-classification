# Semiconductor Wafer Defect Classification Pipeline

An end-to-end industrial deep learning and edge deployment pipeline for automated classification of semiconductor wafer map defects using the **WM-811K (LSWMD)** benchmark dataset.

The system features **MobileNetV3-Small** transfer learning, hardware-agnostic **ONNX graph optimization (Opset 12)**, **Bayesian post-hoc logit calibration** for severe class imbalance, an **interactive web cleanroom inspection dashboard**, a **real-time local inference server**, and automated CLI demonstration engines.

---

## 🌟 Key Features

* **Lightweight Deep Learning Backbone:** Fine-tuned `MobileNetV3-Small` architecture with depthwise separable convolutions and Squeeze-and-Excitation (SE) attention modules (< 1.5M parameters, ~0.06 GFLOPs).
* **Hardware-Agnostic ONNX Optimization:** PyTorch model exported to ONNX format (Opset 12) with constant folding and dynamic batch dimensions (compact 5.84 MB binary).
* **Sub-2ms CPU Inference Latency:** Executes on standard edge CPU hardware without requiring dedicated GPU infrastructure, achieving over **650 wafers/second** throughput.
* **Bayesian Logit Calibration:** Mitigates the 81:1 training class imbalance (117,891 clean vs rare defect classes) to achieve high recall on borderline defects like *Edge-Ring* and *Scratch* while maintaining 100% clean specificity.
* **Interactive Cleanroom Web Demo (`demo.html`):** Modern semiconductor fab inspection dashboard featuring a 300mm wafer canvas, die pass/fail heatmap, drag-and-drop custom image tester, 8-class probability telemetry, and a 25-wafer batch lot simulator.
* **Real-Time Python Inference Server (`server.py`):** Standalone HTTP inference server with `/predict` endpoint that connects the web interface directly to the ONNX runtime engine.
* **Automated Terminal Benchmark Runner (`demo_cli.py`):** Rapid verification tool testing sample wafer maps across all 8 classes in the terminal with 100% pass rate.
* **Complete Review & Presentation Package:** Includes a 16:9 widescreen PowerPoint deck (`Semiconductor_Defect_Classification_Review.pptx`), an interactive browser slide deck (`index.html`), and a defense guide (`PRESENTATION_AND_DEFENSE_GUIDE.md`).

---

## 📁 Repository Structure

```text
├── LSWMD.pkl.zip                                # Git LFS pointer for dataset
├── output/
│   ├── models/
│   │   ├── model.onnx                           # Exported ONNX model (5.84 MB)
│   │   └── mobilenetv3_lswmd.pth                # Trained PyTorch weights (6.24 MB)
│   └── results/
│       ├── confusion_matrix.png                 # Test set confusion matrix plot
│       ├── metrics_summary.txt                  # Precision, recall, and classification report
│       └── training_logs.txt                    # Per-epoch training & validation loss logs
├── wafer_images/                                # Sample wafer images across all 8 classes
│   ├── center/
│   ├── clean/
│   ├── donut/
│   ├── edge_loc/
│   ├── edge_ring/
│   ├── loc/
│   ├── other/
│   └── scratch/
├── server.py                                    # Real-time HTTP ONNX inference server (/predict)
├── demo.html                                    # Interactive cleanroom web demonstration
├── demo.css                                     # Fab-aesthetic styles for web demo
├── demo.js                                      # Telemetry, wafer canvas & upload client logic
├── demo_cli.py                                  # Automated 8-class terminal benchmark runner
├── run_inference.py                             # Single-image ONNX inference script
├── open_web_demo.bat                            # 1-Click launcher: starts server.py & opens demo.html
├── run_demo.bat                                 # 1-Click launcher: runs demo_cli.py in terminal
├── generate_powerpoint.py                       # Script generating 16:9 widescreen PowerPoint deck
├── Semiconductor_Defect_Classification_Review.pptx # Official PowerPoint presentation deck
├── index.html                                   # 11-slide presentation web deck with speaker notes
├── app.js                                       # Slide deck controller
├── styles.css                                   # Presentation deck styles
├── PRESENTATION_AND_DEFENSE_GUIDE.md            # Comprehensive defense guide & viva Q&A
├── train_lswmd_mobilenetv3_onnx.py              # Main training & export pipeline script
├── convert_pkl_to_images.py                     # Data extraction script from raw pickle
├── requirements.txt                             # Python dependencies
└── README.md                                    # Project documentation
```

---

## 🛠️ Environment & Tools

* **Operating System:** Windows 10 / 11, Linux (Ubuntu 20.04+)
* **Programming Language:** Python 3.7+ (Tested on Python 3.12 & 3.14)
* **Deep Learning & Inference:** PyTorch (`torch`, `torchvision`), ONNX (`onnx`), ONNX Runtime (`onnxruntime`)
* **Computer Vision & Data Processing:** OpenCV (`opencv-python`), Pillow (`PIL`), NumPy, Pandas, Scikit-learn
* **Presentation Tools:** Python-pptx, Google Fonts (`Inter`, `JetBrains Mono`), FontAwesome

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/noopamariarojan/semiconductor-defect-classification.git
   cd semiconductor-defect-classification
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Or install manually:*
   ```bash
   pip install numpy pandas opencv-python torch torchvision scikit-learn matplotlib seaborn onnx onnxruntime pillow python-pptx tqdm
   ```

---

## 💻 Running the Demonstrations

### 1. Interactive Web Cleanroom Dashboard (Recommended)

Start the local inference server and open the web dashboard:

```bash
python server.py
```
* Or double-click **`open_web_demo.bat`** in Windows File Explorer.
* The browser will automatically open: **[http://localhost:8000/demo.html](http://localhost:8000/demo.html)**.
* **Features:**
  * Click any of the 8 defect preset cards (*Center*, *Clean*, *Donut*, *Edge-Loc*, *Edge-Ring*, *Loc*, *Other*, *Scratch*) for instant inspection.
  * Drag and drop any custom wafer image into the dropzone to run real-time inference.
  * Click **"Run Lot Inspection Simulation"** to screen a batch of 25 wafers with live yield statistics.
  * View model evaluation metrics and confusion matrix in the modal.

---

### 2. Automated Terminal Benchmark (CLI)

Run the automated test runner across all 8 defect classes:

```bash
python demo_cli.py
```
* Or double-click **`run_demo.bat`** in Windows File Explorer.
* **Sample Output:**
  ```text
  ==============================================================================
    SEMICONDUCTOR WAFER DEFECT CLASSIFICATION - WORKING PROJECT DEMO (CLI)
    Powered by MobileNetV3-Small & ONNX Runtime (CPU Execution)
  ==============================================================================
  [*] Loaded Model        : output/models/model.onnx (5.84 MB)
  [*] Hardware Execution  : ONNX Runtime CPU Provider
  ------------------------------------------------------------------------------
  #   | True Class  | Predicted   | Confidence | Latency  | Status
  ------------------------------------------------------------------------------
  1   | Center      | CENTER      |   99.39%  |  2.41 ms | [PASS]
  2   | Clean       | CLEAN       |  100.00%  |  1.49 ms | [PASS]
  3   | Donut       | DONUT       |  100.00%  |  1.51 ms | [PASS]
  4   | Edge-Loc    | EDGE_LOC    |   99.80%  |  1.20 ms | [PASS]
  5   | Edge-Ring   | EDGE_RING   |   93.20%  |  1.03 ms | [PASS]
  6   | Loc         | LOC         |   99.80%  |  2.18 ms | [PASS]
  7   | Other       | OTHER       |   99.99%  |  1.24 ms | [PASS]
  8   | Scratch     | SCRATCH     |   99.98%  |  1.08 ms | [PASS]
  ------------------------------------------------------------------------------
  [SUMMARY] Total Tested   : 8 sample wafer maps | Class Match: 8/8 (100.0%)
  [SUMMARY] Average Latency: 1.52 ms per wafer  | Peak Throughput: ~658 wafers/sec
  ==============================================================================
  ```

---

### 3. Single Wafer Image Inference

Test any individual image with top-3 probability outputs:

```bash
python run_inference.py wafer_images/scratch/scratch_129086.png
```

---

## 📊 Experimental Results

Evaluated across **121,719 test wafers** from the WM-811K benchmark dataset:

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Overall Accuracy** | **98.02%** | High overall classification accuracy |
| **Macro Precision** | **0.6897** | Unweighted across all 8 classes |
| **Macro Recall** | **0.6177** | Balanced across rare and dominant classes |
| **Weighted F1-Score** | **0.9800** | Population-weighted harmonic mean |
| **Model Size (ONNX)** | **5.84 MB** | >16× smaller than ResNet-50 (98 MB) |
| **Inference Latency** | **~1.5 ms** | Sub-2ms execution on standard CPU |

### Target Defect Classes (8)
1. `Center`: Concentric defect clusters from spin-coating nozzle misalignment.
2. `Clean`: Nominal defect-free silicon wafers (100% yield).
3. `Donut`: Annular ring defects from thermal annealing gradients.
4. `Edge-Loc`: Localized edge clusters from clamp contact or bead removal.
5. `Edge-Ring`: Continuous perimeter rings from chemical-mechanical planarization (CMP).
6. `Loc`: Isolated internal die clusters from gas line particulate contamination.
7. `Other`: Catastrophic gross yield collapse (random dispersion or near-full failure).
8. `Scratch`: Linear/curvilinear streaks caused by wafer-handling robot end-effectors.

---

## 👥 Project Team & Contributions

* **Institution:** Saintgits College of Engineering (Autonomous)
* **Program:** B.Tech Mini Project Final Internal Review
* **Supervisor:** Er. Justin Mathew, Assistant Professor, Department of Computer Science & Engineering

### Team Members:
* **ADITHYAN B (Roll No: MGP23EVLSI001):** Data Engineering, WM-811K dataset parsing, legacy Pandas index unpickling compatibility patches, 224×224 image standardization, and 70/15/15 stratified data partitioning.
* **ANNA BOBAN (Roll No: MGP23EVLSI002):** Deep Learning Architecture, MobileNetV3-Small selection, custom 8-class transfer learning head, Adam optimizer training loop, loss curves, and epoch validation checkpoints.
* **NOOPA MARIA ROJAN (Roll No: MGP23EVLSI009):** ONNX Graph Optimization (Opset 12), Bayesian logit calibration, real-time HTTP server (`server.py`), CLI engines (`demo_cli.py`, `run_inference.py`), and interactive cleanroom web dashboard (`demo.html`).
