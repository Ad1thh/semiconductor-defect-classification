# Project Presentation, Demonstration & Viva Defense Guide
## End-to-End Deep Learning Pipeline for Automated Semiconductor Wafer Defect Classification with ONNX Optimization

* **Institution:** Saintgits College of Engineering (Autonomous)
* **Program:** B.Tech Mini Project Final Internal Review
* **Supervisor:** Er. Justin Mathew, Assistant Professor, Department of Computer Science & Engineering
* **Project Team:**
  1. **ADITHYAN B** (Roll No: MGP23EVLSI001)
  2. **ANNA BOBAN** (Roll No: MGP23EVLSI002)
  3. **NOOPA MARIA ROJAN** (Roll No: MGP23EVLSI009)

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Brief Explanation of Core Concepts](#2-brief-explanation-of-core-concepts)
   * 2.1 [Problem Statement](#21-problem-statement)
   * 2.2 [Project Objectives](#22-project-objectives)
   * 2.3 [Methodology & Architecture](#23-methodology--architecture)
   * 2.4 [Implementation Pipeline](#24-implementation-pipeline)
   * 2.5 [Experimental Outcomes & Results](#25-experimental-outcomes--results)
3. [Individual Team Member Contributions & Speaking Scripts](#3-individual-team-member-contributions--speaking-scripts)
   * 3.1 [Adithyan B (Data Engineering & Preprocessing)](#31-adithyan-b-mgp23evlsi001)
   * 3.2 [Anna Boban (Deep Learning Architecture & Training)](#32-anna-boban-mgp23evlsi002)
   * 3.3 [Noopa Maria Rojan (ONNX Optimization, Inference & Demo)](#33-noopa-maria-rojan-mgp23evlsi009)
4. [Working Project Demonstration Walkthrough](#4-working-project-demonstration-walkthrough)
   * 4.1 [Interactive Web Application Demo (`demo.html`)](#41-interactive-web-application-demo)
   * 4.2 [High-Performance CLI Demo (`demo_cli.py` & `run_inference.py`)](#42-cli-engine-demo)
5. [PowerPoint Presentation Overview (`.pptx` & Web Deck)](#5-powerpoint-presentation-overview)
6. [Top 15 Anticipated Viva / Review Questions & Model Answers](#6-top-15-anticipated-viva--review-questions--model-answers)

---

## 1. Executive Summary

This project delivers an industrial-grade, end-to-end deep learning framework for automated classification of semiconductor wafer map defect patterns using the benchmark **WM-811K (LSWMD)** dataset. 

By combining transfer learning on **MobileNetV3-Small** with hardware-agnostic **ONNX graph optimization (Opset 12)**, our pipeline achieves:
* **98.02% Test Accuracy** across 121,719 test wafers.
* **Ultra-compact binary footprint of 5.84 MB** (over 16× smaller than conventional ResNet-50).
* **Sub-2ms CPU inference latency (~1.53 ms/wafer)**, enabling automated test equipment (ATE) lines to process over **650 wafers per second** without requiring dedicated GPU infrastructure.

---

## 2. Brief Explanation of Core Concepts

### 2.1 Problem Statement
In semiconductor fabrication, wafer defect pattern identification is essential for rapid root-cause diagnosis of manufacturing equipment failures. However:
1. **Manual Visual Inspection Limitations:** Human inspection under electron or optical microscopes is subjective, prone to fatigue, and cannot scale to high-throughput 300mm wafer fabrication lines.
2. **Complex Spatial Topologies:** Wafer failure modes manifest in distinct spatial distributions (e.g., *Center* defects from spin-coating nozzle misalignment, *Donut* patterns from thermal gradients, *Scratches* from handling robot end-effectors, and *Edge-Ring* defects from chemical-mechanical planarization). Classical machine learning with manual feature extraction fails to generalize across high spatial variance.
3. **Legacy Dataset Hurdles:** The industrial benchmark WM-811K dataset is stored in legacy pickle formats with outdated Pandas indexing conventions that cause runtime deserialization crashes in modern environments.
4. **Edge Deployment Constraints:** Conventional deep CNNs (e.g., VGG, ResNet) require gigabytes of GPU VRAM, making them impractical for low-power edge units integrated into Automated Optical Inspection (AOI) machines.

### 2.2 Project Objectives
* **Objective 1:** Build a robust, automated data pipeline to parse legacy `LSWMD.pkl` (811,472 wafer records) and convert raw arrays into standardized 224×224 8-bit grayscale images across 8 target classes.
* **Objective 2:** Design a lightweight CNN classification architecture based on `MobileNetV3-Small` utilizing depthwise separable convolutions to minimize computational complexity (< 1.5M parameters).
* **Objective 3:** Implement stratified 70% Train / 15% Validation / 15% Test partitioning and optimize transfer learning using Adam ($lr = 3 \times 10^{-4}$) and Cross-Entropy loss.
* **Objective 4:** Export trained PyTorch weights to open ONNX format (Opset 12) with dynamic batch dimensions and constant folding, reducing the runtime memory footprint to < 6 MB.
* **Objective 5:** Deliver production-ready execution engines: a standalone CLI runner (`run_inference.py`), an automated batch tester (`demo_cli.py`), and an interactive web demo dashboard (`demo.html`).

### 2.3 Methodology & Architecture
The system architecture follows four sequential engineering stages:
```
[Stage A: Data Pipeline] 
  Raw LSWMD.pkl (811k) ──> Compatibility Patch ──> Bilinear 224x224 ──> Stratified Split (70/15/15)
                                                                                  │
[Stage B: Deep Learning]                                                          ▼
  MobileNetV3-Small Backbone ──> Depthwise Convs + SE ──> Custom 8-Class Head ──> PyTorch Training Loop
                                                                                  │
[Stage C: Optimization]                                                           ▼
  PyTorch Weights (.pth) ──> Constant Folding ──> Dynamic Batch ──> ONNX Opset 12 Export (5.84 MB)
                                                                                  │
[Stage D: Deployment]                                                             ▼
  ONNX Runtime CPU Engine ──> CLI Runner (run_inference.py) + Interactive Web Visualizer (demo.html)
```

### 2.4 Implementation Pipeline
* **Software Stack:** Python 3.12, PyTorch (`torch`, `torchvision`), ONNX (`onnx`), ONNX Runtime (`onnxruntime`), OpenCV (`cv2`), Pillow, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn.
* **Dataset Organization:** The 8 target categories are:
  1. `center`: Concentric core clusters
  2. `clean`: Nominal defect-free wafers (none)
  3. `donut`: Annular ring failure
  4. `edge_loc`: Perimeter localized failure
  5. `edge_ring`: Peripheral boundary circle
  6. `loc`: Isolated interior clusters
  7. `other`: Random or near-full catastrophic failures
  8. `scratch`: Linear mechanical streaks
* **Model Configuration:** MobileNetV3-Small replaces the standard final linear layer (`classifier[3]`) with an 8-output linear classifier (`Linear(1024, 8)`).

### 2.5 Experimental Outcomes & Results
* **Overall Test Accuracy:** **98.02%** across 121,719 test samples.
* **Macro Precision:** **0.6897** (~69.0%)
* **Macro Recall:** **0.6177** (~61.8%)
* **Weighted F1-Score:** **0.9800** (98.0%)
* **Model File Size:** **5.84 MB** (ONNX), **6.24 MB** (PyTorch state dict).
* **Inference Latency:** **1.53 ms** average per wafer on standard CPU.
* **Throughput:** **~653 wafers/second** peak CPU throughput.

---

## 3. Individual Team Member Contributions & Speaking Scripts

### 3.1 ADITHYAN B (Roll No: MGP23EVLSI001)
* **Domain:** Data Engineering, Dataset Standardization & Preprocessing Architecture

#### Key Engineering Contributions:
1. **Dataset Acquisition & Compatibility:** Analyzed the WM-811K dataset (811,472 wafer records). Developed the dynamic unpickling compatibility patch in `load_and_process_dataframe()` to resolve Python 2 to 3 `pandas.indexes.base._new_Index` deserialization errors.
2. **Image Standardization Pipeline:** Engineered the wafer map array transformation: filtered valid 2D arrays, applied bilinear interpolation to standardize variable-sized arrays to 224×224, and performed Min-Max intensity normalization to 0–255 8-bit grayscale images.
3. **Stratified Data Partitioning:** Implemented stratified train/validation/test partitioning (70% Train, 15% Validation, 15% Test) across all 8 defect classes to maintain representative distributions despite extreme class imbalance.

#### Individual Speech Script (What to Say During Presentation):
> *"Respected Supervisor and evaluators, I am Adithyan B. In this project, I took charge of the data engineering and preprocessing pipeline.*  
> *The industrial WM-811K dataset contains 811,472 wafer maps from 46,393 fabrication lots. A significant engineering hurdle was that the raw dataset was stored in a legacy pickle file whose Pandas index structures are deprecated in modern Python 3. I resolved this by designing an unpickling patch that maps legacy index objects dynamically at runtime.*  
> *Next, because raw wafer maps have varying physical dimensions (e.g., 25×25 to 200×200 dies), I developed an automated transformation pipeline that resizes wafer arrays to a uniform 224×224 resolution using bilinear interpolation and normalizes pixel intensities into 8-bit grayscale PNGs.*  
> *Finally, to address severe class imbalance—where over 96% of wafers are defect-free—I implemented a stratified 70/15/15 data split. This ensured that rare defect classes like Donut and Scratch were preserved proportionally across training, validation, and testing."*

---

### 3.2 ANNA BOBAN (Roll No: MGP23EVLSI002)
* **Domain:** Deep Learning Architecture, Transfer Learning & Model Optimization

#### Key Engineering Contributions:
1. **Model Selection & Architecture:** Investigated CNN backbones (VGG-16, ResNet-50 vs MobileNetV3). Selected `MobileNetV3-Small` for its depthwise separable convolutions and Squeeze-and-Excitation (SE) modules, achieving 1.5M parameters and ~0.06 GFLOPs.
2. **Transfer Learning Head:** Reconfigured the classification head by adapting `model.classifier[3]` from 1,000 ImageNet outputs to an 8-class linear classifier receiving 1,024-dimensional feature embeddings.
3. **Training & Convergence Monitoring:** Formulated the PyTorch training loop with Adam optimizer ($lr = 3 \times 10^{-4}$), Cross-Entropy Loss, and mini-batch size of 32 with CUDA acceleration. Tracked loss curves across 8 epochs and confirmed optimal validation loss at Epoch 4 (0.0496) without overfitting.

#### Individual Speech Script (What to Say During Presentation):
> *"Respected evaluators, I am Anna Boban. My core focus was on deep learning model architecture and training optimization.*  
> *Semiconductor manufacturing requires rapid, low-latency defect detection. Heavy architectures like ResNet-50 possess 25 million parameters and over 4 GFLOPs, making them unsuitable for edge units. I chose MobileNetV3-Small, which relies on depthwise separable convolutions, inverted residual blocks, and Squeeze-and-Excitation attention modules.*  
> *This design reduces computational overhead to just 1.5 million parameters and 0.06 GFLOPs while preserving spatial feature extraction.*  
> *I modified the classifier to an 8-output linear layer and trained the network using Adam optimizer at a learning rate of 3e-4 with Cross-Entropy loss. Over 8 epochs, our validation accuracy reached 98.05% with optimal validation loss at epoch 4, proving rapid convergence without overfitting."*

---

### 3.3 NOOPA MARIA ROJAN (Roll No: MGP23EVLSI009)
* **Domain:** ONNX Graph Optimization, Edge Inference Engine & Working Demonstration

#### Key Engineering Contributions:
1. **ONNX Graph Transformation:** Exported trained PyTorch model weights to ONNX format (Opset 12) with constant folding and dynamic batch axes (`batch_size`), compressing the deployment binary to 5.84 MB.
2. **Edge Inference Engine:** Developed `run_inference.py` and `demo_cli.py` using ONNX Runtime CPU execution, eliminating the multi-gigabyte PyTorch dependency on edge machines.
3. **Rigorous Metric Analysis & Benchmarking:** Evaluated test performance on 121,719 test wafers (98.02% accuracy, confusion matrix generation, macro precision/recall) and benchmarked CPU inference latency at 1.53 ms per wafer.
4. **Interactive Demonstration Web App:** Designed and developed `demo.html`, featuring real-time wafer heatmap rendering, drag-and-drop custom testing, 8-class probability telemetry, and automated lot screening.

#### Individual Speech Script (What to Say During Presentation):
> *"Respected evaluators, I am Noopa Maria Rojan. My responsibilities encompassed ONNX graph optimization, edge deployment, comprehensive model evaluation, and building our interactive demonstration systems.*  
> *To eliminate dependency on bulky PyTorch installations in industrial cleanrooms, I converted our trained model into open ONNX format using Opset 12 with constant folding and dynamic batching. This produced a self-contained model binary of just 5.84 MB.*  
> *I developed our inference engines—a standalone CLI runner and an interactive web demonstration—powered by ONNX Runtime. Benchmarking on standard CPU hardware demonstrates an inference latency of just 1.53 milliseconds per wafer, equivalent to a throughput of over 650 wafers per second.*  
> *Across our full test set of 121,719 wafers, the model achieves 98.02% accuracy. In a moment, I will demonstrate both our terminal inference engine and our interactive cleanroom inspection dashboard live."*

---

## 4. Working Project Demonstration Walkthrough

### 4.1 Interactive Web Application Demo (`demo.html`)
To launch: Open `demo.html` in any web browser (Chrome, Edge, Firefox).

#### Demonstration Steps to Show Faculty:
1. **Cleanroom Telemetry Strip:** Highlight the top metrics: *Inference Latency: ~1.5 ms*, *Test Accuracy: 98.02%*, *Model Footprint: 5.84 MB*, *Hardware Target: Edge CPU*.
2. **Interactive Wafer Canvas:** 
   * Point to the 300mm silicon wafer disk showing 784 dies and orientation notch.
   * Switch between **Heatmap**, **Die Pass/Fail**, and **Greyscale** visualization modes to show die cluster mapping.
3. **Quick-Test Presets (Click each pattern):**
   * Click **Scratch:** Shows linear robot handling scratch (99.92% confidence).
   * Click **Clean:** Shows nominal wafer (100.0% pass).
   * Click **Donut:** Shows thermal anneal annular ring defect (99.99% confidence).
   * Click **Edge-Loc & Center:** Highlights perimeter and central die clusters.
4. **Automated Lot Simulator:**
   * Click **"Run Lot Inspection Simulation (25 Wafers)"**.
   * Faculty will see the system rapidly screen 25 wafers with live animated counters showing wafers screened, yield %, and flagged defects.
5. **Model Evaluation & Confusion Matrix Modal:**
   * Click the **"Model Metrics & CM"** button in the header.
   * Shows the high-resolution confusion matrix heatmap (`output/results/confusion_matrix.png`) and class-wise test metrics.

---

### 4.2 CLI Engine Demo (`demo_cli.py` & `run_inference.py`)
To demonstrate live in PowerShell / Command Prompt:

#### 1. Automated 8-Class Benchmark:
```powershell
python demo_cli.py
```
**Expected Output:**
```
==============================================================================
  SEMICONDUCTOR WAFER DEFECT CLASSIFICATION - WORKING PROJECT DEMO (CLI)
  Powered by MobileNetV3-Small & ONNX Runtime (CPU Execution)
==============================================================================
[*] Loaded Model        : output/models/model.onnx
[*] Model Binary Size   : 5.84 MB
[*] Target Classes (8)  : center, clean, donut, edge_loc, edge_ring, loc, other, scratch
[*] Hardware Execution  : ONNX Runtime CPU Provider
------------------------------------------------------------------------------
#   | True Class  | Predicted   | Confidence | Latency  | Status
------------------------------------------------------------------------------
1   | Center      | CENTER      |   98.01%  |  2.01 ms | [PASS]
2   | Clean       | CLEAN       |  100.00%  |  1.32 ms | [PASS]
3   | Donut       | DONUT       |   99.99%  |  2.23 ms | [PASS]
4   | Edge-Loc    | EDGE_LOC    |   99.29%  |  1.49 ms | [PASS]
5   | Edge-Ring   | EDGE_RING   |   79.41%  |  1.04 ms | [PASS]
6   | Loc         | LOC         |   99.39%  |  1.96 ms | [PASS]
7   | Other       | OTHER       |   99.96%  |  1.17 ms | [PASS]
8   | Scratch     | SCRATCH     |   99.92%  |  1.04 ms | [PASS]
------------------------------------------------------------------------------
[SUMMARY] Total Tested   : 8 sample wafer maps
[SUMMARY] Class Match    : 8/8 (100.0%)
[SUMMARY] Average Latency: 1.53 ms per wafer
[SUMMARY] Peak Throughput: ~653 wafers / second on CPU
==============================================================================
```

#### 2. Single Wafer Inference CLI:
```powershell
python run_inference.py wafer_images/scratch/scratch_129086.png
```
**Output:**
```
[INFO] Running inference on: wafer_images/scratch/scratch_129086.png

Prediction: SCRATCH
Confidence: 99.92%
------------------------------
Top 3 Probabilities:
  scratch   : 99.92%
  clean     : 0.07%
  center    : 0.01%
```

---

## 5. PowerPoint Presentation Overview

The presentation is available in two formats:
1. **Native PowerPoint Presentation File:** [`Semiconductor_Defect_Classification_Review.pptx`](file:///c:/Users/noopa/Downloads/semiconductor-defect-classification/Semiconductor_Defect_Classification_Review.pptx) (16:9 widescreen, custom dark theme, embedded confusion matrix & wafer samples).
2. **Browser Presentation Deck:** [`index.html`](file:///c:/Users/noopa/Downloads/semiconductor-defect-classification/index.html) (11 interactive slides with speaker notes drawer, keyboard navigation, and direct "Live Demo" button).

### Slide Structure Overview:
* **Slide 1:** Title, Team Members (Roll Nos), Supervisor & Institution
* **Slide 2:** Problem Statement & Industrial Challenges
* **Slide 3:** Project Objectives & Scope (5 Core Milestones)
* **Slide 4:** Literature Survey & Research Gaps (Saqlain et al., Nakazawa et al., Howard et al.)
* **Slide 5:** End-to-End System Architecture (Stage A to D)
* **Slide 6:** WM-811K Dataset Profile & Image Preprocessing Pipeline (with embedded wafer images)
* **Slide 7:** MobileNetV3-Small Architecture & Efficiency Comparison vs ResNet-50
* **Slide 8:** Training Methodology, Hyperparameters & Epoch Loss Curves
* **Slide 9:** ONNX Graph Optimization & Edge Export Specs
* **Slide 10:** Experimental Outcomes, Metrics & Embedded Confusion Matrix Heatmap
* **Slide 11:** Working Demo Architecture (CLI Engine & Web Dashboard)
* **Slide 12:** Individual Team Member Contributions & Roles (Adithyan, Anna, Noopa)
* **Slide 13:** Summary, Phase-2 Future Scope & Q&A Discussion

---

## 6. Top 15 Anticipated Viva / Review Questions & Model Answers

### Q1: Why did you choose MobileNetV3-Small over standard CNNs like ResNet-50 or VGG-16?
**Model Answer:**  
*"In semiconductor Automated Optical Inspection (AOI), test decisions must execute in sub-15ms windows directly on factory test station CPUs without requiring bulky $2,000 GPUs. ResNet-50 has over 25.5 million parameters, 4.1 GFLOPs, and a model file size close to 100 MB. In contrast, MobileNetV3-Small employs depthwise separable convolutions, inverted residual bottlenecks, and hardware-friendly Hard-Swish activations. This slashes parameters to ~1.5 million (a 16× reduction) and memory footprint to 5.84 MB, while executing in just 1.53 ms on a standard CPU with 98.02% test accuracy."*

### Q2: What is a depthwise separable convolution and how does it save compute?
**Model Answer:**  
*"A standard convolution applies a 3D filter across all spatial dimensions and all input channels simultaneously. Depthwise separable convolutions split this into two steps:*  
*1. Depthwise Convolution: A single 2D spatial filter is applied to each input channel independently.*  
*2. Pointwise Convolution: A 1×1 convolution computes linear combinations across all channels.*  
*This reduces computational cost by a factor of approximately $\frac{1}{N} + \frac{1}{D_k^2}$, where $D_k$ is the kernel size (typically 3×3) and $N$ is the number of output channels, saving roughly 80–90% of FLOPs with minimal loss in spatial feature representation."*

### Q3: Why is the dataset split into 8 classes when WM-811K originally had 9 failure types?
**Model Answer:**  
*"The raw WM-811K dataset lists 9 labels: Center, Donut, Edge-Loc, Edge-Ring, Loc, Scratch, Random, Near-full, and None. In industrial yield management, 'Random' (scattered particle yield loss) and 'Near-full' (severe whole-wafer breakdown) both represent unpatterned gross yield collapse, so they are mapped into a unified 'other' category. 'None' represents nominal defect-free wafers, mapped to 'clean'. This yields 8 standardized categories, aligning with SEMI industrial standards."*

### Q4: Over 96% of the dataset is 'Clean'. How did you handle class imbalance during training and evaluation, and why did edge-ring defects initially tilt towards clean?
**Model Answer:**  
*"Class imbalance in the WM-811K dataset is extreme: 117,891 wafers are Clean (96.8%) while only 1,452 are Edge-Ring (1.2%)—an 81:1 imbalance ratio. Furthermore, on an edge-ring wafer, the defect ring only affects the outer perimeter (8%–15% of dies), while the remaining 85%–92% of the wafer interior is normal, clean silicon.  
Because standard Cross-Entropy training is unweighted, the model learns an artificial statistical prior bias $P(\text{Clean}) = 0.968$, inflating the Clean logit output by $\sim +1.3$. When an edge-ring wafer is processed, this false prior can barely tip the uncalibrated prediction to e.g. 52% Clean vs 46% Edge-Ring.  
We resolved this using **Bayesian Post-Hoc Prior Logit Calibration**:  
By subtracting the unweighted training prior offset ($\Delta = -1.3$ on the Clean logit index), we recover true class-conditional likelihoods. This immediately boosts Edge-Ring accuracy from 27% to **96.0%** while preserving **100% specificity on clean wafers**."*

### Q5: What is ONNX and what specific advantages does it provide here?
**Model Answer:**  
*"ONNX (Open Neural Network Exchange) is an open graph representation format. When we export from PyTorch to ONNX using Opset 12, the ONNX exporter performs constant folding and dead-code elimination. Deploying with ONNX Runtime allows execution via optimized C++ kernels on CPU without needing PyTorch installed. This shrinks the deployment environment from >2 GB (PyTorch + CUDA) to a lightweight ~15 MB runtime library, which is critical for embedded fabrication hardware."*

### Q6: What does 'dynamic batch axis' mean in your ONNX export?
**Model Answer:**  
*"When exporting to ONNX, static tensors fix the input shape (e.g., exactly 1×3×224×224). By configuring dynamic axes on dimension 0 (`dynamic_axes={'input': {0: 'batch_size'}}`), our exported ONNX model can accept either a single wafer image for real-time edge screening or a batch of 25 to 100 wafers simultaneously when processing an entire cassette/lot."*

### Q7: Why did you resize wafer maps to 224×224 and duplicate grayscale to 3 channels?
**Model Answer:**  
*"Raw wafer maps from different wafer fabs have varying grid dimensions (such as 26×26 or 52×52). Resizing to 224×224 using bilinear interpolation provides a standardized input tensor and smooths binary die clusters into continuous spatial intensity gradients. We duplicated the single grayscale channel into 3 identical RGB channels to leverage pre-trained MobileNetV3 weights trained on ImageNet without having to retrain the initial convolutional feature extractors from scratch."*

### Q8: What was the bug in `run_inference.py` and how did you resolve it?
**Model Answer:**  
*"In PyTorch, `ImageFolder` sorts subdirectories alphabetically: `['center', 'clean', 'donut', 'edge_loc', 'edge_ring', 'loc', 'other', 'scratch']`. The original `run_inference.py` script had listed classes starting with `clean`, which caused an index offset. We corrected the class list to match the exact alphabetical order produced during training, restoring 100% test agreement on our verified benchmark samples."*

### Q9: Why is the precision for 'scratch' (0.48) lower than 'edge_ring' (0.87)?
**Model Answer:**  
*"Scratch defects are thin, curvilinear streaks that occupy very few dies (low spatial density) compared to full wafers. Because scratch samples are rare (only 179 in the test set) and some localized defects ('loc') resemble short scratch fragments, the model occasionally confuses them. Edge-Ring defects, by contrast, possess an unambiguous, prominent circular geometric signature along the wafer boundary, making them much easier to recognize."*

### Q10: What is Squeeze-and-Excitation (SE) in MobileNetV3?
**Model Answer:**  
*"Squeeze-and-Excitation is a channel attention mechanism. The 'Squeeze' step performs global average pooling to aggregate spatial feature maps into a single channel descriptor. The 'Excitation' step uses two fully connected layers with a non-linearity to learn adaptive channel-wise weights. This allows MobileNetV3 to dynamically emphasize feature maps corresponding to defect patterns while suppressing background wafer noise."*

### Q11: How does this system integrate into an actual semiconductor fab?
**Model Answer:**  
*"In a fab, after wafer fabrication, wafers are loaded into Automated Test Equipment (ATE) probers or Automated Optical Inspection (AOI) tools. As dies undergo electrical wafer sort (EWS), a binary wafer map is generated. Our ONNX runtime engine can be invoked via local CLI or a lightweight C++/Python API embedded directly on the inspection tool's controller PC. When a defect pattern is detected, the system flags the lot and sends alerts to yield engineers before the wafer proceeds to expensive packaging."*

### Q12: Why did you train for 8 epochs instead of 50 or 100?
**Model Answer:**  
*"Because we utilized transfer learning with pre-trained MobileNetV3 weights, the convolutional feature extraction layers were already effective at edge and shape detection. As seen in our training logs, validation loss dropped from 0.0593 in Epoch 1 to 0.0496 in Epoch 4, reaching 98.02% accuracy. Beyond Epoch 5, validation loss stabilized while training loss continued decreasing, indicating that 8 epochs achieved convergence without entering an overfitting regime."*

### Q13: What optimizer and learning rate did you select and why?
**Model Answer:**  
*"We selected the Adam optimizer with a learning rate of $3 \times 10^{-4}$ (0.0003). Adam computes adaptive learning rates for each parameter using first and second moments of gradients, providing fast convergence. A moderate learning rate of 3e-4 prevents catastrophic forgetting of pre-trained convolutional features while allowing the new 8-class classification head to learn rapidly."*

### Q14: What is the significance of the orientation notch on a semiconductor wafer?
**Model Answer:**  
*"Silicon wafers have an orientation notch (or flat) at the bottom perimeter. This notch indicates the crystal orientation (e.g., <100> or <111>) of the single-crystal silicon ingot and serves as an alignment reference for photolithography stepper chucks and robotic transfer arms. In our wafer visualizer, we explicitly modeled this alignment notch at the bottom of the canvas."*

### Q15: What are the key milestones planned for Phase-2?
**Model Answer:**  
*"For Phase-2, we plan to implement:*  
*1. Focal Loss and Data Augmentation (e.g., Mixup, CutMix) to boost Scratch and Center recall.*  
*2. Embedded Hardware Deployment: Benchmarking the ONNX model on NVIDIA Jetson Nano / Orin and Raspberry Pi 5 using TensorRT.*  
*3. Multi-Label Defect Classification: Handling mixed defect patterns where a wafer exhibits both Edge-Ring and Scratch simultaneously.*  
*4. Production API: Packaging the engine into a fast gRPC / REST microservice for fab MES integration."*
