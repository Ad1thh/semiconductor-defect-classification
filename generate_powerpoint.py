"""
generate_powerpoint.py
Generates a 16:9 widescreen PowerPoint presentation for the Semiconductor Wafer Defect Classification project.
"""

import os
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = pptx.Presentation()
    # 16:9 widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # blank layout

    # Color Palette
    COLOR_BG = RGBColor(15, 23, 42)        # Slate 900
    COLOR_CARD = RGBColor(30, 41, 59)      # Slate 800
    COLOR_BORDER = RGBColor(51, 65, 85)    # Slate 700
    COLOR_CYAN = RGBColor(14, 165, 233)    # Sky / Cyan
    COLOR_EMERALD = RGBColor(16, 185, 129) # Emerald
    COLOR_AMBER = RGBColor(245, 158, 11)   # Amber
    COLOR_TEXT_MAIN = RGBColor(248, 250, 252) # Slate 50
    COLOR_TEXT_MUTED = RGBColor(148, 163, 184) # Slate 400
    COLOR_WHITE = RGBColor(255, 255, 255)

    def add_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = COLOR_BG
        bg.line.fill.background()
        return bg

    def add_header(slide, title_text, category_text="SEMICONDUCTOR DEFECT CLASSIFICATION"):
        # Category tag
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(0.4))
        tf_c = cat_box.text_frame
        tf_c.word_wrap = True
        p_c = tf_c.paragraphs[0]
        p_c.text = category_text.upper()
        p_c.font.size = Pt(11)
        p_c.font.bold = True
        p_c.font.color.rgb = COLOR_CYAN

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.85), Inches(11.7), Inches(0.8))
        tf_t = title_box.text_frame
        tf_t.word_wrap = True
        p_t = tf_t.paragraphs[0]
        p_t.text = title_text
        p_t.font.size = Pt(24)
        p_t.font.bold = True
        p_t.font.color.rgb = COLOR_TEXT_MAIN

    def add_card(slide, left, top, width, height, title, items, badge_color=COLOR_CYAN, title_size=16, body_size=13):
        # Card background
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD
        card.line.color.rgb = COLOR_BORDER
        card.line.width = Pt(1)

        # Content textbox
        tb = slide.shapes.add_textbox(left + Inches(0.25), top + Inches(0.2), width - Inches(0.5), height - Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True

        if title:
            p_title = tf.paragraphs[0]
            p_title.text = title
            p_title.font.size = Pt(title_size)
            p_title.font.bold = True
            p_title.font.color.rgb = badge_color
            p_title.space_after = Pt(10)
            first = False
        else:
            first = True

        for item in items:
            if first:
                p = tf.paragraphs[0]
                first = False
            else:
                p = tf.add_paragraph()
            p.text = item
            p.font.size = Pt(body_size)
            p.font.color.rgb = COLOR_TEXT_MAIN
            p.space_after = Pt(8)
            p.level = 0
        return card

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    add_background(s1)

    # Title Tag
    tag_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.0), Inches(11.3), Inches(0.5))
    p = tag_box.text_frame.paragraphs[0]
    p.text = "SAINTGITS COLLEGE OF ENGINEERING (AUTONOMOUS) • B.TECH MINI PROJECT"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN

    # Main Project Title
    t_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.3), Inches(1.8))
    tf = t_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "End-to-End Deep Learning Pipeline for Automated Semiconductor Wafer Defect Classification with ONNX Optimization"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEXT_MAIN

    # Subtitle
    sub_box = s1.shapes.add_textbox(Inches(1.0), Inches(3.3), Inches(11.3), Inches(0.8))
    tf = sub_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "A Lightweight MobileNetV3-Small Architecture & Edge Runtime Framework for Real-Time Wafer Map Pattern Inspection"
    p.font.size = Pt(15)
    p.font.color.rgb = COLOR_TEXT_MUTED

    # Presenters Card
    add_card(s1, Inches(1.0), Inches(4.3), Inches(5.4), Inches(2.5), "Presented By", [
        "ADITHYAN B  (Roll No: MGP23EVLSI001)",
        "ANNA BOBAN  (Roll No: MGP23EVLSI002)",
        "NOOPA MARIA ROJAN  (Roll No: MGP23EVLSI009)"
    ], COLOR_CYAN, 16, 14)

    # Supervisor Card
    add_card(s1, Inches(6.8), Inches(4.3), Inches(5.5), Inches(2.5), "Project Supervisor", [
        "ER. JUSTIN MATHEW",
        "Assistant Professor",
        "Department of Computer Science & Engineering",
        "Saintgits College of Engineering (Autonomous)"
    ], COLOR_EMERALD, 16, 14)

    # =========================================================================
    # SLIDE 2: Problem Statement
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_background(s2)
    add_header(s2, "Problem Statement & Industrial Challenges")

    add_card(s2, Inches(0.8), Inches(1.8), Inches(5.6), Inches(2.4),
             "1. Manual Visual Inspection Bottleneck", [
                 "• Manual inspection in high-volume fabs is slow, expensive, and inconsistent.",
                 "• High fatigue causes critical defect escapes to packaging stages.",
                 "• Inability to inspect 100% of wafers in modern sub-micron fabrication nodes."
             ], COLOR_AMBER)

    add_card(s2, Inches(6.8), Inches(1.8), Inches(5.7), Inches(2.4),
             "2. Complex Defect Spatial Geometries", [
                 "• Defect patterns indicate specific root-cause fabrication errors.",
                 "• Ring defects stem from CMP/spin coating; scratches from handling robot arms.",
                 "• Traditional heuristics and manual filters fail to capture diverse spatial topologies."
             ], COLOR_CYAN)

    add_card(s2, Inches(0.8), Inches(4.5), Inches(5.6), Inches(2.4),
             "3. Legacy Dataset Formats & Inconsistencies", [
                 "• The benchmark WM-811K (LSWMD) dataset is stored in legacy pickle format.",
                 "• Discontinued Pandas indexing conventions break modern Python environments.",
                 "• Severe class imbalance where >96% of wafers are defect-free (Clean)."
             ], COLOR_EMERALD)

    add_card(s2, Inches(6.8), Inches(4.5), Inches(5.7), Inches(2.4),
             "4. Edge Latency & Compute Constraints", [
                 "• Conventional heavyweight CNNs (ResNet, VGG) require costly dedicated GPUs.",
                 "• Fab floor test stations (ATE) operate under strict real-time deadlines (< 15 ms).",
                 "• Demand for lightweight, hardware-agnostic cross-platform execution on CPUs/edge."
             ], RGBColor(244, 63, 94))

    # =========================================================================
    # SLIDE 3: Objectives & Scope
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_background(s3)
    add_header(s3, "Project Objectives & Scope")

    objectives = [
        ("01. Automated Ingestion & Standardization",
         "Develop a robust Python preprocessing pipeline to parse raw LSWMD.pkl (811,472 records), apply compatibility patches, and convert wafer arrays into standardized 224x224 8-bit grayscale images across 8 defect classes.",
         COLOR_CYAN),
        ("02. Lightweight Deep Learning Architecture",
         "Fine-tune MobileNetV3-Small with depthwise separable convolutions and custom 8-class classification head using transfer learning for maximum parameter efficiency (< 1.5M parameters).",
         COLOR_EMERALD),
        ("03. Stratified Partitioning & Loss Optimization",
         "Implement a 70% Train, 15% Validation, 15% Test stratified partition with Cross-Entropy Loss and Adam optimizer (lr=3e-4) to ensure robust generalization across rare defect patterns.",
         COLOR_AMBER),
        ("04. Hardware-Agnostic ONNX Optimization",
         "Export trained PyTorch weights to ONNX format (Opset 12) with dynamic batch support and constant folding, slashing runtime footprint to 5.83 MB.",
         COLOR_CYAN),
        ("05. Production-Ready Inference Engine & Interactive Demo",
         "Deploy a standalone CLI runner (run_inference.py) and interactive browser-based dashboard with top-3 confidence scoring and sub-2ms CPU inference latency.",
         COLOR_EMERALD)
    ]

    top_y = 1.8
    for num_title, desc, color in objectives:
        add_card(s3, Inches(0.8), Inches(top_y), Inches(11.7), Inches(0.95), num_title, [desc], color, 14, 12)
        top_y += 1.05

    # =========================================================================
    # SLIDE 4: Literature Review & Gaps
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_background(s4)
    add_header(s4, "Literature Survey & Research Gaps")

    add_card(s4, Inches(0.8), Inches(1.8), Inches(5.6), Inches(2.5),
             "Saqlain et al. (IEEE Access, 2019)", [
                 "• Approach: Voting ensemble classifier combining multiple ML algorithms.",
                 "• Finding: Improved accuracy over single classical classifiers.",
                 "• Gaps: Computationally demanding, high memory footprint, lacking end-to-end edge inference runtime."
             ], COLOR_CYAN)

    add_card(s4, Inches(6.8), Inches(1.8), Inches(5.7), Inches(2.5),
             "Nakazawa & Kulkarni (IEEE TSM, 2018)", [
                 "• Approach: Conventional deep CNNs for wafer map feature extraction.",
                 "• Finding: Proved automated spatial feature learning outperforms manual features.",
                 "• Gaps: Heavy parameter count (>20M params), impractical for real-time edge ATE deployment."
             ], COLOR_EMERALD)

    add_card(s4, Inches(0.8), Inches(4.6), Inches(5.6), Inches(2.4),
             "Howard et al. (ICCV, 2019)", [
                 "• Approach: MobileNetV3 with hardware-aware NAS & depthwise convolutions.",
                 "• Finding: High accuracy on ImageNet with minimal latency.",
                 "• Gaps: General natural images only; not tuned or evaluated for wafer map defect topography."
             ], COLOR_AMBER)

    add_card(s4, Inches(6.8), Inches(4.6), Inches(5.7), Inches(2.4),
             "Our Proposed Contribution", [
                 "• End-to-end automated pipeline from legacy pickle to edge inference.",
                 "• MobileNetV3-Small adapted for 8 wafer classes with ONNX Opset 12 export.",
                 "• Delivers 98.02% test accuracy with ~1.8 ms inference latency on standard CPU."
             ], RGBColor(244, 63, 94))

    # =========================================================================
    # SLIDE 5: System Architecture & Workflow
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_background(s5)
    add_header(s5, "End-to-End System Architecture")

    stages = [
        ("STAGE A: DATA PIPELINE", [
            "• Raw LSWMD.pkl parsing",
            "• Pandas index compatibility patch",
            "• Bilinear resize to 224x224",
            "• Min-Max normalization (0-255)",
            "• 70% / 15% / 15% Stratified split"
        ], COLOR_CYAN, Inches(0.8)),
        ("STAGE B: MODEL TRAINING", [
            "• Backbone: MobileNetV3-Small",
            "• Inverted residual blocks + SE",
            "• Custom 8-class classification head",
            "• Adam optimizer (lr=3e-4)",
            "• Cross-Entropy loss over 8 epochs"
        ], COLOR_EMERALD, Inches(3.8)),
        ("STAGE C: ONNX OPTIMIZATION", [
            "• PyTorch model graph trace",
            "• ONNX Opset 12 export",
            "• Constant folding & prune",
            "• Dynamic batch dimension",
            "• Model size: 5.83 MB"
        ], COLOR_AMBER, Inches(6.8)),
        ("STAGE D: EDGE INFERENCE", [
            "• ONNX Runtime CPU engine",
            "• ImageNet tensor normalization",
            "• Top-3 confidence scoring",
            "• Sub-2ms inference latency",
            "• Web demo & CLI runner"
        ], RGBColor(244, 63, 94), Inches(9.8))
    ]

    for title, items, color, left_pos in stages:
        add_card(s5, left_pos, Inches(1.8), Inches(2.8), Inches(4.2), title, items, color, 14, 12)

    # Bottom Target Classes Bar
    add_card(s5, Inches(0.8), Inches(6.15), Inches(11.8), Inches(0.95),
             "Target Defect Classes (8 Categories)", [
                 "Center  |  Donut  |  Edge-Loc  |  Edge-Ring  |  Loc  |  Scratch  |  Other (Random/Near-full)  |  Clean (None)"
             ], COLOR_CYAN, 13, 13)

    # =========================================================================
    # SLIDE 6: Dataset & Automated Preprocessing
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_background(s6)
    add_header(s6, "WM-811K (LSWMD) Dataset & Image Preprocessing")

    add_card(s6, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.2),
             "Dataset Profile & Challenges", [
                 "• Total Records: 811,472 real semiconductor wafer maps collected from 46,393 lots.",
                 "• Labeled Samples: ~172,950 annotated wafers with failure patterns.",
                 "• Severe Real-World Imbalance: Over 96% of manufactured wafers are defect-free (Clean).",
                 "• Array Representation: Variable resolution 2D matrices where 0 = exterior, 1 = normal die, 2 = defect die.",
                 "• Automated Preprocessing:",
                 "   1. Unpickle with runtime patch for deprecated pandas Int64Index.",
                 "   2. Filter valid 2D wafer arrays and map 9 raw labels to 8 standardized targets.",
                 "   3. Standardize dimensions to 224x224 via bilinear interpolation.",
                 "   4. Scale pixel values to 0-255 grayscale PNG images.",
                 "   5. Generate stratified Train (70%), Validation (15%), Test (15%) partitions."
             ], COLOR_CYAN, 16, 12)

    # Add sample wafer images if available
    samples = [
        ("Center Defect", "wafer_images/center/center_3837.png"),
        ("Donut Defect", "wafer_images/donut/donut_252319.png"),
        ("Edge-Loc Defect", "wafer_images/edge_loc/edge_loc_2565.png"),
        ("Scratch Defect", "wafer_images/scratch/scratch_129086.png")
    ]

    card_right = add_card(s6, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.2),
                          "Representative Wafer Defect Patterns", [
                              "Visual examples of real silicon wafer defect topography:"
                          ], COLOR_EMERALD, 16, 13)

    # Embed sample images in a 2x2 grid inside the card
    pos_grid = [
        (Inches(7.1), Inches(2.6)),
        (Inches(9.8), Inches(2.6)),
        (Inches(7.1), Inches(4.8)),
        (Inches(9.8), Inches(4.8))
    ]

    for (lbl, pth), (img_x, img_y) in zip(samples, pos_grid):
        if os.path.exists(pth):
            s6.shapes.add_picture(pth, img_x, img_y, Inches(2.3), Inches(1.7))
            txt = s6.shapes.add_textbox(img_x, img_y + Inches(1.75), Inches(2.3), Inches(0.3))
            tf = txt.text_frame
            p = tf.paragraphs[0]
            p.text = lbl
            p.font.size = Pt(11)
            p.font.bold = True
            p.font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 7: Deep Learning Model Architecture
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_background(s7)
    add_header(s7, "Model Architecture: MobileNetV3-Small")

    add_card(s7, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.2),
             "Architectural Highlights", [
                 "• Depthwise Separable Convolutions: Drastically reduces multiply-accumulate (MAC) operations compared to standard 3x3 convolutions.",
                 "• Inverted Residual Blocks: Expand-filter-squeeze structure retains high-dimensional spatial features while preserving a compact memory footprint.",
                 "• Squeeze-and-Excitation (SE) Modules: Channel-wise attention mechanisms adaptively recalibrate defect feature map weights.",
                 "• Hard-Swish (h-swish) Activation: Avoids costly sigmoid/exp computations, optimizing execution on mobile and embedded CPU hardware.",
                 "• Custom Transfer Learning Head:",
                 "   - Global Average Pooling (7x7 -> 1x1x1024)",
                 "   - Fully Connected Layer: 1,024 -> 8 logits",
                 "   - Softmax output across 8 defect categories."
             ], COLOR_CYAN, 16, 12)

    add_card(s7, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.2),
             "Architectural Efficiency Comparison", [
                 "Feature                     MobileNetV3-Small    ResNet-50",
                 "-------------------------------------------------------------",
                 "Parameter Count        ~1.5 Million               ~25.5 Million",
                 "Model Size (pth)          6.24 MB                    97.8 MB",
                 "ONNX Size                   5.83 MB                    94.2 MB",
                 "FLOPs                          ~0.06 GFLOPs           ~4.1 GFLOPs",
                 "Inference Latency       1.8 ms (CPU)             22.4 ms (CPU)",
                 "Edge Deployable?        YES (Micro ATE)       High GPU Needed",
                 "",
                 "Key Takeaway: MobileNetV3-Small delivers over 16x reduction in model size and 12x faster CPU inference while maintaining top-tier classification accuracy for industrial wafer defect detection."
             ], COLOR_EMERALD, 16, 12)

    # =========================================================================
    # SLIDE 8: Training & Optimization Setup
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_background(s8)
    add_header(s8, "Training Methodology & Hyperparameter Setup")

    add_card(s8, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.2),
             "Hyperparameters & Training Strategy", [
                 "• Input Resolution: 224 x 224 pixels (3-channel duplicated grayscale)",
                 "• Batch Size: 32 samples per mini-batch",
                 "• Optimization Algorithm: Adam Optimizer",
                 "• Learning Rate: 3e-4 (0.0003) for stable transfer learning convergence",
                 "• Loss Function: Cross-Entropy Loss over 8 categorical defect outputs",
                 "• Training Duration: 8 Epochs with per-epoch validation checkpointing",
                 "• Normalization: Standard ImageNet channel stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
                 "• Hardware: NVIDIA GPU with CUDA acceleration for rapid training."
             ], COLOR_CYAN, 16, 12)

    add_card(s8, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.2),
             "Epoch Convergence History", [
                 "Epoch 1:  Train Loss: 0.0663  |  Val Loss: 0.0593  |  Val Acc: 97.74%",
                 "Epoch 2:  Train Loss: 0.0537  |  Val Loss: 0.0543  |  Val Acc: 97.92%",
                 "Epoch 3:  Train Loss: 0.0490  |  Val Loss: 0.0518  |  Val Acc: 97.98%",
                 "Epoch 4:  Train Loss: 0.0449  |  Val Loss: 0.0496  |  Val Acc: 98.02%",
                 "Epoch 5:  Train Loss: 0.0415  |  Val Loss: 0.0513  |  Val Acc: 98.05%",
                 "Epoch 6:  Train Loss: 0.0379  |  Val Loss: 0.0538  |  Val Acc: 98.02%",
                 "Epoch 7:  Train Loss: 0.0343  |  Val Loss: 0.0534  |  Val Acc: 98.05%",
                 "Epoch 8:  Train Loss: 0.0309  |  Val Loss: 0.0577  |  Val Acc: 97.97%",
                 "",
                 "Analysis: The model achieves optimal validation loss at Epoch 4 (0.0496) with 98.02% accuracy, displaying smooth convergence without severe overfitting."
             ], COLOR_AMBER, 16, 12)

    # =========================================================================
    # SLIDE 9: ONNX Optimization & Edge Export
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_background(s9)
    add_header(s9, "ONNX Graph Optimization & Export")

    add_card(s9, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.2),
             "Why ONNX for Industrial Deployment?", [
                 "• Open Neural Network Exchange (ONNX): An open ecosystem that frees models from proprietary framework lock-in.",
                 "• Graph Optimization: Automatically executes constant folding, dead-node elimination, and node fusion.",
                 "• Hardware Independence: Enables the same binary to execute seamlessly on Intel CPUs, AMD APUs, ARM SoCs, and edge accelerators.",
                 "• Zero PyTorch Dependency: The deployment runtime requires only onnxruntime (< 15 MB) instead of the entire PyTorch library (> 2 GB).",
                 "• Export Specs:",
                 "   - Format: Opset 12",
                 "   - Input Name: 'input' with dynamic batch axis (batch_size, 3, 224, 224)",
                 "   - Output Name: 'output' with dynamic batch axis (batch_size, 8)",
                 "   - Model Binary Size: Exactly 5.83 MB."
             ], COLOR_CYAN, 16, 12)

    add_card(s9, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.2),
             "Inference Pipeline Flow", [
                 "1. Image Acquisition: Read grayscale wafer map from AOI inspection.",
                 "2. Preprocessing: Resize to 224x224, duplicate to 3 channels.",
                 "3. Normalization: Apply ImageNet mean/std scaling in NumPy.",
                 "4. Session Initialization: Load output/models/model.onnx into ONNX Runtime CPU session.",
                 "5. Graph Execution: Execute optimized C++ runtime kernels (session.run).",
                 "6. Softmax & Post-processing: Calculate probability distribution across 8 classes.",
                 "7. Decision Dispatch: Trigger automated fab sorting or alarm if confidence exceeds threshold.",
                 "",
                 "Measured Latency: 1.82 ms per wafer image on standard CPU."
             ], COLOR_EMERALD, 16, 12)

    # =========================================================================
    # SLIDE 10: Performance Evaluation & Confusion Matrix
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_background(s10)
    add_header(s10, "Experimental Results & Performance Metrics")

    add_card(s10, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.2),
             "Comprehensive Evaluation Metrics", [
                 "• Overall Test Accuracy: 98.02% (121,719 test wafers)",
                 "• Macro-Averaged Precision: 0.69 (68.97%)",
                 "• Macro-Averaged Recall: 0.62 (61.77%)",
                 "• Weighted Average F1-Score: 0.98 (98.00%)",
                 "",
                 "Per-Class Performance Summary:",
                 "• Clean Wafers:      99% Precision | 99% Recall (117,891 samples)",
                 "• Edge-Ring Defect: 87% Precision | 76% Recall (1,452 samples)",
                 "• Donut Defect:       73% Precision | 67% Recall (83 samples)",
                 "• Center Defect:       70% Precision | 34% Recall (644 samples)",
                 "• Edge-Loc Defect: 63% Precision | 56% Recall (779 samples)",
                 "• Other Defect:        60% Precision | 70% Recall (152 samples)",
                 "• Loc Defect:           52% Precision | 53% Recall (539 samples)",
                 "• Scratch Defect:     48% Precision | 39% Recall (179 samples)"
             ], COLOR_CYAN, 16, 12)

    # Embed Confusion Matrix if exists
    cm_path = "output/results/confusion_matrix.png"
    if os.path.exists(cm_path):
        card_cm = add_card(s10, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.2),
                           "Confusion Matrix Heatmap", [], COLOR_EMERALD, 16, 12)
        s10.shapes.add_picture(cm_path, Inches(7.1), Inches(2.4), Inches(5.1), Inches(4.3))
    else:
        add_card(s10, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.2),
                 "Confusion Matrix Heatmap", [
                     "Confusion matrix plot generated in output/results/confusion_matrix.png."
                 ], COLOR_EMERALD, 16, 12)

    # =========================================================================
    # SLIDE 11: Working Demo & Deployment Tools
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_background(s11)
    add_header(s11, "Working Project Demonstration & Industrial Integration")

    add_card(s11, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.2),
             "1. Production CLI Engine (run_inference.py)", [
                 "• Lightweight standalone Python script without PyTorch dependencies.",
                 "• Executes ONNX Runtime CPU inference on individual wafer images.",
                 "• Command:",
                 "    python run_inference.py wafer_images/scratch/scratch_129086.png",
                 "• Output:",
                 "    Prediction: SCRATCH",
                 "    Confidence: 99.92%",
                 "    Top-3 Probabilities:",
                 "       scratch   : 99.92%",
                 "       clean     : 0.07%",
                 "       center    : 0.01%",
                 "• Integrated automated batch test runner (demo_cli.py) testing all 8 defect categories."
             ], COLOR_CYAN, 16, 12)

    add_card(s11, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.2),
             "2. Interactive Web Demo Dashboard (demo.html)", [
                 "• Modern high-aesthetic browser interface for live demonstration.",
                 "• Features:",
                 "   - Real-time Interactive Wafer Canvas & Defect Heatmap.",
                 "   - Quick-Test Gallery for all 8 defect categories with 1-click execution.",
                 "   - Custom Image Drag & Drop file uploader.",
                 "   - Live prediction telemetry with top confidence gauge and class bars.",
                 "   - Automated Production Lot Batch Tester (25 wafers/run).",
                 "   - Confusion Matrix & Performance Metric Explorer.",
                 "• Semiconductor Fab Ready: Simulated integration with automated test equipment (ATE) and wafer sorting."
             ], COLOR_EMERALD, 16, 12)

    # =========================================================================
    # SLIDE 12: Individual Team Contributions
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_background(s12)
    add_header(s12, "Individual Team Member Contributions")

    members = [
        ("ADITHYAN B", "Roll No: MGP23EVLSI001", "Data Engineering & Pipeline Standardization", [
            "• Spearheaded WM-811K dataset acquisition (811,472 wafer records).",
            "• Engineered compatibility patches for legacy Pandas unpickling.",
            "• Built automated 224x224 image conversion and Min-Max scaling.",
            "• Designed 70% / 15% / 15% stratified dataset split across all classes.",
            "• Formulated the problem statement and literature gap analysis."
        ], COLOR_CYAN, Inches(0.8)),

        ("ANNA BOBAN", "Roll No: MGP23EVLSI002", "Deep Learning Architecture & Model Training", [
            "• Evaluated and selected MobileNetV3-Small for edge parameter efficiency.",
            "• Implemented PyTorch transfer learning model with custom 8-class head.",
            "• Configured Adam optimizer (lr=3e-4) and Cross-Entropy loss training loop.",
            "• Monitored epoch convergence, loss curves, and validation checkpoints.",
            "• Analyzed computational complexity and FLOPs reduction vs ResNet."
        ], COLOR_EMERALD, Inches(4.8)),

        ("NOOPA MARIA ROJAN", "Roll No: MGP23EVLSI009", "ONNX Optimization, Inference Engine & Demo", [
            "• Formulated ONNX Opset 12 export with constant folding (5.83 MB binary).",
            "• Developed standalone CLI inference engine (run_inference.py).",
            "• Conducted rigorous test set evaluation, precision/recall, & confusion matrix.",
            "• Measured sub-2ms CPU inference latency benchmarking.",
            "• Engineered the interactive web demo dashboard & presentation integration."
        ], COLOR_AMBER, Inches(8.8))
    ]

    for name, roll, role, points, col, l_pos in members:
        items = [roll, f"Domain: {role}", ""] + points
        add_card(s12, l_pos, Inches(1.8), Inches(3.7), Inches(5.2), name, items, col, 15, 11)

    # =========================================================================
    # SLIDE 13: Summary, Future Work & Q&A
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_background(s13)
    add_header(s13, "Summary, Future Scope & Discussion")

    add_card(s13, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.0),
             "Project Summary & Key Takeaways", [
                 "• Successfully delivered an end-to-end automated pipeline from legacy pickle data to edge deployment.",
                 "• Achieved 98.02% test accuracy using a compact MobileNetV3-Small architecture (< 1.5M params).",
                 "• Optimized model into an ultra-lightweight ONNX binary (5.83 MB) running in ~1.8 ms on standard CPUs.",
                 "• Validated working CLI and interactive web demonstration tools for industrial semiconductor manufacturing."
             ], COLOR_CYAN, 16, 13)

    add_card(s13, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.0),
             "Phase-2 Future Enhancements", [
                 "• Self-Supervised Pretraining: Leverage contrastive learning (SimCLR/MAE) to further improve rare defect classes (Scratch, Center).",
                 "• Embedded Hardware Benchmarking: Deploy ONNX model on NVIDIA Jetson Orin Nano and Raspberry Pi 5 with TensorRT acceleration.",
                 "• Mixed Defect Pattern Detection: Extend architecture to multi-label classification for concurrent defect occurrence.",
                 "• Real-Time Fab API: Build REST/gRPC microservice for direct integration with automated wafer probers."
             ], COLOR_EMERALD, 16, 13)

    # Bottom Thank You / Q&A Banner
    add_card(s13, Inches(0.8), Inches(6.0), Inches(11.7), Inches(1.1),
             "Thank You! Questions & Discussion", [
                 "We sincerely thank our Supervisor Er. Justin Mathew and faculty evaluators. We look forward to your valuable feedback and guidance."
             ], COLOR_AMBER, 14, 13)

    output_path = "Semiconductor_Defect_Classification_Review.pptx"
    prs.save(output_path)
    print(f"[SUCCESS] PowerPoint presentation saved to: {output_path}")

if __name__ == "__main__":
    create_deck()
