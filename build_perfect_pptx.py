import os
import win32com.client

TEMPLATE_PATH = os.path.abspath(r"C:\Users\noopa\Downloads\Mini Project PPT format (1).pptx")
OUT_PATHS = [
    os.path.abspath(r"C:\Users\noopa\Downloads\Semiconductor_Defect_Classification_Review.pptx"),
    os.path.abspath(r"c:\Users\noopa\Downloads\semiconductor-defect-classification\Semiconductor_Defect_Classification_Review.pptx"),
    os.path.abspath(r"C:\Users\noopa\Downloads\Final_Review_Mini_Project_Semiconductor_Defects.pptx"),
    os.path.abspath(r"c:\Users\noopa\Downloads\semiconductor-defect-classification\Final_Review_Mini_Project_Semiconductor_Defects.pptx")
]

# Absolute image paths
IMG_CENTER = os.path.abspath(r"c:\Users\noopa\Downloads\semiconductor-defect-classification\wafer_images\center\center_3837.png")
IMG_DONUT = os.path.abspath(r"c:\Users\noopa\Downloads\semiconductor-defect-classification\wafer_images\donut\donut_252319.png")
IMG_EDGELOC = os.path.abspath(r"c:\Users\noopa\Downloads\semiconductor-defect-classification\wafer_images\edge_loc\edge_loc_2565.png")
IMG_SCRATCH = os.path.abspath(r"c:\Users\noopa\Downloads\semiconductor-defect-classification\wafer_images\scratch\scratch_129086.png")
IMG_CM = os.path.abspath(r"c:\Users\noopa\Downloads\semiconductor-defect-classification\output\results\confusion_matrix.png")

# Start PowerPoint
ppt = win32com.client.Dispatch("PowerPoint.Application")
try:
    pres = ppt.Presentations.Open(TEMPLATE_PATH, ReadOnly=False, Untitled=True, WithWindow=False)
    print("Opened template via PowerPoint COM. Initial slides:", pres.Slides.Count)

    # 1. Delete slides 13, 11, 9, 8, 7, 6 (1-based index from highest to lowest)
    to_delete = [13, 11, 9, 8, 7, 6]
    for idx in to_delete:
        pres.Slides(idx).Delete()
    print("Remaining slides after deleting 6 slides:", pres.Slides.Count) # 9 slides

    # 2. Duplicate Slide 5 twice to create slides 6 and 7
    # Slide 5 has the official layout, logos, and bottom banner
    dup1 = pres.Slides(5).Duplicate()
    dup1.MoveTo(6)
    dup2 = pres.Slides(5).Duplicate()
    dup2.MoveTo(7)
    print("Total slides after duplicating slides 6 and 7:", pres.Slides.Count) # 11 slides

    # Colors
    RGB_NAVY = 0x2A170F     # BGR in COM: 15, 23, 42
    RGB_TEXT = 0x3B291E     # BGR in COM: 30, 41, 59
    RGB_BLUE = 0xC78402     # BGR in COM: 2, 132, 199
    RGB_CARD = 0x3B291E     # Slate card
    RGB_WHITE = 0xFFFFFF

    # Helper: set text on shape
    def set_title(slide, text):
        for sh in slide.Shapes:
            if sh.HasTextFrame and sh.Top < 150 and sh.Left < 400:
                sh.TextFrame.TextRange.Text = text
                sh.TextFrame.TextRange.Font.Name = "Times New Roman"
                sh.TextFrame.TextRange.Font.Bold = True
                sh.TextFrame.TextRange.Font.Size = 34
                sh.TextFrame.TextRange.Font.Color.RGB = 0x241C18
                return sh
        return None

    def clear_content_shapes(slide):
        # Clear any old content box
        for sh in list(slide.Shapes):
            if sh.HasTextFrame and sh.Top > 100 and sh.Top < 450:
                t = sh.TextFrame.TextRange.Text.strip()
                if "LEARN" not in t and "INTRODUCTION" not in t and "PROBLEM" not in t and "NEED" not in t and "OBJECTIVES" not in t and "SOFTWARE" not in t and "EXPECTED" not in t and "REFERENCES" not in t and "THANK" not in t:
                    sh.Delete()

    # --------------------------------------------------------------------------
    # SLIDE 1: Title Slide
    # --------------------------------------------------------------------------
    s1 = pres.Slides(1)
    for sh in s1.Shapes:
        if sh.HasTextFrame:
            t = sh.TextFrame.TextRange.Text
            if "PROJECT TITLE" in t:
                sh.TextFrame.TextRange.Text = "AUTOMATED SEMICONDUCTOR WAFER DEFECT CLASSIFICATION USING DEEP LEARNING FOR REAL-TIME AOI SYSTEMS"
                sh.TextFrame.TextRange.Font.Name = "Times New Roman"
                sh.TextFrame.TextRange.Font.Size = 24
                sh.TextFrame.TextRange.Font.Bold = True
                sh.TextFrame.TextRange.Font.Color.RGB = 0x241C18
            elif "STUDENT NAME" in t:
                sh.TextFrame.TextRange.Text = (
                    "PROJECT ASSOCIATES:\n"
                    "• ADITHYAN B (Reg No: MGP23EVLSI001)\n"
                    "• ANNA BOBAN (Reg No: MGP23EVLSI002)\n"
                    "• NOOPA MARIA ROJAN (Reg No: MGP23EVLSI009)\n\n"
                    "Department of Electronics & Communication Engineering\n"
                    "Saintgits College of Engineering (Autonomous), Kottayam\n"
                    "B.Tech Mini Project Final Internal Review • October 2026"
                )
                sh.TextFrame.TextRange.Font.Name = "Times New Roman"
                sh.TextFrame.TextRange.Font.Size = 14
                sh.TextFrame.TextRange.Font.Color.RGB = 0x241C18
            elif "GUIDED BY" in t:
                sh.TextFrame.TextRange.Text = (
                    "PROJECT SUPERVISOR:\n"
                    "Er. JUSTIN MATHEW\n"
                    "Assistant Professor, Dept. of ECE"
                )
                sh.TextFrame.TextRange.Font.Name = "Times New Roman"
                sh.TextFrame.TextRange.Font.Size = 15
                sh.TextFrame.TextRange.Font.Bold = True
                sh.TextFrame.TextRange.Font.Color.RGB = 0xC78402
    print("Slide 1 formatted.")

    # Helper to add standard bullet text box
    def add_bullets(slide, items):
        clear_content_shapes(slide)
        tb = slide.Shapes.AddTextbox(1, 50, 140, 780, 360) # 1 = msoTextOrientationHorizontal
        tf = tb.TextFrame
        tf.WordWrap = -1
        full_text = ""
        for heading, bullets in items:
            full_text += f"{heading}\n"
            for b in bullets:
                full_text += f"•  {b}\n"
            full_text += "\n"
        tf.TextRange.Text = full_text.strip()
        tf.TextRange.Font.Name = "Times New Roman"
        tf.TextRange.Font.Size = 15
        tf.TextRange.Font.Color.RGB = 0x241C18
        
        # Style headings bold blue
        for heading, _ in items:
            idx = tf.TextRange.Text.find(heading)
            if idx != -1:
                r = tf.TextRange.Characters(idx + 1, len(heading))
                r.Font.Bold = True
                r.Font.Size = 17
                r.Font.Color.RGB = 0xC78402

    # --------------------------------------------------------------------------
    # SLIDE 2: INTRODUCTION
    # --------------------------------------------------------------------------
    s2 = pres.Slides(2)
    set_title(s2, "INTRODUCTION")
    add_bullets(s2, [
        ("Semiconductor Fabrication Complexity", [
            "Modern integrated circuits are manufactured on 300 mm silicon wafers spanning 500+ delicate chemical-mechanical processing steps at sub-10 nm nodes.",
            "A single microscopic defect during lithography, etch, or chemical-mechanical planarization (CMP) compromises hundreds of microchips."
        ]),
        ("Spatial Wafer Defect Signatures & WM-811K Benchmark", [
            "Equipment failures deposit spatially correlated defective dies across the substrate, generating characteristic topological clusters (Center, Donut, Scratch, Edge-Ring, Loc).",
            "We utilize the real-world WM-811K dataset (811,457 commercial wafer maps) to capture authentic industrial semiconductor defect patterns."
        ]),
        ("Automated Optical Inspection (AOI)", [
            "Instant automated defect classification enables immediate fabrication tool root-cause diagnosis, saving millions in scrap and engineering turnaround time."
        ])
    ])
    print("Slide 2 formatted.")

    # --------------------------------------------------------------------------
    # SLIDE 3: PROBLEM STATEMENT
    # --------------------------------------------------------------------------
    s3 = pres.Slides(3)
    set_title(s3, "PROBLEM STATEMENT")
    add_bullets(s3, [
        ("High-Throughput Fabrication Bottlenecks", [
            "Commercial fabs produce tens of thousands of wafers monthly; manual optical and SEM microscopy inspection is slow, subjective, and prone to severe human fatigue."
        ]),
        ("Extreme Industrial Class Imbalance", [
            "In commercial fabrication, nominal (clean) wafers exceed 78% of production lots, leaving minority defect topologies difficult for conventional algorithms to detect reliably."
        ]),
        ("Edge Computational Constraints in Fab Lines", [
            "Heavy deep learning backbones (e.g., ResNet-50, VGG-16) suffer from multi-hundred-megabyte footprints, heavy GPU dependence, and high inference latency (>50 ms)."
        ]),
        ("Core Problem Statement", [
            "Need for an ultra-lightweight, edge-deployable deep learning classification engine achieving >98% accuracy and sub-2 ms latency across multi-class wafer defect topologies on standard CPU hardware."
        ])
    ])
    print("Slide 3 formatted.")

    # --------------------------------------------------------------------------
    # SLIDE 4: NEED FOR THE PROJECT
    # --------------------------------------------------------------------------
    s4 = pres.Slides(4)
    set_title(s4, "NEED FOR THE PROJECT")
    add_bullets(s4, [
        ("Critical Yield Economics & Cost Reduction", [
            "Defect detection at wafer probe testing prevents defective dies from progressing into expensive packaging, wire-bonding, and encapsulation phases, saving millions annually."
        ]),
        ("Equipment Root-Cause Traceability", [
            "Topological signatures correlate directly with specific fab tool malfunctions:",
            "  - Center Cluster: CMP slurry buildup or spin-coat nozzle misalignment.",
            "  - Donut Ring: Radial thermal gradient during rapid thermal annealing (RTA).",
            "  - Scratch: Mechanical robot end-effector abrasion or wafer cassette handling damage.",
            "  - Edge-Ring: Plasma etching nonuniformity or edge-bead removal (EBR) solvent splash."
        ]),
        ("Edge Inline Autonomous Screening", [
            "Eliminates bandwidth bottlenecks and cloud data-security risks by executing inference directly on local fab equipment controllers with zero cloud latency."
        ])
    ])
    print("Slide 4 formatted.")

    # --------------------------------------------------------------------------
    # SLIDE 5: OBJECTIVES
    # --------------------------------------------------------------------------
    s5 = pres.Slides(5)
    set_title(s5, "OBJECTIVES")
    add_bullets(s5, [
        ("Primary Project Objectives", [
            "Objective 1: Curate, filter, and normalize the large-scale WM-811K benchmark into balanced 8-class topological defect representations.",
            "Objective 2: Design and fine-tune an ultra-lightweight MobileNetV3-Small deep convolutional neural network leveraging depthwise separable convolutions and Squeeze-and-Excitation (SE) attention.",
            "Objective 3: Export and optimize the trained model into an ONNX runtime engine achieving sub-2 ms CPU inference latency and <6 MB storage footprint.",
            "Objective 4: Build a full-stack, interactive AOI demonstration dashboard featuring live 300 mm wafer visualization, die pass/fail matrix telemetry, and automated inspection."
        ])
    ])
    print("Slide 5 formatted.")

    # --------------------------------------------------------------------------
    # SLIDE 6: END-TO-END SYSTEM ARCHITECTURE (Added Slide from Page 5 of review)
    # --------------------------------------------------------------------------
    s6 = pres.Slides(6)
    set_title(s6, "END-TO-END SYSTEM ARCHITECTURE")
    clear_content_shapes(s6)

    # Add 4 Pipeline Stage Cards
    stages = [
        ("STAGE A: DATA PIPELINE", [
            "• Raw LSWMD.pkl parsing",
            "• Pandas index patch",
            "• Bilinear resize 224x224",
            "• Min-Max normalization",
            "• 70% / 15% / 15% Split"
        ], 35),
        ("STAGE B: MODEL TRAINING", [
            "• Backbone: MobileNetV3",
            "• Inverted residuals + SE",
            "• Custom 8-class linear head",
            "• Adam optimizer (3e-4)",
            "• Cross-Entropy loss"
        ], 235),
        ("STAGE C: ONNX ENGINE", [
            "• PyTorch model graph trace",
            "• ONNX Opset 12 export",
            "• Constant folding & prune",
            "• Dynamic batch dimension",
            "• Model size: 5.84 MB"
        ], 435),
        ("STAGE D: EDGE INFERENCE", [
            "• ONNX Runtime CPU",
            "• ImageNet tensor norm",
            "• Top-3 confidence scoring",
            "• Sub-2ms inference latency",
            "• Standalone REST runner"
        ], 635)
    ]

    for title, items, left_x in stages:
        # Rounded rectangle shape
        card = s6.Shapes.AddShape(5, left_x, 140, 185, 230) # 5 = msoShapeRoundedRectangle
        card.Fill.Solid()
        card.Fill.ForeColor.RGB = 0xF5F0EB # clean light card
        card.Line.ForeColor.RGB = 0xC78402 # blue border
        card.Line.Weight = 1.5

        # Text box inside card
        tb = s6.Shapes.AddTextbox(1, left_x + 5, 145, 175, 220)
        tb.TextFrame.WordWrap = -1
        t_str = f"{title}\n" + "\n".join(items)
        tb.TextFrame.TextRange.Text = t_str
        tb.TextFrame.TextRange.Font.Name = "Times New Roman"
        tb.TextFrame.TextRange.Font.Size = 12
        tb.TextFrame.TextRange.Font.Color.RGB = 0x241C18

        # Style title
        idx = tb.TextFrame.TextRange.Text.find(title)
        if idx != -1:
            r = tb.TextFrame.TextRange.Characters(idx + 1, len(title))
            r.Font.Bold = True
            r.Font.Size = 13
            r.Font.Color.RGB = 0xC78402

    # Bottom Target Classes bar
    bar = s6.Shapes.AddShape(5, 35, 385, 785, 45)
    bar.Fill.Solid()
    bar.Fill.ForeColor.RGB = 0x241C18
    bar.TextFrame.TextRange.Text = "Target Defect Classes:  Center  |  Donut  |  Edge-Loc  |  Edge-Ring  |  Loc  |  Scratch  |  Other  |  Clean (Nominal)"
    bar.TextFrame.TextRange.Font.Name = "Times New Roman"
    bar.TextFrame.TextRange.Font.Size = 13
    bar.TextFrame.TextRange.Font.Bold = True
    bar.TextFrame.TextRange.Font.Color.RGB = 0xFFFFFF
    print("Slide 6 formatted.")

    # --------------------------------------------------------------------------
    # SLIDE 7: WM-811K DATASET & DEFECT PATTERNS (Added Slide from Page 6 with pictures)
    # --------------------------------------------------------------------------
    s7 = pres.Slides(7)
    set_title(s7, "WM-811K DATASET & DEFECT PATTERNS")
    clear_content_shapes(s7)

    # Left card: Dataset description
    card_left = s7.Shapes.AddShape(5, 35, 140, 370, 290)
    card_left.Fill.Solid()
    card_left.Fill.ForeColor.RGB = 0xF5F0EB
    card_left.Line.ForeColor.RGB = 0xC78402
    card_left.Line.Weight = 1.5

    tb_left = s7.Shapes.AddTextbox(1, 45, 145, 350, 280)
    tb_left.TextFrame.WordWrap = -1
    ds_text = (
        "Dataset Profile & Industrial Challenges\n"
        "• Total Records: 811,472 real semiconductor wafer maps from 46,393 lots.\n"
        "• Labeled Samples: ~172,950 annotated wafers with failure patterns.\n"
        "• Severe Imbalance: Over 96% of manufactured wafers are defect-free (Clean).\n"
        "• Array Representation: Variable resolution 2D matrices where 0 = exterior, 1 = normal die, 2 = defect die.\n"
        "• Preprocessing Pipeline:\n"
        "   1. Patch deprecated pandas Int64Index unpickling.\n"
        "   2. Map 9 raw labels to 8 standardized classes.\n"
        "   3. Standardize dimensions to 224x224 (bilinear).\n"
        "   4. 70% Train, 15% Validation, 15% Test split."
    )
    tb_left.TextFrame.TextRange.Text = ds_text
    tb_left.TextFrame.TextRange.Font.Name = "Times New Roman"
    tb_left.TextFrame.TextRange.Font.Size = 12
    tb_left.TextFrame.TextRange.Font.Color.RGB = 0x241C18

    # Style heading
    r = tb_left.TextFrame.TextRange.Characters(1, len("Dataset Profile & Industrial Challenges"))
    r.Font.Bold = True
    r.Font.Size = 14
    r.Font.Color.RGB = 0xC78402

    # Right side: 4 sample wafer pictures!
    pics = [
        (IMG_CENTER, "Center Defect", 430, 140),
        (IMG_DONUT, "Donut Defect", 630, 140),
        (IMG_EDGELOC, "Edge-Loc Defect", 430, 285),
        (IMG_SCRATCH, "Scratch Defect", 630, 285)
    ]

    for img_p, label, left_x, top_y in pics:
        if os.path.exists(img_p):
            p_obj = s7.Shapes.AddPicture(img_p, False, True, left_x, top_y, 175, 115)
            # Label box
            lbl = s7.Shapes.AddTextbox(1, left_x, top_y + 115, 175, 20)
            lbl.TextFrame.TextRange.Text = label
            lbl.TextFrame.TextRange.Font.Name = "Times New Roman"
            lbl.TextFrame.TextRange.Font.Size = 11
            lbl.TextFrame.TextRange.Font.Bold = True
            lbl.TextFrame.TextRange.Font.Color.RGB = 0x241C18
            lbl.TextFrame.TextRange.ParagraphFormat.Alignment = 2 # center
    print("Slide 7 formatted with 4 wafer defect pictures.")

    # --------------------------------------------------------------------------
    # SLIDE 8: SOFTWARE REQUIREMENTS & METHODOLOGY
    # --------------------------------------------------------------------------
    s8 = pres.Slides(8)
    set_title(s8, "SOFTWARE REQUIREMENTS & METHODOLOGY")
    add_bullets(s8, [
        ("Software Requirements & Frameworks", [
            "Operating System: Windows 11 / Linux (Ubuntu 22.04 LTS)",
            "Programming Language: Python 3.12 / 3.10",
            "Deep Learning Framework: PyTorch 2.x, Torchvision",
            "Edge Runtime Engine: ONNX Runtime 1.18+",
            "Computer Vision & Analytics: OpenCV (cv2), NumPy, Scikit-learn, Matplotlib, Seaborn",
            "Web Application & API: HTML5 Canvas, Vanilla CSS3, JavaScript ES6, Python HTTP Server"
        ]),
        ("4-Phase Implementation Methodology", [
            "Phase 1: Dataset Curation - WM-811K parsing, 224x224 bicubic resizing, ImageNet tensor normalization.",
            "Phase 2: Transfer Learning - Pre-trained MobileNetV3-Small backbone with custom 8-class linear head.",
            "Phase 3: Loss Optimization - Cross-entropy loss minimization with Adam optimizer (lr=3e-4) & early stopping.",
            "Phase 4: ONNX Deployment - Model graph trace, CPU graph optimizations, and interactive web visualizer."
        ])
    ])
    print("Slide 8 formatted.")

    # --------------------------------------------------------------------------
    # SLIDE 9: EXPERIMENTAL RESULTS & OUTCOMES (with Confusion Matrix picture)
    # --------------------------------------------------------------------------
    s9 = pres.Slides(9)
    set_title(s9, "EXPERIMENTAL RESULTS & OUTCOMES")
    clear_content_shapes(s9)

    # Left text box
    tb9 = s9.Shapes.AddTextbox(1, 35, 140, 430, 290)
    tb9.TextFrame.WordWrap = -1
    res_text = (
        "Key Performance Metrics\n"
        "• Overall Test Set Accuracy: 98.02% (121,719 test wafers).\n"
        "• Macro Precision: 0.6897  |  Macro Recall: 0.6177\n"
        "• Inference Latency: 1.53 ms on CPU (zero GPU requirement).\n"
        "• Model Footprint: 5.84 MB (ultra-compact ONNX binary).\n\n"
        "Per-Class Defect Verification Highlights\n"
        "• Clean (Nominal): 99.98% Precision, 100% Detection Rate.\n"
        "• Scratch: 99.92% Confidence with clear linear streak isolation.\n"
        "• Donut: 99.99% Confidence detecting radial thermal rings.\n"
        "• Center Cluster: 98.01% Confidence detecting CMP slurry anomalies.\n"
        "• Edge-Loc & Loc: >99.2% Confidence on localized boundary dies."
    )
    tb9.TextFrame.TextRange.Text = res_text
    tb9.TextFrame.TextRange.Font.Name = "Times New Roman"
    tb9.TextFrame.TextRange.Font.Size = 13
    tb9.TextFrame.TextRange.Font.Color.RGB = 0x241C18

    # Style headings
    r1 = tb9.TextFrame.TextRange.Characters(1, len("Key Performance Metrics"))
    r1.Font.Bold = True
    r1.Font.Size = 16
    r1.Font.Color.RGB = 0xC78402

    idx2 = res_text.find("Per-Class Defect Verification Highlights")
    if idx2 != -1:
        r2 = tb9.TextFrame.TextRange.Characters(idx2 + 1, len("Per-Class Defect Verification Highlights"))
        r2.Font.Bold = True
        r2.Font.Size = 16
        r2.Font.Color.RGB = 0xC78402

    # Embed Confusion Matrix on right side
    if os.path.exists(IMG_CM):
        s9.Shapes.AddPicture(IMG_CM, False, True, 485, 140, 335, 280)
    print("Slide 9 formatted with Confusion Matrix picture.")

    # --------------------------------------------------------------------------
    # SLIDE 10: REFERENCES
    # --------------------------------------------------------------------------
    s10 = pres.Slides(10)
    set_title(s10, "REFERENCES")
    clear_content_shapes(s10)

    tb10 = s10.Shapes.AddTextbox(1, 45, 140, 770, 300)
    tb10.TextFrame.WordWrap = -1
    refs = (
        "1. M.-J. Wu, J.-S. R. Jang, and J.-L. Chen, 'Wafer map failure pattern recognition and similarity ranking for large-scale semiconductor manufacturing,' IEEE Transactions on Semiconductor Manufacturing, vol. 28, no. 1, pp. 1–12, 2015.\n\n"
        "2. A. Howard, M. Sandler, G. Chu, L.-C. Chen, B. Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan et al., 'Searching for MobileNetV3,' in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 1314–1324.\n\n"
        "3. T. Nakazawa and D. V. Kulkarni, 'Wafer map defect pattern classification and retrieval using convolutional neural networks,' IEEE Transactions on Semiconductor Manufacturing, vol. 31, no. 2, pp. 309–314, 2018.\n\n"
        "4. S. Cheon, H. Lee, C. O. Kim, and S. H. Lee, 'Convolutional neural network for wafer surface defect classification and the detection of unknown defect classes,' IEEE Transactions on Semiconductor Manufacturing, vol. 32, no. 2, pp. 163–170, 2019.\n\n"
        "5. ONNX Runtime Developers, 'ONNX Runtime: Cross-platform, high performance ML inferencing and training accelerator,' Linux Foundation / Microsoft, 2024. [Online]. Available: https://onnxruntime.ai"
    )
    tb10.TextFrame.TextRange.Text = refs
    tb10.TextFrame.TextRange.Font.Name = "Times New Roman"
    tb10.TextFrame.TextRange.Font.Size = 13
    tb10.TextFrame.TextRange.Font.Color.RGB = 0x241C18
    print("Slide 10 formatted.")

    # --------------------------------------------------------------------------
    # SLIDE 11: THANK YOU (ONLY THANK YOU - Prominently Centered!)
    # --------------------------------------------------------------------------
    s11 = pres.Slides(11)
    # Clear all content shapes on slide 11
    for sh in list(s11.Shapes):
        if sh.HasTextFrame:
            t = sh.TextFrame.TextRange.Text.strip()
            if "LEARN" in t:
                # Keep bottom motto banner
                pass
            else:
                sh.Delete()

    # Create one big, bold, centered THANK YOU box
    tb11 = s11.Shapes.AddTextbox(1, 50, 190, 755, 120)
    tb11.TextFrame.WordWrap = -1
    tb11.TextFrame.TextRange.Text = "THANK YOU"
    tb11.TextFrame.TextRange.Font.Name = "Times New Roman"
    tb11.TextFrame.TextRange.Font.Size = 64
    tb11.TextFrame.TextRange.Font.Bold = True
    tb11.TextFrame.TextRange.Font.Color.RGB = 0x241C18
    tb11.TextFrame.TextRange.ParagraphFormat.Alignment = 2 # centered!
    print("Slide 11 formatted (ONLY THANK YOU).")

    # --------------------------------------------------------------------------
    # SAVE AS CLEAN PPTX NATIVELY VIA POWERPOINT
    # --------------------------------------------------------------------------
    for pth in OUT_PATHS:
        pres.SaveAs(pth)
        print(f"Saved: {pth}")

    pres.Close()
    print("Presentation saved and closed cleanly!")
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    ppt.Quit()
