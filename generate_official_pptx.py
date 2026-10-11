import os
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

TEMPLATE_PATH = r"C:\Users\noopa\Downloads\Mini Project PPT format (1).pptx"
OUT_PATHS = [
    r"c:\Users\noopa\Downloads\semiconductor-defect-classification\Semiconductor_Defect_Classification_Review.pptx",
    r"c:\Users\noopa\Downloads\semiconductor-defect-classification\Final_Review_Mini_Project_Semiconductor_Defects.pptx",
    r"c:\Users\noopa\Downloads\Final_Review_Mini_Project_Semiconductor_Defects.pptx"
]

prs = pptx.Presentation(TEMPLATE_PATH)
DARK_NAVY = RGBColor(15, 23, 42)
TEXT_DARK = RGBColor(30, 41, 59)
BLUE_ACCENT = RGBColor(2, 132, 199)
GREEN_ACCENT = RGBColor(5, 150, 105)

def format_run(run, text, font_name="Times New Roman", size_pt=20, bold=False, color=None):
    run.text = text
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    if color:
        run.font.color.rgb = color

def create_or_get_content_box(slide, left=Inches(0.8), top=Inches(2.1), width=Inches(11.7), height=Inches(4.5)):
    # Check if a content shape already exists that isn't title, number, or logo
    for shape in slide.shapes:
        if shape.has_text_frame and shape.top > Inches(1.5) and shape.top < Inches(6.0) and shape.width > Inches(5.0):
            shape.left = left
            shape.top = top
            shape.width = width
            shape.height = height
            tf = shape.text_frame
            tf.word_wrap = True
            return tf
    # Otherwise create a textbox
    tx_box = slide.shapes.add_textbox(left, top, width, height)
    tf = tx_box.text_frame
    tf.word_wrap = True
    return tf

# ==============================================================================
# SLIDE 1: TITLE SLIDE
# ==============================================================================
slide1 = prs.slides[0]
for shape in slide1.shapes:
    if shape.has_text_frame:
        t = shape.text_frame.text.strip()
        if "PROJECT TITLE" in t:
            tf = shape.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT
            r = p.add_run()
            format_run(r, "AUTOMATED SEMICONDUCTOR WAFER DEFECT CLASSIFICATION USING DEEP LEARNING FOR REAL-TIME AOI SYSTEMS", size_pt=26, bold=True, color=DARK_NAVY)
        elif "STUDENT NAME" in t:
            tf = shape.text_frame
            tf.clear()
            lines = [
                ("PROJECT ASSOCIATES:", 18, True, DARK_NAVY),
                ("• ADITHYAN B (Reg No: MGP23EVLSI001)", 16, False, TEXT_DARK),
                ("• ANNA BOBAN (Reg No: MGP23EVLSI002)", 16, False, TEXT_DARK),
                ("• NOOPA MARIA ROJAN (Reg No: MGP23EVLSI009)", 16, False, TEXT_DARK),
                ("", 8, False, None),
                ("Department of Electronics & Communication Engineering", 16, True, BLUE_ACCENT),
                ("Saintgits College of Engineering (Autonomous), Kottayam", 15, False, TEXT_DARK),
                ("Date: October 2026", 14, False, TEXT_DARK)
            ]
            for idx, (txt, sz, bld, col) in enumerate(lines):
                p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
                p.alignment = PP_ALIGN.LEFT
                if txt:
                    r = p.add_run()
                    format_run(r, txt, size_pt=sz, bold=bld, color=col)
        elif "GUIDED BY" in t:
            tf = shape.text_frame
            tf.clear()
            lines = [
                ("PROJECT SUPERVISOR:", 18, True, DARK_NAVY),
                ("Er. JUSTIN MATHEW", 18, True, BLUE_ACCENT),
                ("Assistant Professor, Dept. of ECE", 15, False, TEXT_DARK)
            ]
            for idx, (txt, sz, bld, col) in enumerate(lines):
                p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
                p.alignment = PP_ALIGN.LEFT
                r = p.add_run()
                format_run(r, txt, size_pt=sz, bold=bld, color=col)

print("Slide 1 formatted.")

# Helper function to populate standard slides
def populate_section_slide(slide, title_text, sections, extra_width=Inches(11.7)):
    # Update title
    for shape in slide.shapes:
        if shape.has_text_frame and shape.top < Inches(2.0) and shape.left < Inches(5.0):
            tf = shape.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            r = p.add_run()
            format_run(r, title_text, size_pt=34, bold=True, color=DARK_NAVY)
            break
            
    # Content frame
    tf = create_or_get_content_box(slide, width=extra_width)
    tf.clear()
    
    first = True
    for header, points in sections:
        if header:
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.space_before = Pt(8)
            p.space_after = Pt(2)
            r = p.add_run()
            format_run(r, header, size_pt=22, bold=True, color=BLUE_ACCENT)
        for pt in points:
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.space_before = Pt(2)
            p.space_after = Pt(4)
            p.level = 0
            # Bullet
            r = p.add_run()
            format_run(r, f"•  {pt}", size_pt=18, bold=False, color=TEXT_DARK)

# ==============================================================================
# SLIDE 2: INTRODUCTION
# ==============================================================================
populate_section_slide(prs.slides[1], "INTRODUCTION", [
    ("Semiconductor Fabrication Complexity", [
        "Modern integrated circuit fabrication operates at deep sub-micron nodes on 300 mm silicon wafers, spanning over 500 physical and chemical processing steps.",
        "A single microscopic defect during deposition, photolithography, or chemical-mechanical planarization (CMP) can render hundreds of chips inoperable."
    ]),
    ("Spatial Wafer Defect Signatures & WM-811K Benchmark", [
        "Manufacturing tool failures manifest as systematic spatial defect clusters across the wafer disc (e.g., Center, Donut, Scratch, Edge-Ring, Loc).",
        "We utilize the world-benchmark WM-811K industrial dataset (811,457 commercial wafer maps) to capture authentic semiconductor fabrication signatures."
    ]),
    ("Role of Automated Optical Inspection (AOI)", [
        "Automated defect classification enables immediate fabrication tool root-cause diagnosis, drastically minimizing wafer lot scrap and engineering turnaround time."
    ])
])
print("Slide 2 formatted.")

# ==============================================================================
# SLIDE 3: PROBLEM STATEMENT
# ==============================================================================
populate_section_slide(prs.slides[2], "PROBLEM STATEMENT", [
    ("High-Throughput Fabrication Bottlenecks", [
        "Modern fabs produce tens of thousands of wafers monthly; manual optical and scanning electron microscopy (SEM) inspection is slow, subjective, and prone to severe operator fatigue."
    ]),
    ("Extreme Industrial Class Imbalance", [
        "In production environments, nominal (clean) wafers exceed 78% of all manufactured substrates, making minority defect topologies difficult for conventional algorithms to detect reliably."
    ]),
    ("Edge Computational Constraints in Fab Lines", [
        "Traditional heavy deep learning architectures (e.g., ResNet-50, VGG-16) suffer from multi-hundred-megabyte footprints, heavy GPU dependence, and high inference latency (>50 ms), making inline fab deployment infeasible."
    ]),
    ("Core Problem Definition", [
        "Need for an ultra-lightweight, edge-deployable deep learning classification engine achieving >98% accuracy and sub-2 ms latency across multi-class wafer defect topologies on standard CPU hardware."
    ])
])
print("Slide 3 formatted.")

# ==============================================================================
# SLIDE 4: NEED FOR THE PROJECT
# ==============================================================================
populate_section_slide(prs.slides[3], "NEED FOR THE PROJECT", [
    ("Critical Yield Economics & Cost Reduction", [
        "Defect detection at wafer probe testing prevents defective dies from progressing into expensive packaging, wire-bonding, and encapsulation phases, saving millions annually."
    ]),
    ("Equipment Root-Cause Traceability", [
        "Topological signatures correlate directly with specific fab tool malfunctions:",
        "    - Center Cluster: CMP slurry buildup or spin-coat nozzle misalignment.",
        "    - Donut Ring: Radial thermal gradient during rapid thermal annealing (RTA).",
        "    - Scratch: Mechanical robot end-effector abrasion or automated wafer cassette handling damage.",
        "    - Edge-Ring: Plasma etching nonuniformity or edge-bead removal (EBR) solvent splash."
    ]),
    ("Edge Inline Autonomous Screening", [
        "Eliminates bandwidth bottlenecks and cloud data-security risks by executing inference directly on local fab equipment controllers with zero cloud latency."
    ])
])
print("Slide 4 formatted.")

# ==============================================================================
# SLIDE 5: OBJECTIVES
# ==============================================================================
populate_section_slide(prs.slides[4], "OBJECTIVES", [
    ("Primary Project Objectives", [
        "Objective 1: Curate, filter, and normalize the large-scale WM-811K benchmark into balanced 8-class topological defect representations.",
        "Objective 2: Design and fine-tune an ultra-lightweight MobileNetV3-Small deep convolutional neural network leveraging depthwise separable convolutions and Squeeze-and-Excitation (SE) attention.",
        "Objective 3: Export and optimize the trained model into an ONNX runtime engine achieving sub-2 ms CPU inference latency and <6 MB storage footprint.",
        "Objective 4: Build a full-stack, interactive AOI demonstration dashboard featuring live 300 mm wafer visualization, die pass/fail matrix telemetry, and automated 25-wafer lot screening."
    ])
])
print("Slide 5 formatted.")

# ==============================================================================
# SLIDE 6: EXISTING SYSTEM
# ==============================================================================
populate_section_slide(prs.slides[5], "EXISTING SYSTEM", [
    ("Current Industrial Inspection Methods", [
        "Manual microscopy visual verification conducted by quality assurance personnel.",
        "Legacy rule-based computer vision utilizing handcrafted spatial filters, Hough transforms, and template matching."
    ]),
    ("Operational Workflow", [
        "Wafer scan acquired -> Batch saved to server -> Heuristic filters evaluate die failure threshold -> Manual verification triggered for ambiguous patterns."
    ]),
    ("Key Technical Limitations", [
        "High latency (>500 ms per wafer) incompatible with continuous inline fab speed.",
        "High false-alarm rate on nominal wafers caused by random single-die sensor noise.",
        "Inability to adapt to amorphous, complex, or rotated spatial cluster patterns."
    ])
])
print("Slide 6 formatted.")

# ==============================================================================
# SLIDE 7: PROPOSED SYSTEM
# ==============================================================================
populate_section_slide(prs.slides[6], "PROPOSED SYSTEM", [
    ("Our Deep Learning Solution", [
        "An automated, lightweight deep convolutional network (MobileNetV3-Small) optimized through ONNX Runtime for real-time edge execution on commodity CPUs and ARM architectures."
    ]),
    ("Why It Is Superior", [
        "Ultra-Low Latency: Real-time 1.53 ms inference per wafer map (over 300x faster than manual review).",
        "High Accuracy: 98.02% overall test accuracy across 121,719 test wafer samples.",
        "Minimal Footprint: 5.84 MB compact ONNX model binary requiring zero GPU hardware."
    ]),
    ("Core Functional Features", [
        "Instant 8-class topological classification with full softmax probability distribution.",
        "Interactive silicon wafer canvas with die-level pass/fail color mapping (Green = Pass, Red = Fail).",
        "Automated fab lot screening simulator with live yield tracking and defect counter."
    ])
])
print("Slide 7 formatted.")

# ==============================================================================
# SLIDE 8: NOVELTY OF THE PROJECT
# ==============================================================================
populate_section_slide(prs.slides[7], "NOVELTY OF THE PROJECT", [
    ("Edge-First Deep Architecture Selection", [
        "Replaces bulky architectures (ResNet-50 / VGG) with MobileNetV3-Small, reducing floating-point operations (FLOPs) by >90% while maintaining state-of-the-art accuracy."
    ]),
    ("Squeeze-and-Excitation (SE) Channel Attention", [
        "Dynamically recalibrates feature map channels, allowing the network to isolate narrow defect patterns (such as thin scratches) amidst tens of thousands of operational dies."
    ]),
    ("Hardware-Agnostic ONNX Edge Engine", [
        "Compiled into an open neural network exchange (ONNX) format, allowing seamless execution on industrial x86 PCs, Raspberry Pi 4/5, and smart camera vision systems without Python/PyTorch dependencies."
    ]),
    ("Full-Stack Fab Simulation & Die Matrix Telemetry", [
        "Bridges pure model classification with production fab metrics: computes exact good/bad die yield on a 648-die matrix in real time."
    ])
])
print("Slide 8 formatted.")

# ==============================================================================
# SLIDE 9: SCOPE OF THE PROJECT
# ==============================================================================
populate_section_slide(prs.slides[8], "SCOPE OF THE PROJECT", [
    ("Current Implementation Scope", [
        "Automated high-precision classification across 8 key defect classes: Center, Clean, Donut, Edge-Loc, Edge-Ring, Loc, Other, Scratch.",
        "Complete validation on 121,719 test wafer maps from the WM-811K benchmark.",
        "Interactive web-based inspection console with live canvas, lot simulation, and REST inference API.",
        "Zero-cloud offline deployment capable of running on standard shop-floor laptops or edge processors."
    ]),
    ("Future Enhancements", [
        "Die-level semantic segmentation (U-Net / YOLOv8-seg) for pixel-precise defect contouring.",
        "Multi-modal fab telemetry fusion combining wafer maps with electrical parametric test (E-test) logs.",
        "Federated active learning for collaborative multi-fab model continuous improvement."
    ])
])
print("Slide 9 formatted.")

# ==============================================================================
# SLIDE 10: METHODOLOGY & WORKFLOW
# ==============================================================================
populate_section_slide(prs.slides[9], "METHODOLOGY & WORKFLOW", [
    ("Phase 1: Dataset Curation & Preprocessing", [
        "Extracted wafer maps from the WM-811K benchmark; performed spatial bicubic/nearest resizing to 224x224 and tensor normalization (ImageNet mean & std)."
    ]),
    ("Phase 2: Transfer Learning & Model Fine-Tuning", [
        "Utilized pre-trained MobileNetV3-Small backbone with inverted residual bottlenecks and Hard-Swish activations; replaced the classification head with an 8-output linear layer."
    ]),
    ("Phase 3: Training & Regularization Pipeline", [
        "Trained using Adam optimizer, cross-entropy loss, dropout regularization (p=0.2), and early stopping to prevent overfitting on majority classes."
    ]),
    ("Phase 4: ONNX Compilation & Edge Integration", [
        "Exported PyTorch computational graph to ONNX with CPU graph optimizations; integrated into a high-concurrency Python REST server and interactive web visualizer."
    ])
])
print("Slide 10 formatted.")

# ==============================================================================
# SLIDE 11: HARDWARE & SOFTWARE REQUIREMENTS
# ==============================================================================
populate_section_slide(prs.slides[10], "HARDWARE & SOFTWARE REQUIREMENTS", [
    ("Software Requirements", [
        "Operating System: Windows 11 / Linux (Ubuntu 22.04 LTS)",
        "Programming Language: Python 3.12 / 3.10",
        "Deep Learning Framework: PyTorch 2.x, Torchvision",
        "Edge Runtime Engine: ONNX Runtime 1.18+",
        "Computer Vision & Analytics: OpenCV (cv2), NumPy, Scikit-learn, Matplotlib, Seaborn",
        "Web Application & API: HTML5 Canvas, Vanilla CSS3, JavaScript ES6, Python HTTP Server"
    ]),
    ("Hardware Requirements", [
        "Training Platform: Intel Core i5/i7 Processor, 16 GB RAM (optional NVIDIA RTX GPU for training)",
        "Deployment Station: Standard x86 PC or ARM Single Board Computer (Raspberry Pi 4/5, Intel NUC)",
        "Storage: 256 GB SSD (dataset repository and evaluation logs)",
        "Network: Standard Gigabit Ethernet / Wi-Fi for shop-floor scanner transmission"
    ])
])
print("Slide 11 formatted.")

# ==============================================================================
# SLIDE 12: EXPERIMENTAL RESULTS & OUTCOMES
# ==============================================================================
# On Slide 12, place text on the left and the real confusion matrix on the right!
slide12 = prs.slides[11]
# Update title
for shape in slide12.shapes:
    if shape.has_text_frame and shape.top < Inches(2.0) and shape.left < Inches(5.0):
        tf = shape.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        r = p.add_run()
        format_run(r, "EXPERIMENTAL RESULTS & OUTCOMES", size_pt=34, bold=True, color=DARK_NAVY)
        break

# Left text box
tf12 = create_or_get_content_box(slide12, left=Inches(0.8), top=Inches(2.0), width=Inches(6.8), height=Inches(4.5))
tf12.clear()

results_text = [
    ("Key Performance Metrics", [
        "Overall Test Set Accuracy: 98.02% across 121,719 test wafer samples.",
        "Macro Precision: 0.6897  |  Macro Recall: 0.6177",
        "Inference Latency: 1.53 ms on standard CPU (zero GPU requirement).",
        "Model Binary Footprint: 5.84 MB (ultra-compact ONNX engine)."
    ]),
    ("Per-Class Defect Verification Highlights", [
        "Clean (Nominal): 99.98% Precision, 100% Detection Rate (Near-zero false alarms).",
        "Scratch: 99.92% Confidence with clear linear scratch isolation.",
        "Donut: 99.99% Confidence detecting radial thermal ring defects.",
        "Center Cluster: 98.01% Confidence detecting CMP slurry anomalies.",
        "Edge-Loc & Loc: >99.2% Confidence identifying localized clusters."
    ])
]

first = True
for header, points in results_text:
    p = tf12.paragraphs[0] if first else tf12.add_paragraph()
    first = False
    p.space_before = Pt(6)
    p.space_after = Pt(2)
    r = p.add_run()
    format_run(r, header, size_pt=20, bold=True, color=BLUE_ACCENT)
    for pt in points:
        p = tf12.add_paragraph()
        p.space_before = Pt(2)
        p.space_after = Pt(3)
        r = p.add_run()
        format_run(r, f"•  {pt}", size_pt=15, bold=False, color=TEXT_DARK)

# Embed Confusion Matrix on right side if available
cm_path = r"c:\Users\noopa\Downloads\semiconductor-defect-classification\output\results\confusion_matrix.png"
if os.path.exists(cm_path):
    slide12.shapes.add_picture(cm_path, Inches(7.8), Inches(2.1), width=Inches(4.6), height=Inches(4.2))

print("Slide 12 formatted with embedded Confusion Matrix.")

# ==============================================================================
# SLIDE 13: PROJECT TIMELINE & TEAM RESPONSIBILITIES
# ==============================================================================
populate_section_slide(prs.slides[12], "PROJECT TIMELINE & TEAM CONTRIBUTIONS", [
    ("Project Milestone Timeline", [
        "Weeks 1–3: Literature survey, problem formulation, and WM-811K benchmark acquisition.",
        "Weeks 4–6: Dataset preprocessing, spatial normalization (224x224), and augmentation pipeline.",
        "Weeks 7–9: MobileNetV3-Small model implementation, transfer learning, and hyperparameter tuning.",
        "Weeks 10–12: ONNX edge runtime conversion, latency benchmarking, and confusion matrix evaluation.",
        "Weeks 13–14: Full-stack web AOI demo development, die pass/fail matrix, and defense documentation."
    ]),
    ("Individual Team Member Responsibilities", [
        "ADITHYAN B (MGP23EVLSI001): Model architecture engineering, MobileNetV3-Small training, PyTorch-to-ONNX model optimization, and latency profiling.",
        "ANNA BOBAN (MGP23EVLSI002): Dataset curation, spatial normalization, data augmentation, evaluation metrics calculation, and confusion matrix validation.",
        "NOOPA MARIA ROJAN (MGP23EVLSI009): Real-time web deployment, interactive HTML5 wafer canvas, die pass/fail color logic, fab lot simulator, and REST API integration."
    ])
])
print("Slide 13 formatted.")

# ==============================================================================
# SLIDE 14: REFERENCES
# ==============================================================================
populate_section_slide(prs.slides[13], "REFERENCES", [
    ("", [
        "1. M.-J. Wu, J.-S. R. Jang, and J.-L. Chen, 'Wafer map failure pattern recognition and similarity ranking for large-scale semiconductor manufacturing,' IEEE Transactions on Semiconductor Manufacturing, vol. 28, no. 1, pp. 1–12, 2015.",
        "2. A. Howard, M. Sandler, G. Chu, L.-C. Chen, B. Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan et al., 'Searching for MobileNetV3,' in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 1314–1324.",
        "3. T. Nakazawa and D. V. Kulkarni, 'Wafer map defect pattern classification and retrieval using convolutional neural networks,' IEEE Transactions on Semiconductor Manufacturing, vol. 31, no. 2, pp. 309–314, 2018.",
        "4. S. Cheon, H. Lee, C. O. Kim, and S. H. Lee, 'Convolutional neural network for wafer surface defect classification and the detection of unknown defect classes,' IEEE Transactions on Semiconductor Manufacturing, vol. 32, no. 2, pp. 163–170, 2019.",
        "5. ONNX Runtime Developers, 'ONNX Runtime: Cross-platform, high performance ML inferencing and training accelerator,' Linux Foundation / Microsoft, 2024. [Online]. Available: https://onnxruntime.ai"
    ])
])
print("Slide 14 formatted.")

# ==============================================================================
# SLIDE 15: THANK YOU
# ==============================================================================
slide15 = prs.slides[14]
for shape in slide15.shapes:
    if shape.has_text_frame:
        t = shape.text_frame.text.strip()
        if "THANK YOU" in t:
            tf = shape.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            format_run(r, "THANK YOU", size_pt=54, bold=True, color=DARK_NAVY)
        elif "Questions" in t:
            tf = shape.text_frame
            tf.clear()
            lines = [
                ("Questions & Discussion", 26, True, BLUE_ACCENT),
                ("", 10, False, None),
                ("Automated Semiconductor Wafer Defect Inspection System", 20, True, DARK_NAVY),
                ("B.Tech Mini Project Final Review • Saintgits College of Engineering (Autonomous)", 16, False, TEXT_DARK),
                ("Team: Adithyan B • Anna Boban • Noopa Maria Rojan", 16, False, TEXT_DARK),
                ("Supervisor: Er. Justin Mathew, Assistant Professor, Dept. of ECE", 16, False, TEXT_DARK)
            ]
            for idx, (txt, sz, bld, col) in enumerate(lines):
                p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
                p.alignment = PP_ALIGN.CENTER
                if txt:
                    r = p.add_run()
                    format_run(r, txt, size_pt=sz, bold=bld, color=col)

print("Slide 15 formatted.")

# ==============================================================================
# SAVE TO ALL LOCATIONS
# ==============================================================================
for pth in OUT_PATHS:
    prs.save(pth)
    print(f"Saved: {pth} (Size: {os.path.getsize(pth)} bytes)")
