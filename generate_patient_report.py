# generate_patient_report.py
# PDF Patient Report Generator using the best trained model

import os
import sys
import json
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from fpdf import FPDF

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from TrainData.transform import val_transform, IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    from transform import val_transform, IMAGENET_MEAN, IMAGENET_STD

from models import MODEL_REGISTRY


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
REPORTS_DIR = os.path.join(BASE_DIR, "patient_reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# Clinical information for each tumor type
CLINICAL_INFO = {
    "glioma": {
        "full_name": "Glioma",
        "description": (
            "Gliomas are tumors that arise from glial cells in the brain or spine. "
            "They are one of the most common types of primary brain tumors. "
            "Gliomas can be low-grade (slow-growing) or high-grade (aggressive)."
        ),
        "severity": "HIGH",
        "severity_color": (220, 50, 50),
        "common_symptoms": [
            "Persistent headaches (often worse in the morning)",
            "Seizures (new onset)",
            "Progressive neurological deficits",
            "Nausea and vomiting",
            "Memory and cognitive changes",
            "Personality or behavior changes"
        ],
        "recommended_actions": [
            "Urgent neurosurgical consultation within 48 hours",
            "Advanced MRI with contrast and spectroscopy",
            "Consider stereotactic biopsy for histological grading",
            "Multidisciplinary tumor board review",
            "Baseline neurocognitive assessment",
            "Discuss treatment options: surgery, radiation, chemotherapy"
        ],
        "prognosis_note": (
            "Prognosis varies significantly by grade. Low-grade gliomas may have "
            "median survival of 5-10+ years; high-grade (GBM) has median survival "
            "of 12-18 months with standard treatment."
        )
    },
    "meningioma": {
        "full_name": "Meningioma",
        "description": (
            "Meningiomas arise from the meninges, the membranes surrounding the brain "
            "and spinal cord. Most meningiomas are benign (WHO Grade I) and slow-growing. "
            "They are the most common primary intracranial tumor."
        ),
        "severity": "MODERATE",
        "severity_color": (230, 150, 30),
        "common_symptoms": [
            "Gradual onset headaches",
            "Vision changes or visual field deficits",
            "Hearing loss (if near auditory nerve)",
            "Weakness in limbs",
            "Seizures (less common)",
            "Balance and coordination difficulties"
        ],
        "recommended_actions": [
            "Neurosurgical consultation within 1-2 weeks",
            "Contrast-enhanced MRI for detailed characterization",
            "Serial imaging surveillance if small and asymptomatic",
            "Consider surgical resection if symptomatic or growing",
            "Ophthalmologic evaluation if near optic pathway",
            "Discuss observation vs. intervention based on size/location"
        ],
        "prognosis_note": (
            "WHO Grade I meningiomas have excellent prognosis with >90% "
            "5-year survival after complete resection. Recurrence rates are "
            "low (7-20%) for benign tumors."
        )
    },
    "pituitary": {
        "full_name": "Pituitary Tumor (Adenoma)",
        "description": (
            "Pituitary adenomas are benign tumors of the pituitary gland. "
            "They can be functioning (hormone-producing) or non-functioning. "
            "They may cause hormonal imbalances or compress nearby structures."
        ),
        "severity": "MODERATE",
        "severity_color": (230, 150, 30),
        "common_symptoms": [
            "Hormonal imbalance symptoms (irregular periods, growth changes)",
            "Visual field deficits (bitemporal hemianopia)",
            "Chronic headaches",
            "Fatigue and weakness",
            "Unexplained weight changes",
            "Galactorrhea (inappropriate milk production)"
        ],
        "recommended_actions": [
            "Endocrinology consultation for hormone panel assessment",
            "Formal visual field testing (Humphrey perimetry)",
            "Dedicated pituitary MRI protocol with contrast",
            "Complete pituitary hormone panel (GH, PRL, ACTH, TSH, FSH/LH)",
            "Consider medical management for prolactinomas (dopamine agonists)",
            "Trans-sphenoidal surgery if medical therapy fails or non-functioning"
        ],
        "prognosis_note": (
            "Pituitary adenomas generally have favorable outcomes. "
            "Surgical cure rates for microadenomas exceed 80%. "
            "Long-term endocrine follow-up is essential."
        )
    },
    "notumor": {
        "full_name": "No Tumor Detected",
        "description": (
            "The MRI scan analysis indicates no evidence of a brain tumor. "
            "The brain parenchyma appears within normal limits based on the "
            "AI classification model's assessment."
        ),
        "severity": "LOW",
        "severity_color": (50, 160, 80),
        "common_symptoms": [],
        "recommended_actions": [
            "Routine follow-up with primary care physician",
            "Address presenting symptoms through differential diagnosis",
            "Consider alternative causes for presenting symptoms",
            "Routine screening MRI if clinically indicated",
            "Patient reassurance and education"
        ],
        "prognosis_note": (
            "No tumor identified on this scan. If symptoms persist or worsen, "
            "clinical re-evaluation and repeat imaging may be warranted."
        )
    }
}


# ──────────────────────────────────────────────
# Load Best Model
# ──────────────────────────────────────────────
def load_best_model():
    """Load the best performing model from evaluation results."""

    best_model_path = os.path.join(RESULTS_DIR, "best_model.json")
    if os.path.exists(best_model_path):
        with open(best_model_path) as f:
            best_info = json.load(f)
        model_name = best_info["best_model"]
        ckpt_path = best_info["checkpoint"]
    else:
        # Default fallback
        print("  ⚠ No best_model.json found. Using EfficientNet-B0.")
        model_name = "EfficientNet-B0"
        ckpt_path = os.path.join(CHECKPOINTS_DIR, "efficientnet_b0.pth")

    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(BASE_DIR, ckpt_path)

    print(f"  Loading best model: {model_name}")
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model, model_name


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────
def predict_image(model, image_path):
    """Run inference on a single MRI image."""

    img = Image.open(image_path).convert("RGB")
    img_tensor = val_transform(img).unsqueeze(0).to(DEVICE)

    start = time.time()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
    inference_time = (time.time() - start) * 1000

    predicted_idx = probabilities.argmax().item()
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = probabilities[predicted_idx].item() * 100

    # All class probabilities
    class_probs = {
        CLASS_NAMES[i]: probabilities[i].item() * 100
        for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "class_probabilities": class_probs,
        "inference_time_ms": inference_time
    }


# ──────────────────────────────────────────────
# Generate Confidence Chart
# ──────────────────────────────────────────────
def generate_confidence_chart(class_probs, predicted_class, save_path):
    """Generate a horizontal bar chart of class probabilities."""

    fig, ax = plt.subplots(figsize=(7, 3))

    classes = list(class_probs.keys())
    probs = list(class_probs.values())

    colors = []
    for cls in classes:
        if cls == predicted_class:
            colors.append("#2196F3")
        else:
            colors.append("#B0BEC5")

    bars = ax.barh(classes, probs, color=colors, edgecolor="white", height=0.6)
    ax.set_xlim([0, 105])
    ax.set_xlabel("Confidence (%)", fontsize=11, fontweight="bold")
    ax.set_title("AI Classification Confidence", fontsize=13, fontweight="bold")

    for bar, val in zip(bars, probs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontweight="bold", fontsize=10)

    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────
# PDF Report Generator
# ──────────────────────────────────────────────
class PatientReportPDF(FPDF):
    """Custom PDF class for patient reports."""

    def header(self):
        # Header bar
        self.set_fill_color(25, 118, 210)  # Blue
        self.rect(0, 0, 210, 25, "F")

        self.set_font("Helvetica", "B", 16)
        self.set_text_color(255, 255, 255)
        self.set_y(5)
        self.cell(0, 10, "Brain Tumor Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")

        self.set_font("Helvetica", "", 8)
        self.cell(0, 5, "AI-Assisted Diagnostic Report | Multimodal Generative AI System",
                  align="C", new_x="LMARGIN", new_y="NEXT")

        self.ln(10)

    def footer(self):
        self.set_y(-20)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(128, 128, 128)
        self.cell(0, 5,
                  "DISCLAIMER: This report is generated by an AI system and is intended "
                  "for informational purposes only. It does not constitute a medical diagnosis.",
                  align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 5,
                  f"Page {self.page_no()}/{{nb}} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                  align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(25, 118, 210)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(25, 118, 210)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def info_row(self, label, value):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(80, 80, 80)
        self.cell(50, 6, label + ":", new_x="RIGHT")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")


def generate_report(
    image_path,
    patient_name="Anonymous Patient",
    patient_id="N/A",
    patient_age="N/A",
    patient_gender="N/A",
    referring_physician="N/A",
    clinical_notes=""
):
    """Generate a comprehensive PDF patient report."""

    print(f"\n{'='*60}")
    print(f"  GENERATING PATIENT REPORT")
    print(f"{'='*60}")

    # Load model & predict
    model, model_name = load_best_model()
    prediction = predict_image(model, image_path)

    predicted_class = prediction["predicted_class"]
    confidence = prediction["confidence"]
    class_probs = prediction["class_probabilities"]
    clinical = CLINICAL_INFO[predicted_class]

    print(f"  Patient: {patient_name} (ID: {patient_id})")
    print(f"  Prediction: {clinical['full_name']} ({confidence:.1f}% confidence)")

    # Generate confidence chart
    chart_path = os.path.join(REPORTS_DIR, "temp_confidence_chart.png")
    generate_confidence_chart(class_probs, predicted_class, chart_path)

    # Create PDF
    pdf = PatientReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=25)
    pdf.add_page()

    # ── Patient Information ──
    pdf.section_title("Patient Information")
    pdf.info_row("Patient Name", patient_name)
    pdf.info_row("Patient ID", patient_id)
    pdf.info_row("Age", patient_age)
    pdf.info_row("Gender", patient_gender)
    pdf.info_row("Referring Physician", referring_physician)
    pdf.info_row("Report Date", datetime.now().strftime("%B %d, %Y at %H:%M"))
    pdf.info_row("Report ID", f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}")
    pdf.ln(5)

    # ── AI Classification Result ──
    pdf.section_title("AI Classification Result")

    # Diagnosis box
    r, g, b = clinical["severity_color"]
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 12,
             f"  Diagnosis: {clinical['full_name']}  |  Severity: {clinical['severity']}",
             fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(40, 40, 40)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, f"Confidence: {confidence:.1f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, f"Model: {model_name} | Inference time: {prediction['inference_time_ms']:.1f}ms",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ── MRI Scan & Confidence Chart ──
    pdf.section_title("MRI Scan & Classification Confidence")

    # MRI image
    try:
        pdf.image(image_path, x=15, w=70, h=70)
    except Exception:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 10, "[MRI Image could not be loaded]", new_x="LMARGIN", new_y="NEXT")

    # Confidence chart
    try:
        current_y = pdf.get_y() - 70
        pdf.image(chart_path, x=100, y=max(current_y, pdf.get_y() - 60), w=100, h=50)
    except Exception:
        pass

    pdf.ln(5)

    # ── Clinical Description ──
    pdf.section_title("Clinical Description")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 5, clinical["description"])
    pdf.ln(3)

    if clinical["prognosis_note"]:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "Prognosis:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, clinical["prognosis_note"])
        pdf.ln(3)

    # ── Common Symptoms ──
    if clinical["common_symptoms"]:
        pdf.section_title("Associated Symptoms")
        pdf.set_font("Helvetica", "", 9)
        for symptom in clinical["common_symptoms"]:
            pdf.cell(5)
            pdf.cell(0, 5, f"  -  {symptom}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    # ── Recommended Actions ──
    pdf.section_title("Recommended Actions")
    pdf.set_font("Helvetica", "", 9)
    for i, action in enumerate(clinical["recommended_actions"], 1):
        pdf.cell(5)
        pdf.cell(0, 5, f"  {i}. {action}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ── Classification Probabilities Table ──
    pdf.section_title("Detailed Classification Probabilities")

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(230, 240, 250)
    pdf.cell(60, 7, "Tumor Class", border=1, fill=True, align="C")
    pdf.cell(40, 7, "Probability (%)", border=1, fill=True, align="C")
    pdf.cell(40, 7, "Status", border=1, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    for cls_name, prob in sorted(class_probs.items(), key=lambda x: x[1], reverse=True):
        status = "PREDICTED" if cls_name == predicted_class else "-"
        if cls_name == predicted_class:
            pdf.set_fill_color(200, 230, 255)
            fill = True
        else:
            fill = False
        pdf.cell(60, 6, cls_name.capitalize(), border=1, fill=fill, align="C")
        pdf.cell(40, 6, f"{prob:.2f}%", border=1, fill=fill, align="C")
        pdf.cell(40, 6, status, border=1, fill=fill, align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(5)

    # ── Clinical Notes ──
    if clinical_notes:
        pdf.section_title("Clinical Notes")
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, clinical_notes)
        pdf.ln(3)

    # ── Save PDF ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = patient_name.replace(" ", "_")
    pdf_filename = f"Report_{safe_name}_{timestamp}.pdf"
    pdf_path = os.path.join(REPORTS_DIR, pdf_filename)

    pdf.output(pdf_path)

    # Cleanup temp chart
    if os.path.exists(chart_path):
        os.remove(chart_path)

    print(f"\n  ✓ Report saved: {pdf_path}")
    print(f"  Report size: {os.path.getsize(pdf_path) / 1024:.1f} KB")

    return pdf_path


# ──────────────────────────────────────────────
# Demo: Generate sample reports
# ──────────────────────────────────────────────
def generate_demo_reports():
    """Generate demo reports using test set images."""

    print("\n  Generating demo patient reports from test set...")

    test_dir = "processed_data/test"
    demo_patients = [
        {
            "class": "glioma",
            "patient_name": "John Anderson",
            "patient_id": "PT-2024-001",
            "patient_age": "54",
            "patient_gender": "Male",
            "referring_physician": "Dr. Sarah Williams",
            "clinical_notes": "Patient presented with persistent headaches for 3 weeks and new-onset seizure."
        },
        {
            "class": "meningioma",
            "patient_name": "Maria Garcia",
            "patient_id": "PT-2024-002",
            "patient_age": "62",
            "patient_gender": "Female",
            "referring_physician": "Dr. James Chen",
            "clinical_notes": "Progressive vision loss in right eye over 6 months. No seizure history."
        },
        {
            "class": "pituitary",
            "patient_name": "David Kim",
            "patient_id": "PT-2024-003",
            "patient_age": "38",
            "patient_gender": "Male",
            "referring_physician": "Dr. Emily Thompson",
            "clinical_notes": "Hormonal imbalance detected in blood work. Bitemporal hemianopia on exam."
        },
        {
            "class": "notumor",
            "patient_name": "Lisa Johnson",
            "patient_id": "PT-2024-004",
            "patient_age": "45",
            "patient_gender": "Female",
            "referring_physician": "Dr. Michael Brown",
            "clinical_notes": "Routine screening MRI. Patient has family history of brain tumors."
        },
    ]

    for patient in demo_patients:
        cls_dir = os.path.join(test_dir, patient["class"])
        if os.path.exists(cls_dir):
            images = os.listdir(cls_dir)
            if images:
                image_path = os.path.join(cls_dir, images[0])
                generate_report(
                    image_path=image_path,
                    patient_name=patient["patient_name"],
                    patient_id=patient["patient_id"],
                    patient_age=patient["patient_age"],
                    patient_gender=patient["patient_gender"],
                    referring_physician=patient["referring_physician"],
                    clinical_notes=patient["clinical_notes"]
                )


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single image mode
        image_path = sys.argv[1]
        patient_name = sys.argv[2] if len(sys.argv) > 2 else "Anonymous Patient"
        patient_id = sys.argv[3] if len(sys.argv) > 3 else "N/A"

        if not os.path.exists(image_path):
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)

        generate_report(
            image_path=image_path,
            patient_name=patient_name,
            patient_id=patient_id
        )
    else:
        # Demo mode: generate sample reports
        generate_demo_reports()
