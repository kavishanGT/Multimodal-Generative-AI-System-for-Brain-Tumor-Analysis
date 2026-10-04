# app.py
# Flask Web Application for Brain Tumor Analysis
# Upload MRI scans, get AI predictions, download HL7 FHIR & PDF reports

import os
import sys
import json
import time
import uuid
from datetime import datetime

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    send_from_directory,
)
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from TrainData.transform import val_transform, IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    from transform import val_transform, IMAGENET_MEAN, IMAGENET_STD

from models import MODEL_REGISTRY
from hl7_report_generator import generate_hl7_pdf_report

# -----------------------------------------------
# App Configuration
# -----------------------------------------------
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "frontend", "templates"),
    static_folder=os.path.join(BASE_DIR, "frontend", "static"),
)
CORS(app)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "frontend", "static", "uploads")
REPORTS_FOLDER = os.path.join(BASE_DIR, "patient_reports")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

# Clinical display info
CLINICAL_DISPLAY = {
    "glioma": {
        "full_name": "Glioma",
        "severity": "HIGH",
        "color": "#dc3545",
        "icon": "exclamation-triangle",
        "description": (
            "Gliomas are tumors that arise from glial cells in the brain or spine. "
            "They are one of the most common types of primary brain tumors."
        ),
        "actions": [
            "Urgent neurosurgical consultation within 48 hours",
            "Advanced MRI with contrast and spectroscopy",
            "Consider stereotactic biopsy for grading",
            "Multidisciplinary tumor board review",
        ],
    },
    "meningioma": {
        "full_name": "Meningioma",
        "severity": "MODERATE",
        "color": "#fd7e14",
        "icon": "exclamation-circle",
        "description": (
            "Meningiomas arise from the meninges. Most are benign (WHO Grade I) "
            "and slow-growing. They are the most common primary intracranial tumor."
        ),
        "actions": [
            "Neurosurgical consultation within 1-2 weeks",
            "Contrast-enhanced MRI for characterization",
            "Serial imaging surveillance if asymptomatic",
            "Ophthalmologic evaluation if near optic pathway",
        ],
    },
    "pituitary": {
        "full_name": "Pituitary Tumor (Adenoma)",
        "severity": "MODERATE",
        "color": "#fd7e14",
        "icon": "exclamation-circle",
        "description": (
            "Pituitary adenomas are benign tumors of the pituitary gland. "
            "They can be functioning (hormone-producing) or non-functioning."
        ),
        "actions": [
            "Endocrinology consultation for hormone panel",
            "Formal visual field testing",
            "Dedicated pituitary MRI protocol",
            "Complete pituitary hormone panel",
        ],
    },
    "notumor": {
        "full_name": "No Tumor Detected",
        "severity": "LOW",
        "color": "#28a745",
        "icon": "check-circle",
        "description": (
            "The MRI scan analysis indicates no evidence of a brain tumor. "
            "Brain parenchyma appears within normal limits."
        ),
        "actions": [
            "Routine follow-up with primary care physician",
            "Address symptoms through differential diagnosis",
            "Routine screening MRI if clinically indicated",
        ],
    },
}


# -----------------------------------------------
# Model Loading (singleton)
# -----------------------------------------------
_model = None
_model_name = None


def get_model():
    """Load the best model (cached singleton)."""
    global _model, _model_name

    if _model is not None:
        return _model, _model_name

    best_model_path = os.path.join(RESULTS_DIR, "best_model.json")
    if os.path.exists(best_model_path):
        with open(best_model_path) as f:
            best_info = json.load(f)
        model_name = best_info["best_model"]
        ckpt_path = best_info["checkpoint"]
    else:
        model_name = "EfficientNet-B0"
        ckpt_path = os.path.join(CHECKPOINTS_DIR, "efficientnet_b0.pth")

    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(BASE_DIR, ckpt_path)

    print(f"Loading model: {model_name} from {ckpt_path}")

    model_class = MODEL_REGISTRY[model_name]
    model = model_class(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    _model = model
    _model_name = model_name
    print(f"Model loaded successfully on {DEVICE}")
    return model, model_name


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict(image_path):
    """Run inference on a single image."""
    model, model_name = get_model()

    img = Image.open(image_path).convert("RGB")
    img_tensor = val_transform(img).unsqueeze(0).to(DEVICE)

    start = time.time()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]
    inference_ms = (time.time() - start) * 1000

    pred_idx = probs.argmax().item()
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx].item() * 100

    class_probs = {
        CLASS_NAMES[i]: round(probs[i].item() * 100, 4)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": pred_class,
        "confidence": round(confidence, 2),
        "class_probabilities": class_probs,
        "inference_time_ms": round(inference_ms, 2),
        "model_name": model_name,
        "clinical_info": CLINICAL_DISPLAY[pred_class],
    }


# -----------------------------------------------
# Routes
# -----------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """Handle MRI image upload and return prediction."""

    if "mri_image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["mri_image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    # Save uploaded file
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(filepath)

    # Run prediction
    try:
        result = predict(filepath)
        result["image_path"] = filepath
        result["image_url"] = f"/static/uploads/{unique_name}"
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/generate-report", methods=["POST"])
def generate_report():
    """Generate HL7 FHIR PDF report and return download link."""

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    predicted_class = data.get("predicted_class", "notumor")
    confidence = data.get("confidence", 0.0)
    class_probs = data.get("class_probabilities", {})
    inference_ms = data.get("inference_time_ms", 0.0)
    model_name = data.get("model_name", "EfficientNet-B0")
    patient_name = data.get("patient_name", "Anonymous Patient")
    patient_id = data.get("patient_id", "N/A")
    patient_age = data.get("patient_age", "N/A")
    patient_gender = data.get("patient_gender", "unknown")
    referring_physician = data.get("referring_physician", "N/A")
    clinical_notes = data.get("clinical_notes", "")
    image_path = data.get("image_path", None)

    # Generate HL7 FHIR PDF report
    try:
        filepath = generate_hl7_pdf_report(
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities=class_probs,
            inference_time_ms=inference_ms,
            model_name=model_name,
            patient_name=patient_name,
            patient_id=patient_id,
            patient_age=patient_age,
            patient_gender=patient_gender,
            referring_physician=referring_physician,
            clinical_notes=clinical_notes,
            image_path=image_path,
            output_dir=REPORTS_FOLDER,
        )
    except Exception as e:
        return jsonify({"error": f"Report generation failed: {str(e)}"}), 500

    filename = os.path.basename(filepath)
    return jsonify({
        "success": True,
        "filename": filename,
        "download_url": f"/download-report/{filename}",
        "view_url": f"/view-report/{filename}",
    })


@app.route("/download-report/<filename>")
def download_report(filename):
    """Download a generated report file."""
    return send_from_directory(
        REPORTS_FOLDER,
        filename,
        as_attachment=True,
    )


@app.route("/view-report/<filename>")
def view_report(filename):
    """View a generated PDF report inline in browser/iframe."""
    return send_from_directory(
        REPORTS_FOLDER,
        filename,
        as_attachment=False,
        mimetype="application/pdf",
    )


@app.route("/api/model-info")
def model_info():
    """Return information about the loaded model."""
    _, model_name = get_model()

    eval_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    eval_data = {}
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            all_evals = json.load(f)
            for e in all_evals:
                if e["model_name"] == model_name:
                    eval_data = e
                    break

    return jsonify({
        "model_name": model_name,
        "device": str(DEVICE),
        "classes": CLASS_NAMES,
        "evaluation": eval_data,
    })


# -----------------------------------------------
# Main
# -----------------------------------------------
if __name__ == "__main__":
    # Pre-load model on startup
    get_model()

    print("\n" + "=" * 50)
    print("  Brain Tumor Analysis Web Application")
    print("  http://localhost:5000")
    print("=" * 50 + "\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )
