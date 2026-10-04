# Multimodal Generative AI System for Brain Tumor Analysis

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-black.svg)](https://flask.palletsprojects.com/)
[![HL7 FHIR](https://img.shields.io/badge/HL7%20FHIR-R4%20Compliant-007ec6.svg)](https://hl7.org/fhir/R4/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, clinically compliant artificial intelligence diagnostic system for automated **brain tumor classification** from MRI scans. The system incorporates deep convolutional and vision transformer architectures, benchmark evidence evaluation, interactive web inference, and automated **HL7 FHIR R4 Diagnostic Report generation (PDF)** with standardized **SNOMED CT** and **LOINC** clinical coding.

---

## Table of Contents

- [Overview](#overview)
- [Model Benchmark & Evidence Scores](#model-benchmark--evidence-scores)
- [Repository Architecture](#repository-architecture)
- [Installation & Environment Setup](#installation--environment-setup)
- [How to Run the System](#how-to-run-the-system)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Model Training](#2-model-training)
  - [3. Model Evaluation & Benchmark Visualizations](#3-model-evaluation--benchmark-visualizations)
  - [4. Launching the Web Application](#4-launching-the-web-application)
  - [5. Standalone Patient Report Generation](#5-standalone-patient-report-generation)
- [HL7 FHIR R4 Compliance & Clinical Coding](#hl7-fhir-r4-compliance--clinical-coding)
- [Web Application API Endpoints](#web-application-api-endpoints)
- [Clinical Disclaimer](#clinical-disclaimer)

---

## Overview

The system classifies brain magnetic resonance imaging (MRI) scans into four distinct diagnostic categories:

1. **Glioma** (`glioma`) — Primary brain tumor originating in glial tissue.
2. **Meningioma** (`meningioma`) — Tumor originating in the meninges surrounding brain and spinal cord.
3. **Pituitary Adenoma** (`pituitary`) — Neoplasm of the pituitary gland.
4. **No Tumor** (`notumor`) — Normal brain tissue / no observable neoplastic lesion.

### Key Capabilities

- **Multi-Model Benchmark**: Evaluates ResNet-50, EfficientNet-B0, and Vision Transformer (DeiT-Small) under identical splits and evaluation metrics.
- **Production Model (EfficientNet-B0)**: Achieves **99.15% Test Accuracy** with sub-30ms inference latency per image.
- **HL7 FHIR R4 Diagnostic Reports**: Generates formal clinical PDF reports structured after the HL7 FHIR `DiagnosticReport`, `Patient`, `Practitioner`, and `Observation` resources with full SNOMED CT and LOINC codings.
- **Modern Clinical Web Interface**: Dark-mode clinical dashboard with drag-and-drop MRI upload, real-time prediction, confidence score rings, probability distributions, recommended actions, and in-browser PDF report preview & download.

---

## Model Benchmark & Evidence Scores

All models were trained on 4,914 images and rigorously evaluated on an independent test set of 1,055 images (70% train / 15% validation / 15% test).

| Metric | EfficientNet-B0 (Selected) | ResNet-50 | DeiT-Small (Vision Transformer) |
|---|:---:|:---:|:---:|
| **Test Accuracy** | **99.15%** | 97.91% | 83.70% |
| **Weighted F1-Score** | **99.15%** | 97.92% | 83.51% |
| **Macro F1-Score** | **99.11%** | 97.83% | 82.78% |
| **Macro ROC-AUC** | **99.90%** | 99.71% | 96.03% |
| **Average Inference Latency** | **29.60 ms** | 81.04 ms | 66.97 ms |
| **Model Size / Parameters** | **4.0M (19.2 MB)** | 23.5M (98.8 MB) | 21.7M (87.8 MB) |

### Per-Class Performance (EfficientNet-B0)

| Class | Precision | Recall | F1-Score | Support | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **Glioma** | 99.59% | 98.36% | **98.97%** | 244 | 99.98% |
| **Meningioma** | 97.23% | 99.60% | **98.40%** | 247 | 99.85% |
| **No Tumor** | 100.00% | 99.67% | **99.83%** | 300 | 100.00% |
| **Pituitary** | 99.62% | 98.86% | **99.24%** | 264 | 99.75% |
| **Overall / Weighted Avg** | **99.16%** | **99.15%** | **99.15%** | **1,055** | **99.90%** |

> **Conclusion**: EfficientNet-B0 outperforms all models across accuracy, F1-score, ROC-AUC, and parameter efficiency, providing over **2.7× faster inference** than ResNet-50.

---

## Repository Architecture

```text
Multimodal-Generative-AI-System-for-Brain-Tumor-Analysis/
├── app.py                      # Flask web application & REST API
├── hl7_report_generator.py     # HL7 FHIR R4 PDF diagnostic report generator
├── generate_patient_report.py  # Standalone CLI PDF report generator
├── clinical_test_build.py      # Synthetic clinical text dataset builder
├── clinical_text_dataset.csv   # Multimodal symptom dataset
├── transform.py                # Root transform redirect (backward compatibility)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── TrainData/                  # Training pipeline & augmentations
│   ├── __init__.py
│   ├── train_all_models.py     # Unified training script for all 3 models
│   └── transform.py            # ImageNet transforms and data augmentations
│
├── EvaluateData/               # Rigorous evaluation & metric plots
│   ├── __init__.py
│   └── evaluate_all_models.py  # Comprehensive evaluation script & plotting
│
├── preprocess_data/            # Data preparation & loading utilities
│   ├── __init__.py
│   ├── data_split.py           # 70/15/15 train/val/test stratified dataset splitter
│   ├── data_loader.py          # PyTorch DataLoader definitions
│   ├── data_visualize.py       # Sample batch visualizer
│   ├── dataset.py              # Raw dataset class checker
│   ├── label.py                # Label saver & class counter
│   └── labels.json             # Saved class label mapping
│
├── models/                     # Deep learning model architectures
│   ├── __init__.py             # Model registry (MODEL_REGISTRY)
│   ├── efficientnet_model.py   # EfficientNet-B0 transfer learning architecture
│   ├── resnet_model.py         # ResNet-50 architecture
│   └── deit_model.py           # Data-efficient Image Transformer (DeiT-Small)
│
├── frontend/                   # Web user interface assets
│   ├── templates/
│   │   └── index.html          # Clinical diagnostic dashboard
│   └── static/
│       ├── css/
│       │   └── style.css       # Medical UI design system & responsive layout
│       ├── js/
│       │   └── app.js          # Client-side state, AJAX inference, PDF preview
│       └── uploads/            # Temporary storage for uploaded scans (.gitkeep)
│
├── checkpoints/                # Trained PyTorch model weights (.pth)
│   ├── efficientnet_b0.pth     # Production model checkpoint
│   ├── resnet_50.pth           # ResNet-50 checkpoint
│   └── deit_small.pth          # DeiT-Small checkpoint
│
├── results/                    # Benchmark metrics & publication plots
│   ├── best_model.json         # Metadata of the selected top model
│   ├── evaluation_results.json # Full evaluation metrics for all models
│   ├── training_summaries.json # Training epoch summaries & convergence stats
│   └── plots/                  # Confusion matrices, ROC curves, training curves
│
└── patient_reports/            # Generated HL7 FHIR PDF reports (.gitkeep)
```

---

## Installation & Environment Setup

### 1. Prerequisites
- Python 3.9 or higher (tested with Python 3.10 and 3.11)
- CUDA-compatible GPU (recommended for training, optional for inference)

### 2. Clone Repository & Create Virtual Environment

```bash
git clone https://github.com/kavishanGT/Multimodal-Generative-AI-System-for-Brain-Tumor-Analysis.git
cd Multimodal-Generative-AI-System-for-Brain-Tumor-Analysis

# Create virtual environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## How to Run the System

### 1. Data Preprocessing

Place raw MRI scans in `data/` structured by folder name:
```text
data/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

Run the stratified splitter to partition into `processed_data/` (70% train, 15% val, 15% test):

```bash
python preprocess_data/data_split.py
```

Verify class distributions and generate `labels.json`:

```bash
python preprocess_data/label.py
```

---

### 2. Model Training

Train all three candidate architectures (EfficientNet-B0, ResNet-50, DeiT-Small) using Cosine Annealing learning rate schedules and early stopping:

```bash
python TrainData/train_all_models.py
```

Checkpoints will be saved to `checkpoints/`, and training summaries will be stored in `results/training_summaries.json`.

---

### 3. Model Evaluation & Benchmark Visualizations

Run full test-set evaluation across all models:

```bash
python EvaluateData/evaluate_all_models.py
```

This generates:
- `results/evaluation_results.json` — Comprehensive metrics (precision, recall, F1, ROC-AUC, latency).
- `results/best_model.json` — Automatic selection of the highest performing model.
- `results/plots/` — Confusion matrix heatmaps, multi-class ROC curves, and training loss/accuracy trajectories.

---

### 4. Launching the Web Application

Start the clinical web server:

```bash
python app.py
```

Open your browser at:
```text
http://localhost:5000
```

#### Web Application Features:
1. **Patient Information Intake**: Input Patient ID, Full Name, Age, Gender, Referring Physician, and Clinical Notes.
2. **MRI Upload & Instant Inference**: Drag and drop any `.jpg`, `.png`, or `.tiff` MRI scan.
3. **Clinical Diagnosis Display**: Tumor category badge, severity indicator (Severe / Moderate / Mild), and confidence percentage.
4. **Probability Distribution**: Real-time animated confidence bars across all 4 diagnostic classes.
5. **Recommended Clinical Actions**: Guideline-based next steps (e.g., stereotactic biopsy, contrast MRI, endocrinology workup).
6. **HL7 FHIR PDF Report Download**: Instant PDF report generation adhering to HL7 FHIR R4 standard.
7. **In-Browser PDF Preview**: View generated clinical PDF reports directly inside the application modal or open in a new tab.

---

### 5. Standalone Patient Report Generation

You can also generate patient diagnostic reports directly from the terminal without the web interface:

```bash
# Generate comprehensive PDF report with MRI scan and clinical findings
python generate_patient_report.py

# Test HL7 FHIR R4 PDF generator standalone
python hl7_report_generator.py
```

Reports are saved to `patient_reports/`.

---

## HL7 FHIR R4 Compliance & Clinical Coding

Reports generated by this system follow the **HL7 FHIR Release 4 (R4)** `DiagnosticReport` standard and use authoritative biomedical ontologies:

### Clinical Terminologies Used:
- **SNOMED CT** (`http://snomed.info/sct`):
  - `393564001`: Glioma of brain (disorder) — Severity: Severe
  - `86049000`: Meningioma (disorder) — Severity: Moderate
  - `254956000`: Pituitary adenoma (disorder) — Severity: Moderate
  - `281900007`: No abnormality detected (finding) — Severity: Mild
  - `241601008`: Magnetic resonance imaging of brain (procedure)
- **LOINC** (`http://loinc.org`):
  - `18748-4`: Diagnostic imaging study
  - `30746-2`: Portable XR and MRI Brain
- **HL7 v2 Table 0074**: Diagnostic service category (`RAD` - Radiology)
- **HL7 v2 Table 0203**: Patient identifier type (`MR` - Medical Record Number)

### FHIR Resource Hierarchy in Generated PDF:
- **Bundle** (`type: document`): Top-level FHIR container with UUID identifier.
- **DiagnosticReport**: Diagnostic report status (`final`), procedure code, conclusion narrative, and conclusionCode.
- **Patient**: Subject identifier (MRN), patient full name, age, birth year estimate, and gender.
- **Practitioner**: Referring physician / reporting clinician.
- **Observation**: Independent resources for each tumor class containing calibrated confidence probabilities (`valueQuantity` with `%` units) and qualitative interpretation (`POS` for primary prediction, `NEG` for alternatives).
- **Extensions**: AI model metadata, inference latency, and prioritized clinical recommendation action lists.

---

## Web Application API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the clinical dashboard interface |
| `POST` | `/predict` | Multipart upload of MRI image; returns JSON prediction, class probabilities, latency, and severity metadata |
| `POST` | `/generate-report` | Accepts patient metadata & prediction data; generates HL7 FHIR PDF report and returns download/view URLs |
| `GET` | `/view-report/<filename>` | Streams generated PDF report inline (`application/pdf`) for modal iframe display or browser preview |
| `GET` | `/download-report/<filename>` | Serves generated PDF report as a downloadable file attachment |
| `GET` | `/api/model-info` | Returns current active model name, execution device, class names, and benchmark validation metrics |

---

## Clinical Disclaimer

> **IMPORTANT**: This software is designed and developed for research, education, and decision-support exploration only. It is **not** a certified medical diagnostic device under FDA/CE-MDR regulations. The artificial intelligence predictions and generated HL7 FHIR reports must always be reviewed, correlated, and confirmed by a board-certified radiologist, neurosurgeon, or licensed medical practitioner before making any patient care or treatment decisions.