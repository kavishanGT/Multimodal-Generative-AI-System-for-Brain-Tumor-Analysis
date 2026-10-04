# hl7_report_generator.py
# HL7 FHIR R4 DiagnosticReport Generator — PDF Output
#
# Generates an HL7 FHIR-structured Diagnostic Report as a professional PDF.
# Follows HL7 FHIR R4 resource structure with SNOMED CT & LOINC coding.
# Reference: https://www.hl7.org/fhir/diagnosticreport.html

import os
import json
import uuid
from datetime import datetime, timezone

from fpdf import FPDF


# -----------------------------------------------
# Clinical coding maps (SNOMED CT & LOINC)
# -----------------------------------------------
TUMOR_SNOMED_CODES = {
    "glioma": {
        "code": "393564001",
        "display": "Glioma of brain (disorder)",
        "severity_code": "24484000",
        "severity_display": "Severe",
    },
    "meningioma": {
        "code": "86049000",
        "display": "Meningioma (disorder)",
        "severity_code": "6736007",
        "severity_display": "Moderate",
    },
    "pituitary": {
        "code": "254956000",
        "display": "Pituitary adenoma (disorder)",
        "severity_code": "6736007",
        "severity_display": "Moderate",
    },
    "notumor": {
        "code": "281900007",
        "display": "No abnormality detected (finding)",
        "severity_code": "255604002",
        "severity_display": "Mild",
    },
}

SEVERITY_COLORS = {
    "Severe": (220, 50, 50),
    "Moderate": (230, 150, 30),
    "Mild": (50, 160, 80),
}

CLINICAL_CONCLUSIONS = {
    "glioma": (
        "AI-assisted analysis of the submitted brain MRI demonstrates findings "
        "consistent with a glioma. The lesion characteristics suggest a glial cell "
        "neoplasm. Histopathological confirmation via stereotactic biopsy is recommended. "
        "Urgent neurosurgical consultation is advised within 48 hours."
    ),
    "meningioma": (
        "AI-assisted analysis of the submitted brain MRI demonstrates findings "
        "consistent with a meningioma. The lesion appears to originate from the meninges. "
        "Most meningiomas are benign (WHO Grade I). Neurosurgical consultation is "
        "recommended within 1-2 weeks for further evaluation and management planning."
    ),
    "pituitary": (
        "AI-assisted analysis of the submitted brain MRI demonstrates findings "
        "consistent with a pituitary adenoma. Endocrine evaluation including a complete "
        "pituitary hormone panel is recommended. Formal visual field testing should be "
        "performed to assess for optic chiasm compression."
    ),
    "notumor": (
        "AI-assisted analysis of the submitted brain MRI shows no evidence of "
        "intracranial neoplasm. Brain parenchyma appears within normal limits. "
        "If symptoms persist, clinical re-evaluation and follow-up imaging may be "
        "warranted to rule out other pathologies."
    ),
}

RECOMMENDED_ACTIONS = {
    "glioma": [
        "Urgent neurosurgical consultation within 48 hours",
        "Advanced MRI with contrast and spectroscopy",
        "Consider stereotactic biopsy for histological grading",
        "Multidisciplinary tumor board review",
        "Baseline neurocognitive assessment",
    ],
    "meningioma": [
        "Neurosurgical consultation within 1-2 weeks",
        "Contrast-enhanced MRI for detailed characterization",
        "Serial imaging surveillance if small and asymptomatic",
        "Ophthalmologic evaluation if near optic pathway",
    ],
    "pituitary": [
        "Endocrinology consultation for hormone panel",
        "Formal visual field testing (Humphrey perimetry)",
        "Dedicated pituitary MRI protocol with contrast",
        "Complete pituitary hormone panel",
    ],
    "notumor": [
        "Routine follow-up with primary care physician",
        "Address presenting symptoms through differential diagnosis",
        "Routine screening MRI if clinically indicated",
    ],
}


def _make_id():
    return str(uuid.uuid4())


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


# -----------------------------------------------
# HL7 FHIR PDF Report Class
# -----------------------------------------------
class HL7ReportPDF(FPDF):
    """Custom PDF with HL7 FHIR-compliant Diagnostic Report layout."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._report_id = ""
        self._timestamp = ""

    def header(self):
        # Top accent bar
        self.set_fill_color(25, 80, 180)
        self.rect(0, 0, 210, 4, "F")

        # Header block
        self.set_fill_color(20, 40, 80)
        self.rect(0, 4, 210, 24, "F")

        self.set_y(7)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(255, 255, 255)
        self.cell(0, 7, "HL7 FHIR Diagnostic Report", align="C", new_x="LMARGIN", new_y="NEXT")

        self.set_font("Helvetica", "", 7)
        self.set_text_color(180, 200, 230)
        self.cell(
            0, 5,
            "HL7 FHIR R4 | DiagnosticReport Resource | AI-Assisted Brain Tumor Analysis",
            align="C", new_x="LMARGIN", new_y="NEXT",
        )
        self.ln(8)

    def footer(self):
        self.set_y(-22)
        self.set_draw_color(180, 180, 180)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)

        self.set_font("Helvetica", "I", 6.5)
        self.set_text_color(128, 128, 128)
        self.cell(
            0, 4,
            "DISCLAIMER: This AI-generated report is for informational purposes only "
            "and does not constitute a medical diagnosis. Clinical correlation is required.",
            align="C", new_x="LMARGIN", new_y="NEXT",
        )
        self.cell(
            0, 4,
            f"Page {self.page_no()}/{{nb}}  |  "
            f"Report ID: {self._report_id}  |  "
            f"Generated: {self._timestamp}",
            align="C",
        )

    # -- helpers ----------------------------------------------------------

    def section_title(self, title, r=25, g=80, b=180):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(r, g, b)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(r, g, b)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def kv_row(self, key, value, bold_val=False):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(80, 80, 80)
        self.cell(55, 5.5, key + ":", new_x="RIGHT")
        self.set_font("Helvetica", "B" if bold_val else "", 9)
        self.set_text_color(30, 30, 30)
        self.cell(0, 5.5, str(value), new_x="LMARGIN", new_y="NEXT")

    def fhir_code_row(self, system, code, display):
        self.set_font("Helvetica", "", 8)
        self.set_text_color(60, 60, 60)
        self.cell(5)
        self.cell(50, 5, f"System: {system}")
        self.cell(30, 5, f"Code: {code}")
        self.cell(0, 5, f"Display: {display}", new_x="LMARGIN", new_y="NEXT")


# -----------------------------------------------
# Main generation function
# -----------------------------------------------
def generate_hl7_pdf_report(
    predicted_class,
    confidence,
    class_probabilities,
    inference_time_ms,
    model_name,
    patient_name="Anonymous Patient",
    patient_id=None,
    patient_age=None,
    patient_gender=None,
    referring_physician=None,
    clinical_notes="",
    image_path=None,
    output_dir="patient_reports",
):
    """
    Generate an HL7 FHIR R4 DiagnosticReport as a PDF file.

    Returns:
        str: File path to the saved PDF.
    """

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(BASE_DIR, output_dir)

    os.makedirs(output_dir, exist_ok=True)

    now_display = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    now_iso = _now_iso()
    report_id = f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    bundle_id = _make_id()
    patient_resource_id = _make_id()

    tumor_info = TUMOR_SNOMED_CODES.get(predicted_class, TUMOR_SNOMED_CODES["notumor"])
    conclusion = CLINICAL_CONCLUSIONS.get(predicted_class, "")
    actions = RECOMMENDED_ACTIONS.get(predicted_class, [])
    sev_display = tumor_info["severity_display"]
    sev_color = SEVERITY_COLORS.get(sev_display, (100, 100, 100))

    gender_map = {"male": "male", "female": "female", "m": "male", "f": "female"}
    fhir_gender = gender_map.get((patient_gender or "").lower(), "unknown")

    pdf = HL7ReportPDF()
    pdf._report_id = report_id
    pdf._timestamp = now_display
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=28)
    pdf.add_page()

    # ================================================================
    # SECTION 1 - FHIR Bundle & Report Identifiers
    # ================================================================
    pdf.section_title("FHIR Bundle / Report Identifiers")
    pdf.kv_row("Resource Type", "Bundle (type: document)")
    pdf.kv_row("Bundle ID", bundle_id)
    pdf.kv_row("Report ID", report_id)
    pdf.kv_row("Report Status", "final")
    pdf.kv_row("Effective Date/Time", now_iso)
    pdf.kv_row("Issued", now_iso)
    pdf.ln(2)

    # Category coding
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5, "Category Coding:", new_x="LMARGIN", new_y="NEXT")
    pdf.fhir_code_row("hl7.org/CodeSystem/v2-0074", "RAD", "Radiology")
    pdf.fhir_code_row("loinc.org", "18748-4", "Diagnostic imaging study")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5, "Report Code:", new_x="LMARGIN", new_y="NEXT")
    pdf.fhir_code_row("loinc.org", "30746-2", "Portable XR and MRI Brain")
    pdf.fhir_code_row("snomed.info/sct", "241601008", "MRI of brain (procedure)")
    pdf.ln(4)

    # ================================================================
    # SECTION 2 - Patient Resource
    # ================================================================
    pdf.section_title("Patient Resource")
    pdf.kv_row("Resource Type", "Patient")
    pdf.kv_row("Patient ID (MRN)", patient_id or "N/A")
    pdf.kv_row("Patient Name", patient_name, bold_val=True)
    pdf.kv_row("Age", patient_age or "N/A")
    pdf.kv_row("Gender (FHIR)", fhir_gender)
    if patient_age:
        try:
            birth_year = datetime.now().year - int(patient_age)
            pdf.kv_row("Approx. Birth Date", f"{birth_year}-01-01")
        except ValueError:
            pass
    pdf.ln(4)

    # ================================================================
    # SECTION 3 - Practitioner Resource
    # ================================================================
    pdf.section_title("Practitioner Resource (Performer)")
    pdf.kv_row("Resource Type", "Practitioner")
    pdf.kv_row("Referring Physician", referring_physician or "N/A")
    pdf.ln(4)

    # ================================================================
    # SECTION 4 - AI Classification Result (ConclusionCode)
    # ================================================================
    pdf.section_title("Conclusion / AI Classification Result", *sev_color)

    # Coloured diagnosis box
    r, g, b = sev_color
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(
        0, 10,
        f"   {tumor_info['display']}   |   Severity: {sev_display}",
        fill=True, new_x="LMARGIN", new_y="NEXT",
    )
    pdf.set_text_color(30, 30, 30)
    pdf.ln(3)

    pdf.kv_row("Confidence", f"{confidence:.2f}%", bold_val=True)
    pdf.kv_row("AI Model", model_name)
    pdf.kv_row("Inference Time", f"{inference_time_ms:.1f} ms")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5, "SNOMED CT ConclusionCode:", new_x="LMARGIN", new_y="NEXT")
    pdf.fhir_code_row("snomed.info/sct", tumor_info["code"], tumor_info["display"])
    pdf.ln(4)

    # ================================================================
    # SECTION 5 - Observation Resources (per-class probabilities)
    # ================================================================
    pdf.section_title("Observation Resources (Classification Probabilities)")

    # Table header
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_fill_color(230, 238, 250)
    pdf.set_text_color(30, 30, 30)
    col_w = [50, 35, 30, 35, 40]
    headers = ["SNOMED Display", "Code", "Prob (%)", "Interpretation", "Observation ID"]
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 6, h, border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    sorted_probs = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)
    for cls_name, prob in sorted_probs:
        cls_info = TUMOR_SNOMED_CODES.get(cls_name, TUMOR_SNOMED_CODES["notumor"])
        is_pred = cls_name == predicted_class
        interp = "POS (Primary)" if is_pred else "NEG"
        obs_id = _make_id()[:8]

        if is_pred:
            pdf.set_fill_color(210, 230, 255)
            fill = True
        else:
            fill = False

        pdf.cell(col_w[0], 5.5, cls_info["display"][:28], border=1, fill=fill, align="L")
        pdf.cell(col_w[1], 5.5, cls_info["code"], border=1, fill=fill, align="C")
        pdf.cell(col_w[2], 5.5, f"{prob:.2f}%", border=1, fill=fill, align="C")
        pdf.cell(col_w[3], 5.5, interp, border=1, fill=fill, align="C")
        pdf.cell(col_w[4], 5.5, obs_id, border=1, fill=fill, align="C")
        pdf.ln()

    pdf.ln(4)

    # ================================================================
    # SECTION 6 - Clinical Conclusion (narrative)
    # ================================================================
    pdf.section_title("Clinical Conclusion (Report Narrative)")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 5, conclusion)
    pdf.ln(4)

    # ================================================================
    # SECTION 7 - Recommended Actions (extensions)
    # ================================================================
    pdf.section_title("Recommended Actions (FHIR Extensions)")
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(40, 40, 40)
    for i, action in enumerate(actions, 1):
        pdf.cell(5)
        pdf.cell(
            0, 5.5,
            f"  {i}. {action}",
            new_x="LMARGIN", new_y="NEXT",
        )
    pdf.ln(4)

    # ================================================================
    # SECTION 8 - Clinical Notes
    # ================================================================
    if clinical_notes:
        pdf.section_title("Clinical Notes")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 5, clinical_notes)
        pdf.ln(4)

    # ================================================================
    # SECTION 9 - MRI Image (if provided)
    # ================================================================
    if image_path and os.path.exists(image_path):
        pdf.section_title("Submitted MRI Scan")
        try:
            pdf.image(image_path, x=55, w=100, h=100)
        except Exception:
            pdf.set_font("Helvetica", "I", 9)
            pdf.cell(0, 6, "[Image could not be embedded]", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    # ================================================================
    # SECTION 10 - FHIR Compliance & Coding Summary
    # ================================================================
    pdf.section_title("FHIR Compliance & Terminology Summary")

    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_fill_color(240, 240, 245)
    t_col = [60, 65, 65]
    t_head = ["Standard", "System URI", "Usage"]
    for i, h in enumerate(t_head):
        pdf.cell(t_col[i], 6, h, border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    rows = [
        ("HL7 FHIR R4", "hl7.org/fhir/R4", "Report structure & resources"),
        ("SNOMED CT", "snomed.info/sct", "Diagnosis & finding codes"),
        ("LOINC", "loinc.org", "Report category & procedure codes"),
        ("HL7 v2 Table 0074", "hl7.org/CodeSystem/v2-0074", "Diagnostic service category"),
        ("HL7 v2 Table 0203", "hl7.org/CodeSystem/v2-0203", "Patient identifier type"),
    ]
    for row in rows:
        for i, val in enumerate(row):
            pdf.cell(t_col[i], 5, val, border=1, align="L")
        pdf.ln()

    pdf.ln(3)

    # ---- Save ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = patient_name.replace(" ", "_")
    filename = f"HL7_FHIR_Report_{safe_name}_{timestamp}.pdf"
    filepath = os.path.join(output_dir, filename)

    pdf.output(filepath)

    file_size = os.path.getsize(filepath) / 1024
    print(f"  HL7 FHIR PDF report saved: {filepath} ({file_size:.1f} KB)")
    return filepath


# -----------------------------------------------
# Standalone test
# -----------------------------------------------
if __name__ == "__main__":
    path = generate_hl7_pdf_report(
        predicted_class="glioma",
        confidence=99.5,
        class_probabilities={
            "glioma": 99.5,
            "meningioma": 0.3,
            "notumor": 0.1,
            "pituitary": 0.1,
        },
        inference_time_ms=29.6,
        model_name="EfficientNet-B0",
        patient_name="John Anderson",
        patient_id="PT-2024-001",
        patient_age="54",
        patient_gender="Male",
        referring_physician="Dr. Sarah Williams",
        clinical_notes="Patient presented with persistent headaches for 3 weeks and new-onset seizure.",
    )
    print(f"\nTest report generated: {path}")
