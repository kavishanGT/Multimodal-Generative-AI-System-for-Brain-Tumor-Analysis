// =============================================
//  Brain Tumor Analysis — Frontend JavaScript
// =============================================

// State
let currentPrediction = null;
let currentReportData = null;

// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const removeImage = document.getElementById('removeImage');
const analyzeBtn = document.getElementById('analyzeBtn');
const btnLoader = document.getElementById('btnLoader');
const btnText = document.getElementById('btnText');
const resultsPanel = document.getElementById('resultsPanel');
const modelStatus = document.getElementById('modelStatus');

// ── Initialize ──
document.addEventListener('DOMContentLoaded', () => {
    loadModelInfo();
    setupUpload();
    setupButtons();
});

// ── Model Info ──
async function loadModelInfo() {
    try {
        const res = await fetch('/api/model-info');
        const data = await res.json();
        modelStatus.textContent = `${data.model_name} ready on ${data.device.toUpperCase()}`;
    } catch (e) {
        modelStatus.textContent = 'Model loading...';
    }
}

// ── Upload Handling ──
function setupUpload() {
    // Click to browse
    uploadArea.addEventListener('click', () => fileInput.click());

    // File selected
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag & Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Remove image
    removeImage.addEventListener('click', () => {
        resetUpload();
    });
}

function handleFile(file) {
    const validTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|bmp|tif|tiff)$/i)) {
        showToast('Invalid file type. Please upload a JPG, PNG, BMP, or TIFF image.', 'error');
        return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        imagePreview.style.display = 'block';
        uploadArea.style.display = 'none';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);

    // Store file for upload
    fileInput._selectedFile = file;
}

function resetUpload() {
    imagePreview.style.display = 'none';
    uploadArea.style.display = 'block';
    previewImg.src = '';
    fileInput.value = '';
    fileInput._selectedFile = null;
    analyzeBtn.disabled = true;
    resultsPanel.style.display = 'none';
    currentPrediction = null;
    currentReportData = null;
}

// ── Button Handlers ──
function setupButtons() {
    analyzeBtn.addEventListener('click', analyzeMRI);

    const downloadBtn = document.getElementById('downloadHL7');
    if (downloadBtn) downloadBtn.addEventListener('click', downloadPDFReport);

    const previewBtn = document.getElementById('previewReport');
    if (previewBtn) previewBtn.addEventListener('click', previewPDFReport);

    const closeModalBtn = document.getElementById('closeModal');
    if (closeModalBtn) closeModalBtn.addEventListener('click', closeModal);

    const closeModalBtn2 = document.getElementById('closeModalBtn');
    if (closeModalBtn2) closeModalBtn2.addEventListener('click', closeModal);

    const modalDownload = document.getElementById('modalDownloadBtn');
    if (modalDownload) modalDownload.addEventListener('click', downloadPDFReport);

    // Close modal on overlay click
    const pdfModal = document.getElementById('pdfModal');
    if (pdfModal) {
        pdfModal.addEventListener('click', (e) => {
            if (e.target === pdfModal) closeModal();
        });
    }

    // Keyboard escape
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();
    });
}

// ── Analyze MRI ──
async function analyzeMRI() {
    const file = fileInput._selectedFile;
    if (!file) {
        showToast('Please upload an MRI scan first.', 'error');
        return;
    }

    // Show loading
    btnLoader.style.display = 'block';
    btnText.textContent = 'Analyzing...';
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append('mri_image', file);

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || 'Prediction failed');
        }

        const result = await res.json();
        currentPrediction = result;

        displayResults(result);
        showToast('Analysis complete!', 'success');

    } catch (e) {
        showToast(`Error: ${e.message}`, 'error');
    } finally {
        btnLoader.style.display = 'none';
        btnText.textContent = 'Analyze MRI Scan';
        analyzeBtn.disabled = false;
    }
}

// ── Display Results ──
function displayResults(result) {
    resultsPanel.style.display = 'block';

    const info = result.clinical_info;

    // Diagnosis title
    document.getElementById('diagnosisTitle').textContent = info.full_name;

    // Severity badge
    const badge = document.getElementById('severityBadge');
    badge.textContent = info.severity;
    badge.className = 'severity-badge ' + info.severity.toLowerCase();

    // Diagnosis icon
    const icon = document.getElementById('diagnosisIcon');
    icon.style.background = info.color + '22';
    const iconSymbols = {
        'HIGH': '<span style="font-size:1.5rem">&#9888;</span>',
        'MODERATE': '<span style="font-size:1.5rem">&#9888;</span>',
        'LOW': '<span style="font-size:1.5rem">&#10004;</span>',
    };
    icon.innerHTML = iconSymbols[info.severity] || '';

    // Description
    document.getElementById('diagnosisDesc').textContent = info.description;

    // Confidence circle
    const conf = result.confidence;
    document.getElementById('confidenceValue').textContent = conf.toFixed(1);
    const ring = document.getElementById('progressRing');
    const circumference = 2 * Math.PI * 42;
    const offset = circumference - (conf / 100) * circumference;

    // Add gradient def to SVG if not exists
    const svg = ring.closest('svg');
    if (!svg.querySelector('#progressGrad')) {
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        defs.innerHTML = `
            <linearGradient id="progressGrad" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#6366f1"/>
                <stop offset="100%" stop-color="#06b6d4"/>
            </linearGradient>
        `;
        svg.insertBefore(defs, svg.firstChild);
    }

    // Animate ring
    ring.style.strokeDasharray = circumference;
    ring.style.strokeDashoffset = circumference;
    requestAnimationFrame(() => {
        ring.style.strokeDashoffset = offset;
    });

    // Probability bars
    const probBars = document.getElementById('probBars');
    probBars.innerHTML = '';

    // Sort by probability descending
    const sortedProbs = Object.entries(result.class_probabilities)
        .sort((a, b) => b[1] - a[1]);

    sortedProbs.forEach(([cls, prob]) => {
        const isPredicted = cls === result.predicted_class;
        const item = document.createElement('div');
        item.className = 'prob-item';
        item.innerHTML = `
            <div class="prob-label">
                <span class="prob-name">${cls}</span>
                <span class="prob-value">${prob.toFixed(2)}%</span>
            </div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill ${isPredicted ? 'predicted' : 'other'}" 
                     style="width: 0%"></div>
            </div>
        `;
        probBars.appendChild(item);

        // Animate bar width
        requestAnimationFrame(() => {
            const fill = item.querySelector('.prob-bar-fill');
            fill.style.width = `${Math.max(prob, 0.5)}%`;
        });
    });

    // Recommended actions
    const actionsList = document.getElementById('actionsList');
    actionsList.innerHTML = '';
    info.actions.forEach((action, i) => {
        const li = document.createElement('li');
        li.innerHTML = `<span class="action-num">${i + 1}</span><span>${action}</span>`;
        actionsList.appendChild(li);
    });

    // Model info
    document.getElementById('modelNameResult').textContent = result.model_name;
    document.getElementById('inferenceTime').textContent = `${result.inference_time_ms.toFixed(1)} ms`;

    // Scroll to results
    resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── HL7 FHIR PDF Report Generation & Preview ──
async function getOrGenerateReport(triggerBtn = null) {
    if (!currentPrediction) {
        showToast('Please analyze an MRI scan first.', 'error');
        return null;
    }

    if (currentReportData) {
        return currentReportData;
    }

    let originalHtml = '';
    if (triggerBtn) {
        originalHtml = triggerBtn.innerHTML;
        triggerBtn.disabled = true;
        triggerBtn.innerHTML = `
            <div class="btn-loader" style="width:14px;height:14px;border-width:2px;display:inline-block;margin-right:6px;"></div>
            Generating PDF...
        `;
    }

    const payload = {
        ...currentPrediction,
        patient_name: document.getElementById('patientName').value.trim() || 'Anonymous Patient',
        patient_id: document.getElementById('patientId').value.trim() || 'N/A',
        patient_age: document.getElementById('patientAge').value.trim() || 'N/A',
        patient_gender: document.getElementById('patientGender').value || 'unknown',
        referring_physician: document.getElementById('referringPhysician').value.trim() || 'N/A',
        clinical_notes: document.getElementById('clinicalNotes').value.trim() || '',
    };

    try {
        const res = await fetch('/generate-report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || 'Failed to generate PDF report');
        }

        const data = await res.json();
        currentReportData = data;
        return data;

    } catch (e) {
        showToast(`Report error: ${e.message}`, 'error');
        return null;
    } finally {
        if (triggerBtn) {
            triggerBtn.disabled = false;
            triggerBtn.innerHTML = originalHtml;
        }
    }
}

async function downloadPDFReport() {
    const downloadBtn = document.getElementById('downloadHL7');
    const data = await getOrGenerateReport(downloadBtn);
    if (!data) return;

    // Trigger browser download
    const link = document.createElement('a');
    link.href = data.download_url;
    link.download = data.filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    showToast('HL7 FHIR PDF report downloaded!', 'success');
}

async function previewPDFReport() {
    const previewBtn = document.getElementById('previewReport');
    const data = await getOrGenerateReport(previewBtn);
    if (!data) return;

    const modal = document.getElementById('pdfModal');
    const frame = document.getElementById('pdfPreviewFrame');
    const openNewTab = document.getElementById('openNewTabBtn');

    if (frame) {
        frame.src = data.view_url;
    }
    if (openNewTab) {
        openNewTab.href = data.view_url;
    }
    if (modal) {
        modal.style.display = 'flex';
    }

    showToast('HL7 FHIR report opened for preview', 'success');
}

function closeModal() {
    const modal = document.getElementById('pdfModal');
    if (modal) {
        modal.style.display = 'none';
        const frame = document.getElementById('pdfPreviewFrame');
        if (frame) {
            frame.src = '';
        }
    }
}

// ── Toast Notifications ──
function showToast(message, type = 'success') {
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        if (toast.parentNode) toast.remove();
    }, 3000);
}
