// MedAI CadX Client-side Pipeline Controller
document.addEventListener('DOMContentLoaded', () => {
    // State Store
    let currentData = {
        patient: {},
        previews: {},
        stats: {},
        features: {},
        evaluations: {},
        prediction: {}
    };

    let featuresChartInstance = null;
    let rocChartInstance = null;

    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const imageFileInput = document.getElementById('imageFileInput');
    const browseFilesBtn = document.getElementById('browseFilesBtn');
    const loadedImagePreview = document.getElementById('loadedImagePreview');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const runPipelineBtn = document.getElementById('runPipelineBtn');
    const quickTrainBtn = document.getElementById('quickTrainBtn');
    const btnExecuteTraining = document.getElementById('btnExecuteTraining');
    const btnRunDiagnosis = document.getElementById('btnRunDiagnosis');
    const selInferenceModel = document.getElementById('selInferenceModel');
    const selTestSplit = document.getElementById('selTestSplit');
    const viewModeTabs = document.getElementById('viewModeTabs');
    const pipelineStepper = document.getElementById('pipelineStepper');

    // Clinical descriptions for features
    const FEATURE_DESCRIPTIONS = {
        "mean_intensity": { category: "Statistical", desc: "Overall tissue radiodensity / brightness" },
        "std_intensity": { category: "Statistical", desc: "Global image contrast across tissue structures" },
        "variance": { category: "Statistical", desc: "Dispersion of pixel intensity values" },
        "skewness": { category: "Statistical", desc: "Asymmetry of pixel intensity distribution" },
        "kurtosis": { category: "Statistical", desc: "Tailedness / peakedness of tissue histogram" },
        "entropy": { category: "Statistical", desc: "Shannon entropy measuring complexity & disorder" },
        "glcm_contrast": { category: "GLCM Texture", desc: "Local variations and intensity difference" },
        "glcm_dissimilarity": { category: "GLCM Texture", desc: "Mean gray level differences between pixel pairs" },
        "glcm_homogeneity": { category: "GLCM Texture", desc: "Closeness of distributed element pairs" },
        "glcm_energy": { category: "GLCM Texture", desc: "Angular second moment; indicates uniformity" },
        "glcm_correlation": { category: "GLCM Texture", desc: "Linear dependency of gray levels in tissue" },
        "roi_area_ratio": { category: "Morphological", desc: "Proportion of candidate anomaly area vs total" },
        "perimeter": { category: "Morphological", desc: "Contour boundary length of lesion / tissue" },
        "circularity": { category: "Morphological", desc: "Roundness metric (1.0 = perfect circle)" },
        "compactness": { category: "Morphological", desc: "Boundary complexity and structural irregularity" },
        "solidity": { category: "Morphological", desc: "Ratio of contour area to convex hull envelope" },
        "hu_moment_1": { category: "Morphological", desc: "1st log-scale invariant moment of mass distribution" },
        "hu_moment_2": { category: "Morphological", desc: "2nd log-scale invariant moment of mass distribution" }
    };

    // Show Notification Toast
    function showToast(msg) {
        const toast = document.getElementById('toastNotification');
        const toastMsg = document.getElementById('toastMessage');
        toastMsg.textContent = msg;
        toast.classList.remove('hidden');
        setTimeout(() => toast.classList.add('hidden'), 3500);
    }

    // Initialize System Status & Default Case
    async function initSystem() {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            if (data.evaluations && data.evaluations.models) {
                currentData.evaluations = data.evaluations;
                renderEvaluationTable(data.evaluations);
            }
            // Load default benchmark case (Brain Tumor)
            await loadSampleCase('brain_tumor');
        } catch (err) {
            console.error("Initialization error:", err);
        }
    }

    // Load Pre-configured Sample Case
    async function loadSampleCase(sampleId) {
        loadingOverlay.style.display = 'flex';
        try {
            const res = await fetch('/api/load-sample', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sample_id: sampleId })
            });
            const data = await res.json();
            if (data.success) {
                updatePipelineViews(data);
                showToast(`Loaded ${data.patient.name} (${sampleId}) successfully.`);
                // Auto trigger diagnosis
                await triggerDiagnosis();
            } else {
                alert(data.error || 'Failed to load sample');
            }
        } catch (err) {
            console.error(err);
            alert('Error communicating with server.');
        } finally {
            loadingOverlay.style.display = 'none';
        }
    }

    // Update UI Elements with Stage 1, 2, 3 Data
    function updatePipelineViews(data) {
        currentData.patient = data.patient;
        currentData.previews = data.previews;
        currentData.stats = data.stats;
        currentData.features = data.features;

        // Stage 1: Demographics & Loaded preview
        document.getElementById('patientId').value = data.patient.patient_id || '';
        document.getElementById('patientName').value = data.patient.name || '';
        document.getElementById('patientAge').value = data.patient.age || '';
        document.getElementById('patientGender').value = data.patient.gender || 'Female';
        document.getElementById('patientModality').value = data.patient.modality || 'Brain MRI (Axial T1/T2)';

        loadedImagePreview.src = data.previews.original;
        document.getElementById('statDim').textContent = data.stats.dimensions;
        document.getElementById('statChannels').textContent = data.stats.color_channels === 1 ? '1 (Monochrome)' : `${data.stats.color_channels} Channels`;
        document.getElementById('statMean').textContent = data.stats.mean_intensity;
        document.getElementById('statStd').textContent = data.stats.std_intensity;

        // Stage 2: Preprocessor Gallery
        document.getElementById('imgPreOriginal').src = data.previews.original;
        document.getElementById('imgPreDenoised').src = data.previews.denoised;
        document.getElementById('imgPreEnhanced').src = data.previews.enhanced;
        document.getElementById('imgPreMask').src = data.previews.segmented_mask;
        document.getElementById('imgPreEdges').src = data.previews.edges;
        document.getElementById('imgPreOverlay').src = data.previews.overlay;

        // Stage 3: Features Table & Chart
        renderFeaturesTable(data.features);
        renderFeaturesChart(data.features);
    }

    // Render Stage 3 Feature Table
    function renderFeaturesTable(feats) {
        const tbody = document.getElementById('featuresTableBody');
        tbody.innerHTML = '';

        for (const [key, val] of Object.entries(feats)) {
            const meta = FEATURE_DESCRIPTIONS[key] || { category: "General", desc: "Extracted radiomic feature" };
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><span class="pill-badge pill-info">${meta.category}</span></td>
                <td><code>${key}</code></td>
                <td><strong>${val}</strong></td>
                <td><small style="color: var(--text-secondary);">${meta.desc}</small></td>
            `;
            tbody.appendChild(tr);
        }
    }

    // Render Stage 3 Radiomic Feature Signature Bar Chart
    function renderFeaturesChart(feats) {
        const ctx = document.getElementById('featuresChart').getContext('2d');
        
        // Select key normalized representative features for visual chart
        const chartKeys = [
            "mean_intensity", "std_intensity", "entropy",
            "glcm_contrast", "glcm_homogeneity", "glcm_energy",
            "roi_area_ratio", "circularity", "solidity"
        ];
        const labels = chartKeys.map(k => k.replace(/_/g, ' '));
        const rawValues = chartKeys.map(k => feats[k] !== undefined ? feats[k] : 0);

        // Normalize for visual comparison
        const maxVal = Math.max(...rawValues, 1.0);
        const normValues = rawValues.map(v => Number((v / maxVal * 100).toFixed(1)));

        if (featuresChartInstance) {
            featuresChartInstance.destroy();
        }

        featuresChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Normalized Biomarker Magnitude (%)',
                    data: normValues,
                    backgroundColor: [
                        'rgba(59, 130, 246, 0.7)',
                        'rgba(99, 102, 241, 0.7)',
                        'rgba(168, 85, 247, 0.7)',
                        'rgba(236, 72, 153, 0.7)',
                        'rgba(244, 63, 94, 0.7)',
                        'rgba(245, 158, 11, 0.7)',
                        'rgba(16, 185, 129, 0.7)',
                        'rgba(20, 184, 166, 0.7)',
                        'rgba(6, 182, 212, 0.7)'
                    ],
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    borderWidth: 1,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            afterLabel: (ctx) => `Raw Value: ${rawValues[ctx.dataIndex]}`
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#9ca3af' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#9ca3af', font: { size: 10 } }
                    }
                }
            }
        });
    }

    // Render Stage 4 Model Evaluation Benchmark Table & Visuals
    function renderEvaluationTable(evalData) {
        if (!evalData || !evalData.models) return;

        const tbody = document.getElementById('evalTableBody');
        tbody.innerHTML = '';
        const bestModel = evalData.best_model || "Random Forest";
        document.getElementById('bestModelBadge').textContent = `Best Model: ${bestModel}`;

        for (const [name, metrics] of Object.entries(evalData.models)) {
            const tr = document.createElement('tr');
            const isBest = name === bestModel;
            tr.innerHTML = `
                <td>
                    <strong>${name}</strong>
                    ${isBest ? '<span class="pill-badge pill-info" style="margin-left:6px;">Top Performer</span>' : ''}
                </td>
                <td><strong class="text-success">${metrics.accuracy}%</strong></td>
                <td>${metrics.precision}%</td>
                <td>${metrics.recall_sensitivity}%</td>
                <td>${metrics.specificity}%</td>
                <td>${metrics.f1_score}%</td>
                <td><code>${metrics.roc_auc}</code></td>
                <td>${metrics.cv_mean}%</td>
                <td>
                    <button class="btn btn-outline btn-sm view-cm-btn" data-model="${name}">
                        <i class="fa-solid fa-eye"></i> View CM
                    </button>
                </td>
            `;
            tbody.appendChild(tr);
        }

        // Add click listener for CM viewer buttons
        document.querySelectorAll('.view-cm-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const modelName = e.currentTarget.getAttribute('data-model');
                updateConfusionMatrix(modelName, evalData.models[modelName]);
            });
        });

        // Set default CM and ROC Chart to best model
        if (evalData.models[bestModel]) {
            updateConfusionMatrix(bestModel, evalData.models[bestModel]);
            renderROCChart(evalData.models);
        }
    }

    // Update Stage 4 Confusion Matrix Display
    function updateConfusionMatrix(modelName, modelMetrics) {
        document.getElementById('cmModelTitle').textContent = `Selected: ${modelName}`;
        const cm = modelMetrics.confusion_matrix;
        document.getElementById('cmTN').textContent = cm.tn;
        document.getElementById('cmFP').textContent = cm.fp;
        document.getElementById('cmFN').textContent = cm.fn;
        document.getElementById('cmTP').textContent = cm.tp;
    }

    // Render Stage 4 ROC Curves Chart
    function renderROCChart(modelsData) {
        const ctx = document.getElementById('rocChart').getContext('2d');
        const colorPalette = {
            "Random Forest": "#3b82f6",
            "Support Vector Machine (SVM)": "#10b981",
            "K-Nearest Neighbors (KNN)": "#f59e0b",
            "Logistic Regression": "#ec4899",
            "Decision Tree": "#8b5cf6"
        };

        const datasets = [];
        for (const [name, data] of Object.entries(modelsData)) {
            if (data.roc_points && data.roc_points.fpr) {
                const points = data.roc_points.fpr.map((fpr, i) => ({
                    x: fpr,
                    y: data.roc_points.tpr[i]
                }));

                datasets.push({
                    label: `${name} (AUC = ${data.roc_auc})`,
                    data: points,
                    borderColor: colorPalette[name] || '#3b82f6',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    showLine: true,
                    tension: 0.1,
                    pointRadius: 2
                });
            }
        }

        // Add reference diagonal chance line (AUC = 0.5)
        datasets.push({
            label: 'Random Chance (AUC = 0.50)',
            data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            borderColor: '#6b7280',
            borderDash: [5, 5],
            borderWidth: 1,
            pointRadius: 0,
            fill: false
        });

        if (rocChartInstance) {
            rocChartInstance.destroy();
        }

        rocChartInstance = new Chart(ctx, {
            type: 'scatter',
            data: { datasets: datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#9ca3af', font: { size: 10 } }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'False Positive Rate (1 - Specificity)', color: '#9ca3af' },
                        min: 0,
                        max: 1,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#9ca3af' }
                    },
                    y: {
                        title: { display: true, text: 'True Positive Rate (Sensitivity / Recall)', color: '#9ca3af' },
                        min: 0,
                        max: 1,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#9ca3af' }
                    }
                }
            }
        });
    }

    // Trigger Patient Diagnosis (Stage 5)
    async function triggerDiagnosis() {
        const selectedModel = selInferenceModel.value;
        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_name: selectedModel })
            });
            const data = await res.json();
            if (data.success) {
                currentData.prediction = data.prediction;
                renderDiagnosisView(data.prediction);
            } else {
                alert(data.error || 'Diagnosis failed.');
            }
        } catch (err) {
            console.error('Prediction error:', err);
        }
    }

    // Update Stage 5 UI with Diagnosis & Clinical Preview
    function renderDiagnosisView(pred) {
        const banner = document.getElementById('diagnosisBanner');
        const icon = document.getElementById('diagIcon');
        const title = document.getElementById('diagTitle');
        const riskBadge = document.getElementById('diagRiskBadge');
        const modelBadge = document.getElementById('diagModelBadge');
        const patientSummary = document.getElementById('diagPatientSummary');
        const confidenceVal = document.getElementById('confidenceValue');
        const confidenceBar = document.getElementById('confidenceBar');
        const probNorm = document.getElementById('probNormal');
        const probPath = document.getElementById('probPathology');
        const recText = document.getElementById('clinicalRecommendationText');

        // Report preview card elements
        const rptFindingAlert = document.getElementById('rptFindingAlert');
        const rptFindingTitle = document.getElementById('rptFindingTitle');
        const rptFindingDetails = document.getElementById('rptFindingDetails');
        const rptName = document.getElementById('rptName');
        const rptId = document.getElementById('rptId');
        const rptDemog = document.getElementById('rptDemog');
        const rptModality = document.getElementById('rptModality');
        const rptDateStamp = document.getElementById('rptDateStamp');

        const pt = currentData.patient;
        patientSummary.textContent = `Patient: ${pt.name || 'Anonymous'} | ${pt.patient_id || 'ID-N/A'}`;
        title.textContent = pred.diagnosis;
        modelBadge.textContent = `Model: ${pred.model_used}`;
        riskBadge.textContent = `Risk Level: ${pred.risk_level}`;

        confidenceVal.textContent = `${pred.confidence}%`;
        confidenceBar.style.width = `${pred.confidence}%`;
        probNorm.textContent = `${pred.prob_normal}%`;
        probPath.textContent = `${pred.prob_pathology}%`;
        recText.textContent = pred.clinical_recommendation;

        // Visual styling for Normal vs Pathological
        if (pred.prediction_label === 1) {
            // Pathological
            banner.classList.remove('normal-case');
            icon.className = 'fa-solid fa-triangle-exclamation';
            riskBadge.className = 'pill-badge pill-danger';
            confidenceBar.style.background = 'linear-gradient(90deg, #f59e0b, #ef4444)';

            rptFindingAlert.style.background = '#fee2e2';
            rptFindingAlert.style.borderColor = '#f87171';
            rptFindingAlert.style.color = '#991b1b';
            rptFindingTitle.textContent = 'ABNORMAL / PATHOLOGICAL LESION IDENTIFIED';
            rptFindingDetails.textContent = `Classifier detected significant lesion morphology with ${pred.confidence}% diagnostic confidence. ${pred.clinical_recommendation}`;
        } else {
            // Normal
            banner.classList.add('normal-case');
            icon.className = 'fa-solid fa-circle-check';
            riskBadge.className = 'pill-badge pill-info';
            riskBadge.style.color = '#10b981';
            confidenceBar.style.background = 'linear-gradient(90deg, #3b82f6, #10b981)';

            rptFindingAlert.style.background = '#dcfce7';
            rptFindingAlert.style.borderColor = '#86efac';
            rptFindingAlert.style.color = '#166534';
            rptFindingTitle.textContent = 'NO SIGNIFICANT ABNORMALITIES DETECTED';
            rptFindingDetails.textContent = `Anatomical tissue characteristics and GLCM texture vectors conform to healthy physiological baseline standards (${pred.confidence}% confidence).`;
        }

        // Update Report Preview Box
        rptName.textContent = pt.name || 'Anonymous Patient';
        rptId.textContent = pt.patient_id || 'PT-9021';
        rptDemog.textContent = `${pt.age || '40'} yrs / ${pt.gender || 'Female'}`;
        rptModality.textContent = pt.modality || 'Brain MRI';
        rptDateStamp.textContent = `Date: ${new Date().toLocaleDateString('en-US', { day: 'numeric', month: 'short', year: 'numeric' })}`;
    }

    // Trigger Model Training from UI
    async function executeModelTraining() {
        loadingOverlay.style.display = 'flex';
        const splitVal = parseFloat(selTestSplit.value);
        showToast('Initiating ML training and cross-validation across 5 algorithms...');

        try {
            const res = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ test_size: splitVal })
            });
            const data = await res.json();
            if (data.success) {
                currentData.evaluations = data.results;
                renderEvaluationTable(data.results);
                showToast(`Training complete! Best model: ${data.results.best_model}`);
                // Re-run diagnosis with best model
                selInferenceModel.value = data.results.best_model;
                await triggerDiagnosis();
            } else {
                alert(data.error || 'Training failed');
            }
        } catch (err) {
            console.error('Training execution error:', err);
            alert('Failed to connect to training engine.');
        } finally {
            loadingOverlay.style.display = 'none';
        }
    }

    // Handle Upload via Drag & Drop or File Input
    async function handleFileUpload(file) {
        if (!file) return;
        const formData = new FormData();
        formData.append('image', file);
        formData.append('patient_id', document.getElementById('patientId').value);
        formData.append('patient_name', document.getElementById('patientName').value);
        formData.append('age', document.getElementById('patientAge').value);
        formData.append('gender', document.getElementById('patientGender').value);
        formData.append('modality', document.getElementById('patientModality').value);

        loadingOverlay.style.display = 'flex';
        try {
            const res = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (data.success) {
                updatePipelineViews(data);
                showToast(`Patient scan uploaded & processed successfully!`);
                await triggerDiagnosis();
            } else {
                alert(data.error || 'Upload failed');
            }
        } catch (err) {
            console.error(err);
            alert('Error uploading file to server.');
        } finally {
            loadingOverlay.style.display = 'none';
        }
    }

    // Event Listeners: Presets
    document.querySelectorAll('.sample-pill').forEach(pill => {
        pill.addEventListener('click', (e) => {
            document.querySelectorAll('.sample-pill').forEach(p => p.classList.remove('active-sample'));
            pill.classList.add('active-sample');
            const sampleId = pill.getAttribute('data-sample');
            loadSampleCase(sampleId);
        });
    });

    // Event Listeners: File Upload
    browseFilesBtn.addEventListener('click', () => imageFileInput.click());
    imageFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    // Run Pipeline Button (re-runs feature extraction on current scan)
    runPipelineBtn.addEventListener('click', () => {
        showToast('Preprocessing & feature extraction refreshed.');
        triggerDiagnosis();
    });

    // Model Retraining Buttons
    btnExecuteTraining.addEventListener('click', executeModelTraining);
    quickTrainBtn.addEventListener('click', executeModelTraining);

    // Inference Model Selection Change
    selInferenceModel.addEventListener('change', triggerDiagnosis);
    btnRunDiagnosis.addEventListener('click', triggerDiagnosis);

    // Stepper Navigation clicks
    pipelineStepper.querySelectorAll('.step-item').forEach(item => {
        item.addEventListener('click', () => {
            pipelineStepper.querySelectorAll('.step-item').forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            const targetId = item.getAttribute('data-target');
            const targetEl = document.getElementById(targetId);
            if (targetEl) {
                targetEl.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });

    // View Mode Switcher (All-In-One Frame vs Step-by-Step)
    viewModeTabs.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            viewModeTabs.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const mode = btn.getAttribute('data-view');
            const sections = document.querySelectorAll('.pipeline-section');
            if (mode === 'all') {
                sections.forEach(s => s.style.display = 'block');
            } else {
                // In step mode, keep active section visible
                sections.forEach((s, idx) => {
                    s.style.display = idx === 0 ? 'block' : 'none';
                });
                showToast('Step mode enabled. Click steps on top bar to navigate.');
            }
        });
    });

    // Boot system
    initSystem();
});

