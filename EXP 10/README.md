# MedAI CadX: Patient Medical Image Diagnostic & Machine Learning Platform

> **An End-to-End Clinical Computer-Aided Diagnostic (CADx) System**  
> Covering **Load Input Image**, **Preprocessor**, **Feature Extraction**, **Training**, and **Performance Evaluation** all integrated within a unified, responsive single-frame web application.

---

## 🌟 Key Highlights & Project Architecture

```
[1. Load Input Scan & Patient Data]
            │
            ▼
[2. Preprocessor Pipeline]
   ├── 2.1 Fixed-Resolution Standardization (256x256)
   ├── 2.2 Grayscale Transformation
   ├── 2.3 Gaussian Spatial Denoising (σ = 1.2, 5x5 kernel)
   ├── 2.4 CLAHE Adaptive Contrast Enhancement
   ├── 2.5 Otsu Dynamic ROI Thresholding & Morphological Closing
   └── 2.6 Canny Edge Detection & Diagnostic Overlay
            │
            ▼
[3. Radiomic Feature Extraction (18 Dimensions)]
   ├── 1st Order Statistics: Mean, Std Dev, Variance, Skewness, Kurtosis, Shannon Entropy
   ├── GLCM Haralick Textures: Contrast, Dissimilarity, Homogeneity, Energy, Correlation
   └── Morphological Descriptors: Area Ratio, Perimeter, Circularity, Compactness, Solidity, Hu Moments
            │
            ▼
[4. Machine Learning Algorithm Training & Performance Evaluation]
   ├── Algorithms: Random Forest, SVM (RBF), KNN, Logistic Regression, Decision Tree
   ├── Standardized Scaling (Z-Score Normalization)
   ├── Stratified Train/Test Split & 5-Fold Cross-Validation
   └── Metrics: Accuracy, Precision, Sensitivity/Recall, Specificity, F1-Score, ROC-AUC, Confusion Matrix
            │
            ▼
[5. Real-Time Patient Diagnosis & Clinical PDF Report Generation]
   ├── Live Probability Distribution & Confidence Scoring
   ├── Clinical Risk Classification (Low / Moderate / High)
   └── Printable / Downloadable Hospital-Grade Diagnostic PDF Report (ReportLab)
```

---

## 📋 System Requirements & Prerequisites

- **Operating System:** Windows 10/11, Linux, or macOS
- **Python Version:** Python 3.9, 3.10, 3.11, or 3.12
- **Required Libraries:**
  - `Flask` (Web framework and REST APIs)
  - `OpenCV (opencv-python)` (Computer vision, CLAHE, morphological operations, edges)
  - `Scikit-Learn` (Machine learning classifiers, metrics, train/test split, cross-validation)
  - `NumPy` & `SciPy` (Matrix operations, GLCM texture calculations, statistical moments)
  - `ReportLab` (Clinical PDF diagnostic report generation)
  - `Matplotlib` & `Pillow` (Image handling and numerical graphing)
  - `Chart.js` (Loaded via CDN in web UI for interactive charts)

---

## 🚀 How to Compile and Run the Code

### Step 1: Open Terminal / Windows PowerShell
Open your terminal or PowerShell and navigate to the project directory:
```powershell
cd d:\lab
```

### Step 2: Install Required Dependencies
Run the following command to ensure all required libraries are installed:
```powershell
pip install -r requirements.txt
```

### Step 3: Generate Dataset and Sample Scans (Initial Setup)
Run the dataset builder to generate 100 benchmark training images (50 Normal, 50 Pathological) and 4 clinical sample test cases:
```powershell
python generate_dataset.py
```

### Step 4: Train and Evaluate the ML Models
Train all 5 machine learning models and calculate performance metrics:
```powershell
python -c "from model_trainer import MedicalModelManager; m = MedicalModelManager(); m.train_and_evaluate_all()"
```

### Step 5: Start the Web Application
Launch the Flask web server:
```powershell
python app.py
```
*(Or on Windows, simply double-click `run.bat`)*

### Step 6: Access the Web Application
Open your web browser (Chrome, Edge, Firefox, Brave) and navigate to:
```
http://127.0.0.1:5000
```

---

## 🔬 Deep-Dive: The 5 Processing Stages

### Stage 1: Load Input Image & Patient Demographics
- **User Upload:** Accepts drag-and-drop or file selection of any DICOM-converted or standard radiological image (`.png`, `.jpg`, `.jpeg`).
- **Benchmark Presets:** 4 clinical demo cases available with one click:
  1. Brain MRI: Healthy / Normal Parenchyma
  2. Brain MRI: Glioblastoma Lesion
  3. Chest Radiograph: Healthy Clear Lungs
  4. Chest Radiograph: Lobar Pneumonia Consolidation
- **Metadata Registration:** Captures Patient ID, Patient Name, Age, Gender, and Scan Modality.

### Stage 2: Preprocessor Pipeline
Medical images contain scanner noise, low contrast, and non-standard intensity distributions. MedAI CadX executes 6 sequential transformations:
1. **Resizing:** Standardizes input to $256 \times 256$ pixels.
2. **Grayscale Conversion:** Reduces 3-channel RGB to luminance matrix $I(x, y)$.
3. **Gaussian Filter Smoothing:**
   $$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$
   Removes high-frequency scanner noise while preserving edge boundaries.
4. **CLAHE (Contrast Limited Adaptive Histogram Equalization):**
   Prevents over-amplification of noise in homogeneous regions and enhances subtle soft-tissue contrast between healthy parenchyma and lesions.
5. **Otsu's Automatic Thresholding & Morphological Closing:**
   Calculates optimal threshold $t^*$ maximizing inter-class variance $\sigma_B^2(t)$ to delineate dense lesion tissue candidates.
6. **Canny Edge Detection & Diagnostic Overlay:**
   Detects anatomical margins and highlights candidate lesions with a semi-transparent diagnostic red overlay.

### Stage 3: Quantitative Radiomic Feature Extraction
Extracts an 18-dimensional feature vector $\vec{x} \in \mathbb{R}^{18}$ per image:
- **Statistical Moments:**
  - Mean Intensity ($\mu$)
  - Standard Deviation ($\sigma$) & Variance ($\sigma^2$)
  - Skewness ($\gamma_1 = \frac{E[(X-\mu)^3]}{\sigma^3}$)
  - Kurtosis ($\beta_2 = \frac{E[(X-\mu)^4]}{\sigma^4}$)
  - Shannon Entropy ($H = -\sum p(i) \log_2 p(i)$)
- **Haralick GLCM Texture Features (32-level quantization):**
  - **Contrast:** $\sum_{i,j} |i - j|^2 P(i,j)$ (local tissue variation)
  - **Dissimilarity:** $\sum_{i,j} |i - j| P(i,j)$
  - **Homogeneity:** $\sum_{i,j} \frac{P(i,j)}{1 + |i - j|^2}$
  - **Energy (ASM):** $\sum_{i,j} P(i,j)^2$ (uniformity of tissue)
  - **Correlation:** $\sum_{i,j} \frac{(i - \mu_i)(j - \mu_j) P(i,j)}{\sigma_i \sigma_j}$
- **Morphological & Shape Descriptors:**
  - ROI Area Ratio ($\frac{\text{Lesion Pixels}}{\text{Total Pixels}}$)
  - Boundary Perimeter ($P$)
  - Circularity / Roundness ($4\pi \frac{\text{Area}}{P^2}$)
  - Compactness ($\frac{P^2}{\text{Area}}$)
  - Solidity ($\frac{\text{Contour Area}}{\text{Convex Hull Area}}$)
  - Hu Invariant Moments ($h_1, h_2$)

### Stage 4: Machine Learning Model Training & Evaluation
The system benchmarks 5 distinct supervised classification algorithms:
1. **Random Forest Classifier:** Ensemble of decision trees with bagging and feature subsampling.
2. **Support Vector Machine (SVM):** Radial Basis Function (RBF) kernel mapping features into high-dimensional hyperplane.
3. **K-Nearest Neighbors (KNN):** Distance-weighted Euclidean neighborhood classification ($k=5$).
4. **Logistic Regression:** L2-regularized logistic classification.
5. **Decision Tree Classifier:** Information gain / Gini impurity partitioning.

#### Performance Evaluation Metrics Computed:
- **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision (PPV):** $\frac{TP}{TP + FP}$
- **Sensitivity / Recall:** $\frac{TP}{TP + FN}$ *(Critical in healthcare: minimizes missed pathology)*
- **Specificity (TNR):** $\frac{TN}{TN + FP}$ *(Minimizes false alarms)*
- **F1-Score:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **ROC-AUC Score & Curves:** Evaluates True Positive Rate vs False Positive Rate across all thresholds.
- **Confusion Matrix:** True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN).

### Stage 5: Live Patient Diagnosis & PDF Report Generation
- **Real-Time Classification:** Evaluates the loaded patient image against the chosen or top-performing ML model.
- **Confidence Meter:** Displays predicted probabilities $P(\text{Normal})$ and $P(\text{Pathological})$.
- **Risk Stratification:** Categorizes case as Low, Moderate, or High risk.
- **Automated Clinical Report (PDF):**
  Clicking **"Download Official Clinical PDF Report"** generates a verified medical diagnostic document including:
  - Hospital Letterhead & Verification ID
  - Complete Patient Demographics
  - Side-by-side Image Visualizations (Raw, Preprocessed Overlay, Canny Edges)
  - Quantitative Radiomic Bio-Profile Table
  - Machine Learning Classification & Medical Guidance
  - Radiologist Sign-off Box & Disclaimer

---

## 📁 Project Directory Structure

```
d:\lab\
├── app.py                     # Flask web server & REST API routes
├── preprocessor.py            # Image preprocessing pipeline (Gaussian, CLAHE, Otsu, Canny)
├── feature_extractor.py       # Radiomic feature extraction (GLCM, moments, morphology)
├── model_trainer.py           # ML model training, benchmarking, ROC curves, inference
├── report_generator.py        # Clinical PDF report generator (ReportLab)
├── generate_dataset.py        # Procedural medical image and sample case generator
├── requirements.txt           # Python package specifications
├── run.bat                    # Windows 1-click execution script
├── README.md                  # Complete technical manual and execution guide
├── templates\
│   └── index.html             # Single-frame responsive medical UI
├── static\
│   ├── css\
│   │   └── style.css          # Clinical dark/light theme, modern cards, and responsive grids
│   ├── js\
│   │   └── app.js             # Interactive pipeline state controller and Chart.js integration
│   └── placeholder.png        # Default initial viewport placeholder
├── sample_data\               # 4 Pre-packaged clinical test cases (Normal/Tumor/Pneumonia)
├── dataset\                   # 100 training images (Normal / Pathological)
└── models\                    # Persisted trained models (scaler.pkl, models.pkl, evaluation.pkl)
```

---

## 🌐 REST API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Loads the one-frame single-page application dashboard |
| `GET` | `/api/status` | Returns pipeline status, dataset counts, and model metrics |
| `GET` | `/api/samples` | Lists available pre-loaded clinical sample cases |
| `POST` | `/api/load-sample` | Loads a sample case and executes preprocessing + feature extraction |
| `POST` | `/api/upload` | Uploads a custom patient scan image with clinical demographics |
| `POST` | `/api/train` | Triggers retraining of all 5 ML models with custom train/test split |
| `POST` | `/api/predict` | Classifies the current patient case with chosen ML algorithm |
| `GET` | `/api/download-report` | Generates and streams clinical PDF diagnostic report |

---

## 🏥 Clinical Disclaimer
This system is developed for **academic, research, and educational demonstration** of machine learning pipelines in biomedical image processing. In clinical environments, AI predictions must be corroborated by a board-certified radiologist and diagnostic biopsy.

