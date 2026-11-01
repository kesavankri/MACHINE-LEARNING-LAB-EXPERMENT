import os
import io
import cv2
import json
import base64
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

from preprocessor import MedicalImagePreprocessor, encode_img_to_base64
from feature_extractor import MedicalFeatureExtractor
from model_trainer import MedicalModelManager
from report_generator import create_patient_pdf_report

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize core pipeline engines
preprocessor = MedicalImagePreprocessor()
extractor = MedicalFeatureExtractor()
model_manager = MedicalModelManager()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Clinical sample cases bundled with the application
SAMPLE_CASES = {
    "brain_normal": {
        "id": "brain_normal",
        "file": os.path.join(BASE_DIR, "sample_data", "patient_brain_normal.png"),
        "title": "Case #101: Normal Brain MRI",
        "modality": "Brain MRI (Axial T1)",
        "patient": {"name": "Emily Davis", "age": 34, "gender": "Female", "patient_id": "MRI-2026-081"},
        "expected": "Normal / Healthy"
    },
    "brain_tumor": {
        "id": "brain_tumor",
        "file": os.path.join(BASE_DIR, "sample_data", "patient_brain_tumor.png"),
        "title": "Case #202: Glioblastoma Brain Lesion",
        "modality": "Brain MRI (Axial Contrast)",
        "patient": {"name": "Robert Miller", "age": 58, "gender": "Male", "patient_id": "MRI-2026-114"},
        "expected": "Pathological (Tumor)"
    },
    "chest_normal": {
        "id": "chest_normal",
        "file": os.path.join(BASE_DIR, "sample_data", "patient_chest_normal.png"),
        "title": "Case #303: Healthy Chest Radiograph",
        "modality": "Chest X-Ray (Posteroanterior)",
        "patient": {"name": "Michael Chen", "age": 29, "gender": "Male", "patient_id": "CXR-2026-309"},
        "expected": "Normal / Clear Lungs"
    },
    "chest_pneumonia": {
        "id": "chest_pneumonia",
        "file": os.path.join(BASE_DIR, "sample_data", "patient_chest_pneumonia.png"),
        "title": "Case #404: Lobar Pneumonia Infiltrate",
        "modality": "Chest X-Ray (Posteroanterior)",
        "patient": {"name": "Sarah Jenkins", "age": 67, "gender": "Female", "patient_id": "CXR-2026-442"},
        "expected": "Pathological (Pneumonia)"
    }
}

# Temporary session store for current loaded patient
current_session = {
    "patient_info": {},
    "prep_results": None,
    "features_result": None,
    "prediction_result": None
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Returns training status, dataset summary, and current model evaluations."""
    return jsonify({
        "is_trained": model_manager.is_trained,
        "best_model": model_manager.best_model_name,
        "available_models": list(model_manager.models.keys()) if model_manager.models else [],
        "evaluations": model_manager.evaluation_results
    })

@app.route('/api/samples', methods=['GET'])
def list_samples():
    """Returns list of pre-loaded clinical sample cases."""
    summary = []
    for k, v in SAMPLE_CASES.items():
        summary.append({
            "id": v["id"],
            "title": v["title"],
            "modality": v["modality"],
            "patient": v["patient"],
            "expected": v["expected"]
        })
    return jsonify({"samples": summary})

@app.route('/api/load-sample', methods=['POST'])
def load_sample():
    """Loads a sample patient image, processes it, and extracts features."""
    data = request.get_json() or {}
    sample_id = data.get("sample_id", "brain_tumor")
    
    if sample_id not in SAMPLE_CASES:
        return jsonify({"error": "Sample case not found"}), 404

    sample = SAMPLE_CASES[sample_id]
    img = cv2.imread(sample["file"])
    if img is None:
        return jsonify({"error": "Failed to read sample image file"}), 500

    # 1. Preprocess image
    prep_res = preprocessor.process(img)

    # 2. Extract features
    feat_res = extractor.extract_all(
        prep_res["arrays"]["gray"],
        prep_res["arrays"]["segmented_mask"]
    )

    # 3. Store in current session
    current_session["patient_info"] = sample["patient"]
    current_session["patient_info"]["modality"] = sample["modality"]
    current_session["prep_results"] = prep_res
    current_session["features_result"] = feat_res
    current_session["prediction_result"] = None

    return jsonify({
        "success": True,
        "patient": current_session["patient_info"],
        "previews": prep_res["previews"],
        "stats": prep_res["stats"],
        "features": feat_res["features_dict"]
    })

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Receives user-uploaded medical image and processes it through the pipeline."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided in request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # Read and process image
    img = cv2.imread(save_path)
    if img is None:
        return jsonify({"error": "Failed to decode uploaded image. Please ensure valid PNG/JPG."}), 400

    prep_res = preprocessor.process(img)
    feat_res = extractor.extract_all(
        prep_res["arrays"]["gray"],
        prep_res["arrays"]["segmented_mask"]
    )

    # Parse patient metadata from form
    patient_name = request.form.get("patient_name", "Anonymous Patient")
    patient_id = request.form.get("patient_id", f"PT-{os.urandom(2).hex().upper()}")
    age = request.form.get("age", "45")
    gender = request.form.get("gender", "Unspecified")
    modality = request.form.get("modality", "Radiological Scan")

    current_session["patient_info"] = {
        "name": patient_name,
        "patient_id": patient_id,
        "age": age,
        "gender": gender,
        "modality": modality
    }
    current_session["prep_results"] = prep_res
    current_session["features_result"] = feat_res
    current_session["prediction_result"] = None

    return jsonify({
        "success": True,
        "patient": current_session["patient_info"],
        "previews": prep_res["previews"],
        "stats": prep_res["stats"],
        "features": feat_res["features_dict"]
    })

@app.route('/api/train', methods=['POST'])
def trigger_training():
    """Triggers ML model training and full evaluation on the dataset."""
    try:
        data = request.get_json() or {}
        test_size = float(data.get("test_size", 0.25))
        eval_results = model_manager.train_and_evaluate_all(test_size=test_size)
        return jsonify({
            "success": True,
            "message": "All ML models trained and evaluated successfully!",
            "results": eval_results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def run_prediction():
    """Classifies the currently loaded patient image using the trained ML model."""
    if current_session["features_result"] is None:
        return jsonify({"error": "No patient image loaded or features extracted. Please load or upload an image first."}), 400

    data = request.get_json() or {}
    selected_model = data.get("model_name", None)

    try:
        feature_vector = current_session["features_result"]["features_vector"]
        pred_res = model_manager.predict_patient_case(feature_vector, selected_model_name=selected_model)
        current_session["prediction_result"] = pred_res

        return jsonify({
            "success": True,
            "patient": current_session["patient_info"],
            "prediction": pred_res
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/download-report', methods=['GET'])
def download_report():
    """Generates and downloads the clinical PDF diagnosis report."""
    if current_session["prep_results"] is None or current_session["prediction_result"] is None:
        return jsonify({"error": "Please process an image and run prediction before generating a report."}), 400

    try:
        pdf_bytes = create_patient_pdf_report(
            patient_info=current_session["patient_info"],
            prep_arrays=current_session["prep_results"]["arrays"],
            features_dict=current_session["features_result"]["features_dict"],
            pred_result=current_session["prediction_result"]
        )

        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"Medical_Diagnosis_Report_{current_session['patient_info'].get('patient_id', 'case')}.pdf"
        )
    except Exception as e:
        return jsonify({"error": f"Report generation failed: {str(e)}"}), 500

if __name__ == '__main__':
    print("================================================================")
    print(" MedAI CadX: Patient Image Diagnostic & ML Pipeline Platform")
    print(" Server launching on http://127.0.0.1:5000")
    print("================================================================")
    app.run(host='0.0.0.0', port=5000, debug=False)

