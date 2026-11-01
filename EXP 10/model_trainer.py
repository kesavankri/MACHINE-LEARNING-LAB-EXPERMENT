import os
import glob
import cv2
import joblib
import warnings
import numpy as np

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from preprocessor import MedicalImagePreprocessor
from feature_extractor import MedicalFeatureExtractor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class MedicalModelManager:
    def __init__(self, models_dir=None):
        self.models_dir = models_dir or os.path.join(BASE_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.preprocessor = MedicalImagePreprocessor()
        self.extractor = MedicalFeatureExtractor()
        
        self.scaler = None
        self.models = {}
        self.best_model_name = "Random Forest"
        self.evaluation_results = {}
        self.is_trained = False
        
        # Check if saved models exist
        self.load_saved_models()

    def build_dataset_features(self, dataset_dir=None):
        """
        Loads all training images from dataset/normal and dataset/pathological,
        runs preprocessing, extracts features, and builds X, y matrices.
        """
        if dataset_dir is None:
            dataset_dir = os.path.join(BASE_DIR, "dataset")
        X = []
        y = []
        file_paths = []

        # 0: Normal, 1: Pathological
        categories = [("normal", 0), ("pathological", 1)]

        for cat_name, label in categories:
            cat_path = os.path.join(dataset_dir, cat_name)
            images = glob.glob(os.path.join(cat_path, "*.png"))
            for img_path in images:
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    prep_res = self.preprocessor.process(img)
                    feats = self.extractor.extract_all(
                        prep_res["arrays"]["gray"],
                        prep_res["arrays"]["segmented_mask"]
                    )
                    X.append(feats["features_vector"])
                    y.append(label)
                    file_paths.append(img_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        return X, y

    def train_and_evaluate_all(self, dataset_dir=None, test_size=0.25, random_state=42):
        """
        Extracts features, trains 5 ML algorithms, evaluates all performance metrics,
        and saves models to disk.
        """
        if dataset_dir is None:
            dataset_dir = os.path.join(BASE_DIR, "dataset")
        X, y = self.build_dataset_features(dataset_dir)
        if len(X) == 0:
            raise ValueError("No images found in dataset directory.")

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Algorithm Suite
        algorithms = {
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state),
            "Support Vector Machine (SVM)": SVC(kernel='rbf', C=1.5, probability=True, random_state=random_state),
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=random_state)
        }

        results = {}
        best_f1 = -1.0
        best_name = "Random Forest"

        for name, clf in algorithms.items():
            # Train
            clf.fit(X_train_scaled, y_train)
            self.models[name] = clf

            # Predict test set
            y_pred = clf.predict(X_test_scaled)
            y_proba = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else y_pred

            # Evaluation Metrics
            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0)) # Sensitivity
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            
            # Confusion Matrix: [[TN, FP], [FN, TP]]
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
            
            # Specificity = TN / (TN + FP)
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc_val = float(auc(fpr, tpr))

            # 5-fold Cross-validation
            cv_scores = cross_val_score(clf, self.scaler.transform(X), y, cv=5)
            cv_mean = float(np.mean(cv_scores))

            results[name] = {
                "accuracy": round(acc * 100, 2),
                "precision": round(prec * 100, 2),
                "recall_sensitivity": round(rec * 100, 2),
                "specificity": round(spec * 100, 2),
                "f1_score": round(f1 * 100, 2),
                "roc_auc": round(roc_auc_val, 4),
                "cv_mean": round(cv_mean * 100, 2),
                "confusion_matrix": {
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp)
                },
                "roc_points": {
                    "fpr": [round(float(x), 3) for x in fpr],
                    "tpr": [round(float(y), 3) for y in tpr]
                }
            }

            if f1 > best_f1:
                best_f1 = f1
                best_name = name

        self.best_model_name = best_name
        self.evaluation_results = {
            "dataset_summary": {
                "total_samples": len(X),
                "normal_samples": int(np.sum(y == 0)),
                "pathological_samples": int(np.sum(y == 1)),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "num_features": X.shape[1]
            },
            "best_model": best_name,
            "models": results
        }
        self.is_trained = True

        # Save artifacts
        self.save_models()
        return self.evaluation_results

    def save_models(self):
        joblib.dump(self.scaler, os.path.join(self.models_dir, "scaler.pkl"))
        joblib.dump(self.models, os.path.join(self.models_dir, "models.pkl"))
        joblib.dump(self.evaluation_results, os.path.join(self.models_dir, "evaluation.pkl"))

    def load_saved_models(self):
        scaler_file = os.path.join(self.models_dir, "scaler.pkl")
        models_file = os.path.join(self.models_dir, "models.pkl")
        eval_file = os.path.join(self.models_dir, "evaluation.pkl")

        if os.path.exists(scaler_file) and os.path.exists(models_file):
            try:
                self.scaler = joblib.load(scaler_file)
                self.models = joblib.load(models_file)
                if os.path.exists(eval_file):
                    self.evaluation_results = joblib.load(eval_file)
                    self.best_model_name = self.evaluation_results.get("best_model", "Random Forest")
                self.is_trained = True
                return True
            except Exception as e:
                print(f"Error loading saved models: {e}")
                return False
        return False

    def predict_patient_case(self, feature_vector, selected_model_name=None):
        """
        Runs inference on single patient feature vector.
        Returns diagnosis, confidence, probabilities, and clinical risk category.
        """
        if not self.is_trained or not self.models:
            raise RuntimeError("Models are not yet trained. Please train the ML pipeline first.")

        model_name = selected_model_name if (selected_model_name and selected_model_name in self.models) else self.best_model_name
        model = self.models[model_name]

        # Reshape and scale
        X_case = np.array(feature_vector).reshape(1, -1)
        X_case_scaled = self.scaler.transform(X_case)

        # Predict
        pred_label = int(model.predict(X_case_scaled)[0])
        
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_case_scaled)[0]
            prob_normal = round(float(proba[0]) * 100, 2)
            prob_pathology = round(float(proba[1]) * 100, 2)
        else:
            prob_normal = 0.0 if pred_label == 1 else 100.0
            prob_pathology = 100.0 if pred_label == 1 else 0.0

        if pred_label == 1:
            diagnosis = "PATHOLOGICAL / LESION DETECTED"
            confidence = prob_pathology
            if confidence >= 85:
                risk_level = "High"
                risk_color = "#dc2626"
            else:
                risk_level = "Moderate"
                risk_color = "#f59e0b"
            clinical_recommendation = "Immediate radiological correlation and clinical biopsy/follow-up recommended."
        else:
            diagnosis = "NORMAL / NO SIGNIFICANT PATHOLOGY DETECTED"
            confidence = prob_normal
            risk_level = "Low"
            risk_color = "#10b981"
            clinical_recommendation = "Anatomical structures appear within normal physiological limits. Routine observation."

        return {
            "model_used": model_name,
            "prediction_label": pred_label,
            "diagnosis": diagnosis,
            "confidence": round(confidence, 1),
            "prob_normal": prob_normal,
            "prob_pathology": prob_pathology,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "clinical_recommendation": clinical_recommendation
        }
