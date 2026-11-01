@echo off
title MedAI CadX Medical Machine Learning Platform
color 0B
echo ======================================================================
echo          MedAI CadX: Patient Image Diagnostic & ML Pipeline
echo ======================================================================
echo.

:: Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.9+ from https://www.python.org/
    pause
    exit /b 1
)

:: Check if dataset exists, generate if needed
if not exist "sample_data\patient_brain_tumor.png" (
    echo [INFO] Generating medical synthetic dataset and sample cases...
    python generate_dataset.py
)

:: Check if models exist, train if needed
if not exist "models\models.pkl" (
    echo [INFO] Initializing and training Machine Learning models...
    python -c "from model_trainer import MedicalModelManager; m = MedicalModelManager(); m.train_and_evaluate_all()"
)

echo.
echo [INFO] Launching MedAI CadX Web Application...
echo [INFO] Access the application in your browser at: http://127.0.0.1:5000
echo.
python app.py

pause

