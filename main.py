from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import os

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Classification API",
    description="API for predicting various heart conditions using machine learning models",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store loaded models and preprocessing components
models = {}
preprocessors = {}  # Store scalers, selectors, PCA, etc.

# Pydantic models for input validation
class CoronaryInput(BaseModel):
    exang: int = Field(..., description="Exercise induced angina (1 = yes; 0 = no)")
    thal: int = Field(..., description="Thalassemia (0-3)")
    restecg: int = Field(..., description="Resting electrocardiographic results (0-2)")
    slope: int = Field(..., description="Slope of peak exercise ST segment (0-2)")
    thalach: int = Field(..., description="Maximum heart rate achieved")
    chol: int = Field(..., description="Serum cholesterol in mg/dl")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    trestbps: int = Field(..., description="Resting blood pressure")
    cp: int = Field(..., description="Chest pain type (0-3)")
    ca: int = Field(..., description="Number of major vessels (0-3)")

class HeartAttackInput(BaseModel):
    age: int = Field(..., description="Age in years")
    sex: int = Field(..., description="Sex (1 = male; 0 = female)")
    chest_pain_type: int = Field(..., description="Chest pain type (0-3)")
    trestbps: int = Field(..., description="Resting blood pressure")
    chol: int = Field(..., description="Serum cholesterol in mg/dl")
    fbs: int = Field(..., description="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)")
    restecg: int = Field(..., description="Resting electrocardiographic results (0-2)")
    thalach: int = Field(..., description="Maximum heart rate achieved")
    exercise_angina: int = Field(..., description="Exercise induced angina (1 = yes; 0 = no)")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    st_slope: int = Field(..., description="Slope of peak exercise ST segment (0-2)")

class HeartFailureInput(BaseModel):
    age: int = Field(..., description="Age in years")
    sex: int = Field(..., description="Sex (1 = male; 0 = female)")
    chest_pain_type: int = Field(..., description="Chest pain type (0-3)")
    resting_bp: int = Field(..., description="Resting blood pressure")
    cholesterol: int = Field(..., description="Serum cholesterol in mg/dl")
    fasting_bs: int = Field(..., description="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)")
    resting_ecg: int = Field(..., description="Resting electrocardiographic results (0-2)")
    max_hr: int = Field(..., description="Maximum heart rate achieved")
    exercise_angina: int = Field(..., description="Exercise induced angina (1 = yes; 0 = no)")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    st_slope: int = Field(..., description="Slope of peak exercise ST segment (0-2)")

class HypertensionInput(BaseModel):
    male: int = Field(..., description="Sex (1 = male; 0 = female)")
    age: int = Field(..., description="Age in years")
    education: int = Field(..., description="Education level (1-4)")
    current_smoker: int = Field(..., description="Current smoker (1 = yes; 0 = no)")
    cigs_per_day: int = Field(..., description="Cigarettes per day")
    bp_meds: int = Field(..., description="On blood pressure medication (1 = yes; 0 = no)")
    prevalent_stroke: int = Field(..., description="Prevalent stroke (1 = yes; 0 = no)")
    prevalent_hyp: int = Field(..., description="Prevalent hypertension (1 = yes; 0 = no)")
    diabetes: int = Field(..., description="Diabetes (1 = yes; 0 = no)")
    tot_chol: int = Field(..., description="Total cholesterol")
    sys_bp: int = Field(..., description="Systolic blood pressure")
    dia_bp: int = Field(..., description="Diastolic blood pressure")
    bmi: float = Field(..., description="Body mass index")
    heart_rate: int = Field(..., description="Heart rate")
    glucose: int = Field(..., description="Glucose level")

class NormalHeartInput(BaseModel):
    age: int = Field(..., description="Age in years")
    sex: int = Field(..., description="Sex (1 = male; 0 = female)")
    cp: int = Field(..., description="Chest pain type (0-3)")
    trestbps: int = Field(..., description="Resting blood pressure")
    chol: int = Field(..., description="Serum cholesterol in mg/dl")
    fbs: int = Field(..., description="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)")
    restecg: int = Field(..., description="Resting electrocardiographic results (0-2)")
    thalach: int = Field(..., description="Maximum heart rate achieved")
    exang: int = Field(..., description="Exercise induced angina (1 = yes; 0 = no)")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    slope: int = Field(..., description="Slope of peak exercise ST segment (0-2)")
    ca: int = Field(..., description="Number of major vessels (0-3)")
    thal: int = Field(..., description="Thalassemia (0-3)")

class AllModelsInput(BaseModel):
    # Combined input for all models - using optional fields
    # Common fields
    age: Optional[int] = None
    sex: Optional[int] = None
    
    # Coronary specific
    exang: Optional[int] = None
    thal: Optional[int] = None
    restecg: Optional[int] = None
    slope: Optional[int] = None
    thalach: Optional[int] = None
    chol: Optional[int] = None
    oldpeak: Optional[float] = None
    trestbps: Optional[int] = None
    cp: Optional[int] = None
    ca: Optional[int] = None
    
    # Heart Attack specific
    chest_pain_type: Optional[int] = None
    fbs: Optional[int] = None
    exercise_angina: Optional[int] = None
    st_slope: Optional[int] = None
    
    # Heart Failure specific
    resting_bp: Optional[int] = None
    cholesterol: Optional[int] = None
    fasting_bs: Optional[int] = None
    resting_ecg: Optional[int] = None
    max_hr: Optional[int] = None
    
    # Hypertension specific
    male: Optional[int] = None
    education: Optional[int] = None
    current_smoker: Optional[int] = None
    cigs_per_day: Optional[int] = None
    bp_meds: Optional[int] = None
    prevalent_stroke: Optional[int] = None
    prevalent_hyp: Optional[int] = None
    diabetes: Optional[int] = None
    tot_chol: Optional[int] = None
    sys_bp: Optional[int] = None
    dia_bp: Optional[int] = None
    bmi: Optional[float] = None
    heart_rate: Optional[int] = None
    glucose: Optional[int] = None

class PredictionResponse(BaseModel):
    model_name: str
    prediction: int
    probability: Optional[float] = None
    risk_level: str

# Load models on startup
@app.on_event("startup")
async def load_models():
    """Load all models into memory on application startup"""
    model_files = {
        "coronary": "final_models/best_coronary_xgboost_dedicated.pkl",
        "heart_attack": "final_models/best_heartattack_model.pkl",
        "heart_failure": "final_models/best_heartfailure_model.pkl",
        "hypertension": "final_models/hypertension_model_compressed.pkl",
        "normal_heart": "final_models/normal_heart.pkl"
    }
    
    # Models that are saved as dictionaries (need to extract 'model' key)
    dict_models = {"heart_attack", "heart_failure", "hypertension"}
    
    for model_name, model_path in model_files.items():
        try:
            if not os.path.exists(model_path):
                print(f"✗ Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            loaded_obj = joblib.load(model_path)
            
            # Extract model from dictionary if needed
            if model_name in dict_models:
                if isinstance(loaded_obj, dict) and 'model' in loaded_obj:
                    models[model_name] = loaded_obj['model']
                    # Store preprocessing components
                    preprocessors[model_name] = {
                        'scaler': loaded_obj.get('scaler'),
                        'selector': loaded_obj.get('selector'),
                        'pca': loaded_obj.get('pca')
                    }
                    print(f"✓ Loaded {model_name} model successfully (from dict)")
                    if loaded_obj.get('selector'):
                        print(f"  - Selector: expects {loaded_obj.get('n_input_features', 'unknown')} input features")
                else:
                    models[model_name] = loaded_obj
                    preprocessors[model_name] = {}
                    print(f"✓ Loaded {model_name} model successfully (direct)")
            else:
                models[model_name] = loaded_obj
                preprocessors[model_name] = {}
                print(f"✓ Loaded {model_name} model successfully")
        except Exception as e:
            print(f"✗ Error loading {model_name} model from {model_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

def get_risk_level(prediction: int, probability: float = None) -> str:
    """Determine risk level based on prediction and probability"""
    if prediction == 0:
        return "Low Risk"
    elif probability is not None:
        if probability >= 0.8:
            return "High Risk"
        elif probability >= 0.6:
            return "Moderate Risk"
        else:
            return "Mild Risk"
    else:
        return "At Risk"

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API is running and models are loaded"""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "total_models": len(models)
    }

# Coronary prediction endpoint
@app.post("/predict/coronary", response_model=PredictionResponse)
async def predict_coronary(data: CoronaryInput):
    """Predict coronary artery disease"""
    try:
        # Prepare input data in correct order
        input_data = np.array([[
            data.exang, data.thal, data.restecg, data.slope, data.thalach,
            data.chol, data.oldpeak, data.trestbps, data.cp, data.ca
        ]])
        
        # Make prediction
        prediction = int(models["coronary"].predict(input_data)[0])
        
        # Get probability if available
        probability = None
        if hasattr(models["coronary"], "predict_proba"):
            proba = models["coronary"].predict_proba(input_data)[0]
            probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
        
        return PredictionResponse(
            model_name="Coronary Artery Disease",
            prediction=prediction,
            probability=probability,
            risk_level=get_risk_level(prediction, probability)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Heart Attack prediction endpoint
@app.post("/predict/heart-attack", response_model=PredictionResponse)
async def predict_heart_attack(data: HeartAttackInput):
    """Predict heart attack (myocardial infarction) risk"""
    try:
        # Prepare input data
        input_data = np.array([[
            data.age, data.sex, data.chest_pain_type, data.trestbps, data.chol,
            data.fbs, data.restecg, data.thalach, data.exercise_angina,
            data.oldpeak, data.st_slope
        ]])
        
        # Apply preprocessing pipeline if available
        processed = input_data
        if 'heart_attack' in preprocessors:
            if preprocessors['heart_attack'].get('scaler'):
                processed = preprocessors['heart_attack']['scaler'].transform(processed)
            if preprocessors['heart_attack'].get('selector'):
                processed = preprocessors['heart_attack']['selector'].transform(processed)
            if preprocessors['heart_attack'].get('pca'):
                processed = preprocessors['heart_attack']['pca'].transform(processed)
        
        # Make prediction
        prediction = int(models["heart_attack"].predict(processed)[0])
        
        # Get probability if available
        probability = None
        if hasattr(models["heart_attack"], "predict_proba"):
            proba = models["heart_attack"].predict_proba(processed)[0]
            probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
        
        return PredictionResponse(
            model_name="Heart Attack",
            prediction=prediction,
            probability=probability,
            risk_level=get_risk_level(prediction, probability)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Heart Failure prediction endpoint
@app.post("/predict/heart-failure", response_model=PredictionResponse)
async def predict_heart_failure(data: HeartFailureInput):
    """Predict heart failure risk"""
    try:
        # Prepare input data
        input_data = np.array([[
            data.age, data.sex, data.chest_pain_type, data.resting_bp,
            data.cholesterol, data.fasting_bs, data.resting_ecg, data.max_hr,
            data.exercise_angina, data.oldpeak, data.st_slope
        ]])
        
        # Apply preprocessing pipeline if available
        processed = input_data
        if 'heart_failure' in preprocessors:
            if preprocessors['heart_failure'].get('selector'):
                processed = preprocessors['heart_failure']['selector'].transform(processed)
        
        # Make prediction
        prediction = int(models["heart_failure"].predict(processed)[0])
        
        # Get probability if available
        probability = None
        if hasattr(models["heart_failure"], "predict_proba"):
            proba = models["heart_failure"].predict_proba(processed)[0]
            probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
        
        return PredictionResponse(
            model_name="Heart Failure",
            prediction=prediction,
            probability=probability,
            risk_level=get_risk_level(prediction, probability)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Hypertension prediction endpoint
@app.post("/predict/hypertension", response_model=PredictionResponse)
async def predict_hypertension(data: HypertensionInput):
    """Predict 10-year risk of coronary heart disease (hypertension)"""
    try:
        # Prepare input data
        input_data = np.array([[
            data.male, data.age, data.education, data.current_smoker,
            data.cigs_per_day, data.bp_meds, data.prevalent_stroke,
            data.prevalent_hyp, data.diabetes, data.tot_chol, data.sys_bp,
            data.dia_bp, data.bmi, data.heart_rate, data.glucose
        ]])
        
        # Make prediction
        prediction = int(models["hypertension"].predict(input_data)[0])
        
        # Get probability if available
        probability = None
        if hasattr(models["hypertension"], "predict_proba"):
            proba = models["hypertension"].predict_proba(input_data)[0]
            probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
        
        return PredictionResponse(
            model_name="Hypertension (10-Year CHD Risk)",
            prediction=prediction,
            probability=probability,
            risk_level=get_risk_level(prediction, probability)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Normal Heart prediction endpoint
@app.post("/predict/normal-heart", response_model=PredictionResponse)
async def predict_normal_heart(data: NormalHeartInput):
    """Predict normal heart condition"""
    try:
        # Prepare input data
        input_data = np.array([[
            data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs,
            data.restecg, data.thalach, data.exang, data.oldpeak, data.slope,
            data.ca, data.thal
        ]])
        
        # Make prediction
        prediction = int(models["normal_heart"].predict(input_data)[0])
        
        # Get probability if available
        probability = None
        if hasattr(models["normal_heart"], "predict_proba"):
            proba = models["normal_heart"].predict_proba(input_data)[0]
            probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
        
        return PredictionResponse(
            model_name="Normal Heart",
            prediction=prediction,
            probability=probability,
            risk_level=get_risk_level(prediction, probability)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Combined prediction endpoint
@app.post("/predict/all")
async def predict_all(data: AllModelsInput):
    """Run predictions through all 5 models and return combined results"""
    results = {}
    
    # Coronary prediction
    try:
        if all([data.exang is not None, data.thal is not None, data.restecg is not None,
                data.slope is not None, data.thalach is not None, data.chol is not None,
                data.oldpeak is not None, data.trestbps is not None, data.cp is not None,
                data.ca is not None]):
            coronary_data = CoronaryInput(
                exang=data.exang, thal=data.thal, restecg=data.restecg,
                slope=data.slope, thalach=data.thalach, chol=data.chol,
                oldpeak=data.oldpeak, trestbps=data.trestbps, cp=data.cp, ca=data.ca
            )
            results["coronary"] = await predict_coronary(coronary_data)
    except Exception as e:
        results["coronary"] = {"error": str(e)}
    
    # Heart Attack prediction
    try:
        if all([data.age is not None, data.sex is not None, data.chest_pain_type is not None,
                data.trestbps is not None, data.chol is not None, data.fbs is not None,
                data.restecg is not None, data.thalach is not None, data.exercise_angina is not None,
                data.oldpeak is not None, data.st_slope is not None]):
            heart_attack_data = HeartAttackInput(
                age=data.age, sex=data.sex, chest_pain_type=data.chest_pain_type,
                trestbps=data.trestbps, chol=data.chol, fbs=data.fbs,
                restecg=data.restecg, thalach=data.thalach, exercise_angina=data.exercise_angina,
                oldpeak=data.oldpeak, st_slope=data.st_slope
            )
            results["heart_attack"] = await predict_heart_attack(heart_attack_data)
    except Exception as e:
        results["heart_attack"] = {"error": str(e)}
    
    # Heart Failure prediction
    try:
        if all([data.age is not None, data.sex is not None, data.chest_pain_type is not None,
                data.resting_bp is not None, data.cholesterol is not None, data.fasting_bs is not None,
                data.resting_ecg is not None, data.max_hr is not None, data.exercise_angina is not None,
                data.oldpeak is not None, data.st_slope is not None]):
            heart_failure_data = HeartFailureInput(
                age=data.age, sex=data.sex, chest_pain_type=data.chest_pain_type,
                resting_bp=data.resting_bp, cholesterol=data.cholesterol, fasting_bs=data.fasting_bs,
                resting_ecg=data.resting_ecg, max_hr=data.max_hr, exercise_angina=data.exercise_angina,
                oldpeak=data.oldpeak, st_slope=data.st_slope
            )
            results["heart_failure"] = await predict_heart_failure(heart_failure_data)
    except Exception as e:
        results["heart_failure"] = {"error": str(e)}
    
    # Hypertension prediction
    try:
        if all([data.male is not None, data.age is not None, data.education is not None,
                data.current_smoker is not None, data.cigs_per_day is not None, data.bp_meds is not None,
                data.prevalent_stroke is not None, data.prevalent_hyp is not None, data.diabetes is not None,
                data.tot_chol is not None, data.sys_bp is not None, data.dia_bp is not None,
                data.bmi is not None, data.heart_rate is not None, data.glucose is not None]):
            hypertension_data = HypertensionInput(
                male=data.male, age=data.age, education=data.education,
                current_smoker=data.current_smoker, cigs_per_day=data.cigs_per_day,
                bp_meds=data.bp_meds, prevalent_stroke=data.prevalent_stroke,
                prevalent_hyp=data.prevalent_hyp, diabetes=data.diabetes,
                tot_chol=data.tot_chol, sys_bp=data.sys_bp, dia_bp=data.dia_bp,
                bmi=data.bmi, heart_rate=data.heart_rate, glucose=data.glucose
            )
            results["hypertension"] = await predict_hypertension(hypertension_data)
    except Exception as e:
        results["hypertension"] = {"error": str(e)}
    
    # Normal Heart prediction
    try:
        if all([data.age is not None, data.sex is not None, data.cp is not None,
                data.trestbps is not None, data.chol is not None, data.fbs is not None,
                data.restecg is not None, data.thalach is not None, data.exang is not None,
                data.oldpeak is not None, data.slope is not None, data.ca is not None,
                data.thal is not None]):
            normal_heart_data = NormalHeartInput(
                age=data.age, sex=data.sex, cp=data.cp, trestbps=data.trestbps,
                chol=data.chol, fbs=data.fbs, restecg=data.restecg, thalach=data.thalach,
                exang=data.exang, oldpeak=data.oldpeak, slope=data.slope, ca=data.ca, thal=data.thal
            )
            results["normal_heart"] = await predict_normal_heart(normal_heart_data)
    except Exception as e:
        results["normal_heart"] = {"error": str(e)}
    
    return {
        "message": "Predictions completed",
        "results": results,
        "models_run": len([k for k, v in results.items() if "error" not in v])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
