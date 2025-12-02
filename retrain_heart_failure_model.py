"""
Retrain Heart Failure Model with Correct Preprocessing
This script retrains the model to match the 11-feature input format expected by the API
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

print("="*70)
print("RETRAINING HEART FAILURE MODEL")
print("="*70)

# Set random seed
np.random.seed(42)

# Load data
df = pd.read_csv("Dataset/heart_failure/heart_failure.csv")
print(f"\n[1] Loaded dataset: {df.shape}")
print(f"    Columns: {list(df.columns)}")

# Check for categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if 'HeartDisease' in categorical_cols:
    categorical_cols.remove('HeartDisease')

print(f"\n[2] Categorical columns found: {categorical_cols}")

# Encode categorical variables manually to ensure correct feature count
df_encoded = df.copy()

# Map categorical to numeric (ordinal encoding, not one-hot)
if 'Sex' in df_encoded.columns:
    df_encoded['Sex'] = (df_encoded['Sex'] == 'M').astype(int)
    print("    - Encoded Sex: M=1, F=0")

if 'ChestPainType' in df_encoded.columns:
    chest_pain_map = {'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3}
    df_encoded['ChestPainType'] = df_encoded['ChestPainType'].map(chest_pain_map)
    print("    - Encoded ChestPainType: ATA=0, NAP=1, ASY=2, TA=3")

if 'RestingECG' in df_encoded.columns:
    ecg_map = {'Normal': 0, 'ST': 1, 'LVH': 2}
    df_encoded['RestingECG'] = df_encoded['RestingECG'].map(ecg_map)
    print("    - Encoded RestingECG: Normal=0, ST=1, LVH=2")

if 'ExerciseAngina' in df_encoded.columns:
    df_encoded['ExerciseAngina'] = (df_encoded['ExerciseAngina'] == 'Y').astype(int)
    print("    - Encoded ExerciseAngina: Y=1, N=0")

if 'ST_Slope' in df_encoded.columns:
    slope_map = {'Up': 0, 'Flat': 1, 'Down': 2}
    df_encoded['ST_Slope'] = df_encoded['ST_Slope'].map(slope_map)
    print("    - Encoded ST_Slope: Up=0, Flat=1, Down=2")

# Separate features and target
y = df_encoded["HeartDisease"].astype(int)
X = df_encoded.drop(columns=["HeartDisease"])

print(f"\n[3] Final feature matrix: {X.shape}")
print(f"    Feature columns: {list(X.columns)}")
print(f"    Target distribution: {y.value_counts().to_dict()}")

# Store column names for later use
feature_columns = list(X.columns)

# Convert to numpy array and impute
X = SimpleImputer(strategy="median").fit_transform(X)
print(f"\n[4] After imputation: {X.shape}")

# Define balancing techniques
methods = {
    "SMOTE": SMOTE(random_state=42),
    "ADASYN": ADASYN(random_state=42),
    "TOMEK": SMOTETomek(random_state=42)
}

# Define models
models = {
    "XGB": XGBClassifier(eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    "GradientBoost": GradientBoostingClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(class_weight="balanced", random_state=42)
}

best_model = None
best_selector = None
best_auc = -1
best_name = ""
best_metrics = {}
best_resampling = ""

print("\n[5] Training models with different balancing techniques...")
print("="*70)

for m, resamp in methods.items():
    Xr, yr = resamp.fit_resample(X, y)
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, stratify=yr, random_state=42)

    # Feature Selection
    fs = SelectKBest(mutual_info_classif, k=min(20, Xtr.shape[1]))
    Xtr2 = fs.fit_transform(Xtr, ytr)
    Xte2 = fs.transform(Xte)

    print(f"\n--- [{m}] --- (Selected {Xtr2.shape[1]} features)")
    
    for name, model in models.items():
        model.fit(Xtr2, ytr)

        prob = model.predict_proba(Xte2)[:, 1]
        pred = (prob > 0.5).astype(int)

        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred)
        auc = roc_auc_score(yte, prob)
        mAP = average_precision_score(yte, prob)

        print(f"  {name:<15} | ACC={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f} | mAP={mAP:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_resampling = m
            best_model = model
            best_selector = fs
            best_metrics = {"ACC": acc, "F1": f1, "AUC": auc, "mAP": mAP}

print("\n" + "="*70)
print(f"[6] BEST MODEL SELECTED")
print("="*70)
print(f"    Model: {best_name}")
print(f"    Resampling: {best_resampling}")
print(f"    Accuracy:  {best_metrics['ACC']:.4f}")
print(f"    F1 Score:  {best_metrics['F1']:.4f}")
print(f"    AUC:       {best_metrics['AUC']:.4f}")
print(f"    mAP:       {best_metrics['mAP']:.4f}")

# Save the model with all components
model_dict = {
    'model': best_model,
    'selector': best_selector,
    'feature_columns': feature_columns,
    'n_input_features': len(feature_columns),
    'ACC': best_metrics['ACC'],
    'F1': best_metrics['F1'],
    'AUC': best_metrics['AUC'],
    'mAP': best_metrics['mAP'],
    'resampling_method': best_resampling,
    'model_name': best_name
}

output_path = 'final_models/best_heartfailure_model.pkl'
joblib.dump(model_dict, output_path)
print(f"\n[7] Model saved to: {output_path}")
print(f"    Keys in saved dict: {list(model_dict.keys())}")
print(f"    Expected input features: {model_dict['n_input_features']}")

# Test the saved model
print("\n[8] Testing saved model...")
loaded_dict = joblib.load(output_path)
loaded_model = loaded_dict['model']
loaded_selector = loaded_dict['selector']

test_input = np.array([[60, 1, 1, 145, 270, 1, 0, 140, 1, 2.0, 2]])
print(f"    Test input shape: {test_input.shape}")

# Apply selector
test_selected = loaded_selector.transform(test_input)
print(f"    After selector: {test_selected.shape}")

# Predict
pred = loaded_model.predict(test_selected)
prob = loaded_model.predict_proba(test_selected)

print(f"    Prediction: {pred[0]}")
print(f"    Probability: Class 0={prob[0][0]:.4f}, Class 1={prob[0][1]:.4f}")

print("\n" + "="*70)
print("RETRAINING COMPLETE!")
print("="*70)
print(f"\nModel can now accept {len(feature_columns)} features:")
for i, col in enumerate(feature_columns):
    print(f"  {i}: {col}")
