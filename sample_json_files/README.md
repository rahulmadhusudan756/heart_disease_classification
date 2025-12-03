# Sample JSON Files for Heart Disease Classification

This folder contains sample JSON files for testing the HeartGuard AI 2-phase assessment system.

## File Naming Convention
- `_neg` = Negative (healthy/no disease predicted)
- `_pos` = Positive (disease/risk predicted)

## Usage

### Phase 1 - Initial Screening
Start by uploading `nh_neg.json` or `nh_pos.json` for the initial heart health screening.

### Phase 2 - Detailed Analysis
Upload any file below. The system auto-detects compatible models.

| Abbrev | Disease | Negative | Positive |
|--------|---------|----------|----------|
| **nh** | Normal Heart | `nh_neg.json` | `nh_pos.json` |
| **ha** | Heart Attack | `ha_neg.json` | `ha_pos.json` |
| **hf** | Heart Failure | `hf_neg.json` | `hf_pos.json` |
| **ht** | Hypertension (10-yr CHD) | `ht_neg.json` | `ht_pos.json` |
| **cad** | Coronary Artery Disease | `cad_neg.json` | `cad_pos.json` |

## Required Fields per Model

### nh (Normal Heart) - Phase 1
`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`

### ha (Heart Attack)
`age, sex, chest_pain_type, trestbps, chol, fbs, restecg, thalach, exercise_angina, oldpeak, st_slope`

### hf (Heart Failure)
`age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope`

### ht (Hypertension)
`male, age, education, current_smoker, cigs_per_day, bp_meds, prevalent_stroke, prevalent_hyp, diabetes, tot_chol, sys_bp, dia_bp, bmi, heart_rate, glucose`

### cad (Coronary Artery Disease)
`exang, thal, restecg, slope, thalach, chol, oldpeak, trestbps, cp, ca`
