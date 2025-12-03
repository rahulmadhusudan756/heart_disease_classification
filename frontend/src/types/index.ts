export interface PredictionResponse {
  model_name: string;
  prediction: number;
  probability: number | null;
  risk_level: string;
}

export interface ModelPrediction {
  modelId: string;
  result: PredictionResponse;
}

export interface Message {
  id: string;
  type: 'user' | 'bot' | 'system';
  content: string;
  timestamp: Date;
  data?: any;
  predictions?: ModelPrediction[];
  isLoading?: boolean;
}

export interface ModelInfo {
  id: string;
  name: string;
  endpoint: string;
  description: string;
  requiredFields: string[];
}

export const MODELS: ModelInfo[] = [
  {
    id: 'coronary',
    name: 'Coronary Artery Disease',
    endpoint: '/predict/coronary',
    description: 'Predicts coronary artery disease risk',
    requiredFields: ['exang', 'thal', 'restecg', 'slope', 'thalach', 'chol', 'oldpeak', 'trestbps', 'cp', 'ca']
  },
  {
    id: 'heart_attack',
    name: 'Heart Attack',
    endpoint: '/predict/heart-attack',
    description: 'Predicts heart attack (myocardial infarction) risk',
    requiredFields: ['age', 'sex', 'chest_pain_type', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exercise_angina', 'oldpeak', 'st_slope']
  },
  {
    id: 'heart_failure',
    name: 'Heart Failure',
    endpoint: '/predict/heart-failure',
    description: 'Predicts heart failure risk',
    requiredFields: ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 'fasting_bs', 'resting_ecg', 'max_hr', 'exercise_angina', 'oldpeak', 'st_slope']
  },
  {
    id: 'hypertension',
    name: 'Hypertension (10-Year CHD)',
    endpoint: '/predict/hypertension',
    description: 'Predicts 10-year coronary heart disease risk',
    requiredFields: ['male', 'age', 'education', 'current_smoker', 'cigs_per_day', 'bp_meds', 'prevalent_stroke', 'prevalent_hyp', 'diabetes', 'tot_chol', 'sys_bp', 'dia_bp', 'bmi', 'heart_rate', 'glucose']
  },
  {
    id: 'normal_heart',
    name: 'Normal Heart',
    endpoint: '/predict/normal-heart',
    description: 'Predicts normal heart condition',
    requiredFields: ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
  }
];
