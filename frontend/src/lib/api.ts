import { PredictionResponse, MODELS } from '@/types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function checkHealth(): Promise<{ status: string; models_loaded: string[]; total_models: number }> {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error('API is not healthy');
  }
  return response.json();
}

export async function predictSingle(modelId: string, data: Record<string, any>): Promise<PredictionResponse> {
  const model = MODELS.find(m => m.id === modelId);
  if (!model) {
    throw new Error(`Unknown model: ${modelId}`);
  }

  const response = await fetch(`${API_BASE_URL}${model.endpoint}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Prediction failed');
  }

  return response.json();
}

export async function predictAll(data: Record<string, any>): Promise<{
  message: string;
  results: Record<string, PredictionResponse | { error: string }>;
  models_run: number;
}> {
  const response = await fetch(`${API_BASE_URL}/predict/all`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Prediction failed');
  }

  return response.json();
}

export function detectCompatibleModels(data: Record<string, any>): string[] {
  const compatibleModels: string[] = [];
  
  for (const model of MODELS) {
    const hasAllFields = model.requiredFields.every(field => 
      data[field] !== undefined && data[field] !== null
    );
    if (hasAllFields) {
      compatibleModels.push(model.id);
    }
  }
  
  return compatibleModels;
}

export function getMissingFields(modelId: string, data: Record<string, any>): string[] {
  const model = MODELS.find(m => m.id === modelId);
  if (!model) return [];
  
  return model.requiredFields.filter(field => 
    data[field] === undefined || data[field] === null
  );
}
