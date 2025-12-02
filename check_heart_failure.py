"""
Check the expected features for the heart failure model
"""
import joblib
import numpy as np

# Load the heart failure model
model_dict = joblib.load('final_models/best_heartfailure_model.pkl')
model = model_dict['model'] if isinstance(model_dict, dict) and 'model' in model_dict else model_dict

print("Heart Failure Model Structure:")
print("="*60)
print(f"Keys: {list(model_dict.keys())}")
print()

# Check selector
if 'selector' in model_dict:
    selector = model_dict['selector']
    print(f"Selector type: {type(selector).__name__}")
    if hasattr(selector, 'n_features_in_'):
        print(f"Selector expects: {selector.n_features_in_} features")
    if hasattr(selector, 'get_support'):
        print(f"Selected features mask: {selector.get_support()}")
        print(f"Number of selected features: {sum(selector.get_support())}")

# Check model
if 'model' in model_dict:
    model = model_dict['model']
    print(f"\nModel type: {type(model).__name__}")
    if hasattr(model, 'n_features_in_'):
        print(f"Model expects: {model.n_features_in_} features")
    if hasattr(model, 'feature_names_'):
        print(f"Feature names: {model.feature_names_}")

# Check metrics
print(f"\nModel Performance Metrics:")
if 'ACC' in model_dict:
    print(f"  Accuracy: {model_dict['ACC']:.4f}")
if 'F1' in model_dict:
    print(f"  F1 Score: {model_dict['F1']:.4f}")
if 'AUC' in model_dict:
    print(f"  AUC: {model_dict['AUC']:.4f}")

# Let's also check the columns if available
if 'columns' in model_dict:
    print(f"\nExpected columns: {model_dict['columns']}")
    print(f"Number of columns: {len(model_dict['columns'])}")
