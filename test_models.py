import pickle
import traceback

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
        print(f"\nLoading {model_name} from {model_path}...")
        with open(model_path, 'rb') as f:
            loaded_obj = pickle.load(f)
        
        # Extract model from dictionary if needed
        if model_name in dict_models:
            if isinstance(loaded_obj, dict) and 'model' in loaded_obj:
                model = loaded_obj['model']
                print(f"✓ Successfully loaded {model_name} (extracted from dict)")
                print(f"  Model type: {type(model)}")
                print(f"  Dict keys: {list(loaded_obj.keys())}")
            else:
                model = loaded_obj
                print(f"✓ Successfully loaded {model_name} (direct model)")
                print(f"  Model type: {type(model)}")
        else:
            model = loaded_obj
            print(f"✓ Successfully loaded {model_name}")
            print(f"  Model type: {type(model)}")
    except Exception as e:
        print(f"✗ Failed to load {model_name}")
        print(f"  Error: {str(e)}")
        traceback.print_exc()
