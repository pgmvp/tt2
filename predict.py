import pandas as pd
import joblib
import argparse
import numpy as np

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def make_predictions(model, data_path, output_path):
    # Load the test data
    test_data = pd.read_csv(data_path)
    
    # Make predictions
    predictions = model.predict(test_data)
    
    # Save predictions to file
    np.savetxt(output_path, predictions, delimiter=",")
    print(f"Predictions saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make predictions with a trained XGBoost model.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the test data (CSV file).')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save predictions.')
    
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model_path)
    
    # Make predictions
    make_predictions(model, args.data_path, args.output_path)
