import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
import joblib
import argparse

def train_model(data_path, model_path, params):

    print("Loading training data")
    # Load the dataset
    df = pd.read_csv(data_path)
    
    # Split features and target
    X_train = df.drop(columns=['target'])
    y_train = df['target']
    
    # Initialize and train XGBoost model with specified parameters
    model = XGBRegressor(**params, objective='reg:squarederror', random_state=42)

    print("Training the model")
    model.fit(X_train, y_train)
    
    # Evaluate model on train set
    y_train_pred = model.predict(X_train)
    rmse = root_mean_squared_error(y_train, y_train_pred)
    print(f"Train RMSE: {rmse:.4f}")
    
    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an XGBoost regression model.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the training data (CSV file).')
    parser.add_argument('--model_path', type=str, required=True, help='Path to save the trained model.')
    
    # Default hyperparameters with the option to override
    parser.add_argument('--max_depth', type=int, default=9, help='Maximum depth of the trees.')
    parser.add_argument('--min_child_weight', type=int, default=7, help='Minimum sum of instance weight (hessian) needed in a child.')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate (step size).')
    parser.add_argument('--n_estimators', type=int, default=1000, help='Number of boosting rounds.')
    parser.add_argument('--subsample', type=float, default=0.8, help='Subsample ratio of the training data.')
    parser.add_argument('--colsample_bytree', type=float, default=0.8, help='Subsample ratio of columns when constructing each tree.')
    parser.add_argument('--reg_alpha', type=float, default=0.5, help='L1 regularization term on weights.')
    parser.add_argument('--reg_lambda', type=float, default=10, help='L2 regularization term on weights.')

    args = parser.parse_args()
    
    # Assemble parameters from arguments
    params = {
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'n_estimators': args.n_estimators,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'alpha': args.reg_alpha,
        'lambda': args.reg_lambda
    }
    
    train_model(args.data_path, args.model_path, params)
