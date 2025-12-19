"""
Training script for SageMaker
Runs inside SageMaker training container
"""
import os
import json
import joblib
import argparse
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=5)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--test-size', type=float, default=0.2)
    
    # SageMaker specific paths
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))
    
    return parser.parse_args()

def load_data():
    """Load and prepare Iris dataset"""
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y, iris.target_names

def train_model(X_train, y_train, args):
    """Train RandomForest model"""
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, target_names):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(report['weighted avg']['precision']),
        'recall': float(report['weighted avg']['recall']),
        'f1_score': float(report['weighted avg']['f1-score'])
    }
    return metrics

def save_model(model, model_dir):
    """Save model using joblib"""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def save_metrics(metrics, output_dir):
    """Save metrics to JSON"""
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")

def main():
    args = parse_args()
    
    print("Loading data...")
    X, y, target_names = load_data()
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    
    print(f"Training model with n_estimators={args.n_estimators}, max_depth={args.max_depth}...")
    model = train_model(X_train, y_train, args)
    
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test, target_names)
    
    print("Saving model and metrics...")
    save_model(model, args.model_dir)
    save_metrics(metrics, args.output_data_dir)
    
    print("Training complete!")

if __name__ == '__main__':
    main()