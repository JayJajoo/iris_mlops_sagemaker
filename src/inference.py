"""
Custom inference handler for SageMaker endpoint
Optional: Use if you need custom pre/post processing
"""
import os
import json
import joblib
import numpy as np

# SageMaker model directory
MODEL_PATH = '/opt/ml/model'

def model_fn(model_dir):
    """
    Load model from the model directory
    Called once when endpoint starts
    """
    model_path = os.path.join(model_dir, 'model.joblib')
    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    """
    Parse input data
    """
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        
        # Handle single prediction
        if 'features' in data:
            features = np.array(data['features']).reshape(1, -1)
        # Handle batch predictions
        elif 'instances' in data:
            features = np.array(data['instances'])
        else:
            raise ValueError("Invalid input format. Expected 'features' or 'instances' key")
        
        return features
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Make predictions
    """
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    return predictions, probabilities

def output_fn(prediction, response_content_type):
    """
    Format output
    """
    predictions, probabilities = prediction
    
    # Class names
    class_names = ['setosa', 'versicolor', 'virginica']
    
    # Format response
    if len(predictions) == 1:
        # Single prediction
        response = {
            'prediction': class_names[predictions[0]],
            'prediction_label': int(predictions[0]),
            'probabilities': {
                class_names[i]: float(probabilities[0][i])
                for i in range(len(class_names))
            }
        }
    else:
        # Batch predictions
        response = {
            'predictions': [
                {
                    'prediction': class_names[pred],
                    'prediction_label': int(pred),
                    'probabilities': {
                        class_names[i]: float(probabilities[idx][i])
                        for i in range(len(class_names))
                    }
                }
                for idx, pred in enumerate(predictions)
            ]
        }
    
    return json.dumps(response)