"""
Test SageMaker endpoint with sample predictions
"""
import json
import boto3
import argparse
from sklearn.datasets import load_iris

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint-name', type=str, default='iris-endpoint', help='Endpoint name')
    parser.add_argument('--region', type=str, default='us-east-1', help='AWS region')
    return parser.parse_args()

def get_test_data():
    """Get sample data from Iris dataset"""
    iris = load_iris()
    
    # Get one sample from each class
    test_samples = [
        {
            'features': iris.data[0].tolist(),  # Setosa
            'expected': 'setosa'
        },
        {
            'features': iris.data[50].tolist(),  # Versicolor
            'expected': 'versicolor'
        },
        {
            'features': iris.data[100].tolist(),  # Virginica
            'expected': 'virginica'
        }
    ]
    
    return test_samples

def invoke_endpoint(endpoint_name, payload, region):
    """Invoke SageMaker endpoint"""
    
    runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    
    result = json.loads(response['Body'].read().decode())
    return result

def test_single_prediction(endpoint_name, region):
    """Test single prediction"""
    
    print("\n" + "="*50)
    print("Testing Single Prediction")
    print("="*50)
    
    test_samples = get_test_data()
    
    passed = 0
    failed = 0
    
    for i, sample in enumerate(test_samples):
        print(f"\nTest {i+1}:")
        print(f"Features: {sample['features']}")
        print(f"Expected: {sample['expected']}")
        
        # Create payload
        payload = {'features': sample['features']}
        
        try:
            result = invoke_endpoint(endpoint_name, payload, region)
            prediction = result['prediction']
            print(f"Predicted: {prediction}")
            print(f"Probabilities: {json.dumps(result['probabilities'], indent=2)}")
            
            if prediction == sample['expected']:
                print("✓ PASSED")
                passed += 1
            else:
                print("✗ FAILED")
                failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    return passed, failed

def test_batch_prediction(endpoint_name, region):
    """Test batch prediction"""
    
    print("\n" + "="*50)
    print("Testing Batch Prediction")
    print("="*50)
    
    test_samples = get_test_data()
    
    # Create batch payload
    instances = [sample['features'] for sample in test_samples]
    payload = {'instances': instances}
    
    print(f"\nBatch size: {len(instances)}")
    
    try:
        result = invoke_endpoint(endpoint_name, payload, region)
        predictions = result['predictions']
        
        print(f"\nReceived {len(predictions)} predictions")
        
        passed = 0
        failed = 0
        
        for i, (pred, sample) in enumerate(zip(predictions, test_samples)):
            print(f"\nSample {i+1}:")
            print(f"Expected: {sample['expected']}")
            print(f"Predicted: {pred['prediction']}")
            
            if pred['prediction'] == sample['expected']:
                print("✓ PASSED")
                passed += 1
            else:
                print("✗ FAILED")
                failed += 1
        
        return passed, failed
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return 0, len(test_samples)

def test_error_handling(endpoint_name, region):
    """Test error handling with invalid input"""
    
    print("\n" + "="*50)
    print("Testing Error Handling")
    print("="*50)
    
    invalid_payloads = [
        {'invalid_key': [1, 2, 3, 4]},  # Wrong key
        {'features': [1, 2, 3]},  # Wrong feature count
        {'features': 'not a list'},  # Wrong type
    ]
    
    for i, payload in enumerate(invalid_payloads):
        print(f"\nTest {i+1}: {payload}")
        try:
            result = invoke_endpoint(endpoint_name, payload, region)
            print("✗ FAILED: Expected error but got result")
        except Exception as e:
            print(f"✓ PASSED: Got expected error - {str(e)[:100]}")

def main():
    args = parse_args()
    
    print(f"Testing endpoint: {args.endpoint_name}")
    print(f"Region: {args.region}")
    
    # Test single predictions
    single_passed, single_failed = test_single_prediction(args.endpoint_name, args.region)
    
    # Test batch predictions
    batch_passed, batch_failed = test_batch_prediction(args.endpoint_name, args.region)
    
    # Test error handling
    test_error_handling(args.endpoint_name, args.region)
    
    # Summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    print(f"Single Predictions: {single_passed} passed, {single_failed} failed")
    print(f"Batch Predictions: {batch_passed} passed, {batch_failed} failed")
    
    total_passed = single_passed + batch_passed
    total_failed = single_failed + batch_failed
    
    print(f"\nTotal: {total_passed} passed, {total_failed} failed")
    
    if total_failed > 0:
        print("\n✗ Some tests failed")
        exit(1)
    else:
        print("\n✓ All tests passed!")
        exit(0)

if __name__ == '__main__':
    main()