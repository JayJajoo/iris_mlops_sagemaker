"""
Deploy trained model to SageMaker endpoint
"""
import os
import json
import time
import boto3
import argparse
from datetime import datetime
from sagemaker.sklearn import SKLearnModel
from sagemaker import Session

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-data', type=str, required=True, help='S3 path to model.tar.gz')
    parser.add_argument('--role-arn', type=str, required=True, help='SageMaker execution role ARN')
    parser.add_argument('--endpoint-name', type=str, default='iris-endpoint', help='Endpoint name')
    parser.add_argument('--instance-type', type=str, default='ml.t2.medium')
    parser.add_argument('--instance-count', type=int, default=1)
    parser.add_argument('--update-endpoint', action='store_true', help='Update existing endpoint')
    return parser.parse_args()

def check_endpoint_exists(endpoint_name):
    """Check if endpoint already exists"""
    client = boto3.client('sagemaker')
    
    try:
        response = client.describe_endpoint(EndpointName=endpoint_name)
        return True, response['EndpointStatus']
    except client.exceptions.ClientError:
        return False, None

def create_model(args):
    """Create SageMaker Model"""
    
    session = Session()
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    model_name = f'iris-model-{timestamp}'
    
    print(f"Creating model: {model_name}")
    
    model = SKLearnModel(
        model_data=args.model_data,
        role=args.role_arn,
        entry_point='inference.py',
        source_dir='src',
        framework_version='1.2-1',
        py_version='py3',
        name=model_name,
        sagemaker_session=session
    )
    
    return model

def deploy_new_endpoint(model, args):
    """Deploy new endpoint"""
    
    print(f"Deploying new endpoint: {args.endpoint_name}")
    
    predictor = model.deploy(
        endpoint_name=args.endpoint_name,
        instance_type=args.instance_type,
        initial_instance_count=args.instance_count,
        wait=True
    )
    
    print(f"Endpoint {args.endpoint_name} deployed successfully!")
    return predictor

def update_existing_endpoint(model, args):
    """Update existing endpoint with new model"""
    
    print(f"Updating existing endpoint: {args.endpoint_name}")
    
    # Create endpoint config
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    endpoint_config_name = f'iris-endpoint-config-{timestamp}'
    
    client = boto3.client('sagemaker')
    
    # Create new endpoint config
    client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                'VariantName': 'AllTraffic',
                'ModelName': model.name,
                'InstanceType': args.instance_type,
                'InitialInstanceCount': args.instance_count,
                'InitialVariantWeight': 1
            }
        ]
    )
    
    # Update endpoint
    client.update_endpoint(
        EndpointName=args.endpoint_name,
        EndpointConfigName=endpoint_config_name
    )
    
    print("Waiting for endpoint update to complete...")
    
    # Wait for update to complete
    while True:
        response = client.describe_endpoint(EndpointName=args.endpoint_name)
        status = response['EndpointStatus']
        print(f"Status: {status}")
        
        if status == 'InService':
            print(f"Endpoint {args.endpoint_name} updated successfully!")
            break
        elif status == 'Failed':
            failure_reason = response.get('FailureReason', 'Unknown')
            raise Exception(f"Endpoint update failed: {failure_reason}")
        
        time.sleep(30)

def save_endpoint_info(endpoint_name, instance_type):
    """Save endpoint info for testing"""
    
    endpoint_info = {
        'endpoint_name': endpoint_name,
        'instance_type': instance_type,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('endpoint_info.json', 'w') as f:
        json.dump(endpoint_info, f, indent=2)
    
    print("Endpoint info saved to endpoint_info.json")

def main():
    args = parse_args()
    
    # Check if endpoint exists
    exists, status = check_endpoint_exists(args.endpoint_name)
    
    # Create model
    model = create_model(args)
    
    # Deploy or update
    if exists and args.update_endpoint:
        print(f"Endpoint {args.endpoint_name} exists with status: {status}")
        update_existing_endpoint(model, args)
    elif exists and not args.update_endpoint:
        print(f"Endpoint {args.endpoint_name} already exists. Use --update-endpoint to update it.")
        exit(1)
    else:
        deploy_new_endpoint(model, args)
    
    # Save endpoint info
    save_endpoint_info(args.endpoint_name, args.instance_type)
    
    print("Deployment complete!")

if __name__ == '__main__':
    main()