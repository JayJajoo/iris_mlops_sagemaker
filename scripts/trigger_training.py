"""
Trigger SageMaker training job from CI/CD
"""
import os
import json
import time
import boto3
import argparse
from datetime import datetime
from sagemaker.sklearn import SKLearn
from sagemaker import get_execution_role, Session

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket-name', type=str, required=True, help='S3 bucket for artifacts')
    parser.add_argument('--role-arn', type=str, required=True, help='SageMaker execution role ARN')
    parser.add_argument('--instance-type', type=str, default='ml.m5.large')
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=5)
    parser.add_argument('--wait', action='store_true', help='Wait for training to complete')
    return parser.parse_args()

def create_training_job(args):
    """Create and start SageMaker training job"""
    
    # Initialize SageMaker session
    session = Session()
    
    # Timestamp for unique job name
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    job_name = f'iris-training-{timestamp}'
    
    print(f"Creating training job: {job_name}")
    
    # Define estimator
    estimator = SKLearn(
        entry_point='train.py',
        source_dir='src',  # Directory containing train.py
        role=args.role_arn,
        instance_type=args.instance_type,
        instance_count=1,
        framework_version='1.2-1',  # sklearn version
        py_version='py3',
        output_path=f's3://{args.bucket_name}/model-artifacts',
        code_location=f's3://{args.bucket_name}/code',
        base_job_name='iris-training',
        hyperparameters={
            'n-estimators': args.n_estimators,
            'max-depth': args.max_depth,
            'random-state': 42,
            'test-size': 0.2
        },
        sagemaker_session=session,
        enable_sagemaker_metrics=True,
        metric_definitions=[
            {'Name': 'accuracy', 'Regex': 'accuracy: ([0-9.]+)'},
        ]
    )
    
    print("Starting training job...")
    estimator.fit(wait=False, job_name=job_name)
    
    return job_name, estimator

def wait_for_training(job_name, session):
    """Wait for training job to complete and return status"""
    
    client = session.sagemaker_client
    
    print(f"Waiting for training job {job_name} to complete...")
    
    while True:
        response = client.describe_training_job(TrainingJobName=job_name)
        status = response['TrainingJobStatus']
        
        print(f"Status: {status}")
        
        if status in ['Completed', 'Failed', 'Stopped']:
            break
        
        time.sleep(30)
    
    if status == 'Completed':
        print("Training completed successfully!")
        model_artifacts = response['ModelArtifacts']['S3ModelArtifacts']
        print(f"Model artifacts: {model_artifacts}")
        return True, model_artifacts
    else:
        print(f"Training failed with status: {status}")
        if 'FailureReason' in response:
            print(f"Failure reason: {response['FailureReason']}")
        return False, None

def get_metrics_from_s3(model_artifacts_path, bucket_name):
    """Retrieve metrics from S3"""
    
    s3 = boto3.client('s3')
    
    # Extract key from S3 path
    # model_artifacts_path format: s3://bucket/path/to/model.tar.gz
    key_prefix = '/'.join(model_artifacts_path.split('/')[3:-1])  # Remove bucket and model.tar.gz
    metrics_key = f"{key_prefix}/output/metrics.json"
    
    try:
        print(f"Fetching metrics from s3://{bucket_name}/{metrics_key}")
        response = s3.get_object(Bucket=bucket_name, Key=metrics_key)
        metrics = json.loads(response['Body'].read().decode('utf-8'))
        print(f"Metrics retrieved: {json.dumps(metrics, indent=2)}")
        return metrics
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return None

def save_job_info(job_name, model_artifacts, metrics):
    """Save job information for downstream steps"""
    
    job_info = {
        'job_name': job_name,
        'model_artifacts': model_artifacts,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to file for CI/CD to read
    with open('training_job_info.json', 'w') as f:
        json.dump(job_info, f, indent=2)
    
    print("Job info saved to training_job_info.json")

def main():
    args = parse_args()
    
    # Create and start training job
    job_name, estimator = create_training_job(args)
    
    # Wait for completion if requested
    if args.wait:
        session = Session()
        success, model_artifacts = wait_for_training(job_name, session)
        
        if success:
            # Get metrics
            metrics = get_metrics_from_s3(model_artifacts, args.bucket_name)
            
            # Save job info
            save_job_info(job_name, model_artifacts, metrics)
            
            # Exit with success
            exit(0)
        else:
            exit(1)
    else:
        print(f"Training job {job_name} started. Not waiting for completion.")
        exit(0)

if __name__ == '__main__':
    main()