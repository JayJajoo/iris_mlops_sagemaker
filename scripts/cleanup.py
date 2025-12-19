"""
Cleanup script to remove SageMaker resources and save costs
"""
import boto3
import argparse
from datetime import datetime, timedelta

def parse_args():
    parser = argparse.ArgumentParser(description='Cleanup SageMaker resources')
    parser.add_argument('--endpoint-name', type=str, help='Endpoint name to delete')
    parser.add_argument('--delete-all-endpoints', action='store_true', help='Delete all endpoints with iris prefix')
    parser.add_argument('--delete-old-models', action='store_true', help='Delete models older than N days')
    parser.add_argument('--days', type=int, default=7, help='Days threshold for old models')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without deleting')
    parser.add_argument('--region', type=str, default='us-east-1')
    return parser.parse_args()

def delete_endpoint(endpoint_name, dry_run=False):
    """Delete a SageMaker endpoint"""
    client = boto3.client('sagemaker')
    
    try:
        # Get endpoint details
        response = client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_config_name = response['EndpointConfigName']
        
        if dry_run:
            print(f"[DRY RUN] Would delete endpoint: {endpoint_name}")
            print(f"[DRY RUN] Would delete endpoint config: {endpoint_config_name}")
            return
        
        # Delete endpoint
        print(f"Deleting endpoint: {endpoint_name}")
        client.delete_endpoint(EndpointName=endpoint_name)
        
        # Delete endpoint config
        print(f"Deleting endpoint config: {endpoint_config_name}")
        client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        
        print(f"✓ Deleted endpoint {endpoint_name}")
        
    except client.exceptions.ClientError as e:
        if 'Could not find endpoint' in str(e):
            print(f"Endpoint {endpoint_name} not found")
        else:
            print(f"Error deleting endpoint: {e}")

def list_and_delete_iris_endpoints(dry_run=False):
    """List and delete all endpoints with 'iris' prefix"""
    client = boto3.client('sagemaker')
    
    response = client.list_endpoints(
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=100
    )
    
    iris_endpoints = [
        ep for ep in response['Endpoints']
        if 'iris' in ep['EndpointName'].lower()
    ]
    
    if not iris_endpoints:
        print("No iris endpoints found")
        return
    
    print(f"Found {len(iris_endpoints)} iris endpoints:")
    for ep in iris_endpoints:
        print(f"  - {ep['EndpointName']} (Status: {ep['EndpointStatus']})")
    
    if not dry_run:
        for ep in iris_endpoints:
            delete_endpoint(ep['EndpointName'], dry_run=False)
    else:
        print("\n[DRY RUN] Use --delete-all-endpoints without --dry-run to delete")

def delete_old_models(days_threshold, dry_run=False):
    """Delete models older than N days"""
    client = boto3.client('sagemaker')
    
    cutoff_date = datetime.now() - timedelta(days=days_threshold)
    
    response = client.list_models(
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=100
    )
    
    old_models = [
        model for model in response['Models']
        if model['CreationTime'].replace(tzinfo=None) < cutoff_date
        and 'iris' in model['ModelName'].lower()
    ]
    
    if not old_models:
        print(f"No iris models older than {days_threshold} days found")
        return
    
    print(f"Found {len(old_models)} old iris models:")
    for model in old_models:
        age = (datetime.now() - model['CreationTime'].replace(tzinfo=None)).days
        print(f"  - {model['ModelName']} ({age} days old)")
    
    if dry_run:
        print(f"\n[DRY RUN] Would delete {len(old_models)} models")
        return
    
    for model in old_models:
        try:
            print(f"Deleting model: {model['ModelName']}")
            client.delete_model(ModelName=model['ModelName'])
            print(f"✓ Deleted {model['ModelName']}")
        except Exception as e:
            print(f"Error deleting {model['ModelName']}: {e}")

def show_current_resources():
    """Display current SageMaker resources"""
    client = boto3.client('sagemaker')
    
    print("\n" + "="*60)
    print("Current SageMaker Resources")
    print("="*60)
    
    # Endpoints
    endpoints = client.list_endpoints(MaxResults=100)
    iris_endpoints = [ep for ep in endpoints['Endpoints'] if 'iris' in ep['EndpointName'].lower()]
    print(f"\nEndpoints: {len(iris_endpoints)}")
    for ep in iris_endpoints:
        print(f"  - {ep['EndpointName']} ({ep['EndpointStatus']})")
    
    # Models
    models = client.list_models(MaxResults=100)
    iris_models = [m for m in models['Models'] if 'iris' in m['ModelName'].lower()]
    print(f"\nModels: {len(iris_models)}")
    for model in iris_models[:5]:  # Show first 5
        age = (datetime.now() - model['CreationTime'].replace(tzinfo=None)).days
        print(f"  - {model['ModelName']} ({age} days old)")
    if len(iris_models) > 5:
        print(f"  ... and {len(iris_models) - 5} more")
    
    # Training jobs
    training_jobs = client.list_training_jobs(MaxResults=10, SortBy='CreationTime', SortOrder='Descending')
    iris_jobs = [j for j in training_jobs['TrainingJobSummaries'] if 'iris' in j['TrainingJobName'].lower()]
    print(f"\nRecent Training Jobs: {len(iris_jobs)}")
    for job in iris_jobs[:5]:
        age = (datetime.now() - job['CreationTime'].replace(tzinfo=None)).days
        print(f"  - {job['TrainingJobName']} ({job['TrainingJobStatus']}, {age} days ago)")
    
    print()

def estimate_cost_savings(dry_run_results):
    """Estimate monthly cost savings from cleanup"""
    # Rough estimates
    endpoint_cost_per_hour = 0.05  # ml.t2.medium
    hours_per_month = 730
    
    num_endpoints = dry_run_results.get('endpoints', 0)
    monthly_savings = num_endpoints * endpoint_cost_per_hour * hours_per_month
    
    print(f"\nEstimated Monthly Savings: ${monthly_savings:.2f}")
    print(f"  ({num_endpoints} endpoints × ${endpoint_cost_per_hour}/hour × {hours_per_month} hours)")

def main():
    args = parse_args()
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No resources will be deleted\n")
    
    # Show current resources
    show_current_resources()
    
    # Delete specific endpoint
    if args.endpoint_name:
        delete_endpoint(args.endpoint_name, dry_run=args.dry_run)
    
    # Delete all iris endpoints
    if args.delete_all_endpoints:
        list_and_delete_iris_endpoints(dry_run=args.dry_run)
    
    # Delete old models
    if args.delete_old_models:
        delete_old_models(args.days, dry_run=args.dry_run)
    
    # Show updated state
    if not args.dry_run and (args.endpoint_name or args.delete_all_endpoints or args.delete_old_models):
        print("\n" + "="*60)
        print("Updated Resources")
        print("="*60)
        show_current_resources()
    
    if args.dry_run:
        print("\n💡 Run without --dry-run to actually delete resources")

if __name__ == '__main__':
    main()