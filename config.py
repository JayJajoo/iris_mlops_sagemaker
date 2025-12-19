"""
Centralized configuration for SageMaker pipeline
"""
import os

class Config:
    """Pipeline configuration"""
    
    # AWS Configuration
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    S3_BUCKET = os.getenv('S3_BUCKET', 'your-sagemaker-bucket')
    SAGEMAKER_ROLE_ARN = os.getenv('SAGEMAKER_ROLE_ARN')
    
    # Training Configuration
    TRAINING_INSTANCE_TYPE = os.getenv('TRAINING_INSTANCE_TYPE', 'ml.m5.large')
    TRAINING_INSTANCE_COUNT = 1
    
    # Model Hyperparameters
    N_ESTIMATORS = int(os.getenv('N_ESTIMATORS', 100))
    MAX_DEPTH = int(os.getenv('MAX_DEPTH', 5))
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # Deployment Configuration
    ENDPOINT_NAME = os.getenv('ENDPOINT_NAME', 'iris-endpoint')
    ENDPOINT_INSTANCE_TYPE = os.getenv('ENDPOINT_INSTANCE_TYPE', 'ml.t2.medium')
    ENDPOINT_INSTANCE_COUNT = 1
    
    # Model Quality Gate
    ACCURACY_THRESHOLD = float(os.getenv('ACCURACY_THRESHOLD', 0.85))
    
    # Framework Configuration
    FRAMEWORK_VERSION = '1.2-1'
    PYTHON_VERSION = 'py3'
    
    # S3 Paths
    @property
    def s3_model_artifacts_path(self):
        return f's3://{self.S3_BUCKET}/model-artifacts'
    
    @property
    def s3_code_path(self):
        return f's3://{self.S3_BUCKET}/code'
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.SAGEMAKER_ROLE_ARN:
            raise ValueError("SAGEMAKER_ROLE_ARN environment variable is required")
        
        if not cls.S3_BUCKET or cls.S3_BUCKET == 'your-sagemaker-bucket':
            raise ValueError("S3_BUCKET environment variable is required")
        
        print("✓ Configuration validated")
    
    @classmethod
    def display(cls):
        """Display current configuration"""
        print("\nCurrent Configuration:")
        print(f"  AWS Region: {cls.AWS_REGION}")
        print(f"  S3 Bucket: {cls.S3_BUCKET}")
        print(f"  Training Instance: {cls.TRAINING_INSTANCE_TYPE}")
        print(f"  Endpoint Instance: {cls.ENDPOINT_INSTANCE_TYPE}")
        print(f"  Endpoint Name: {cls.ENDPOINT_NAME}")
        print(f"  Accuracy Threshold: {cls.ACCURACY_THRESHOLD}")
        print(f"  Model: RandomForest(n_estimators={cls.N_ESTIMATORS}, max_depth={cls.MAX_DEPTH})")
        print()

if __name__ == '__main__':
    # Test configuration
    try:
        Config.validate()
        Config.display()
    except ValueError as e:
        print(f"Configuration Error: {e}")