#!/usr/bin/env python3
"""
AWS Infrastructure Setup for Yale Alumni 100K Records Processing
================================================================

This script sets up AWS infrastructure for processing 100K alumni records with:
1. EC2 instances for parallel processing 
2. RDS PostgreSQL for scalable database
3. S3 for data storage and caching
4. Lambda for serverless processing tasks
5. SQS for job queuing
6. CloudWatch for monitoring
"""

import boto3
import json
import time
from botocore.exceptions import ClientError

class YaleAlumniAWSSetup:
    def __init__(self):
        self.ec2 = boto3.client('ec2')
        self.rds = boto3.client('rds')
        self.s3 = boto3.client('s3')
        self.lambda_client = boto3.client('lambda')
        self.sqs = boto3.client('sqs')
        self.cloudwatch = boto3.client('cloudwatch')
        self.iam = boto3.client('iam')
        
        # Configuration
        self.config = {
            'region': 'us-west-2',
            'db_instance_class': 'db.r5.xlarge',  # 4 vCPU, 32 GB RAM
            'ec2_instance_type': 't3.2xlarge',    # 8 vCPU, 32 GB RAM
            'bucket_name': 'yale-alumni-processing-2025',
            'db_name': 'yale_alumni_prod',
            'db_username': 'yale_admin'
        }

    def create_s3_bucket(self):
        """Create S3 bucket for data storage"""
        try:
            self.s3.create_bucket(
                Bucket=self.config['bucket_name'],
                CreateBucketConfiguration={'LocationConstraint': self.config['region']}
            )
            print(f"✓ Created S3 bucket: {self.config['bucket_name']}")
            
            # Set up bucket versioning and lifecycle
            self.s3.put_bucket_versioning(
                Bucket=self.config['bucket_name'],
                VersioningConfiguration={'Status': 'Enabled'}
            )
            
            lifecycle_config = {
                'Rules': [{
                    'ID': 'yale-alumni-lifecycle',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': 'processed/'},
                    'Transitions': [{
                        'Days': 30,
                        'StorageClass': 'STANDARD_IA'
                    }, {
                        'Days': 90,
                        'StorageClass': 'GLACIER'
                    }]
                }]
            }
            
            self.s3.put_bucket_lifecycle_configuration(
                Bucket=self.config['bucket_name'],
                LifecycleConfiguration=lifecycle_config
            )
            
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
                print(f"✓ S3 bucket already exists: {self.config['bucket_name']}")
                return True
            else:
                print(f"✗ Failed to create S3 bucket: {e}")
                return False

    def create_vpc_and_security_groups(self):
        """Create VPC and security groups for secure networking"""
        try:
            # Create VPC
            vpc_response = self.ec2.create_vpc(CidrBlock='10.0.0.0/16')
            vpc_id = vpc_response['Vpc']['VpcId']
            
            # Tag VPC
            self.ec2.create_tags(
                Resources=[vpc_id],
                Tags=[{'Key': 'Name', 'Value': 'yale-alumni-vpc'}]
            )
            
            # Create internet gateway
            igw_response = self.ec2.create_internet_gateway()
            igw_id = igw_response['InternetGateway']['InternetGatewayId']
            
            # Attach internet gateway to VPC
            self.ec2.attach_internet_gateway(
                InternetGatewayId=igw_id,
                VpcId=vpc_id
            )
            
            # Create public subnet
            subnet_response = self.ec2.create_subnet(
                VpcId=vpc_id,
                CidrBlock='10.0.1.0/24',
                AvailabilityZone=f"{self.config['region']}a"
            )
            subnet_id = subnet_response['Subnet']['SubnetId']
            
            # Create private subnet for RDS
            private_subnet_response = self.ec2.create_subnet(
                VpcId=vpc_id,
                CidrBlock='10.0.2.0/24',
                AvailabilityZone=f"{self.config['region']}b"
            )
            private_subnet_id = private_subnet_response['Subnet']['SubnetId']
            
            # Create security group for EC2 instances
            ec2_sg_response = self.ec2.create_security_group(
                GroupName='yale-alumni-ec2-sg',
                Description='Security group for Yale Alumni processing EC2 instances',
                VpcId=vpc_id
            )
            ec2_sg_id = ec2_sg_response['GroupId']
            
            # Create security group for RDS
            rds_sg_response = self.ec2.create_security_group(
                GroupName='yale-alumni-rds-sg', 
                Description='Security group for Yale Alumni RDS instance',
                VpcId=vpc_id
            )
            rds_sg_id = rds_sg_response['GroupId']
            
            # Configure security group rules
            self.ec2.authorize_security_group_ingress(
                GroupId=ec2_sg_id,
                IpPermissions=[
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 22,
                        'ToPort': 22,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    },
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 80,
                        'ToPort': 80,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    }
                ]
            )
            
            self.ec2.authorize_security_group_ingress(
                GroupId=rds_sg_id,
                IpPermissions=[{
                    'IpProtocol': 'tcp',
                    'FromPort': 5432,
                    'ToPort': 5432,
                    'UserIdGroupPairs': [{'GroupId': ec2_sg_id}]
                }]
            )
            
            print(f"✓ Created VPC: {vpc_id}")
            print(f"✓ Created subnets: {subnet_id}, {private_subnet_id}")
            print(f"✓ Created security groups: {ec2_sg_id}, {rds_sg_id}")
            
            return {
                'vpc_id': vpc_id,
                'subnet_id': subnet_id,
                'private_subnet_id': private_subnet_id,
                'ec2_sg_id': ec2_sg_id,
                'rds_sg_id': rds_sg_id
            }
            
        except ClientError as e:
            print(f"✗ Failed to create VPC infrastructure: {e}")
            return None

    def create_rds_instance(self, network_config):
        """Create RDS PostgreSQL instance for scalable database"""
        try:
            # Create DB subnet group
            self.rds.create_db_subnet_group(
                DBSubnetGroupName='yale-alumni-subnet-group',
                DBSubnetGroupDescription='Subnet group for Yale Alumni RDS',
                SubnetIds=[network_config['subnet_id'], network_config['private_subnet_id']]
            )
            
            # Create RDS instance
            response = self.rds.create_db_instance(
                DBName=self.config['db_name'],
                DBInstanceIdentifier='yale-alumni-db',
                DBInstanceClass=self.config['db_instance_class'],
                Engine='postgres',
                EngineVersion='15.4',
                MasterUsername=self.config['db_username'],
                MasterUserPassword='YaleAlumni2025!',  # Change this in production!
                AllocatedStorage=100,
                MaxAllocatedStorage=1000,
                VpcSecurityGroupIds=[network_config['rds_sg_id']],
                DBSubnetGroupName='yale-alumni-subnet-group',
                BackupRetentionPeriod=7,
                MultiAZ=True,
                StorageType='gp2',
                StorageEncrypted=True,
                DeletionProtection=False,  # Set to True in production
                Tags=[
                    {'Key': 'Project', 'Value': 'YaleAlumni'},
                    {'Key': 'Environment', 'Value': 'Production'}
                ]
            )
            
            db_endpoint = response['DBInstance']['Endpoint']['Address']
            print(f"✓ Created RDS instance: {db_endpoint}")
            print("⏳ RDS instance is being created (this may take 10-15 minutes)")
            
            return db_endpoint
            
        except ClientError as e:
            print(f"✗ Failed to create RDS instance: {e}")
            return None

    def create_processing_instances(self, network_config, count=3):
        """Create EC2 instances for parallel processing"""
        try:
            # User data script for instance setup
            user_data = """#!/bin/bash
            yum update -y
            yum install -y python3 python3-pip postgresql-devel gcc git
            pip3 install pandas numpy scipy scikit-learn psycopg2-binary boto3
            pip3 install sentence-transformers faiss-cpu transformers spacy
            
            # Clone processing scripts (replace with your repository)
            cd /home/ec2-user
            mkdir yale-alumni-processing
            cd yale-alumni-processing
            
            # Create processing script
            cat > process_alumni.py << 'EOF'
import boto3
import psycopg2
import pandas as pd
import json
from datetime import datetime

def process_alumni_batch(s3_key, db_config):
    \"\"\"Process a batch of alumni records\"\"\"
    s3 = boto3.client('s3')
    
    # Download data from S3
    local_file = f'/tmp/{s3_key.split("/")[-1]}'
    s3.download_file('yale-alumni-processing-2025', s3_key, local_file)
    
    # Process data
    df = pd.read_csv(local_file)
    processed_records = []
    
    for _, row in df.iterrows():
        # Add your processing logic here
        processed_record = {
            'id': row.get('id'),
            'name': row.get('name'),
            'processed_at': datetime.utcnow().isoformat(),
            'status': 'processed'
        }
        processed_records.append(processed_record)
    
    # Save to database
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    
    for record in processed_records:
        cur.execute(
            "INSERT INTO processed_alumni (id, name, processed_at, status) VALUES (%s, %s, %s, %s)",
            (record['id'], record['name'], record['processed_at'], record['status'])
        )
    
    conn.commit()
    conn.close()
    
    return len(processed_records)

if __name__ == "__main__":
    # Configuration would be passed via environment variables
    db_config = {
        'host': os.environ.get('DB_HOST'),
        'database': 'yale_alumni_prod',
        'user': 'yale_admin',
        'password': os.environ.get('DB_PASSWORD')
    }
    
    # Start processing
    print("Alumni processing instance ready")
EOF
            
            chmod +x process_alumni.py
            """
            
            # Launch instances
            response = self.ec2.run_instances(
                ImageId='ami-0c02fb55956c7d316',  # Amazon Linux 2 AMI
                MinCount=count,
                MaxCount=count,
                InstanceType=self.config['ec2_instance_type'],
                KeyName='yale-alumni-key',  # Create this key pair first!
                SecurityGroupIds=[network_config['ec2_sg_id']],
                SubnetId=network_config['subnet_id'],
                UserData=user_data,
                TagSpecifications=[{
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': f'yale-alumni-processor-{i+1}'},
                        {'Key': 'Project', 'Value': 'YaleAlumni'}
                    ]
                } for i in range(count)],
                IamInstanceProfile={'Name': 'yale-alumni-instance-profile'}
            )
            
            instance_ids = [instance['InstanceId'] for instance in response['Instances']]
            print(f"✓ Created {count} processing instances: {instance_ids}")
            
            return instance_ids
            
        except ClientError as e:
            print(f"✗ Failed to create EC2 instances: {e}")
            return None

    def create_sqs_queue(self):
        """Create SQS queue for job processing"""
        try:
            response = self.sqs.create_queue(
                QueueName='yale-alumni-processing-queue',
                Attributes={
                    'MessageRetentionPeriod': '1209600',  # 14 days
                    'VisibilityTimeoutSeconds': '300',    # 5 minutes
                    'ReceiveMessageWaitTimeSeconds': '20' # Long polling
                }
            )
            
            queue_url = response['QueueUrl']
            print(f"✓ Created SQS queue: {queue_url}")
            
            return queue_url
            
        except ClientError as e:
            print(f"✗ Failed to create SQS queue: {e}")
            return None

    def create_lambda_function(self):
        """Create Lambda function for orchestrating processing"""
        lambda_code = '''
import json
import boto3

def lambda_handler(event, context):
    """
    Lambda function to orchestrate alumni data processing
    Triggered by S3 events when new data is uploaded
    """
    
    s3 = boto3.client('s3')
    sqs = boto3.client('sqs')
    
    # Process S3 event
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        
        # Send processing job to SQS
        message = {
            'bucket': bucket,
            'key': key,
            'timestamp': record['eventTime'],
            'action': 'process_alumni_batch'
        }
        
        sqs.send_message(
            QueueUrl='https://sqs.us-west-2.amazonaws.com/123456789012/yale-alumni-processing-queue',
            MessageBody=json.dumps(message)
        )
    
    return {
        'statusCode': 200,
        'body': json.dumps(f'Processed {len(event["Records"])} records')
    }
'''
        
        try:
            # Create deployment package
            import zipfile
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
                with zipfile.ZipFile(tmp.name, 'w') as zf:
                    zf.writestr('lambda_function.py', lambda_code)
                
                # Create Lambda function
                with open(tmp.name, 'rb') as f:
                    self.lambda_client.create_function(
                        FunctionName='yale-alumni-orchestrator',
                        Runtime='python3.9',
                        Role='arn:aws:iam::123456789012:role/yale-alumni-lambda-role',  # Create this role!
                        Handler='lambda_function.lambda_handler',
                        Code={'ZipFile': f.read()},
                        Description='Orchestrates Yale Alumni data processing',
                        Timeout=300,
                        MemorySize=512,
                        Environment={
                            'Variables': {
                                'SQS_QUEUE_URL': 'yale-alumni-processing-queue',
                                'S3_BUCKET': self.config['bucket_name']
                            }
                        }
                    )
                
                os.unlink(tmp.name)
                
            print("✓ Created Lambda orchestrator function")
            return True
            
        except ClientError as e:
            print(f"✗ Failed to create Lambda function: {e}")
            return False

    def setup_monitoring(self):
        """Set up CloudWatch monitoring and alarms"""
        try:
            # Create CloudWatch dashboard
            dashboard_body = {
                "widgets": [
                    {
                        "type": "metric",
                        "properties": {
                            "metrics": [
                                ["AWS/EC2", "CPUUtilization"],
                                ["AWS/RDS", "CPUUtilization"],
                                ["AWS/SQS", "NumberOfMessagesReceived"]
                            ],
                            "period": 300,
                            "stat": "Average",
                            "region": self.config['region'],
                            "title": "Yale Alumni Processing Metrics"
                        }
                    }
                ]
            }
            
            self.cloudwatch.put_dashboard(
                DashboardName='YaleAlumniProcessing',
                DashboardBody=json.dumps(dashboard_body)
            )
            
            print("✓ Created CloudWatch dashboard")
            return True
            
        except ClientError as e:
            print(f"✗ Failed to setup monitoring: {e}")
            return False

    def generate_deployment_summary(self):
        """Generate deployment summary and next steps"""
        summary = f"""
🎉 AWS Infrastructure Setup Complete!
=====================================

Created Resources:
- S3 Bucket: {self.config['bucket_name']}
- RDS Instance: {self.config['db_instance_class']} PostgreSQL
- EC2 Instances: 3x {self.config['ec2_instance_type']} 
- SQS Queue: yale-alumni-processing-queue
- Lambda Function: yale-alumni-orchestrator
- CloudWatch Dashboard: YaleAlumniProcessing

Next Steps:
1. Upload 100K records to S3 bucket in batches
2. Configure database connection strings
3. Set up IAM roles and permissions
4. Test processing pipeline with sample data
5. Monitor via CloudWatch dashboard

Cost Estimation (Monthly):
- RDS: ~$400-500
- EC2: ~$600-800 (3 instances)
- S3: ~$50-100 
- Lambda/SQS: ~$10-20
- Total: ~$1,060-1,420/month

Security Notes:
- Change default RDS password
- Configure proper IAM roles
- Set up VPC endpoints for private communication
- Enable CloudTrail for audit logging
"""
        print(summary)
        
        # Save to file
        with open('aws_deployment_summary.txt', 'w') as f:
            f.write(summary)

def main():
    """Main deployment function"""
    print("🚀 Starting AWS Infrastructure Setup for Yale Alumni 100K Processing")
    print("=" * 70)
    
    aws_setup = YaleAlumniAWSSetup()
    
    # Step 1: Create S3 bucket
    print("\n1. Setting up S3 storage...")
    if not aws_setup.create_s3_bucket():
        return False
    
    # Step 2: Create networking
    print("\n2. Setting up VPC and networking...")
    network_config = aws_setup.create_vpc_and_security_groups()
    if not network_config:
        return False
    
    # Step 3: Create RDS
    print("\n3. Setting up RDS database...")
    db_endpoint = aws_setup.create_rds_instance(network_config)
    if not db_endpoint:
        return False
    
    # Step 4: Create processing instances
    print("\n4. Setting up EC2 processing instances...")
    instance_ids = aws_setup.create_processing_instances(network_config)
    if not instance_ids:
        return False
    
    # Step 5: Create SQS queue
    print("\n5. Setting up SQS queue...")
    queue_url = aws_setup.create_sqs_queue()
    if not queue_url:
        return False
    
    # Step 6: Create Lambda function
    print("\n6. Setting up Lambda orchestrator...")
    if not aws_setup.create_lambda_function():
        print("⚠️  Lambda creation failed - manual setup required")
    
    # Step 7: Setup monitoring
    print("\n7. Setting up monitoring...")
    if not aws_setup.setup_monitoring():
        print("⚠️  Monitoring setup failed - manual setup required")
    
    # Step 8: Generate summary
    print("\n8. Generating deployment summary...")
    aws_setup.generate_deployment_summary()
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ AWS infrastructure setup completed successfully!")
        print("📋 Check 'aws_deployment_summary.txt' for detailed information")
    else:
        print("\n❌ AWS infrastructure setup failed")
        print("🔧 Check AWS credentials and permissions")