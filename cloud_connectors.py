
"""
Cloud Storage and Notification Connectors for Jarvis AI
- AWS S3, GCS, Azure Blob
- Slack and email notifications
"""

# AWS S3 Connector
class S3Connector:
    def __init__(self, bucket):
        import boto3
        self.s3 = boto3.client('s3')
        self.bucket = bucket

    def upload(self, file_path, key):
        self.s3.upload_file(file_path, self.bucket, key)

    def download(self, key, file_path):
        self.s3.download_file(self.bucket, key, file_path)


# Slack Notification
class SlackNotifier:
    def __init__(self, token):
        from slack_sdk import WebClient
        self.client = WebClient(token=token)

    def send(self, channel, message):
        self.client.chat_postMessage(channel=channel, text=message)



# Google Cloud Storage Connector
class GCSConnector:
    def __init__(self, bucket_name):
        from google.cloud import storage
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def upload(self, file_path, blob_name):
        blob = self.bucket.blob(blob_name)
        blob.upload_from_filename(file_path)

    def download(self, blob_name, file_path):
        blob = self.bucket.blob(blob_name)
        blob.download_to_filename(file_path)


# Azure Blob Storage Connector
class AzureBlobConnector:
    def __init__(self, connection_string, container_name):
        from azure.storage.blob import BlobServiceClient
        self.service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.service_client.get_container_client(container_name)

    def upload(self, file_path, blob_name):
        with open(file_path, "rb") as data:
            self.container_client.upload_blob(name=blob_name, data=data, overwrite=True)

    def download(self, blob_name, file_path):
        with open(file_path, "wb") as file:
            stream = self.container_client.download_blob(blob_name)
            file.write(stream.readall())


# Email Notification
class EmailNotifier:
    def __init__(self, smtp_server, smtp_port, username, password):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password

    def send(self, to_email, subject, message):
        import smtplib
        from email.message import EmailMessage
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.username
        msg["To"] = to_email
        msg.set_content(message)
        with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
            server.login(self.username, self.password)
            server.send_message(msg)


# Azure Functions Connector for Serverless Scaling
class AzureFunctionsConnector:
    def __init__(self, subscription_id, resource_group, function_app_name):
        from azure.identity import DefaultAzureCredential
        from azure.mgmt.web import WebSiteManagementClient
        
        self.credential = DefaultAzureCredential()
        self.web_client = WebSiteManagementClient(self.credential, subscription_id)
        self.resource_group = resource_group
        self.function_app_name = function_app_name

    def scale_function(self, instance_count):
        """Scale Azure Function app instances"""
        # Note: Azure Functions scaling is typically handled by App Service Plans
        # This is a simplified example
        return f"Scaled {self.function_app_name} to {instance_count} instances"

    def deploy_function(self, function_code, function_name):
        """Deploy function code (simplified)"""
        return f"Deployed {function_name} to {self.function_app_name}"


# AWS Lambda Connector for Serverless Scaling
class AWSLambdaConnector:
    def __init__(self, region='us-east-1'):
        import boto3
        self.lambda_client = boto3.client('lambda', region_name=region)

    def invoke_function(self, function_name, payload):
        """Invoke Lambda function"""
        response = self.lambda_client.invoke(
            FunctionName=function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        return json.loads(response['Payload'].read())

    def update_concurrency(self, function_name, concurrency_limit):
        """Update function concurrency for scaling"""
        self.lambda_client.put_function_concurrency(
            FunctionName=function_name,
            ReservedConcurrentExecutions=concurrency_limit
        )
        return f"Updated concurrency for {function_name} to {concurrency_limit}"


# GCP Cloud Functions Connector
class GCPFunctionsConnector:
    def __init__(self, project_id):
        from google.cloud import functions_v1
        self.client = functions_v1.CloudFunctionsServiceClient()
        self.project_id = project_id

    def deploy_function(self, function_name, source_code, trigger):
        """Deploy GCP Cloud Function"""
        # Simplified deployment
        return f"Deployed {function_name} to GCP project {self.project_id}"

    def call_function(self, function_name, data):
        """Call GCP Cloud Function"""
        # Simplified call
        return f"Called {function_name} with data: {data}"
