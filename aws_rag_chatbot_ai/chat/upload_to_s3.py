import boto3
from typing import Any

async def upload_to_s3(file_content: bytes, bucket_name: str, object_name: str) -> None:
    try:
        s3_client = boto3.client('s3')
        s3_client.put_object(
            Bucket=bucket_name,
            Key=object_name,
            Body=file_content
        )
    except Exception as e:
        raise Exception(f"Failed to upload to S3: {str(e)}")