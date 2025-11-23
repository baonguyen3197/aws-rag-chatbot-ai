import os
import boto3
from typing import Optional

# Centralized boto3 helpers. Keep minimal and importable from other modules.
boto3.setup_default_session()

def make_client(name: str, region: Optional[str] = None, **kwargs):
    region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    endpoint = os.getenv('AWS_ENDPOINT_URL')
    if endpoint:
        return boto3.client(name, region_name=region, endpoint_url=endpoint, **kwargs)
    return boto3.client(name, region_name=region, **kwargs)

def make_resource(name: str, region: Optional[str] = None, **kwargs):
    region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    endpoint = os.getenv('AWS_ENDPOINT_URL')
    if endpoint:
        return boto3.resource(name, region_name=region, endpoint_url=endpoint, **kwargs)
    return boto3.resource(name, region_name=region, **kwargs)

_chat_table = None

def chat_table_name():
    return os.environ.get("CHAT_TABLE_NAME", "ChatSession")

def get_chat_table():
    global _chat_table
    if _chat_table is None:
        dynamodb = make_resource("dynamodb")
        _chat_table = dynamodb.Table(chat_table_name())
    return _chat_table

def aws_user_id():
    return os.environ.get("AWS_USER_ID", "local-user")
