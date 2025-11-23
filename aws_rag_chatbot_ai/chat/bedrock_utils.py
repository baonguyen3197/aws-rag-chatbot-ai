import os
from typing import Dict, Any

DEFAULT_BEDROCK_MODEL = os.environ.get(
    "BEDROCK_MODEL", "arn:aws:bedrock:ap-northeast-1::foundation-model/amazon.nova-micro-v1:0"
)

def build_bedrock_payload(prompt: str, model: str = None, max_tokens: int = 1024) -> Dict[str, Any]:
    model = model or DEFAULT_BEDROCK_MODEL
    payload = {
        "modelId": model,
        "input": prompt,
        "maxTokens": max_tokens,
    }
    return payload
