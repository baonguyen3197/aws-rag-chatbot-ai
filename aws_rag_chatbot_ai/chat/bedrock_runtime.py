import os
import json
from typing import Any, Dict

from aws_rag_chatbot_ai.chat.aws_utils import make_client
from aws_rag_chatbot_ai.chat.bedrock_utils import DEFAULT_BEDROCK_MODEL


def invoke_bedrock_model(prompt: str, model_override: str = None) -> str:
    """Invoke Bedrock Runtime and return a best-effort textual answer.

    This isolates the Bedrock invocation shape so callers (like `state.py`) can
    import a single function and avoid embedding protocol details.
    """
    aws_region = os.getenv('AWS_DEFAULT_REGION') or os.getenv('AWS_REGION')
    client = make_client("bedrock-runtime", region=aws_region)

    # Selection order: explicit override -> env vars (multiple aliases) -> default
    chosen = (
        model_override
        or os.getenv('BEDROCK_MODEL_ARN')
        or os.getenv('BEDROCK_MODEL_ID')
        or os.getenv('BEDROCK_INFERENCE_PROFILE')
        or os.getenv('FALLBACK_MODEL')
        or os.getenv('BEDROCK_PREFERRED')
    )

    model_id = chosen or DEFAULT_BEDROCK_MODEL
    # Normalize if an ARN or path was provided
    if isinstance(model_id, str) and (model_id.startswith('arn:') or '/' in model_id):
        try:
            model_id = model_id.split('/')[-1]
        except Exception:
            pass

    model_id = str(model_id).strip()
    lower_mid = (model_id or '').lower()

    if 'nova' in lower_mid or 'titan' in lower_mid:
        payload = json.dumps({"input": prompt})
    elif 'claude' in lower_mid or 'anthropic' in lower_mid:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        sonnet_payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": int(os.getenv("BEDROCK_MAX_TOKENS", "512")),
            "messages": messages,
            "temperature": float(os.getenv("BEDROCK_TEMPERATURE", "0.5")),
        }
        payload = json.dumps(sonnet_payload)
    else:
        payload = json.dumps({"input": prompt})

    response = client.invoke_model(
        modelId=model_id,
        body=payload,
        contentType="application/json",
        accept="application/json",
    )

    raw = b""
    try:
        body_obj = response.get('body') if isinstance(response, dict) else response
        if hasattr(body_obj, 'read'):
            raw = body_obj.read()
        elif isinstance(body_obj, (bytes, str)):
            raw = body_obj if isinstance(body_obj, bytes) else str(body_obj).encode('utf-8', errors='replace')
    except Exception:
        pass

    decoded = raw.decode('utf-8', errors='replace') if raw else ''
    parsed: Dict[str, Any] = {}
    try:
        parsed = json.loads(decoded) if decoded else {}
    except Exception:
        parsed = {}

    answer = ''
    if isinstance(parsed, dict):
        for key in ('completion', 'output', 'result', 'text'):
            v = parsed.get(key)
            if isinstance(v, str) and v.strip():
                answer = v.strip()
                break

        if not answer:
            content = parsed.get('outputs') or parsed.get('messages') or parsed.get('content')
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict):
                    answer = first.get('text') or first.get('content') or ''
                    if isinstance(answer, list) and answer and isinstance(answer[0], dict):
                        answer = answer[0].get('text', '')
                    if isinstance(answer, str):
                        answer = answer.strip()

    if not answer:
        answer = decoded.strip()

    return answer


def get_bedrock_model_id() -> str:
    """Return the Bedrock model id string that will be used by default.

    This is a convenience so other parts of the code can display or log
    which model is configured without invoking the runtime.
    """
    chosen = (
        os.getenv('BEDROCK_MODEL_ARN')
        or os.getenv('BEDROCK_MODEL_ID')
        or os.getenv('BEDROCK_INFERENCE_PROFILE')
        or os.getenv('FALLBACK_MODEL')
        or os.getenv('BEDROCK_PREFERRED')
    )
    model_id = chosen or DEFAULT_BEDROCK_MODEL
    if isinstance(model_id, str) and (model_id.startswith('arn:') or '/' in model_id):
        try:
            model_id = model_id.split('/')[-1]
        except Exception:
            pass
    return str(model_id)
