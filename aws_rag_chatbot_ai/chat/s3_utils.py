import os
import io
from typing import List

from aws_rag_chatbot_ai.chat.aws_utils import make_client

try:
    import pdfplumber  # type: ignore
    _HAS_PDFPLUMBER = True
except Exception:
    _HAS_PDFPLUMBER = False


def get_s3_client(region: str = None):
    region = region or os.getenv('AWS_DEFAULT_REGION') or os.getenv('AWS_REGION')
    return make_client('s3', region=region)


def list_keys(bucket: str, prefix: str = '') -> List[str]:
    client = get_s3_client()
    keys = []
    try:
        resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        for obj in resp.get('Contents', []):
            k = obj.get('Key')
            if k:
                keys.append(k)
    except Exception:
        try:
            resp = client.list_objects_v2(Bucket=bucket)
            for obj in resp.get('Contents', []):
                k = obj.get('Key')
                if k and (not prefix or prefix in k):
                    keys.append(k)
        except Exception:
            return []
    return keys


def get_object_text(bucket: str, key: str, max_bytes: int = 200000) -> str:
    client = get_s3_client()
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        raw = resp['Body'].read()
        if isinstance(raw, (bytes, bytearray)) and raw[:4] == b'%PDF':
            if _HAS_PDFPLUMBER:
                try:
                    with pdfplumber.open(io.BytesIO(raw)) as pdf:
                        pages = [p.extract_text() or '' for p in pdf.pages]
                        return '\n\n'.join(pages)
                except Exception:
                    return raw[:max_bytes].decode('utf-8', errors='replace')
            else:
                return '[PDF_BINARY_CONTENT] (install pdfplumber to extract text)\n' + raw[:max_bytes].decode('utf-8', errors='replace')
        return raw.decode('utf-8', errors='replace')[:max_bytes]
    except Exception:
        return ''
