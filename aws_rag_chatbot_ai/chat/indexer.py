import os
import math
from typing import List, Dict, Any

from aws_rag_chatbot_ai.chat.s3_utils import list_keys, get_object_text
from aws_rag_chatbot_ai.chat.embeddings_utils import embed_texts, build_faiss_index


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == length:
            break
        start = max(0, end - overlap)
    return chunks


def parse_mdx_frontmatter(text: str) -> Dict[str, Any]:
    meta = {}
    if text.startswith('---'):
        parts = text.split('---', 2)
        if len(parts) >= 3:
            fm = parts[1]
            for line in fm.splitlines():
                if ':' in line:
                    k, v = line.split(':', 1)
                    meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta


def build_index_from_s3(bucket: str, prefix: str = '', chunk_size: int = 800):
    keys = list_keys(bucket, prefix)
    if not keys:
        print("No keys found to index")
        return

    passages = []
    metas = []
    for key in keys:
        text = get_object_text(bucket, key)
        meta = parse_mdx_frontmatter(text) if key.lower().endswith(('.mdx', '.md')) else {}
        chunks = chunk_text(text, chunk_size=chunk_size)
        for i, c in enumerate(chunks):
            metas.append({
                'source': key,
                'text': c,
                'meta': meta,
                'chunk_index': i,
            })
            passages.append(c)

    if not passages:
        print("No passages extracted from S3 files")
        return

    embeddings = embed_texts(passages)
    build_faiss_index(embeddings, metas)
    print(f"Built index with {len(passages)} passages")


def index_single_object(bucket: str, key: str, chunk_size: int = 800):
    """Index a single S3 object (used for re-index-on-upload).

    Fetches the object, chunks it, embeds passages, and appends them to the FAISS index.
    """
    text = get_object_text(bucket, key)
    if not text:
        return 0

    meta = parse_mdx_frontmatter(text) if key.lower().endswith(('.mdx', '.md')) else {}
    chunks = chunk_text(text, chunk_size=chunk_size)
    if not chunks:
        return 0

    metas = []
    for i, c in enumerate(chunks):
        metas.append({
            'source': key,
            'text': c,
            'meta': meta,
            'chunk_index': i,
        })

    embeddings = embed_texts(chunks)
    # Append to existing index (or create if none)
    try:
        from aws_rag_chatbot_ai.chat.embeddings_utils import append_to_index
        append_to_index(embeddings, metas)
        return len(chunks)
    except Exception as e:
        # If appending fails, fallback to building a new index for now
        try:
            build_faiss_index(embeddings, metas)
            return len(chunks)
        except Exception:
            return 0


if __name__ == '__main__':
    bucket = os.getenv('S3_BUCKET_NAME')
    prefix = os.getenv('S3_OBJECT_NAME', '')
    if not bucket:
        print('Set S3_BUCKET_NAME environment variable')
    else:
        build_index_from_s3(bucket, prefix)
