import os
import json
from typing import List, Dict, Any

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss
except Exception:
    # Defer import errors until functions are used so the app can run without these libs
    np = None
    SentenceTransformer = None
    faiss = None

_MODEL = None
_MODEL_NAME = None
_INDEX_PATH = os.path.join(os.path.dirname(__file__), "vector_index.faiss")
_META_PATH = os.path.join(os.path.dirname(__file__), "vector_index_meta.json")
# Extra info file to record index-level metadata (model, dim, count)
_INFO_PATH = os.path.join(os.path.dirname(__file__), "vector_index_info.json")


def ensure_model(model_name: str = None):
    """Ensure the SentenceTransformer model is loaded.

    If `model_name` is not provided, the function consults the
    `EMBEDDING_MODEL` environment variable. If that is not set, it falls
    back to the historical default `all-MiniLM-L6-v2` to avoid surprising
    behavior changes.
    """
    global _MODEL, _MODEL_NAME
    if model_name is None:
        model_name = os.getenv("EMBEDDING_MODEL")
    if not model_name:
        model_name = "all-MiniLM-L6-v2"

    if _MODEL is None or (_MODEL_NAME and _MODEL_NAME != model_name):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed; install via requirements.txt")
        _MODEL = SentenceTransformer(model_name)
        _MODEL_NAME = model_name
    return _MODEL


def get_embedding_model_name() -> str:
    """Return the name of the embedding model currently configured or loaded."""
    global _MODEL_NAME
    if _MODEL_NAME:
        return _MODEL_NAME
    return os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def embed_texts(texts: List[str]) -> Any:
    model = ensure_model()
    emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    # normalize for cosine similarity via inner product
    if np is None:
        raise RuntimeError("numpy required for embeddings")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms
    return emb.astype('float32')


def save_index(index, metas: List[Dict[str, Any]]):
    if faiss is None:
        raise RuntimeError("faiss not available")
    faiss.write_index(index, _INDEX_PATH)
    with open(_META_PATH, 'w', encoding='utf-8') as f:
        json.dump(metas, f, ensure_ascii=False)
    # Write an info file with model and dimension metadata to aid index management
    try:
        info = {
            "model": get_embedding_model_name(),
            "count": len(metas),
            "dim": getattr(index, 'd', None),
        }
        with open(_INFO_PATH, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False)
    except Exception:
        # Non-critical: if writing info fails, proceed silently
        pass


def load_index():
    if faiss is None:
        raise RuntimeError("faiss not available")
    if not os.path.exists(_INDEX_PATH) or not os.path.exists(_META_PATH):
        return None, None
    index = faiss.read_index(_INDEX_PATH)
    with open(_META_PATH, 'r', encoding='utf-8') as f:
        metas = json.load(f)
    # Attempt to load index info if present
    info = None
    try:
        if os.path.exists(_INFO_PATH):
            with open(_INFO_PATH, 'r', encoding='utf-8') as f:
                info = json.load(f)
    except Exception:
        info = None
    return index, metas


def build_faiss_index(embeddings, metas: List[Dict[str, Any]]):
    if faiss is None or np is None:
        raise RuntimeError("faiss/numpy required to build index")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    save_index(index, metas)
    return index


def append_to_index(embeddings, metas: List[Dict[str, Any]]):
    """Append embeddings and corresponding metas to an existing index, or create a new one.

    embeddings: numpy array of shape (n, d)
    metas: list of metadata dicts of length n
    """
    if faiss is None or np is None:
        raise RuntimeError("faiss/numpy required to append to index")

    index, existing_metas = load_index()
    if index is None or existing_metas is None:
        # Build new index
        return build_faiss_index(embeddings, metas)

    # Ensure dimensionality matches
    try:
        d = embeddings.shape[1]
    except Exception:
        raise RuntimeError("Invalid embeddings shape")

    # If existing index has different dimension, rebuild (simple fallback)
    if index.d != d:  # type: ignore
        # Create a new index with concatenated embeddings
        # Load existing embeddings is not supported; rebuild required
        raise RuntimeError("Index dimensionality mismatch; rebuild required")

    index.add(embeddings)
    existing_metas.extend(metas)
    save_index(index, existing_metas)
    return index


def vector_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Return top_k passages for the query using FAISS index.

    Each result is a dict: {source, text, score, meta}
    """
    if SentenceTransformer is None or faiss is None or np is None:
        raise RuntimeError("Embedding or FAISS libraries not installed")
    index, metas = load_index()
    if index is None or metas is None:
        return []
    q_emb = embed_texts([query])
    D, I = index.search(q_emb, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metas):
            continue
        meta = metas[idx]
        results.append({
            "source": meta.get('source'),
            "text": meta.get('text'),
            "score": float(dist),
            "meta": meta.get('meta', {}),
        })
    return results
