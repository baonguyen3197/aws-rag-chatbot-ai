from typing import List
import re

def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"\w+", (text or "").lower())
    return [t for t in tokens if len(t) > 2]

def _score_document(question_tokens: List[str], doc_tokens: List[str]) -> float:
    if not question_tokens or not doc_tokens:
        return 0.0
    qset = set(question_tokens)
    dset = set(doc_tokens)
    overlap = qset & dset
    from math import log
    return len(overlap) / (log(len(doc_tokens) + 2))

def _concise_answer_from_snippet(snippet: str, max_sentences: int = 2) -> str:
    if not snippet:
        return "(No local information found.)"
    lsnip = snippet.strip().lower()
    if lsnip.startswith("(local mock)") or "no relevant files" in lsnip or "no files found" in lsnip:
        return "(No relevant local documents found.)"

    lines = [l for l in snippet.splitlines()]
    source = None
    content = snippet
    if lines and lines[0].lower().startswith("file:"):
        source = lines[0][5:].strip()
        content = "\n".join(lines[1:]).strip()

    sentences = re.split(r'(?<=[.!?])\s+', content)
    chosen = []
    char_count = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        chosen.append(s)
        char_count += len(s)
        if len(chosen) >= max_sentences or char_count > 800:
            break

    if not chosen:
        head = content[:1000]
        last_space = head.rfind(' ')
        if last_space > 0:
            return head[:last_space].strip() + (f"\n\nSource: {source}" if source else "")
        return head.strip()

    answer = " ".join(chosen)
    if source:
        answer = f"{answer}\n\nSource: {source}"
    return answer
