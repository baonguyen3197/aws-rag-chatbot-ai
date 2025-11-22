"""Prompt templates and helper builders for RAG answers.

This file provides copy-paste-ready system prompts, retrieval-aware
instruction templates, output templates, and small helpers to assemble
the final prompt sent to the LLM. Import and use `build_retrieval_prompt`
to compose the instruction + query + top-K snippets payload.

Keep strings short to avoid token bloat in the composed prompt.
"""
from typing import List, Dict

# RAG constants (tune as needed)
RAG_TOP_K = 5
RAG_SIM_THRESHOLD = 0.75
RAG_RERANK_TOP = 10
RAG_RERANK_KEEP = 5


# -- System prompt --
SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Use ONLY the provided retrieved "
    "snippets as evidence for factual claims. If the KB is insufficient, be "
    "explicit: say 'I don't have enough information in the knowledge base to "
    "answer that confidently.' Begin with a 1–2 sentence summary, then Evidence, "
    "then Explanation, then Sources. Be concise and avoid hallucination."
)


# -- Retrieval-aware instruction: appended to the query+snippets block --
RETRIEVAL_INSTRUCTION = (
    "You will receive:\n"
    "1) Query: {{query}}\n"
    "2) Top-K Snippets: a numbered list of snippet objects. Each snippet has: "
    "source_id, title, section_heading, text, score.\n"
    "Processing rules:\n"
    "- If no snippet has score >= {threshold}, reply exactly:\n"
    "  \"I don't have enough information in the knowledge base to answer that confidently. "
    "Would you like me to search more, upload a document, or answer from general knowledge (may be less reliable)?\"\n"
    "- Otherwise, produce an answer with this structure:\n"
    "  Short Answer (1–3 sentences).\n"
    "  Evidence Summary: for each factual claim, include a bullet with a quoted snippet and its metadata.\n"
    "  Explanation/Steps: use sequentially numbered steps; include transitions between steps.\n"
    "  Confidence & Gaps: High/Medium/Low and what is missing.\n"
    "  Sources: short list with title — source_id — 1-line note.\n"
    "- Never assert facts not supported by snippets; flag unsupported claims.\n"
    "Answer the query now."
).format(threshold=RAG_SIM_THRESHOLD)


# Output template (for human-readable guidance / validation)
OUTPUT_TEMPLATE = (
    "Short Answer:\n"
    "- <1–2 sentence summary>\n\n"
    "Details:\n"
    "- **Main points:**\n"
    "  1. ...\n"
    "- **Procedure / Steps:**\n"
    "  1. Step one. After step 1, do...\n"
    "\n"
    "Evidence & Citations:\n"
    "- Claim A:\n"
    "  - Evidence: \"...\" — `<title> | <section> | chunk=<id>`\n"
    "\n"
    "Confidence: High / Medium / Low\n"
    "Gaps / What’s missing:\n"
    "- Missing: <short list>\n"
    "\n"
    "Sources:\n"
    "- <title> — <source_id> — supports: <claim>\n"
)


def format_snippet(sn: Dict) -> str:
    """Format a single snippet dict into a compact textual block.

    Expected keys: source_id, title, section_heading, score, text
    """
    return (
        f"source_id: {sn.get('source_id')} | title: {sn.get('title')} | "
        f"section: {sn.get('section_heading')} | score: {sn.get('score')}\n"
        f"text: {sn.get('text')}\n"
    )


def format_snippets_for_prompt(snippets: List[Dict]) -> str:
    """Return the Top-K snippets block formatted for prompt insertion."""
    lines: List[str] = []
    for i, s in enumerate(snippets[:RAG_TOP_K], start=1):
        lines.append(f"{i}) {format_snippet(s)}")
    return "\n".join(lines)


def build_retrieval_prompt(query: str, snippets: List[Dict]) -> str:
    """Compose the full retrieval-aware prompt to send to the LLM.

    The returned string includes the query, the formatted Top-K snippets,
    and the retrieval instruction. Keep the snippet text trimmed if needed
    upstream to avoid overlong prompts.
    """
    header = f"Query: {query}\n\nTop-K Snippets:\n"
    snippets_block = format_snippets_for_prompt(snippets)
    return f"{header}{snippets_block}\n\n{RETRIEVAL_INSTRUCTION}"


# Example: simple helper for nmap query (illustrative usage)
EXAMPLE_NMAP_QUERY = "How can I detect open ports on a host using nmap?"


if __name__ == "__main__":
    # Quick local smoke test for developers
    sample_snippet = {
        "source_id": "chunk-001",
        "title": "Nmap Quickstart",
        "section_heading": "Scan Types",
        "score": 0.87,
        "text": "To perform a TCP SYN scan use: nmap -sS <target>."
    }
    print(build_retrieval_prompt(EXAMPLE_NMAP_QUERY, [sample_snippet]))
