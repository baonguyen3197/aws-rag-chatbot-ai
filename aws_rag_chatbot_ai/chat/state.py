import os
import reflex as rx
import boto3
import json
from datetime import datetime, timezone
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import re
import mimetypes
from aws_rag_chatbot_ai.chat.text_utils import _concise_answer_from_snippet, _tokenize, _score_document
from aws_rag_chatbot_ai.chat.bedrock_runtime import invoke_bedrock_model
from aws_rag_chatbot_ai.chat.prompt_templates import (
    OUTPUT_TEMPLATE,
    RAG_SIM_THRESHOLD,
)
import io
import logging

# optional PDF text extractor for better PDF support in KB
try:
    import pdfplumber  # type: ignore
    _HAS_PDFPLUMBER = True
except Exception:
    _HAS_PDFPLUMBER = False

# Load environment variables from .env file
load_dotenv()

# Note: logging removed for clarity per project preference
if not load_dotenv():
    pass

# Logging removed by request — all logger calls have been deleted.

# Silence noisy AWS SDK debug logs (botocore/boto3) so console doesn't show
# full DynamoDB/request payloads. We keep warnings/errors visible.
for _lib in ("boto3", "botocore", "boto3.resources", "botocore.exceptions"):
    try:
        logging.getLogger(_lib).setLevel(logging.ERROR)
    except Exception:
        pass

"""
Initialize boto3 using environment variables when available. 
Don't raise on failure to call STS during import
time — fall back to a safe default identity so the module can be imported in
containers that can't reach AWS immediately.
"""
# Prefer explicit env vars
aws_region = os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-1')
aws_endpoint = os.getenv('AWS_ENDPOINT_URL')
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

# Configure a default session
boto3.setup_default_session(
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_region,
)

dynamodb = boto3.resource('dynamodb', region_name=aws_region)

# Table name can be overridden via env for portability
chat_table_name = os.getenv('CHAT_TABLE_NAME', 'ChatSession')
chat_table = dynamodb.Table(chat_table_name)

# Simple client factory (uses AWS_ENDPOINT_URL when present)
def make_client(service_name: str, region: str = None, **kwargs):
    region = region or aws_region
    endpoint = os.getenv('AWS_ENDPOINT_URL')
    if endpoint:
        return boto3.client(service_name, region_name=region, endpoint_url=endpoint, **kwargs)
    return boto3.client(service_name, region_name=region, **kwargs)

# Determine a default user ARN for the application to use when no caller identity is available
# Prefer the env var `AWS_USER_ARN`; otherwise fall back to the known user ARN provided by the operator.
aws_user_id = os.getenv('AWS_USER_ARN', "arn:aws:iam::906034468113:user/nhqb-iam-user")

def make_resource(resource_name: str, region: str = None, **kwargs):
    region = region or aws_region
    endpoint = os.getenv('AWS_ENDPOINT_URL')
    if endpoint:
        return boto3.resource(resource_name, region_name=region, endpoint_url=endpoint, **kwargs)
    return boto3.resource(resource_name, region_name=region, **kwargs)


def _item_is_deleted(item: dict) -> bool:
    """Return True if the DynamoDB item indicates a soft-delete.

    Handles booleans and string representations ('true', '1').
    """
    if not item:
        return False
    v = item.get('isDeleted')
    if isinstance(v, bool):
        return v
    try:
        return str(v).lower() in ("true", "1", "yes")
    except Exception:
        return False

# Simple in-memory cache of S3 documents to avoid re-reading on every question.
# Key: bucket/key -> content string
_s3_doc_cache: Dict[str, str] = {} 

# Determine whether Bedrock calls should be allowed in this environment.
def bedrock_allowed() -> bool:
    """Return False when DISABLE_BEDROCK=1 is set."""
    # Explicit disable takes highest precedence
    if os.getenv("DISABLE_BEDROCK", "0") == "1":
        return False

    return True

def create_chat_session_table():
    try:
        dynamodb_client = boto3.client('dynamodb')
        dynamodb_client.describe_table(TableName='ChatSession')
    except dynamodb_client.exceptions.ResourceNotFoundException:
        dynamodb.create_table(
            TableName='ChatSession',
            KeySchema=[
                {'AttributeName': 'user_id', 'KeyType': 'HASH'},  # Partition key
                {'AttributeName': 'session_id', 'KeyType': 'RANGE'}  # Sort key
            ],
            AttributeDefinitions=[
                {'AttributeName': 'user_id', 'AttributeType': 'S'},
                {'AttributeName': 'session_id', 'AttributeType': 'S'}
            ],
            BillingMode='PAY_PER_REQUEST'
        )
        dynamodb.meta.client.get_waiter('table_exists').wait(TableName='ChatSession')
        

create_chat_session_table()

class QA(BaseModel):
    question: str
    answer: str
    source: Optional[str] = None

DEFAULT_CHATS = {
    "Intros": [],
}

class State(rx.State):
    """The app state."""
    chats: Dict[str, List[QA]] = DEFAULT_CHATS
    current_chat: str = "Intros"
    question: str = ""
    processing: bool = False
    new_chat_name: str = ""
    uploaded_files: List[str] = []
    upload_error: str = ""
    uploading: bool = False
    progress: int = 0
    total_bytes: int = 0
    user_id: str = aws_user_id
    session_ids: Dict[str, str] = {}
    session_load_message: str = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{k: v for k, v in kwargs.items() if k != 'parent_state'})
        self.load_session()

    # Reflex will auto-generate setters by default; newer versions deprecate
    # implicit auto-setters. Define explicit setter to avoid deprecation warnings
    # and future breakage.
    def set_new_chat_name(self, value: str):
        self.new_chat_name = value

    def create_chat(self):
        
        if not self.new_chat_name.strip():
            
            return
        chat_name = self.new_chat_name.strip()
        if chat_name in self.chats:
            
            return
        self.chats[chat_name] = []
        self.current_chat = chat_name
        self.new_chat_name = ""

        session_id = f"Session#{datetime.now(timezone.utc).isoformat()}Z"
        item = {
            "user_id": self.user_id,
            "session_id": session_id,
            "chat_name": chat_name,
            "messages": [],  # Start with empty message list
            "isDeleted": False,
        }
        try:
            chat_table.put_item(Item=item)
            self.session_ids[chat_name] = session_id # Store session_id for new chat
            # Record this as the last active chat for the user
            try:
                chat_table.put_item(Item={"user_id": self.user_id, "session_id": "meta#last_active", "last_active_chat": chat_name})
            except Exception:
                pass
        except Exception as e:
            raise

    def delete_chat(self):
        # Soft-delete: prefer marking the specific session_id for this chat as deleted.
        if self.current_chat not in self.chats:
            return

        chat_titles = list(self.chats.keys())
        current_index = chat_titles.index(self.current_chat)

        # First try to mark the known session id for this chat (most reliable)
        session_id = self.session_ids.get(self.current_chat)
        if session_id:
            try:
                chat_table.update_item(
                    Key={"user_id": self.user_id, "session_id": session_id},
                    UpdateExpression="SET isDeleted = :val",
                    ExpressionAttributeValues={":val": True},
                )
            except Exception:
                # Fall back to scanning by chat_name below
                session_id = None

        # If we didn't have a session_id or the update failed, mark any items matching chat_name
        if not session_id:
            try:
                response = chat_table.query(
                    KeyConditionExpression="user_id = :uid",
                    ExpressionAttributeValues={":uid": self.user_id}
                )
                for item in response.get("Items", []):
                    try:
                        if item.get("chat_name") == self.current_chat:
                            chat_table.update_item(
                                Key={"user_id": self.user_id, "session_id": item.get("session_id")},
                                UpdateExpression="SET isDeleted = :val",
                                ExpressionAttributeValues={":val": True},
                            )
                    except Exception:
                        pass
            except Exception:
                pass

        # Remove from UI view
        try:
            del self.chats[self.current_chat]
        except Exception:
            pass

        # Ensure there's always at least one chat visible for UI
        if not self.chats:
            self.chats = DEFAULT_CHATS.copy()
            self.current_chat = "Intros"
            session_id = f"Intros#{datetime.now(timezone.utc).isoformat()}Z"
            try:
                chat_table.put_item(
                    Item={
                        "user_id": self.user_id,
                        "session_id": session_id,
                        "chat_name": "Intros",
                        "chat_history": [],
                        "isDeleted": False,
                    }
                )
            except Exception:
                pass
        else:
            remaining_chats = list(self.chats.keys())
            new_index = min(current_index, len(remaining_chats) - 1) if current_index < len(remaining_chats) else 0
            self.current_chat = remaining_chats[new_index]

        self.chats = self.chats

    def set_chat(self, chat_name: str):
        if chat_name not in self.chats:
            if not self.chats:
                self.chats = DEFAULT_CHATS.copy()
                self.current_chat = "Intros"
                session_id = f"Intros#{datetime.now(timezone.utc).isoformat()}Z"
                try:
                    chat_table.put_item(
                        Item={
                            "user_id": self.user_id,
                            "session_id": session_id,
                            "chat_name": "Intros",
                            "chat_history": [],
                            "isDeleted": False,
                        }
                    )
                except Exception:
                    pass
            else:
                self.current_chat = list(self.chats.keys())[0]
            self.chats = self.chats
            return
        self.current_chat = chat_name
        # Persist last active chat selection so reloads keep the same chat
        try:
            chat_table.put_item(Item={"user_id": self.user_id, "session_id": "meta#last_active", "last_active_chat": chat_name})
        except Exception:
            pass

    def reset_session(self):
        
        self.chats = DEFAULT_CHATS.copy()
        self.current_chat = "Intros"
        self.processing = False
        
        try:
            response = chat_table.scan(FilterExpression="user_id = :uid", ExpressionAttributeValues={":uid": self.user_id})
            
            for item in response.get("Items", []):
                chat_table.delete_item(Key={"user_id": self.user_id, "session_id": item["session_id"]})
            session_id = f"Intros#{datetime.now(timezone.utc).isoformat()}Z"
            chat_table.put_item(
                Item={
                    "user_id": self.user_id,
                    "session_id": session_id,
                    "chat_name": "Intros",
                    "chat_history": [],
                    "isDeleted": False,
                }
            )
            
        except Exception as e:
            pass
        self.chats = self.chats

    @rx.var(cache=True)
    def chat_titles(self) -> List[str]:
        titles = list(self.chats.keys())
        return titles

    async def process_question(self, form_data: Dict[str, Any]):
        """Process a submitted question: call Bedrock (or mock), store result in DynamoDB, update state."""
        
        question = form_data.get("question", "").strip()
        if not question:
            return
            return

        qa = QA(question=question, answer="")
        self.chats.setdefault(self.current_chat, []).append(qa)
        self.processing = True
        yield

        knowledge_base = await self.get_knowledge_base()

        # Retrieve top-N relevant snippets from S3 and use them to build the retrieval prompt.
        try:
            top_snips = await self.find_relevant_snippets(question, top_n=3, max_chars=1200)
        except Exception:
            top_snips = []

        not_enough_msg = (
            "I don't have enough information in the knowledge base to answer that confidently. "
            "Would you like me to search more, upload a document, or answer from general knowledge (may be less reliable)?"
        )

        skip_model = False
        custom_prompt_template = form_data.get('prompt_template') or form_data.get('prompt')

        if not top_snips:
            answer = not_enough_msg
            skip_model = True
        else:
            top_score = top_snips[0].get('score', 0.0)

            # Build structured snippet objects for the retrieval prompt so the model
            # can cite Sources in its output. Each snippet contains: source_id, title, section_heading, text, score
            snippets_for_prompt = []
            for s in top_snips:
                key = s.get('key')
                excerpt = s.get('excerpt', '')
                score = s.get('score', 0.0)
                snippets_for_prompt.append({
                    'source_id': key,
                    'title': os.path.basename(key) if key else '',
                    'section_heading': '',
                    'text': excerpt,
                    'score': score,
                })

            # If a custom prompt template was provided and references $search_results$, preserve that behavior.
            if custom_prompt_template and isinstance(custom_prompt_template, str) and "$search_results$" in custom_prompt_template:
                # Keep backward compatibility for simple custom templates
                search_results_parts = []
                for i, s in enumerate(top_snips, start=1):
                    search_results_parts.append(f"{i}. File: {s['key']}\n{s['excerpt']}")
                search_results_text = "\n\n".join(search_results_parts)
                prompt = custom_prompt_template.replace("$search_results$", search_results_text).replace("{question}", question)
                if "$output_format_instructions$" in prompt:
                    prompt = prompt.replace("$output_format_instructions$", OUTPUT_TEMPLATE)
            else:
                # Default: build a retrieval-aware prompt that enforces using snippets and emitting Sources
                from aws_rag_chatbot_ai.chat.prompt_templates import build_retrieval_prompt
                prompt = build_retrieval_prompt(question, snippets_for_prompt) + "\n\n" + OUTPUT_TEMPLATE

            # Only invoke the model if the top score meets threshold
            if top_score < RAG_SIM_THRESHOLD:
                answer = not_enough_msg
                skip_model = True
            else:
                try:
                    answer = invoke_bedrock_model(prompt)
                except Exception:
                    # If bedrock fails, fall back to local snippet
                    try:
                        snips = await self.find_relevant_snippets(question, top_n=1, max_chars=1200)
                        if snips:
                            snippet = snips[0].get('excerpt', '')
                            answer = _concise_answer_from_snippet(snippet, max_sentences=5)
                        else:
                            answer = 'Sorry, I encountered an error while processing your request.'
                    except Exception:
                        answer = 'Sorry, I encountered an error while processing your request.'

        # Update the QA and persist to DynamoDB
        # Ensure the answer includes explicit Sources when we have retrievals
        try:
            sources_list = [s.get('key') for s in (top_snips or []) if s.get('key')]
        except Exception:
            sources_list = []

        # Normalize the answer: strip a leading repeated question to avoid duplication.
        # Handles cases where the model echoes the question on the same line
        # (e.g. "Q? A...") or as a separate first line.
        try:
            if answer and question:
                import re

                a_strip = answer.lstrip()
                q_norm = question.strip()
                if q_norm:
                    # Build a regex to remove a leading question occurrence, allowing
                    # optional punctuation and whitespace after the question. Case-insensitive.
                    pattern = re.compile(r'^\s*' + re.escape(q_norm) + r'[\:\?\.!\-–—]?\s*', re.IGNORECASE)
                    new_answer = pattern.sub('', a_strip, count=1)
                    # If substitution removed text, use the cleaned answer.
                    if new_answer != a_strip:
                        answer = new_answer
                    else:
                        # Fallback: if the first line exactly equals the question, drop it.
                        a_first = a_strip.splitlines()[0].strip() if a_strip.splitlines() else ''
                        if a_first and a_first.lower() == q_norm.lower():
                            rest = "\n".join(a_strip.splitlines()[1:]).lstrip()
                            answer = rest
        except Exception:
            pass

        lower_answer = (answer or "").lower()
        # If the model didn't include explicit Sources, append them for traceability
        if sources_list and not any(k in lower_answer for k in ("source:", "sources:", "source_id")):
            sources_lines = "\n\nSources:\n" + "\n".join([f"- {s}" for s in sources_list])
            answer = (answer or "") + sources_lines

        # NOTE: Evidence blocks are intentionally removed per user preference.
        # We do not append quoted Evidence excerpts to model answers anymore.

        self.chats[self.current_chat][-1].answer = answer
        self.processing = False

        session_id = self.session_ids.get(self.current_chat)
        if not session_id:
            session_id = f"Session#{datetime.now(timezone.utc).isoformat()}Z"
            self.session_ids[self.current_chat] = session_id

        try:
            existing_item = {}
            try:
                resp = chat_table.get_item(Key={"user_id": self.user_id, "session_id": session_id})
                existing_item = resp.get("Item", {})
            except Exception:
                existing_item = {}

            existing_messages = existing_item.get("messages", [])
            # Persist sources separately for better observability
            try:
                persisted_sources = sources_list
            except Exception:
                persisted_sources = []
            existing_messages.append({"question": qa.question, "answer": qa.answer, "sources": persisted_sources})

            chat_table.put_item(
                Item={
                    "user_id": self.user_id,
                    "session_id": session_id,
                    "chat_name": self.current_chat,
                    "messages": existing_messages,
                    "isDeleted": False,
                }
            )
            
        except Exception as e:
            pass

        self.chats = self.chats
        yield

    def load_session(self):
        """Load chat sessions from DynamoDB for the current user."""
        try:
            # Query DynamoDB for all items with the user's ID
            response = chat_table.query(
                KeyConditionExpression="user_id = :uid",
                ExpressionAttributeValues={":uid": self.user_id}
            )
            items = response.get("Items", [])

            if not items:
                # No sessions found, initialize with default "Intros" chat
                session_id = f"Session#{datetime.now(timezone.utc).isoformat()}Z"
                self.chats["Intros"] = [QA(question="", answer="")]
                self.session_ids["Intros"] = session_id
                self.session_load_message = "No sessions found; initialized default chat."
            else:
                # Load existing sessions. Also detect per-user metadata item
                # with session_id == 'meta#last_active' to prefer last active chat.
                self.chats = {}
                self.session_ids = {}
                meta_item = None
                for item in items:
                    if item.get('session_id') == 'meta#last_active':
                        meta_item = item
                        break
                for item in items:
                    if item.get('session_id') == 'meta#last_active':
                        continue
                    # Skip sessions that were soft-deleted
                    if _item_is_deleted(item):
                        continue

                    chat_name = item.get("chat_name", "Intros")
                    session_id = item.get("session_id")
                    # NOTE: previously we hid chats present in `meta#hidden_chats`.
                    # That caused cases where all chats became invisible if the
                    # metadata was incorrect. To be safe, always load chats from
                    # the DB and keep `hidden_chats` only as metadata (not enforced).
                    messages = item.get("messages", []) or item.get("chat_history", [])
                    unique_chat_name = chat_name if chat_name not in self.chats else f"{chat_name}_{session_id}"
                    # Sanitize loaded answers: remove any previously persisted "Evidence" blocks
                    sanitized_qas = []
                    for m in messages:
                        try:
                            q_text = m.get("question", "") if isinstance(m, dict) else (m.question if hasattr(m, 'question') else "")
                        except Exception:
                            q_text = ""
                        try:
                            a_text = m.get("answer", "") if isinstance(m, dict) else (m.answer if hasattr(m, 'answer') else "")
                        except Exception:
                            a_text = ""
                        try:
                            # Remove block-form Evidence: lines that start with 'Evidence:' followed by quoted lines starting with '>'
                            a_text = re.sub(r"\n\s*Evidence:\n(?:>.*\n?)+", "\n", a_text or "", flags=re.IGNORECASE)
                            # Also remove any trailing single-line 'Evidence: ...' occurrences
                            a_text = re.sub(r"\n\s*Evidence:\s*.*", "\n", a_text or "", flags=re.IGNORECASE)
                        except Exception:
                            pass
                        sanitized_qas.append(QA(question=q_text, answer=a_text))
                        self.chats[unique_chat_name] = sanitized_qas
                        # remember session id mapping for this chat name
                        try:
                            self.session_ids[unique_chat_name] = session_id
                        except Exception:
                            pass

                    # Prefer the last active chat if present in metadata and exists in loaded chats
                    if meta_item and meta_item.get('last_active_chat'):
                        desired = meta_item.get('last_active_chat')
                        if desired in self.chats:
                            self.current_chat = desired
                        else:
                            # Try prefix match if names were made unique with session ids
                            found = next((k for k in self.chats.keys() if k.startswith(desired)), None)
                            self.current_chat = found or (list(self.chats.keys())[0] if self.chats else 'Intros')
                    else:
                        self.current_chat = list(self.chats.keys())[0] if self.chats else 'Intros'

                # Brief non-verbose status for UI/console
                self.session_load_message = "Sessions loaded."

                # Ensure UI updates with loaded data
                self.chats = self.chats

        except ClientError:
            self.chats = DEFAULT_CHATS.copy()
            self.current_chat = "Intros"
            self.session_ids = {"Intros": f"Session#{datetime.now(timezone.utc).isoformat()}Z"}
            self.session_load_message = "Failed to load sessions (client error); using defaults."
        except Exception:
            self.chats = DEFAULT_CHATS.copy()
            self.current_chat = "Intros"
            self.session_ids = {"Intros": f"Session#{datetime.now(timezone.utc).isoformat()}Z"}
            self.session_load_message = "Failed to load sessions; using defaults."    

    async def get_knowledge_base(self) -> str:
        """Retrieve content from all files under the specified S3 prefix."""
        # Build S3 client honoring endpoint (prefer helper so endpoint handling is consistent)
        s3_client = make_client('s3', region=os.getenv('AWS_DEFAULT_REGION', aws_region))
        bucket_name = os.getenv("S3_BUCKET_NAME")
        prefix = os.getenv("S3_OBJECT_NAME", "")

        if not bucket_name:
            return "No S3 bucket configured."

        if not prefix:
            pass

        knowledge_base = []
        tried_prefixes = []
        # Try several prefix variants to handle nested folder keys
        if prefix:
            tried_prefixes = [prefix, prefix.rstrip('/') + '/']
        else:
            tried_prefixes = ['']

        found_keys = []
        try:
            for p in tried_prefixes:
                if p in found_keys:
                    continue
                try:
                    resp = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=p)
                except Exception:
                    continue
                contents = resp.get('Contents', [])
                for obj in contents:
                    k = obj.get('Key')
                    if k and k not in found_keys:
                        found_keys.append(k)

            # As a fallback, if no keys found for the prefixes, list all and filter contains
            if not found_keys:
                resp = s3_client.list_objects_v2(Bucket=bucket_name)
                for obj in resp.get('Contents', []):
                    k = obj.get('Key')
                    if k and (not prefix or prefix in k):
                        found_keys.append(k)

            if not found_keys:
                return f"No files found under '{prefix}' in S3 bucket {bucket_name}."

            for key in found_keys:
                try:
                    file_response = s3_client.get_object(Bucket=bucket_name, Key=key)
                    raw = file_response['Body'].read()
                    # Detect PDF by header
                    if isinstance(raw, (bytes, bytearray)) and raw[:4] == b'%PDF':
                        if _HAS_PDFPLUMBER:
                            try:
                                with pdfplumber.open(io.BytesIO(raw)) as pdf:
                                    pages = [p.extract_text() or '' for p in pdf.pages]
                                    content = '\n\n'.join(pages)
                            except Exception:
                                content = raw[:400].decode('utf-8', errors='replace')
                        else:
                            content = '[PDF_BINARY_CONTENT] (install pdfplumber to extract text)\n' + raw[:400].decode('utf-8', errors='replace')
                    else:
                        content = raw.decode('utf-8', errors='replace')
                    knowledge_base.append(f"File: {key}\n{content}")
                except Exception:
                    continue
        except Exception:
            return "Error accessing S3 bucket."

        return "\n\n".join(knowledge_base) if knowledge_base else "No knowledge base available."

    async def find_relevant_snippets(self, question: str, top_n: int = 3, max_chars: int = 800) -> List[Dict[str, Any]]:
        """Return top_n relevant excerpts from S3 documents with simple scoring.

        Each returned dict contains: key, score, excerpt
        """
        bucket_name = os.getenv("S3_BUCKET_NAME")
        prefix = os.getenv("S3_OBJECT_NAME", "")
        if not bucket_name:
            return []

        s3_client = make_client('s3', region=os.getenv('AWS_DEFAULT_REGION', aws_region))

        # Try vector search first (if index exists). This improves semantic recall.
        try:
            from aws_rag_chatbot_ai.chat.embeddings_utils import vector_search
            vec_results = []
            try:
                vec_results = vector_search(question, top_k=top_n)
            except Exception:
                vec_results = []
            if vec_results:
                results = []
                for r in vec_results:
                    results.append({"key": r.get('source'), "score": r.get('score', 0.0), "excerpt": (r.get('text') or '')[:max_chars]})
                return results
        except Exception:
            # embedding libs or index might not be installed; fall back to lexical method
            pass

        candidate_keys = []
        tried_prefixes = [p for p in ([prefix, prefix.rstrip('/') + '/'] if prefix else ['']) if p]
        try:
            for p in tried_prefixes:
                try:
                    resp = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=p)
                except Exception as e:
                    continue
                for obj in resp.get('Contents', []):
                    k = obj.get('Key')
                    if k and k not in candidate_keys:
                        candidate_keys.append(k)

            if not candidate_keys:
                resp = s3_client.list_objects_v2(Bucket=bucket_name)
                for obj in resp.get('Contents', []):
                    k = obj.get('Key')
                    if k and (not prefix or prefix in k) and k not in candidate_keys:
                        candidate_keys.append(k)

            if not candidate_keys:
                return []

            question_tokens = _tokenize(question)
            scored = []
            for key in candidate_keys:
                cache_key = f"{bucket_name}/{key}"
                if cache_key not in _s3_doc_cache:
                    try:
                        file_response = s3_client.get_object(Bucket=bucket_name, Key=key)
                        content = file_response['Body'].read().decode('utf-8', errors='replace')
                        _s3_doc_cache[cache_key] = content
                    except Exception as e:
                        _s3_doc_cache[cache_key] = ""

                doc_text = _s3_doc_cache.get(cache_key, "")
                doc_tokens = _tokenize(doc_text)
                score = _score_document(question_tokens, doc_tokens)
                scored.append((score, key))

            scored.sort(key=lambda x: x[0], reverse=True)
            results: List[Dict[str, Any]] = []
            for s, k in scored[:top_n]:
                excerpt = _s3_doc_cache.get(f"{bucket_name}/{k}", "")[:max_chars]
                results.append({"key": k, "score": s, "excerpt": excerpt})

            return results
        except Exception as e:
            return []

    @rx.event
    async def handle_upload(self, files: List[rx.UploadFile]):
        if not files:
            self.upload_error = "Please select a file before uploading."
            return
        bucket_name = os.getenv("S3_BUCKET_NAME")
        object_prefix = "knowledge-based/"
        if not bucket_name:
            self.upload_error = "S3 configuration error. Contact support."
            return
        try:
            file = files[0]  # Only one file due to max_files=1
            clean_filename = file.filename.lstrip("./")  # Remove ./ from filename

            # Classify by extension -> place under knowledge-based/{type}/
            _, ext = os.path.splitext(clean_filename)
            ext = ext.lower().lstrip('.')
            if ext == 'pdf':
                type_prefix = 'knowledge-based/pdf/'
            elif ext in ('mdx', 'md'):
                type_prefix = 'knowledge-based/mdx/'
            elif ext == 'txt':
                type_prefix = 'knowledge-based/txt/'
            else:
                type_prefix = object_prefix  # fallback to generic prefix

            object_name = f"{type_prefix}{clean_filename}"
            
            # Read file content
            content = await file.read()
            
            if not content:
                self.upload_error = "File appears to be empty"
                self.uploading = False
                return

            # Use make_client so endpoint/region selection is consistent
            s3_client = make_client('s3')
            # Try to set a reasonable ContentType based on filename
            content_type, _ = mimetypes.guess_type(clean_filename)
            put_kwargs = dict(Bucket=bucket_name, Key=object_name, Body=content)
            if content_type:
                put_kwargs['ContentType'] = content_type
            s3_client.put_object(**put_kwargs)
            
            self.total_bytes += len(content)
            self.uploaded_files.append(object_name)
            self.uploaded_files = self.uploaded_files
            self.upload_error = ""
            self.uploading = False

            # Trigger background re-indexing for the newly uploaded object so it's quickly searchable
            try:
                from threading import Thread

                def _index_bg(bucket, key):
                    try:
                        from aws_rag_chatbot_ai.chat.indexer import index_single_object
                        index_single_object(bucket, key)
                    except Exception:
                        pass

                Thread(target=_index_bg, args=(bucket_name, object_name), daemon=True).start()
            except Exception:
                pass

            # Ensure the current chat has a session entry so the UI will remain
            # on the same chat after redirect/load. Create or update a minimal
            # session item for the current chat in DynamoDB.
            try:
                session_id = self.session_ids.get(self.current_chat)
                if not session_id:
                    session_id = f"Session#{datetime.now(timezone.utc).isoformat()}Z"
                    self.session_ids[self.current_chat] = session_id

                # Build messages list from in-memory QA objects
                messages = []
                for qa in self.chats.get(self.current_chat, []):
                    # qa may be a pydantic model or dict-like
                    try:
                        q_text = qa.question
                        a_text = qa.answer
                    except Exception:
                        q_text = qa.get('question') if isinstance(qa, dict) else ''
                        a_text = qa.get('answer') if isinstance(qa, dict) else ''
                    messages.append({"question": q_text, "answer": a_text})

                chat_table.put_item(
                    Item={
                        "user_id": self.user_id,
                        "session_id": session_id,
                        "chat_name": self.current_chat,
                        "messages": messages,
                        "isDeleted": False,
                    }
                )
                # Also update last_active metadata so UI remains on this chat after reload
                try:
                    chat_table.put_item(Item={"user_id": self.user_id, "session_id": "meta#last_active", "last_active_chat": self.current_chat})
                except Exception:
                    pass
            except Exception as e:
                    pass

            return rx.redirect("/chat")
        except Exception as e:
            self.upload_error = f"Upload failed: {str(e)}"
            self.uploading = False
            return

    def handle_upload_progress(self, progress: dict):
        """Update progress during upload."""
        self.uploading = True
        self.progress = round(progress["progress"] * 100)
        if self.progress >= 100:
            self.uploading = False
    
    @rx.event
    def cancel_upload(self):
        """Cancel the upload process."""
        self.uploading = False
        self.progress = 0
        self.upload_error = "Upload cancelled."
        return rx.cancel_upload("upload_s3")