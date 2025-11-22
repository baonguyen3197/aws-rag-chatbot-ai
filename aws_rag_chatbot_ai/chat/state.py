import os
import reflex as rx
import boto3
import logging
import json
from datetime import datetime, timezone
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import re
import mimetypes
from aws_rag_chatbot_ai.chat.prompt_templates import (
    OUTPUT_TEMPLATE,
    RAG_SIM_THRESHOLD,
)
import io

# optional PDF text extractor for better PDF support in KB
try:
    import pdfplumber  # type: ignore
    _HAS_PDFPLUMBER = True
except Exception:
    _HAS_PDFPLUMBER = False

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.debug("Starting state.py execution")
if not load_dotenv():
    logger.error("Failed to load .env file - ensure it exists in the project root")
else:
    logger.debug(f"Environment variables loaded: AWS_REGION={os.getenv('AWS_DEFAULT_REGION')}")

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
# aws_session_token = os.getenv('AWS_SESSION_TOKEN')

# Configure a default session
boto3.setup_default_session(
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    # aws_session_token=aws_session_token,
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
logger.info("Resolved AWS user ARN for app: %s", aws_user_id)

def make_resource(resource_name: str, region: str = None, **kwargs):
    region = region or aws_region
    endpoint = os.getenv('AWS_ENDPOINT_URL')
    if endpoint:
        return boto3.resource(resource_name, region_name=region, endpoint_url=endpoint, **kwargs)
    return boto3.resource(resource_name, region_name=region, **kwargs)

# Simple in-memory cache of S3 documents to avoid re-reading on every question.
# Key: bucket/key -> content string
_s3_doc_cache: Dict[str, str] = {}

def _tokenize(text: str) -> List[str]:
    # Lowercase, remove non-word characters, split on whitespace
    tokens = re.findall(r"\w+", text.lower())
    # Remove very short tokens
    return [t for t in tokens if len(t) > 2]

def _score_document(question_tokens: List[str], doc_tokens: List[str]) -> float:
    if not question_tokens or not doc_tokens:
        return 0.0
    qset = set(question_tokens)
    dset = set(doc_tokens)
    overlap = qset & dset
    # score by overlap / log(len(doc_tokens)+2) to prefer concise matches
    from math import log
    return len(overlap) / (log(len(doc_tokens) + 2))

def _concise_answer_from_snippet(snippet: str, max_sentences: int = 2) -> str:
    """Return a short, sentence-complete answer extracted from a snippet.

    The snippet format is expected to start with a 'File: <name>' line followed by content.
    We extract up to `max_sentences` sentences and append a short source line when available.
    """
    if not snippet:
        return "(No local information found.)"

    # Normalize some common no-results messages returned by the snippet fetcher
    lsnip = snippet.strip().lower()
    if lsnip.startswith("(local mock)") or "no relevant files" in lsnip or "no files found" in lsnip:
        return "(No relevant local documents found.)"

    lines = [l for l in snippet.splitlines()]
    source = None
    content = snippet
    if lines and lines[0].lower().startswith("file:"):
        source = lines[0][5:].strip()
        content = "\n".join(lines[1:]).strip()

    # Split into sentences (simple heuristic) and take up to max_sentences
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
        # Fallback: return a trimmed prefix without cutting mid-word
        head = content[:1000]
        last_space = head.rfind(' ')
        if last_space > 0:
            return head[:last_space].strip() + (f"\n\nSource: {source}" if source else "")
        return head.strip()

    answer = " ".join(chosen)
    if source:
        answer = f"{answer}\n\nSource: {source}"
    return answer


def invoke_bedrock_model(prompt: str) -> str:
    """Invoke Bedrock Runtime synchronously and return the model's textual answer.

    This centralizes model selection, payload shaping and response parsing so
    the main flow in `process_question` stays concise and avoids nested try/except
    mismatches.
    """
    client = make_client("bedrock-runtime", region=os.getenv('AWS_DEFAULT_REGION', aws_region))

    env_model_arn = os.getenv('BEDROCK_MODEL_ARN')
    env_model_id = os.getenv('BEDROCK_MODEL_ID')
    env_inference_profile = os.getenv('BEDROCK_INFERENCE_PROFILE')

    chosen = env_model_arn or env_model_id or env_inference_profile or os.getenv('FALLBACK_MODEL')
    if not chosen:
        chosen = os.getenv('BEDROCK_PREFERRED', '')

    model_id = chosen
    if isinstance(model_id, str) and (model_id.startswith('arn:') or '/' in model_id):
        try:
            model_id = model_id.split('/')[-1]
        except Exception:
            pass

    model_id = (model_id or 'amazon.nova-micro-v1:0').strip()
    logger.info("Invoking Bedrock model: %s (env_model_arn=%s env_model_id=%s inference_profile=%s)", model_id, env_model_arn, env_model_id, env_inference_profile)

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

    logger.debug("Payload body (first 1000 chars): %s", payload[:1000])

    response = client.invoke_model(
        modelId=model_id,
        body=payload,
        contentType="application/json",
        accept="application/json",
    )

    # read response body safely
    raw = b""
    try:
        body_obj = response.get('body') if isinstance(response, dict) else response
        if hasattr(body_obj, 'read'):
            raw = body_obj.read()
        elif isinstance(body_obj, (bytes, str)):
            raw = body_obj if isinstance(body_obj, bytes) else str(body_obj).encode('utf-8', errors='replace')
    except Exception as e_raw:
        logger.debug('Failed reading response body: %s', e_raw)

    decoded = raw.decode('utf-8', errors='replace') if raw else ''
    parsed = {}
    try:
        parsed = json.loads(decoded) if decoded else {}
    except Exception:
        parsed = {}

    # Try common locations for the model output
    answer = ''
    if isinstance(parsed, dict):
        for key in ('completion', 'output', 'result', 'text'):
            v = parsed.get(key)
            if isinstance(v, str) and v.strip():
                answer = v.strip()
                break
        # Anthropic-style outputs under 'messages' or 'outputs'
        if not answer:
            content = parsed.get('outputs') or parsed.get('messages') or parsed.get('content')
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict):
                    # try common nested shapes
                    answer = first.get('text') or first.get('content') or ''
                    if isinstance(answer, list) and answer and isinstance(answer[0], dict):
                        answer = answer[0].get('text', '')
                    if isinstance(answer, str):
                        answer = answer.strip()

    if not answer:
        answer = decoded.strip()

    logger.info('Received answer from Bedrock (len=%d)', len(answer))
    return answer

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
        logger.debug("Table 'ChatSession' already exists")
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
        logger.info("Created 'ChatSession' table in DynamoDB")

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{k: v for k, v in kwargs.items() if k != 'parent_state'})
        self.load_session()

    # Reflex will auto-generate setters by default; newer versions deprecate
    # implicit auto-setters. Define explicit setter to avoid deprecation warnings
    # and future breakage.
    def set_new_chat_name(self, value: str):
        self.new_chat_name = value

    def create_chat(self):
        logger.debug(f"Attempting to create chat with name: {self.new_chat_name}")
        if not self.new_chat_name.strip():
            logger.warning("New chat name is empty")
            return
        chat_name = self.new_chat_name.strip()
        if chat_name in self.chats:
            logger.warning(f"Chat '{chat_name}' already exists")
            return
        self.chats[chat_name] = []
        self.current_chat = chat_name
        self.new_chat_name = ""
        logger.info(f"Created new chat in state: {chat_name}")

        session_id = f"Session#{datetime.now(timezone.utc).isoformat()}Z"
        item = {
            "user_id": self.user_id,
            "session_id": session_id,
            "chat_name": chat_name,
            "messages": []  # Start with empty message list
        }
        try:
            chat_table.put_item(Item=item)
            self.session_ids[chat_name] = session_id # Store session_id for new chat
            logger.info(f"Saved new chat '{chat_name}' to DynamoDB with session_id: {session_id}")
            # Record this as the last active chat for the user
            try:
                chat_table.put_item(Item={"user_id": self.user_id, "session_id": "meta#last_active", "last_active_chat": chat_name})
            except Exception:
                logger.debug("Failed to write last_active_chat metadata during create_chat")
        except Exception as e:
            logger.error(f"Failed to save chat to DynamoDB: {str(e)}", exc_info=True)
            raise

    def delete_chat(self):
        logger.debug(f"Attempting to delete chat: {self.current_chat}")
        if self.current_chat not in self.chats:
            logger.warning(f"Attempted to delete non-existent chat: {self.current_chat}")
            return

        chat_titles = list(self.chats.keys())
        current_index = chat_titles.index(self.current_chat)

        try:
            response = chat_table.query(
                KeyConditionExpression="user_id = :uid AND begins_with(session_id, :sid)",
                ExpressionAttributeValues={":uid": self.user_id, ":sid": self.current_chat}
            )
            logger.debug(f"Query response for deletion: {response}")
            for item in response.get("Items", []):
                if item["chat_name"] == self.current_chat:
                    chat_table.delete_item(
                        Key={"user_id": self.user_id, "session_id": item["session_id"]}
                    )
                    logger.info(f"Deleted chat '{self.current_chat}' from DynamoDB")
                    break
        except Exception as e:
            logger.error(f"Failed to delete chat from DynamoDB: {str(e)}", exc_info=True)

        del self.chats[self.current_chat]
        logger.info(f"Deleted chat from state: {self.current_chat}")

        if not self.chats:
            self.chats = DEFAULT_CHATS.copy()
            self.current_chat = "Intros"
            logger.info("No chats remain, created new default 'Intros'")
            session_id = f"Intros#{datetime.now(timezone.utc).isoformat()}Z"
            try:
                chat_table.put_item(
                    Item={
                        "user_id": self.user_id,
                        "session_id": session_id,
                        "chat_name": "Intros",
                        "chat_history": [],
                    }
                )
                logger.info("Saved default 'Intros' to DynamoDB")
            except Exception as e:
                logger.error(f"Failed to save default 'Intros' to DynamoDB: {str(e)}", exc_info=True)
        else:
            remaining_chats = list(self.chats.keys())
            new_index = min(current_index, len(remaining_chats) - 1) if current_index < len(remaining_chats) else 0
            self.current_chat = remaining_chats[new_index]
            logger.info(f"Switched to chat: {self.current_chat}")

        self.chats = self.chats

    def set_chat(self, chat_name: str):
        logger.debug(f"Attempting to set chat to: {chat_name}")
        if chat_name not in self.chats:
            logger.warning(f"Chat '{chat_name}' does not exist")
            if not self.chats:
                self.chats = DEFAULT_CHATS.copy()
                self.current_chat = "Intros"
                logger.info("Chat history empty, created new default 'Intros'")
                session_id = f"Intros#{datetime.now(timezone.utc).isoformat()}Z"
                try:
                    chat_table.put_item(
                        Item={
                            "user_id": self.user_id,
                            "session_id": session_id,
                            "chat_name": "Intros",
                            "chat_history": [],
                        }
                    )
                    logger.info("Saved default 'Intros' to DynamoDB")
                except Exception as e:
                    logger.error(f"Failed to save default 'Intros' to DynamoDB: {str(e)}", exc_info=True)
            else:
                self.current_chat = list(self.chats.keys())[0]
                logger.info(f"Chat '{chat_name}' deleted or invalid, switched to: {self.current_chat}")
            self.chats = self.chats
            return
        self.current_chat = chat_name
        logger.info(f"Switched to chat: {chat_name}")
        # Persist last active chat selection so reloads keep the same chat
        try:
            chat_table.put_item(Item={"user_id": self.user_id, "session_id": "meta#last_active", "last_active_chat": chat_name})
        except Exception:
            logger.debug("Failed to write last_active_chat metadata during set_chat")

    def reset_session(self):
        logger.debug("Attempting to reset session")
        self.chats = DEFAULT_CHATS.copy()
        self.current_chat = "Intros"
        self.processing = False
        logger.info("Session reset to default state in memory")
        try:
            response = chat_table.scan(FilterExpression="user_id = :uid", ExpressionAttributeValues={":uid": self.user_id})
            logger.debug(f"Scan response for reset: {response}")
            for item in response.get("Items", []):
                chat_table.delete_item(Key={"user_id": self.user_id, "session_id": item["session_id"]})
            session_id = f"Intros#{datetime.now(timezone.utc).isoformat()}Z"
            chat_table.put_item(
                Item={
                    "user_id": self.user_id,
                    "session_id": session_id,
                    "chat_name": "Intros",
                    "chat_history": [],
                }
            )
            logger.info(f"Reset DynamoDB session for user {self.user_id}")
        except Exception as e:
            logger.error(f"Failed to reset DynamoDB session: {str(e)}", exc_info=True)
        self.chats = self.chats

    @rx.var(cache=True)
    def chat_titles(self) -> List[str]:
        titles = list(self.chats.keys())
        logger.debug(f"Chat titles retrieved: {titles}")
        return titles

    async def process_question(self, form_data: Dict[str, Any]):
        """Process a submitted question: call Bedrock (or mock), store result in DynamoDB, update state."""
        logger.debug(f"Processing question: {form_data}")
        question = form_data.get("question", "").strip()
        if not question:
            logger.warning("Question is empty, skipping processing")
            return

        qa = QA(question=question, answer="")
        self.chats.setdefault(self.current_chat, []).append(qa)
        self.processing = True
        logger.info(f"Added question to chat '{self.current_chat}': {question}")
        yield

        knowledge_base = await self.get_knowledge_base()

        # Retrieve top-N relevant snippets from S3 and use them to build the retrieval prompt.
        try:
            top_snips = await self.find_relevant_snippets(question, top_n=3, max_chars=1200)
        except Exception as e:
            logger.error("Error finding relevant snippets: %s", e, exc_info=True)
            top_snips = []

        not_enough_msg = (
            "I don't have enough information in the knowledge base to answer that confidently. "
            "Would you like me to search more, upload a document, or answer from general knowledge (may be less reliable)?"
        )

        skip_model = False
        custom_prompt_template = form_data.get('prompt_template') or form_data.get('prompt')

        if not top_snips:
            logger.info("No candidate snippets found in KB for question: %s", question)
            answer = not_enough_msg
            skip_model = True
        else:
            top_score = top_snips[0].get('score', 0.0)
            logger.debug("Top snippet scores: %s", [s.get('score') for s in top_snips])

            # Build enumerated search results block used in prompts
            search_results_parts = []
            for i, s in enumerate(top_snips, start=1):
                search_results_parts.append(f"{i}. File: {s['key']}\n{s['excerpt']}")
            search_results_text = "\n\n".join(search_results_parts)

            # If user provided a custom prompt template, prefer it. If it contains
            # "$search_results$", substitute the search results. Note: custom
            # templates that omit citation/output instructions will not include the
            # standard `OUTPUT_TEMPLATE`; the caller should include any desired
            # output guidance in their template.
            if custom_prompt_template and isinstance(custom_prompt_template, str):
                logger.info("Using custom prompt template provided by caller")
                prompt = custom_prompt_template.replace("$search_results$", search_results_text).replace("{question}", question)
                # If custom prompt includes $output_format_instructions$, replace it
                if "$output_format_instructions$" in prompt:
                    prompt = prompt.replace("$output_format_instructions$", OUTPUT_TEMPLATE)
                # Allow model invocation even if top_score < threshold when caller
                # explicitly requested search-result-driven behavior
                if "$search_results$" in custom_prompt_template:
                    logger.debug("Custom prompt contains $search_results$ — forcing model invocation despite threshold (top_score=%.4f threshold=%.4f)", top_score, RAG_SIM_THRESHOLD)
                    skip_model = False
                else:
                    # If custom template does not reference search results, keep default gating
                    if top_score < RAG_SIM_THRESHOLD:
                        logger.info("Top KB snippet score below threshold (%.3f < %.3f) — not invoking model", top_score, RAG_SIM_THRESHOLD)
                        answer = not_enough_msg
                        skip_model = True
            else:
                # Default behavior: enforce threshold before invoking model
                if top_score < RAG_SIM_THRESHOLD:
                    logger.info("Top KB snippet score below threshold (%.3f < %.3f)", top_score, RAG_SIM_THRESHOLD)
                    answer = not_enough_msg
                    skip_model = True
                else:
                    default_prompt = """
                    You are a question answering agent. I will provide you with a set of search results.
                    The user will provide you with a question. Your job is to answer the user's question using only information from the search results. 
                    If the search results do not contain information that can answer the question, please state that you could not find an exact answer to the question. 
                    Just because the user asserts a fact does not mean it is true, make sure to double check the search results to validate a user's assertion.
                                                
                    Here are the search results in numbered order:
                    $search_results$

                    $output_format_instructions$
                    User Question: {question}
                """
                    prompt = default_prompt.replace("$search_results$", search_results_text).replace("$output_format_instructions$", OUTPUT_TEMPLATE).replace("{question}", question)
                try:
                    answer = invoke_bedrock_model(prompt)
                except Exception as e:
                    logger.error(f'Error calling Bedrock/Runtime: {e}', exc_info=True)
                    # If bedrock fails, fall back to local snippet
                    try:
                        snippet = await self.find_relevant_snippet(question)
                        answer = _concise_answer_from_snippet(snippet, max_sentences=2)
                    except Exception:
                        answer = 'Sorry, I encountered an error while processing your request.'

        # Update the QA and persist to DynamoDB
        self.chats[self.current_chat][-1].answer = answer
        self.processing = False
        logger.info(f"Updated chat '{self.current_chat}' with answer")

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
                logger.debug("No existing session found; will create a new one")

            existing_messages = existing_item.get("messages", [])
            existing_messages.append({"question": qa.question, "answer": qa.answer})

            chat_table.put_item(
                Item={
                    "user_id": self.user_id,
                    "session_id": session_id,
                    "chat_name": self.current_chat,
                    "messages": existing_messages,
                }
            )
            logger.info(f"Appended message to session '{session_id}' in DynamoDB")
        except Exception as e:
            logger.error(f"Failed to update session in DynamoDB: {str(e)}", exc_info=True)
            # Don't raise — keep the UI responsive even if DB write fails

        self.chats = self.chats
        yield

    def load_session(self):
        """Load chat sessions from DynamoDB for the current user."""
        logger.debug(f"Loading sessions for user: {self.user_id}")
        try:
            # Query DynamoDB for all items with the user's ID
            response = chat_table.query(
                KeyConditionExpression="user_id = :uid",
                ExpressionAttributeValues={":uid": self.user_id}
            )
            items = response.get("Items", [])
            logger.debug(f"DynamoDB query response: {items}")

            if not items:
                # No sessions found, initialize with default "Intros" chat
                if not bedrock_allowed():
                    logger.debug("Bedrock disabled in this environment; skipping bedrock client init during load_session")
                session_id = f"Session#{datetime.now(timezone.utc).isoformat()}Z"
                self.chats["Intros"] = [QA(question="", answer="")]
                self.session_ids["Intros"] = session_id
                logger.info(f"Created default 'Intros' session for user {self.user_id}")
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
                    chat_name = item.get("chat_name", "Intros")
                    session_id = item.get("session_id")
                    messages = item.get("messages", [])
                    unique_chat_name = chat_name if chat_name not in self.chats else f"{chat_name}_{session_id}"
                    self.chats[unique_chat_name] = [QA(question=m.get("question", ""), answer=m.get("answer", "")) for m in messages]
                    self.session_ids[unique_chat_name] = session_id
                    logger.debug(f"Loaded chat '{unique_chat_name}' with {len(messages)} messages")

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

                logger.info(f"Loaded sessions for user {self.user_id}: {list(self.chats.keys())}")

            # Ensure UI updates with loaded data
            self.chats = self.chats
        except ClientError as e:
            logger.error(f"DynamoDB error: {str(e)}")
            self.chats = DEFAULT_CHATS.copy()
            self.current_chat = "Intros"
            self.session_ids = {"Intros": f"Session#{datetime.now(timezone.utc).isoformat()}Z"}
        except Exception as e:
            logger.error(f"Unexpected error loading sessions: {str(e)}")
            self.chats = DEFAULT_CHATS.copy()
            self.current_chat = "Intros"
            self.session_ids = {"Intros": f"Session#{datetime.now(timezone.utc).isoformat()}Z"}

    async def get_knowledge_base(self) -> str:
        """Retrieve content from all files under the specified S3 prefix."""
        # Build S3 client honoring endpoint (prefer helper so endpoint handling is consistent)
        s3_client = make_client('s3', region=os.getenv('AWS_DEFAULT_REGION', aws_region))
        bucket_name = os.getenv("S3_BUCKET_NAME")
        prefix = os.getenv("S3_OBJECT_NAME", "")

        if not bucket_name:
            logger.error("S3_BUCKET_NAME not set in environment variables.")
            return "No S3 bucket configured."

        if not prefix:
            logger.debug("S3_OBJECT_NAME not set; fetching from bucket root.")

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
                except Exception as e:
                    logger.debug(f"list_objects_v2 failed for prefix '{p}': {e}")
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
                            except Exception as e:
                                content = f"[PDF_READ_ERROR] {e}\n" + raw[:400].decode('utf-8', errors='replace')
                        else:
                            content = '[PDF_BINARY_CONTENT] (install pdfplumber to extract text)\n' + raw[:400].decode('utf-8', errors='replace')
                    else:
                        content = raw.decode('utf-8', errors='replace')
                    knowledge_base.append(f"File: {key}\n{content}")
                except Exception as e:
                    logger.error(f"Error fetching {key} from S3: {e}")
        except Exception as e:
            logger.error(f"Error accessing S3 bucket {bucket_name} with prefix {prefix}: {e}")
            return "Error accessing S3 bucket."

        return "\n\n".join(knowledge_base) if knowledge_base else "No knowledge base available."

    async def find_relevant_snippet(self, question: str, max_chars: int = 800) -> str:
        """Find the most relevant S3 document for the question and return an excerpt.

        This is a cheap local fallback when Bedrock is disabled. It lists objects in
        the configured S3 bucket, caches their contents in memory, scores them by
        simple token overlap with the question, and returns a small excerpt from
        the best matching document.
        """
        bucket_name = os.getenv("S3_BUCKET_NAME")
        prefix = os.getenv("S3_OBJECT_NAME", "")
        if not bucket_name:
            logger.error("S3_BUCKET_NAME not set in environment variables.")
            return "No S3 bucket configured."

        # Use the consistent client factory (respects AWS_ENDPOINT_URL)
        s3_client = make_client('s3', region=os.getenv('AWS_DEFAULT_REGION', aws_region))

        # Build candidate key list by trying prefix variants and then a filtered full list
        candidate_keys = []
        tried_prefixes = [p for p in ([prefix, prefix.rstrip('/') + '/'] if prefix else ['']) if p]

        try:
            for p in tried_prefixes:
                try:
                    resp = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=p)
                except Exception as e:
                    logger.debug(f"list_objects_v2 failed for prefix '{p}': {e}")
                    continue
                for obj in resp.get('Contents', []):
                    k = obj.get('Key')
                    if k and k not in candidate_keys:
                        candidate_keys.append(k)

            if not candidate_keys:
                # Fallback: list all and include keys that contain the prefix as substring
                resp = s3_client.list_objects_v2(Bucket=bucket_name)
                for obj in resp.get('Contents', []):
                    k = obj.get('Key')
                    if k and (not prefix or prefix in k) and k not in candidate_keys:
                        candidate_keys.append(k)

            if not candidate_keys:
                return f"No relevant files found in bucket {bucket_name}."

            question_tokens = _tokenize(question)
            # Detect date-like tokens in the question to boost documents that contain those strings
            date_matches = re.findall(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b", question)

            scored = []
            for key in candidate_keys:
                cache_key = f"{bucket_name}/{key}"
                if cache_key not in _s3_doc_cache:
                    try:
                        file_response = s3_client.get_object(Bucket=bucket_name, Key=key)
                        raw = file_response['Body'].read()
                        if isinstance(raw, (bytes, bytearray)) and raw[:4] == b'%PDF':
                            if _HAS_PDFPLUMBER:
                                try:
                                    with pdfplumber.open(io.BytesIO(raw)) as pdf:
                                        pages = [p.extract_text() or '' for p in pdf.pages]
                                        content = '\n\n'.join(pages)
                                except Exception as e:
                                    logger.error(f"PDF extraction error for {key}: {e}")
                                    content = ''
                            else:
                                # store a short decoded prefix so text-search still works somewhat
                                content = raw[:1000].decode('utf-8', errors='replace')
                        else:
                            content = raw.decode('utf-8', errors='replace')
                        _s3_doc_cache[cache_key] = content
                    except Exception as e:
                        logger.error(f"Error fetching {key} from S3 for local retrieval: {e}")
                        _s3_doc_cache[cache_key] = ""

                doc_text = _s3_doc_cache.get(cache_key, "")
                doc_tokens = _tokenize(doc_text)
                base_score = _score_document(question_tokens, doc_tokens)

                # Boost if any detected date token appears verbatim in the document
                boost = 0.0
                for d in date_matches:
                    if d in doc_text:
                        boost += 2.0

                final_score = base_score + boost
                scored.append((final_score, key))

            # Sort descending
            scored.sort(key=lambda x: x[0], reverse=True)

            best_score, best_key = scored[0]

            # If best score is very small, return top-3 summaries instead of a single tiny match
            if best_score < 0.05:
                top_n = [k for s, k in scored[:3] if s > 0]
                if not top_n:
                    # No substantive overlap; return a short bucket listing
                    return f"Found files: {', '.join(candidate_keys[:10])}"
                excerpts = []
                for k in top_n:
                    text = _s3_doc_cache.get(f"{bucket_name}/{k}", "").strip()[:max_chars]
                    excerpts.append(f"File: {k}\n{text}")
                return "\n\n---\n\n".join(excerpts)

            # Return an excerpt from the best document
            excerpt_text = _s3_doc_cache.get(f"{bucket_name}/{best_key}", "").strip()
            excerpt_text = excerpt_text[:max_chars]
            return f"File: {best_key}\n{excerpt_text}"

        except Exception as e:
            logger.error(f"Error during local S3 retrieval: {e}")
            return "Error accessing S3 during local retrieval."

    async def find_relevant_snippets(self, question: str, top_n: int = 3, max_chars: int = 800) -> List[Dict[str, Any]]:
        """Return top_n relevant excerpts from S3 documents with simple scoring.

        Each returned dict contains: key, score, excerpt
        """
        bucket_name = os.getenv("S3_BUCKET_NAME")
        prefix = os.getenv("S3_OBJECT_NAME", "")
        if not bucket_name:
            logger.error("S3_BUCKET_NAME not set in environment variables.")
            return []

        s3_client = make_client('s3', region=os.getenv('AWS_DEFAULT_REGION', aws_region))

        candidate_keys = []
        tried_prefixes = [p for p in ([prefix, prefix.rstrip('/') + '/'] if prefix else ['']) if p]
        try:
            for p in tried_prefixes:
                try:
                    resp = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=p)
                except Exception as e:
                    logger.debug(f"list_objects_v2 failed for prefix '{p}': {e}")
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
                        logger.error(f"Error fetching {key} from S3 for local retrieval: {e}")
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

            logger.debug("find_relevant_snippets results: %s", [(r['key'], r['score']) for r in results])
            return results
        except Exception as e:
            logger.error(f"Error during local S3 retrieval: {e}")
            return []

    @rx.event
    async def handle_upload(self, files: List[rx.UploadFile]):
        logger.debug("handle_upload called with files: %s", files)
        if not files:
            logger.warning("No files selected for upload.")
            self.upload_error = "Please select a file before uploading."
            return
        bucket_name = os.getenv("S3_BUCKET_NAME")
        object_prefix = "knowledge-based/"
        if not bucket_name:
            logger.error("S3_BUCKET_NAME not set in environment variables.")
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
            logger.debug(f"Uploading to S3 with bucket: {bucket_name}, object_name: {object_name}")
            
            # Read file content
            content = await file.read()
            logger.debug(f"File content length: {len(content)} bytes")  # Log content length
            
            if not content:
                logger.error("File content is empty after reading")
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
            logger.info(f"Successfully uploaded {file.filename} to S3 at {object_name} with {len(content)} bytes")
            self.upload_error = ""
            self.uploading = False

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
                    }
                )
                # Also update last_active metadata so UI remains on this chat after reload
                try:
                    chat_table.put_item(Item={"user_id": self.user_id, "session_id": "meta#last_active", "last_active_chat": self.current_chat})
                except Exception:
                    logger.debug("Failed to write last_active_chat metadata during upload")
            except Exception as e:
                logger.error(f"Failed to ensure session after upload: {e}", exc_info=True)

            return rx.redirect("/chat")
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}", exc_info=True)
            self.upload_error = f"Upload failed: {str(e)}"
            self.uploading = False
            return

    def handle_upload_progress(self, progress: dict):
        """Update progress during upload."""
        logger.debug("Upload progress: %s", progress)
        self.uploading = True
        self.progress = round(progress["progress"] * 100)
        if self.progress >= 100:
            self.uploading = False
    
    @rx.event
    def cancel_upload(self):
        """Cancel the upload process."""
        logger.debug("Upload cancelled")
        self.uploading = False
        self.progress = 0
        self.upload_error = "Upload cancelled."
        return rx.cancel_upload("upload_s3")