import os
import reflex as rx
import boto3
import logging
import json
from datetime import datetime
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from aws_rag_chatbot_ai.chat.upload_to_s3 import upload_to_s3
from fastapi import UploadFile
import re

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
Initialize boto3 using environment variables when available, and fall back to
LocalStack-friendly defaults. Don't raise on failure to call STS during import
time — fall back to a safe default identity so the module can be imported in
containers that can't reach AWS immediately.
"""
# Prefer explicit env vars but default to LocalStack 'test' credentials for dev
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID') or 'test'
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY') or 'test'
aws_session_token = os.getenv('AWS_SESSION_TOKEN') or None
aws_region = os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-1')
aws_endpoint = os.getenv('AWS_ENDPOINT_URL')

# Configure a default session
boto3.setup_default_session(
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    aws_session_token=aws_session_token,
    region_name=aws_region,
)

# Create DynamoDB resource (use endpoint if provided for LocalStack)
if aws_endpoint:
    dynamodb = boto3.resource('dynamodb', region_name=aws_region, endpoint_url=aws_endpoint)
else:
    dynamodb = boto3.resource('dynamodb', region_name=aws_region)

# Table name can be overridden via env for portability
chat_table_name = os.getenv('CHAT_TABLE_NAME', 'ChatSession')
chat_table = dynamodb.Table(chat_table_name)

# Helper factories so all clients/resources consistently use LocalStack endpoint when set
def make_client(service_name: str, region: str = None, **kwargs):
    region = region or aws_region
    endpoint = os.getenv('AWS_ENDPOINT_URL')
    if endpoint:
        return boto3.client(service_name, region_name=region, endpoint_url=endpoint, **kwargs)
    return boto3.client(service_name, region_name=region, **kwargs)

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

# Determine whether Bedrock calls should be allowed in this environment.
def bedrock_allowed() -> bool:
    """Return False when running against LocalStack or when DISABLE_BEDROCK=1 is set."""
    # Explicit disable takes highest precedence
    if os.getenv("DISABLE_BEDROCK", "0") == "1":
        return False

    # Allow forcing Bedrock (useful for testing against LocalStack)
    if os.getenv("FORCE_BEDROCK", "0") == "1":
        logger.info("FORCE_BEDROCK=1 set — allowing Bedrock calls even when targeting LocalStack")
        return True

    # By default avoid calling Bedrock when the endpoint looks like LocalStack
    endpoint = os.getenv("AWS_ENDPOINT_URL", "").lower()
    if "localstack" in endpoint:
        return False

    return True

# Attempt to fetch caller identity, but tolerate failures in dev/localstack
try:
    if aws_endpoint:
        sts_client = boto3.client('sts', region_name=aws_region, endpoint_url=aws_endpoint)
    else:
        sts_client = boto3.client('sts', region_name=aws_region)

    identity = sts_client.get_caller_identity()
    aws_user_id = identity.get('Arn', f"arn:aws:iam::000000000000:root")
    logger.debug(f"Fetched AWS identity: {identity}")
    logger.debug("Successfully initialized boto3 with static credentials")
except ClientError as e:
    logger.error(f"Failed to initialize boto3 with static credentials: {e}", exc_info=True)
    # Fallback identity for LocalStack / dev environments
    aws_user_id = os.getenv('AWS_USER_ARN', f"arn:aws:iam::000000000000:root")
except Exception as e:
    logger.error(f"Unexpected error while initializing boto3/STS: {e}", exc_info=True)
    aws_user_id = os.getenv('AWS_USER_ARN', f"arn:aws:iam::000000000000:root")

# def create_chat_session_table():
#     try:
#         dynamodb_client = boto3.client('dynamodb')
#         dynamodb_client.describe_table(TableName='ChatSession')
#         logger.debug("Table 'ChatSession' already exists")
#     except dynamodb_client.exceptions.ResourceNotFoundException:
#         dynamodb.create_table(
#             TableName='ChatSession',
#             KeySchema=[
#                 {'AttributeName': 'user_id', 'KeyType': 'HASH'},  # Partition key
#                 {'AttributeName': 'session_id', 'KeyType': 'RANGE'}  # Sort key
#             ],
#             AttributeDefinitions=[
#                 {'AttributeName': 'user_id', 'AttributeType': 'S'},
#                 {'AttributeName': 'session_id', 'AttributeType': 'S'}
#             ],
#             BillingMode='PAY_PER_REQUEST'
#         )
#         dynamodb.meta.client.get_waiter('table_exists').wait(TableName='ChatSession')
#         logger.info("Created 'ChatSession' table in DynamoDB")

# create_chat_session_table()

# class QA(rx.Base):
#     """A question and answer pair."""
#     question: str
#     answer: str

class QA(BaseModel):
    question: str
    answer: str
    source: Optional[str] = None

DEFAULT_CHATS = {
    "Intros": [],
}

class State(rx.State):
    """The app state."""
    state_auto_setters = False
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

    def set_new_chat_name(self, name: str):
        self.new_chat_name = name

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

        session_id = f"Session#{datetime.utcnow().isoformat()}Z"
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
            session_id = f"Intros#{datetime.utcnow().isoformat()}Z"
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
                session_id = f"Intros#{datetime.utcnow().isoformat()}Z"
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
            session_id = f"Intros#{datetime.utcnow().isoformat()}Z"
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
        prompt = (
            "You are a helpful assistant. Use the following information as your knowledge base "
            "to answer the question. If the information below is insufficient, say so and do not "
            "rely on pretrained data.\n\n"
            "Human: Here is the knowledge base:\n"
            f"{knowledge_base}\n\n"
            f"Now, please answer this question: {question}\n\n"
            "Assistant:"
        )

        # Call Bedrock (or LocalStack stub) using helper so endpoint is honored
        if not bedrock_allowed():
            logger.info("Bedrock disabled or running against LocalStack — using local mock response")
            # Use a simple local retrieval to return the most relevant excerpt
            snippet = await self.find_relevant_snippet(question)
            answer = (
                "(Local mock) Bedrock is disabled in this environment. "
                f"Here's the most relevant excerpt I found:\n{snippet}"
            ).strip()
        else:
            try:
                client = make_client("bedrock-runtime", region=os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-1'))
                model_id = "anthropic.claude-v2"
                body = json.dumps({"prompt": prompt, "max_tokens_to_sample": 2000, "temperature": 0.7})

                response = client.invoke_model(
                    modelId=model_id,
                    body=body,
                    contentType="application/json",
                    accept="application/json"
                )
                response_body = json.loads(response["body"].read())
                answer = response_body.get("completion", "").strip()
                logger.info(f"Received answer from Bedrock: {answer[:50]}...")
            except Exception as e:
                logger.error(f"Error calling Bedrock/Runtime: {e}", exc_info=True)
                answer = "Sorry, I encountered an error while processing your request."

        # Update the QA and persist to DynamoDB
        self.chats[self.current_chat][-1].answer = answer
        self.processing = False
        logger.info(f"Updated chat '{self.current_chat}' with answer")

        session_id = self.session_ids.get(self.current_chat)
        if not session_id:
            session_id = f"Session#{datetime.utcnow().isoformat()}Z"
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
                # Don't attempt to initialize Bedrock when running in LocalStack or when disabled
                if not bedrock_allowed():
                    logger.debug("Bedrock disabled in this environment; skipping bedrock client init during load_session")
                session_id = f"Session#{datetime.utcnow().isoformat()}Z"
                self.chats["Intros"] = [QA(question="", answer="")]
                self.session_ids["Intros"] = session_id
                logger.info(f"Created default 'Intros' session for user {self.user_id}")
            else:
                # Load existing sessions
                self.chats = {}
                self.session_ids = {}
                for item in items:
                    chat_name = item["chat_name"]
                    session_id = item["session_id"]
                    messages = item.get("messages", [])
                    # Avoid duplicate chat names by appending session_id if needed
                    unique_chat_name = chat_name if chat_name not in self.chats else f"{chat_name}_{session_id}"
                    self.chats[unique_chat_name] = [QA(question=m["question"], answer=m["answer"]) for m in messages]
                    self.session_ids[unique_chat_name] = session_id
                self.chats = {}
                self.session_ids = {}
                for item in items:
                    chat_name = item["chat_name"]
                    session_id = item["session_id"]
                    messages = item.get("messages", [])
                    # Avoid duplicate chat names by appending session_id if needed
                    unique_chat_name = chat_name if chat_name not in self.chats else f"{chat_name}_{session_id}"
                    self.chats[unique_chat_name] = [QA(question=m["question"], answer=m["answer"]) for m in messages]
                    self.session_ids[unique_chat_name] = session_id
                    logger.debug(f"Loaded chat '{unique_chat_name}' with {len(messages)} messages")
                self.current_chat = list(self.chats.keys())[0]  # Set to first chat
                logger.info(f"Loaded sessions for user {self.user_id}: {list(self.chats.keys())}")

            # Ensure UI updates with loaded data
            self.chats = self.chats
        except ClientError as e:
            logger.error(f"DynamoDB error: {str(e)}")
            self.chats = DEFAULT_CHATS.copy()
            self.current_chat = "Intros"
            self.session_ids = {"Intros": f"Session#{datetime.utcnow().isoformat()}Z"}
        except Exception as e:
            logger.error(f"Unexpected error loading sessions: {str(e)}")
            self.chats = DEFAULT_CHATS.copy()
            self.current_chat = "Intros"
            self.session_ids = {"Intros": f"Session#{datetime.utcnow().isoformat()}Z"}

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
                    content = file_response['Body'].read().decode('utf-8', errors='replace')
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
                return f"(Local mock) Bedrock is disabled in this environment. No relevant files found in bucket {bucket_name}."

            question_tokens = _tokenize(question)
            # Detect date-like tokens in the question to boost documents that contain those strings
            date_matches = re.findall(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b", question)

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
                    return f"(Local mock) Bedrock is disabled in this environment. Found files: {', '.join(candidate_keys[:10])}"
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
    
    async def bedrock_process_question(self, question: str):
        """Get the response from AWS Bedrock using uploaded resources as knowledge base."""
        qa = QA(question=question, answer="")
        self.chats[self.current_chat].append(qa)
        self.processing = True
        yield

        # Fetch knowledge base from uploaded files
        knowledge_base = await self.get_knowledge_base()

        # Build the prompt with the knowledge base
        prompt = (
            "You are a helpful assistant. Use the following information as your knowledge base "
            "to answer the question. If the information below is insufficient, say so and do not "
            "rely on pretrained data.\n\n"
            "Human: Here is the knowledge base:\n"
            f"{knowledge_base}\n\n"
            f"Now, please answer this question: {question}\n\n"
            "Assistant:"
        )

        # If Bedrock is disabled (LocalStack/dev), return a mock answer to keep UI responsive
        if not bedrock_allowed():
            logger.info("Bedrock disabled or running against LocalStack — returning mock answer from bedrock_process_question")
            snippet = await self.find_relevant_snippet(question)
            answer = (
                "(Local mock) Bedrock is disabled in this environment. "
                f"Here's the most relevant excerpt I found:\n{snippet}"
            ).strip()
        else:
            # Initialize the Bedrock Runtime client using make_client so endpoint_url is honored
            client = make_client("bedrock-runtime", region=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'))

            # Define the model ID
            model_id = "anthropic.claude-v2"  # Replace with your desired model ID

            # Prepare the request body
            body = json.dumps({
                "prompt": prompt,
                "max_tokens_to_sample": 2000,
                "temperature": 0.7,
            })

            try:
                # Invoke the Bedrock model
                response = client.invoke_model(
                    modelId=model_id,
                    body=body,
                    contentType="application/json",
                    accept="application/json"
                )
                response_body = json.loads(response["body"].read())
                answer = response_body.get("completion", "").strip()
            except Exception as e:
                logger.error(f"Error calling AWS Bedrock: {e}")
                answer = "Sorry, I encountered an error while processing your request."

        self.chats[self.current_chat][-1].answer = answer
        self.chats = self.chats
        self.processing = False
        yield

    @rx.event
    async def handle_upload(self, files: List[rx.UploadFile]):
        logger.debug("handle_upload called with files: %s", files)
        if not files:
            logger.warning("No files selected for upload.")
            self.upload_error = "Please select a file before uploading."
            return
        bucket_name = os.getenv("S3_BUCKET_NAME")
        object_prefix = "nhqb-cloud-kinetics-bucket/"
        if not bucket_name:
            logger.error("S3_BUCKET_NAME not set in environment variables.")
            self.upload_error = "S3 configuration error. Contact support."
            return
        try:
            file = files[0]  # Only one file due to max_files=1
            clean_filename = file.filename.lstrip("./")  # Remove ./ from filename
            object_name = f"{object_prefix}{clean_filename}"
            logger.debug(f"Uploading to S3 with bucket: {bucket_name}, object_name: {object_name}")
            
            # Read file content
            content = await file.read()
            logger.debug(f"File content length: {len(content)} bytes")  # Log content length
            
            if not content:
                logger.error("File content is empty after reading")
                self.upload_error = "File appears to be empty"
                self.uploading = False
                return

            # Use boto3 directly to upload the content
            s3_client = boto3.client('s3')
            s3_client.put_object(
                Bucket=bucket_name,
                Key=object_name,
                Body=content
            )
            
            self.total_bytes += len(content)
            self.uploaded_files.append(object_name)
            self.uploaded_files = self.uploaded_files  # Trigger state update
            logger.info(f"Successfully uploaded {file.filename} to S3 at {object_name} with {len(content)} bytes")
            self.upload_error = ""
            self.uploading = False
            return rx.redirect("/")  # Redirect on success
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