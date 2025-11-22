import reflex as rx
from aws_rag_chatbot_ai.chat.state import State
from aws_rag_chatbot_ai.components.chat import message, chat, action_bar
from aws_rag_chatbot_ai.components.navbar import navbar

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

color = "rgb(107,99,246)"

def chat_page() -> rx.Component:
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.debug("Rendering chat page")

    return rx.vstack(
        navbar(),
        chat(),
        action_bar(),
        align_items="stretch",
        spacing="0",
        background_color=rx.color("mauve", 1),
        color=rx.color("mauve", 12),
        min_height="100vh",
        on_mount=State.load_session,
    )