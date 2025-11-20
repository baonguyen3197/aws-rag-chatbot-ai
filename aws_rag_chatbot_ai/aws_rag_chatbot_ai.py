"""Welcome to Reflex! This file outlines the steps to create a basic app."""

import reflex as rx
import reflex_chakra as rc

from rxconfig import config
from aws_rag_chatbot_ai.pages.chat_page import chat_page
from aws_rag_chatbot_ai.pages.upload_page import upload_page

class State(rx.State):
    """The app state."""
    state_auto_setters = False

def index() -> rx.Component:
    # Welcome Page (Index)
    return rx.container(
        rx.color_mode.button(position="top-right"),
        rx.vstack(
            rx.heading("Welcome to Reflex!", size="9"),
            rx.text(
                "Get started by ",
                # rx.code(f"{config.app_name}/{config.app_name}.py"),
            rx.link(
                rx.button("Get started!"),
                href="/chat",
                is_external=True,
            ),
                size="5",
            ),
            rx.link(
                rx.button("Check out our docs!"),
                href="https://reflex.dev/docs/getting-started/introduction/",
                is_external=True,
            ),
            spacing="5",
            justify="center",
            min_height="85vh",
        ),
    )

app = rx.App()
app.add_page(index)
app.add_page(chat_page, route="/chat")
app.add_page(upload_page, route="/upload")
