import reflex as rx
from aws_rag_chatbot_ai.chat.state import State
from aws_rag_chatbot_ai.pages.chat_page import chat_page
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

color = "rgb(107,99,246)"

def upload_page() -> rx.Component:
    """A page for uploading resources to S3."""
    return rx.vstack(
        rx.heading("Upload Resource", size="4"),
        rx.upload(
            rx.text("Drag and drop files here or click to select files"),
            id="upload_s3",
            accept={"text/plain": [".txt"], "application/pdf": [".pdf"], "markdown/md": [".mdx"]},
            max_files=1,
            border=f"1px dotted {color}",
            padding="2em",
            width="100%",
        ),
        rx.vstack(
            rx.text("Selected file:", size="2", color="gray", margin_top="1em"),
            rx.foreach(
                rx.selected_files("upload_s3"),
                lambda file: rx.text(file, size="2", color="green"),
            ),
            align_items="center",
        ),
        rx.progress(value=State.progress, max=100, width="100%", margin_top="1em"),
        rx.cond(
            ~State.uploading,
            rx.hstack(
                rx.button(
                    "Upload",
                    on_click=State.handle_upload(
                        rx.upload_files(
                            upload_id="upload_s3",
                            on_upload_progress=State.handle_upload_progress,
                        )
                    ),
                    color=color,
                    bg="white",
                    border=f"1px solid {color}",
                    margin_top="1em",
                ),
                rx.button(
                    "Clear",
                    on_click=rx.clear_selected_files("upload_s3"),
                    color="gray",
                    bg="white",
                    border=f"1px solid gray",
                    margin_top="1em",
                ),
                spacing="2",
            ),
            rx.button(
                "Cancel",
                on_click=State.cancel_upload,
                color="red",
                bg="white",
                border="1px solid red",
                margin_top="1em",
            ),
        ),
        rx.text(
            "Total bytes uploaded: ",
            State.total_bytes,
            size="2",
            margin_top="1em",
        ),
        rx.cond(
            State.upload_error != "",
            rx.text(
                State.upload_error,
                size="2",
                color="red",
                margin_top="1em",
            ),
            rx.text(""),
        ),
        rx.link("Back to Chat", href="/chat", margin_top="2em"),
        align_items="center",
        spacing="4",
        padding="2em",
        background_color=rx.color("mauve", 1),
        min_height="10vh",
    )
    