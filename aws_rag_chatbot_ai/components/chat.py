import reflex as rx
import reflex_chakra as rc
import logging
from aws_rag_chatbot_ai.components.loading_icon import loading_icon
from aws_rag_chatbot_ai.chat.state import QA, State
from aws_rag_chatbot_ai.components.navbar import navbar

message_style = dict(display="inline-block", padding="1em", border_radius="8px", max_width=["30em", "30em", "50em", "50em", "50em", "50em"])

def message(qa: QA, idx: int | None = None) -> rx.Component:
    """Render a single QA message. Accepts optional index to use as a stable key."""
    outer_kwargs = {"width": "100%"}
    if idx is not None:
        outer_kwargs["key"] = f"msg-{idx}"

    return rx.box(
        rx.box(
            rx.markdown(qa.question, background_color=rx.color("mauve", 4), color=rx.color("mauve", 12), **message_style),
            text_align="right",
            margin_top="1em",
        ),
        rx.box(
            rx.markdown(qa.answer, background_color=rx.color("accent", 4), color=rx.color("accent", 12), **message_style),
            text_align="left",
            padding_top="1em",
        ),
        **outer_kwargs,
    )

def chat() -> rx.Component:
    return rx.vstack(
        rx.box(rx.foreach(State.chats[State.current_chat], lambda qa, i: message(qa, i)), width="100%"),
        py="8",
        flex="1",
        width="100%",
        max_width="50em",
        padding_x="4px",
        align_self="center",
        overflow="hidden",
        padding_bottom="5em",
    )

def action_bar() -> rx.Component:
    return rx.center(
        rx.vstack(
            rc.form(
                rc.form_control(
                    rx.hstack(
                        rx.input(
                            rx.input.slot(rx.tooltip(rx.icon("info", size=18), content="Enter a question to get a response.")),
                            placeholder="Type something...",
                            id="question",
                            width=["15em", "20em", "45em", "50em", "50em", "50em"],
                        ),
                        rx.button(
                            rx.cond(State.processing, loading_icon(height="1em"), rx.text("Send")),
                            type="submit",
                        ),
                        align_items="center",
                    ),
                    is_disabled=State.processing,
                ),
                on_submit=State.process_question,
                reset_on_submit=True,
            ),
            rx.text("ReflexGPT may return factually incorrect or misleading responses. Use discretion.", text_align="center", font_size=".75em", color=rx.color("mauve", 10)),
            rx.logo(margin_top="-1em", margin_bottom="-1em"),
            align_items="center",
        ),
        position="sticky",
        bottom="0",
        left="0",
        padding_y="16px",
        backdrop_filter="auto",
        backdrop_blur="lg",
        border_top=f"1px solid {rx.color('mauve', 3)}",
        background_color=rx.color("mauve", 2),
        align_items="stretch",
        width="100%",
    )