import reflex as rx
from aws_rag_chatbot_ai.components.version import get_version

def version_badge(version: str | None = None, size: str = "2", color_scheme: str = "indigo") -> rx.Component:
    """Return a reusable version badge component.

    If `version` is not provided, the centralized `get_version()` is used.
    """
    if version is None:
        version = get_version()
    return rx.badge(version, size=size, color_scheme=color_scheme)
