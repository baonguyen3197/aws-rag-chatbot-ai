import reflex as rx

config = rx.Config(
    app_name="aws_rag_chatbot_ai",
    pages={
        "/": "aws_rag_chatbot_ai.pages.index",
        "/chat": "aws_rag_chatbot_ai.pages.chat.chat_page",
        "/upload": "aws_rag_chatbot_ai.pages.upload_page"
    },
    plugins=[rx.plugins.SitemapPlugin()],
)