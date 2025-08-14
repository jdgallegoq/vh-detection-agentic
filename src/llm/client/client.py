from langchain_openai import ChatOpenAI

from src.core.settings import settings


class LLMClient:
    def __init__(self):
        self.client = self.init_client()

    def init_client(self):
        self.client = ChatOpenAI(
            api_key=settings.openai_api_key,
            api_version=settings.openai_api_version,
            model=settings.openai_api_model,
        )
        return self.client
