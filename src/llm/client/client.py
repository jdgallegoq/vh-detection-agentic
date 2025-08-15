from langchain_openai import ChatOpenAI

from core.settings import settings


class LLMClient:
    def __init__(self):
        self.client = self._init_client()

    def _init_client(self):
        client = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_api_model,
        )
        return client

    def invoke(self, prompt: str):
        return self.client.invoke(prompt)


if __name__ == "__main__":
    client = LLMClient()
    print(client.invoke("What's the weather in Tokyo?"))
