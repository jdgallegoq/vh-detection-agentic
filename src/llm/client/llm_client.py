import boto3
from botocore.config import Config
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock

from core.settings import settings


class OpenAILLMClient:
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

class BedrockLLMClient:
    def __init__(self):
        boto3_session = boto3.Session(
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
        )

        self.bedrock_cli = boto3_session.client(
            service_name="bedrock-runtime",
            config=Config(
                retries={"max_attempts": 3},
                connect_timeout=5,
                read_timeout=30
            )
        )

        self.client = self._init_client()

    def _init_client(self):
        client = ChatBedrock(
            model_id=settings.aws_bedrock_model_id,
            region_name=settings.aws_region,
            client=self.bedrock_cli,
        )
        return client
    
    def invoke(self, prompt: str):
        return self.client.invoke(prompt)

if __name__ == "__main__":
    client = OpenAILLMClient()
    print(client.invoke("What's the weather in Tokyo?"))
