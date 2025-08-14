from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = Field(alias="OPENAI_API_KEY", default=None)
    openai_api_version: str = Field(alias="OPENAI_API_VERSION", default=None)
    openai_api_model: str = Field(alias="OPENAI_API_MODEL", default=None)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()