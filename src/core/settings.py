from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = Field(alias="OPENAI_API_KEY", default=None)
    openai_api_version: str = Field(alias="OPENAI_API_VERSION", default=None)
    openai_api_model: str = Field(alias="OPENAI_API_MODEL", default=None)
    prompt_dir: str = Field(alias="PROMPT_DIR", default="./llm/prompts")
    aws_access_key_id: str = Field(alias="AWS_ACCESS_KEY_ID", default=None)
    aws_secret_access_key: str = Field(alias="AWS_SECRET_ACCESS_KEY", default=None)
    aws_region: str = Field(alias="AWS_REGION", default=None)
    aws_bedrock_model_id: str = Field(alias="AWS_BEDROCK_MODEL_ID", default=None)
    default_llm_client: str = Field(alias="DEFAULT_LLM_CLIENT", default="openai")

    model_config = SettingsConfigDict(env_file="./.env", env_file_encoding="utf-8")


settings = Settings()