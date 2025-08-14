from pydantic import BaseModel


class AgentState(BaseModel):
    image: str
    messages: list[str]