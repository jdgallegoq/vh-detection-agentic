from pydantic import BaseModel


class AgentResponse(BaseModel):
    response: str
    image: str | None = None