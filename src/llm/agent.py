from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from src.llm.client import LLMClient
from src.dto.agent_state import AgentState
from src.dto.agent_response import AgentResponse


class Agent:
    def __init__(self):
        self.client = LLMClient()
        self.graph = self._build_graph()

    def _build_graph(self): -> StateGraph:
        graph = StateGraph(HumanMessage)
        graph.add_node("model", self.client.run)
        graph.add_edge(START, "model")
        graph.add_edge("model", END)
        return graph

    def _verify_image(self, state: AgentState) -> AgentResponse:
        
        return AgentResponse(response="Image is valid")


    def run_graph(self, message: str):
        return self.graph.invoke({"messages": [HumanMessage(content=message)]})