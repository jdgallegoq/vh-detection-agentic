from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from dto.agent_state import AgentState, VehicleType
from llm.prompt_manager import PromptManager
from llm.client import LLMClient
from dto.agent_state import AgentState
from dto.agent_response import VerifyImageResponse, VehicleTypeResponse, ReviewImageResponse


class Agent:
    def __init__(self):
        self.client = LLMClient()
        self.prompt_manager = PromptManager()
        self.graph = self._build_graph()

    def _build_graph(self): -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("verify_image", self._verify_image)
        graph.add_node("get_vehicle_type", self._get_vehicle_type)
        graph.add_node("review_image", self._review_image)
        
        graph.add_edge(START, "verify_image")
        graph.add_conditional_edges(
            "verify_image",
            lambda state: "get_vehicle_type" if state.get("is_vehicle") else END,
            {
                "get_vehicle_type": "get_vehicle_type",
                "END": END
            }
        )
        graph.add_conditional_edges(
            "get_vehicle_type",
            lambda state: "review_image" if state.get("vehicle_type") else END,
            {
                "review_image": "review_image",
                "END": END
            }
        )
        graph.add_edge("review_image", END)
        
        return graph

    def _verify_image(self, state: AgentState):
        parser = PydanticOutputParser(pydantic_object=VerifyImageResponse)
        response = self._ainvoke_client("verify_image", b64_image=state.b64_image)
        response = parser.parse(response["response"])

        return {"is_vehicle": response.is_vehicle}

    def _get_vehicle_type(self, state: AgentState):
        parser = PydanticOutputParser(pydantic_object=VehicleTypeResponse)
        response = self._ainvoke_client("vehicle_type", b64_image=state.b64_image, is_vehicle=state.is_vehicle)
        response = parser.parse(response["response"])

        return {"vehicle_type": response.model_dump()}
    
    def _review_image(self, state: AgentState):
        parser = PydanticOutputParser(pydantic_object=ReviewImageResponse)
        response = self._ainvoke_client(
            "review_image",
            b64_image=state.b64_image,
            vehicle_type=state.vehicle_type,
            is_vehicle=state.is_vehicle
        )
        response = parser.parse(response["response"])
        return {"review": response.review}

    async def _ainvoke_client(self, prompt_name: str, b64_image: str = None, **inputs: any):
        prompt = self._get_prompt(prompt_name, **inputs)
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                }
            ]
        )
        response = await self.client.ainvoke([message])
        return {"response": response.content[0]}


    def _get_prompt(self, prompt_name: str, **inputs: any):
        prompt = self.prompt_manager.get_prompt(prompt_name)
        return prompt.render(**inputs)

    def execute(self, message: str):
        return self.graph.invoke({"messages": [HumanMessage(content=message)]})
