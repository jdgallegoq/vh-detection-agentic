from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
import logging

from dto.agent_state import AgentState
from llm.prompt_manager import PromptManager
from llm.client.llm_client import LLMClient
from dto.agent_response import VerifyImageResponse, VehicleTypeResponse, ReviewImageResponse


class Agent:
    def __init__(self, client: LLMClient, prompt_manager: PromptManager, logger: logging.Logger):
        self.client = client
        self.prompt_manager = prompt_manager
        self.logger = logger
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("verify_image", self._verify_image)
        graph.add_node("get_vehicle_type", self._get_vehicle_type)
        graph.add_node("review_image", self._review_image)
        
        graph.add_edge(START, "verify_image")
        graph.add_conditional_edges(
            "verify_image",
            lambda state: "get_vehicle_type" if state.get("is_vehicle") else "END",
            {
                "get_vehicle_type": "get_vehicle_type",
                "END": END
            }
        )
        graph.add_conditional_edges(
            "get_vehicle_type",
            lambda state: "review_image" if state.get("vehicle_type") else "END",
            {
                "review_image": "review_image",
                "END": END
            }
        )
        graph.add_edge("review_image", END)
        
        return graph.compile()

    def _verify_image(self, state: AgentState):
        parser = PydanticOutputParser(pydantic_object=VerifyImageResponse)
        response = self._invoke_client(
            "verify_image",
            b64_image=state.get("b64_image"),
            format_instructions=parser.get_format_instructions()
        )
        response = parser.parse(response["response"])

        return {"is_vehicle": response.is_vehicle}

    def _get_vehicle_type(self, state: AgentState):
        parser = PydanticOutputParser(pydantic_object=VehicleTypeResponse)
        response = self._invoke_client(
            "vehicle_type",
            b64_image=state.get("b64_image"),
            is_vehicle=state.get("is_vehicle"),
            format_instructions=parser.get_format_instructions()
        )
        response = parser.parse(response["response"])

        return {"vehicle_type": response.vehicle_type}
    
    def _review_image(self, state: AgentState):
        parser = PydanticOutputParser(pydantic_object=ReviewImageResponse)
        response = self._invoke_client(
            "review_image",
            b64_image=state.get("b64_image"),
            vehicle_type=state.get("vehicle_type"),
            is_vehicle=state.get("is_vehicle"),
            format_instructions=parser.get_format_instructions()
        )
        response = parser.parse(response["response"])
        return {"review": response.review}

    def _invoke_client(self, prompt_name: str, b64_image: str = None, **inputs: any):
        # Include b64_image in the inputs for the template
        all_inputs = {**inputs, "b64_image": b64_image}
        prompt = self._get_prompt(prompt_name, **all_inputs)
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
        response = self.client.invoke([message])
        return {"response": response.content}


    def _get_prompt(self, prompt_name: str, **inputs: any):
        prompt = self.prompt_manager.get_prompt(prompt_name)
        return prompt.render(**inputs)

    async def aexecute(self, b64_image: str):
        response = await self.graph.ainvoke(AgentState(b64_image=b64_image))
        result = None
        if response.get("is_vehicle"):
            result = ReviewImageResponse(review=response.get("review", ""))
            return {"content": result.model_dump() if result else None}
        else:
            return {"content": ReviewImageResponse(review="Not a vehicle").model_dump()}

if __name__ == "__main__":
    import asyncio
    from utils.utils import preprocess_image

    client = LLMClient()
    prompt_manager = PromptManager()
    logger = logging.getLogger("Agent")
    agent = Agent(client, prompt_manager, logger)
    b64_image = preprocess_image("./images/image16.jpeg")
    result = asyncio.run(agent.aexecute(b64_image))
    print(result)
