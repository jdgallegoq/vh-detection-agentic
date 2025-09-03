"""
MCP Server for Vehicle Detection Tools
"""
import json
import base64
from typing import Optional, Dict, Any
from mcp.server import FastMCP
from mcp.types import TextContent, ImageContent
from pydantic import BaseModel

from llm.client.llm_client import OpenAILLMClient, BedrockLLMClient
from llm.prompt_manager import PromptManager
from dto.agent_response import VerifyImageResponse, VehicleTypeResponse, ReviewImageResponse
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage
from core.settings import settings

# Initialize MCP server
mcp = FastMCP("VehicleDetection")

# Global instances (will be initialized when server starts)
llm_client: Optional[OpenAILLMClient | BedrockLLMClient] = None
prompt_manager: Optional[PromptManager] = None


def _init_components(client_type: str = None):
    """Initialize LLM client and prompt manager"""
    global llm_client, prompt_manager
    if llm_client is None:
        # Use client_type if provided, otherwise use default from settings
        client = client_type or settings.default_llm_client
        if client.lower() == "bedrock":
            llm_client = BedrockLLMClient()
        else:
            llm_client = OpenAILLMClient()
    if prompt_manager is None:
        prompt_manager = PromptManager()


def _invoke_llm_with_image(prompt_name: str, b64_image: str, client_type: str = None, **inputs: Any) -> str:
    """Helper function to invoke LLM with image and prompt"""
    _init_components(client_type)
    
    # Include b64_image in the inputs for the template
    all_inputs = {**inputs, "b64_image": b64_image}
    prompt = prompt_manager.get_prompt(prompt_name).render(**all_inputs)
    
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
    
    response = llm_client.invoke([message])
    return response.content


@mcp.tool()
def configure_llm_client(client_type: str = None) -> Dict[str, Any]:
    """
    Configure which LLM client to use for subsequent operations.
    
    Args:
        client_type: Type of LLM client to use ("openai" or "bedrock")
        
    Returns:
        Configuration status
    """
    global llm_client
    
    try:
        # Reset the client to force reinitialization
        llm_client = None
        
        # Initialize with the new client type
        _init_components(client_type)
        
        actual_client = client_type or settings.default_llm_client
        return {
            "success": True,
            "client_type": actual_client,
            "message": f"LLM client configured to use {actual_client}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to configure LLM client: {str(e)}"
        }


@mcp.tool()
def verify_image(b64_image: str, format_instructions: str = None, client_type: str = None) -> Dict[str, Any]:
    """
    Verify if the provided image contains a vehicle.
    
    Args:
        b64_image: Base64 encoded image data
        format_instructions: Optional format instructions for the response
        client_type: Type of LLM client to use ("openai" or "bedrock")
        
    Returns:
        Dictionary containing verification result
    """
    try:
        parser = PydanticOutputParser(pydantic_object=VerifyImageResponse)
        format_instructions = format_instructions or parser.get_format_instructions()
        
        response_text = _invoke_llm_with_image(
            "verify_image",
            b64_image=b64_image,
            client_type=client_type,
            format_instructions=format_instructions
        )
        
        response = parser.parse(response_text)
        return {
            "is_vehicle": response.is_vehicle,
            "raw_response": response_text
        }
    except Exception as e:
        return {
            "error": f"Failed to verify image: {str(e)}",
            "is_vehicle": False
        }


@mcp.tool()
def get_vehicle_type(b64_image: str, is_vehicle: bool, format_instructions: str = None, client_type: str = None) -> Dict[str, Any]:
    """
    Determine the type of vehicle in the image.
    
    Args:
        b64_image: Base64 encoded image data
        is_vehicle: Whether the image contains a vehicle (from verify_image)
        format_instructions: Optional format instructions for the response
        client_type: Type of LLM client to use ("openai" or "bedrock")
        
    Returns:
        Dictionary containing vehicle type information
    """
    if not is_vehicle:
        return {
            "vehicle_type": None,
            "message": "No vehicle detected in image"
        }
    
    try:
        parser = PydanticOutputParser(pydantic_object=VehicleTypeResponse)
        format_instructions = format_instructions or parser.get_format_instructions()
        
        response_text = _invoke_llm_with_image(
            "vehicle_type",
            b64_image=b64_image,
            client_type=client_type,
            is_vehicle=is_vehicle,
            format_instructions=format_instructions
        )
        
        response = parser.parse(response_text)
        return {
            "vehicle_type": response.vehicle_type.value,
            "raw_response": response_text
        }
    except Exception as e:
        return {
            "error": f"Failed to determine vehicle type: {str(e)}",
            "vehicle_type": None
        }


@mcp.tool()
def review_image(b64_image: str, vehicle_type: str, is_vehicle: bool, format_instructions: str = None, client_type: str = None) -> Dict[str, Any]:
    """
    Provide a detailed review of the vehicle in the image.
    
    Args:
        b64_image: Base64 encoded image data
        vehicle_type: Type of vehicle detected
        is_vehicle: Whether the image contains a vehicle
        format_instructions: Optional format instructions for the response
        client_type: Type of LLM client to use ("openai" or "bedrock")
        
    Returns:
        Dictionary containing detailed review
    """
    if not is_vehicle:
        return {
            "review": "Not a vehicle",
            "message": "No vehicle detected in image"
        }
    
    try:
        parser = PydanticOutputParser(pydantic_object=ReviewImageResponse)
        format_instructions = format_instructions or parser.get_format_instructions()
        
        response_text = _invoke_llm_with_image(
            "review_image",
            b64_image=b64_image,
            client_type=client_type,
            vehicle_type=vehicle_type,
            is_vehicle=is_vehicle,
            format_instructions=format_instructions
        )
        
        response = parser.parse(response_text)
        return {
            "review": response.review,
            "raw_response": response_text
        }
    except Exception as e:
        return {
            "error": f"Failed to review image: {str(e)}",
            "review": "Error occurred during review"
        }


@mcp.tool()
def process_vehicle_image(b64_image: str, client_type: str = None) -> Dict[str, Any]:
    """
    Complete vehicle detection workflow: verify -> classify -> review.
    
    Args:
        b64_image: Base64 encoded image data
        client_type: Type of LLM client to use ("openai" or "bedrock")
        
    Returns:
        Dictionary containing complete analysis results
    """
    try:
        # Step 1: Verify if image contains a vehicle
        verify_result = verify_image(b64_image, client_type=client_type)
        
        if verify_result.get("error"):
            return verify_result
        
        is_vehicle = verify_result.get("is_vehicle", False)
        
        if not is_vehicle:
            return {
                "is_vehicle": False,
                "review": "Not a vehicle",
                "vehicle_type": None,
                "workflow_completed": True
            }
        
        # Step 2: Get vehicle type
        vehicle_type_result = get_vehicle_type(b64_image, is_vehicle, client_type=client_type)
        
        if vehicle_type_result.get("error"):
            return vehicle_type_result
        
        vehicle_type = vehicle_type_result.get("vehicle_type")
        
        if not vehicle_type:
            return {
                "is_vehicle": True,
                "review": "Vehicle detected but type could not be determined",
                "vehicle_type": None,
                "workflow_completed": True
            }
        
        # Step 3: Review the vehicle
        review_result = review_image(b64_image, vehicle_type, is_vehicle, client_type=client_type)
        
        if review_result.get("error"):
            return review_result
        
        return {
            "is_vehicle": is_vehicle,
            "vehicle_type": vehicle_type,
            "review": review_result.get("review"),
            "workflow_completed": True
        }
        
    except Exception as e:
        return {
            "error": f"Workflow failed: {str(e)}",
            "workflow_completed": False
        }


if __name__ == "__main__":
    mcp.run(transport="stdio")
