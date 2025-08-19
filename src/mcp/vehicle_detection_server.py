"""
MCP Server for Vehicle Detection Tools
"""
import json
import base64
from typing import Optional, Dict, Any
from mcp.server import FastMCP
from mcp.types import TextContent, ImageContent
from pydantic import BaseModel

from llm.client.llm_client import LLMClient
from llm.prompt_manager import PromptManager
from dto.agent_response import VerifyImageResponse, VehicleTypeResponse, ReviewImageResponse
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage

# Initialize MCP server
mcp = FastMCP("VehicleDetection")

# Global instances (will be initialized when server starts)
llm_client: Optional[LLMClient] = None
prompt_manager: Optional[PromptManager] = None


def _init_components():
    """Initialize LLM client and prompt manager"""
    global llm_client, prompt_manager
    if llm_client is None:
        llm_client = LLMClient()
    if prompt_manager is None:
        prompt_manager = PromptManager()


def _invoke_llm_with_image(prompt_name: str, b64_image: str, **inputs: Any) -> str:
    """Helper function to invoke LLM with image and prompt"""
    _init_components()
    
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
def verify_image(b64_image: str, format_instructions: str = None) -> Dict[str, Any]:
    """
    Verify if the provided image contains a vehicle.
    
    Args:
        b64_image: Base64 encoded image data
        format_instructions: Optional format instructions for the response
        
    Returns:
        Dictionary containing verification result
    """
    try:
        parser = PydanticOutputParser(pydantic_object=VerifyImageResponse)
        format_instructions = format_instructions or parser.get_format_instructions()
        
        response_text = _invoke_llm_with_image(
            "verify_image",
            b64_image=b64_image,
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
def get_vehicle_type(b64_image: str, is_vehicle: bool, format_instructions: str = None) -> Dict[str, Any]:
    """
    Determine the type of vehicle in the image.
    
    Args:
        b64_image: Base64 encoded image data
        is_vehicle: Whether the image contains a vehicle (from verify_image)
        format_instructions: Optional format instructions for the response
        
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
def review_image(b64_image: str, vehicle_type: str, is_vehicle: bool, format_instructions: str = None) -> Dict[str, Any]:
    """
    Provide a detailed review of the vehicle in the image.
    
    Args:
        b64_image: Base64 encoded image data
        vehicle_type: Type of vehicle detected
        is_vehicle: Whether the image contains a vehicle
        format_instructions: Optional format instructions for the response
        
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
def process_vehicle_image(b64_image: str) -> Dict[str, Any]:
    """
    Complete vehicle detection workflow: verify -> classify -> review.
    
    Args:
        b64_image: Base64 encoded image data
        
    Returns:
        Dictionary containing complete analysis results
    """
    try:
        # Step 1: Verify if image contains a vehicle
        verify_result = verify_image(b64_image)
        
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
        vehicle_type_result = get_vehicle_type(b64_image, is_vehicle)
        
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
        review_result = review_image(b64_image, vehicle_type, is_vehicle)
        
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
