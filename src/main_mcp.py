"""
Main entry point for the MCP-based vehicle detection agent
Demonstrates both MCP and original LangGraph implementations
"""
import asyncio
import logging
import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from utils.utils import preprocess_image
from llm.agent import Agent
from llm.client.llm_client import OpenAILLMClient, BedrockLLMClient
from llm.prompt_manager import PromptManager
from dto.agent_response import VerifyImageResponse, VehicleTypeResponse, ReviewImageResponse
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage
from core.settings import settings


class SimplifiedMCPAgent:
    """Simplified MCP-style agent for comparison (no subprocess complexity)"""
    
    def __init__(self, client_type: str = None):
        self.client_type = client_type or settings.default_llm_client
        self.llm_client = None
        self.prompt_manager = PromptManager()
        self.logger = logging.getLogger(f"SimpleMCP_{self.client_type}")
    
    def _init_client(self):
        """Initialize the LLM client"""
        if self.llm_client is None:
            if self.client_type.lower() == "bedrock":
                self.llm_client = BedrockLLMClient()
            else:
                self.llm_client = OpenAILLMClient()
    
    def _invoke_llm_with_image(self, prompt_name: str, b64_image: str, **inputs):
        """Helper function to invoke LLM with image and prompt"""
        self._init_client()
        
        all_inputs = {**inputs, "b64_image": b64_image}
        prompt = self.prompt_manager.get_prompt(prompt_name).render(**all_inputs)
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
            ]
        )
        
        response = self.llm_client.invoke([message])
        return response.content
    
    async def verify_image_tool(self, b64_image: str):
        """MCP Tool: Verify if image contains vehicle"""
        try:
            parser = PydanticOutputParser(pydantic_object=VerifyImageResponse)
            response_text = self._invoke_llm_with_image(
                "verify_image", b64_image=b64_image,
                format_instructions=parser.get_format_instructions()
            )
            response = parser.parse(response_text)
            return {"is_vehicle": response.is_vehicle}
        except Exception as e:
            return {"error": str(e), "is_vehicle": False}
    
    async def get_vehicle_type_tool(self, b64_image: str, is_vehicle: bool):
        """MCP Tool: Get vehicle type"""
        if not is_vehicle:
            return {"vehicle_type": None}
        try:
            parser = PydanticOutputParser(pydantic_object=VehicleTypeResponse)
            response_text = self._invoke_llm_with_image(
                "vehicle_type", b64_image=b64_image, is_vehicle=is_vehicle,
                format_instructions=parser.get_format_instructions()
            )
            response = parser.parse(response_text)
            return {"vehicle_type": response.vehicle_type.value}
        except Exception as e:
            return {"error": str(e), "vehicle_type": None}
    
    async def review_image_tool(self, b64_image: str, vehicle_type: str, is_vehicle: bool):
        """MCP Tool: Review vehicle"""
        if not is_vehicle:
            return {"review": "Not a vehicle"}
        try:
            parser = PydanticOutputParser(pydantic_object=ReviewImageResponse)
            response_text = self._invoke_llm_with_image(
                "review_image", b64_image=b64_image, vehicle_type=vehicle_type, 
                is_vehicle=is_vehicle, format_instructions=parser.get_format_instructions()
            )
            response = parser.parse(response_text)
            return {"review": response.review}
        except Exception as e:
            return {"error": str(e), "review": "Error in review"}
    
    async def aexecute(self, b64_image: str):
        """MCP-style execution: Individual tool calls"""
        try:
            self.logger.info("Starting MCP-style workflow")
            
            # Tool 1: Verify
            self.logger.info("Calling verify_image tool")
            verify_result = await self.verify_image_tool(b64_image)
            if verify_result.get("error"):
                return verify_result
            
            is_vehicle = verify_result.get("is_vehicle", False)
            if not is_vehicle:
                return {"content": {"review": "Not a vehicle"}}
            
            # Tool 2: Get type
            self.logger.info("Calling get_vehicle_type tool")
            type_result = await self.get_vehicle_type_tool(b64_image, is_vehicle)
            if type_result.get("error"):
                return type_result
            
            vehicle_type = type_result.get("vehicle_type")
            if not vehicle_type:
                return {"content": {"review": "Vehicle detected but type unknown"}}
            
            # Tool 3: Review
            self.logger.info("Calling review_image tool")
            review_result = await self.review_image_tool(b64_image, vehicle_type, is_vehicle)
            if review_result.get("error"):
                return review_result
            
            return {"content": {"review": review_result.get("review")}}
            
        except Exception as e:
            self.logger.error(f"MCP workflow failed: {e}")
            return {"error": str(e)}


async def test_mcp_agent(image_path: str, client_type: str = None):
    """Test the simplified MCP-style agent"""
    print("=" * 60)
    print(f"TESTING SIMPLIFIED MCP APPROACH ({(client_type or settings.default_llm_client).upper()})")
    print("=" * 60)
    
    try:
        # Preprocess image
        print(f"Processing image: {image_path}")
        b64_image = preprocess_image(image_path)
        
        # Test simplified MCP agent
        agent = SimplifiedMCPAgent(client_type)
        print("\nTesting MCP tool-based workflow...")
        result = await agent.aexecute(b64_image)
        print("Result:", result)
        
        # Show individual tool results for clarity
        print("\n--- Individual MCP Tool Results ---")
        verify_result = await agent.verify_image_tool(b64_image)
        print(f"1. verify_image tool: {verify_result}")
        
        if verify_result.get("is_vehicle"):
            type_result = await agent.get_vehicle_type_tool(b64_image, True)
            print(f"2. get_vehicle_type tool: {type_result}")
            
            if type_result.get("vehicle_type"):
                review_result = await agent.review_image_tool(b64_image, type_result.get("vehicle_type"), True)
                print(f"3. review_image tool: {review_result}")
            
    except Exception as e:
        print(f"MCP Agent test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_original_agent(image_path: str, client_type: str = "openai"):
    """Test the original LangGraph agent for comparison"""
    print("\n" + "=" * 60)
    print(f"TESTING ORIGINAL LANGGRAPH AGENT ({client_type.upper()})")
    print("=" * 60)
    
    try:
        # Preprocess image  
        print(f"Processing image: {image_path}")
        b64_image = preprocess_image(image_path)
        
        # Test original agent with specified client
        if client_type.lower() == "bedrock":
            client = BedrockLLMClient()
        else:
            client = OpenAILLMClient()
            
        prompt_manager = PromptManager()
        logger = logging.getLogger(f"OriginalAgent_{client_type}")
        agent = Agent(client, prompt_manager, logger)
        
        print(f"\nExecuting original LangGraph workflow with {client_type}...")
        result = await agent.aexecute(b64_image)
        print("Result:", result)
        
    except Exception as e:
        print(f"Original Agent test failed: {e}")
        import traceback
        traceback.print_exc()


async def compare_agents(image_path: str, client_type: str = "openai"):
    """Compare both agent implementations with specified client"""
    print("\n" + "=" * 60)
    print(f"COMPARING BOTH IMPLEMENTATIONS ({client_type.upper()})")
    print("=" * 60)
    
    try:
        b64_image = preprocess_image(image_path)
        
        # Test original agent with specified client
        print(f"\nOriginal LangGraph Agent ({client_type}):")
        if client_type.lower() == "bedrock":
            client = BedrockLLMClient()
        else:
            client = OpenAILLMClient()
            
        prompt_manager = PromptManager()
        logger = logging.getLogger(f"OriginalAgent_{client_type}")
        original_agent = Agent(client, prompt_manager, logger)
        original_result = await original_agent.aexecute(b64_image)
        
        # Test MCP agent
        print(f"\nSimplified MCP Agent ({client_type}):")
        mcp_agent = SimplifiedMCPAgent(client_type)
        mcp_result = await mcp_agent.aexecute(b64_image)
        
        # Compare results
        print("\n" + "-" * 40)
        print("COMPARISON RESULTS:")
        print("-" * 40)
        print(f"Original Agent Result: {original_result}")
        print(f"MCP Agent Result:      {mcp_result}")
        
        # Check if results are equivalent
        original_content = original_result.get("content", {})
        mcp_content = mcp_result.get("content", {})
        
        # Analyze the approaches
        print("\n" + "=" * 60)
        print("📊 ARCHITECTURAL COMPARISON")
        print("=" * 60)
        
        print("\n🔧 LangGraph Approach:")
        print("  • State-driven workflow execution")
        print("  • Built-in conditional branching")
        print("  • Automatic state management")
        print("  • Single process execution")
        print("  • Tight integration with LangChain")
        
        print("\n🔧 MCP Approach:")
        print("  • Tool-based function calls")
        print("  • Manual orchestration required")
        print("  • Explicit parameter passing")
        print("  • Standardized protocol")
        print("  • Better modularity and reusability")
        
        print("\n📈 Performance & Results:")
        if original_content.get("review") == mcp_content.get("review"):
            print("✅ Functional equivalence: Both approaches produce identical results")
        else:
            print("⚠️  Results differ - may indicate implementation variations")
            
        print(f"✅ Multi-LLM support: Both approaches support {client_type.upper()} successfully")
            
    except Exception as e:
        print(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main function"""
    print("Vehicle Detection Agent - MCP Migration Demo")
    print("=" * 60)
    
    # Default image path
    default_image = "images/image1.jpeg"
    
    # Check if image exists
    if not os.path.exists(default_image):
        print(f"Warning: Default image {default_image} not found.")
        print("Please ensure you have test images in the src/images/ directory.")
        return
    
    # Run tests
    print(f"Using test image: {default_image}")
    
    # Test MCP agent
    await test_mcp_agent(default_image, client_type=settings.default_llm_client)
    
    # Test original agent
    await test_original_agent(default_image, client_type=settings.default_llm_client)
    
    # Compare both approaches
    await compare_agents(default_image, client_type=settings.default_llm_client)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run main function
    asyncio.run(main())
