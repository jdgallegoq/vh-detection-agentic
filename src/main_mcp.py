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
from vehicle_mcp.mcp_agent import MCPVehicleAgent
from llm.agent import Agent
from llm.client.llm_client import OpenAILLMClient, BedrockLLMClient
from llm.prompt_manager import PromptManager
from core.settings import settings


async def test_mcp_agent(image_path: str):
    """Test the MCP-based agent"""
    print("=" * 60)
    print("TESTING MCP-BASED AGENT")
    print("=" * 60)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("MCPAgent")
    
    try:
        # Preprocess image
        print(f"Processing image: {image_path}")
        b64_image = preprocess_image(image_path)
        
        # Test MCP agent
        async with MCPVehicleAgent(logger) as agent:
            print("\n1. Testing individual tool workflow...")
            result1 = await agent.aexecute(b64_image, use_single_tool=False)
            print("Result:", result1)
            
            print("\n2. Testing single composite tool workflow...")
            result2 = await agent.aexecute(b64_image, use_single_tool=True)
            print("Result:", result2)
            
    except Exception as e:
        print(f"MCP Agent test failed: {e}")
        import traceback
        traceback.print_exc()


def test_original_agent(image_path: str, client_type: str = "openai"):
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
        result = asyncio.run(agent.aexecute(b64_image))
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
        print(f"\nMCP Agent ({client_type}):")
        mcp_logger = logging.getLogger(f"MCPAgent_{client_type}")
        async with MCPVehicleAgent(mcp_logger) as mcp_agent:
            mcp_result = await mcp_agent.aexecute(b64_image, use_single_tool=False)
        
        # Compare results
        print("\n" + "-" * 40)
        print("COMPARISON RESULTS:")
        print("-" * 40)
        print(f"Original Agent Result: {original_result}")
        print(f"MCP Agent Result:      {mcp_result}")
        
        # Check if results are equivalent
        original_content = original_result.get("content", {})
        mcp_content = mcp_result.get("content", {})
        
        if original_content.get("review") == mcp_content.get("review"):
            print("✅ Results match! Migration successful.")
        else:
            print("⚠️  Results differ. Check implementation.")
            
    except Exception as e:
        print(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main function"""
    print("Vehicle Detection Agent - MCP Migration Demo")
    print("=" * 60)
    
    # Default image path
    default_image = "src/images/image1.jpeg"
    
    # Check if image exists
    if not os.path.exists(default_image):
        print(f"Warning: Default image {default_image} not found.")
        print("Please ensure you have test images in the src/images/ directory.")
        return
    
    # Run tests
    print(f"Using test image: {default_image}")
    
    # Test MCP agent
    await test_mcp_agent(default_image)
    
    # Test original agent
    test_original_agent(default_image, client_type=settings.default_llm_client)
    
    # Compare both
    await compare_agents(default_image, client_type=settings.default_llm_client)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run main function
    asyncio.run(main())
