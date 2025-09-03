#!/usr/bin/env python3
"""
Simple vehicle detection test bypassing MCP complexity
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from utils.utils import preprocess_image
from llm.agent import Agent
from llm.client.llm_client import OpenAILLMClient, BedrockLLMClient
from llm.prompt_manager import PromptManager
from core.settings import settings


async def test_direct_llm_clients():
    """Test both LLM clients directly with the LangGraph agent"""
    print("🔍 TESTING LLM CLIENTS DIRECTLY")
    print("=" * 50)
    
    # Check if image exists
    image_path = "images/image1.jpeg"
    if not Path(image_path).exists():
        print(f"❌ Image {image_path} not found")
        return
    
    # Process image
    print(f"📸 Processing image: {image_path}")
    b64_image = preprocess_image(image_path)
    print(f"✅ Image processed. Size: {len(b64_image)} chars")
    
    # Setup common components
    prompt_manager = PromptManager()
    
    # Test OpenAI client
    print("\n" + "-" * 30)
    print("🤖 TESTING OPENAI CLIENT")
    print("-" * 30)
    try:
        openai_client = OpenAILLMClient()
        logger = logging.getLogger("OpenAI_Agent")
        openai_agent = Agent(openai_client, prompt_manager, logger)
        
        print("Executing OpenAI workflow...")
        openai_result = await openai_agent.aexecute(b64_image)
        print("✅ OpenAI Result:")
        print(f"   {openai_result}")
        
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Bedrock client
    print("\n" + "-" * 30)
    print("🌐 TESTING BEDROCK CLIENT")
    print("-" * 30)
    try:
        bedrock_client = BedrockLLMClient()
        logger = logging.getLogger("Bedrock_Agent")
        bedrock_agent = Agent(bedrock_client, prompt_manager, logger)
        
        print("Executing Bedrock workflow...")
        bedrock_result = await bedrock_agent.aexecute(b64_image)
        print("✅ Bedrock Result:")
        print(f"   {bedrock_result}")
        
    except Exception as e:
        print(f"❌ Bedrock test failed: {e}")
        import traceback
        traceback.print_exc()


def test_llm_client_initialization():
    """Test that LLM clients can be initialized"""
    print("\n🔧 TESTING LLM CLIENT INITIALIZATION")
    print("=" * 50)
    
    try:
        print("Testing OpenAI client initialization...")
        openai_client = OpenAILLMClient()
        print("✅ OpenAI client initialized successfully")
    except Exception as e:
        print(f"❌ OpenAI client failed: {e}")
    
    try:
        print("Testing Bedrock client initialization...")
        bedrock_client = BedrockLLMClient()
        print("✅ Bedrock client initialized successfully")
    except Exception as e:
        print(f"❌ Bedrock client failed: {e}")


async def main():
    """Main test function"""
    print("🚗 VEHICLE DETECTION - SIMPLIFIED TEST")
    print("=" * 50)
    print(f"Default LLM client: {settings.default_llm_client}")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test client initialization
    test_llm_client_initialization()
    
    # Test direct clients
    await test_direct_llm_clients()
    
    print("\n" + "=" * 50)
    print("🎉 SIMPLIFIED TEST COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
