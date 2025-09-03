"""
MCP-based Vehicle Detection Agent
Replaces the LangGraph implementation with MCP client
"""
import os
import asyncio
import logging
from typing import Dict, Any, Optional
from mcp import ClientSession, StdioServerParameters, stdio_client

from dto.agent_response import ReviewImageResponse


class MCPVehicleAgent:
    """
    Vehicle detection agent using Model Context Protocol (MCP)
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.client_session: Optional[ClientSession] = None

        current_dir = os.path.dirname(__file__)
        server_path = os.path.join(current_dir, "vehicle_detection_server.py")
        self.server_params = StdioServerParameters(
            command="python",
            args=[server_path]
        )
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_session()
    
    async def start_session(self):
        """Initialize MCP client session"""
        try:
            self.client_session = await stdio_client(self.server_params)
            await self.client_session.__aenter__()
            self.logger.info("MCP client session started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start MCP client session: {e}")
            raise
    
    async def close_session(self):
        """Close MCP client session"""
        if self.client_session:
            try:
                await self.client_session.__aexit__(None, None, None)
                self.logger.info("MCP client session closed")
            except Exception as e:
                self.logger.error(f"Error closing MCP client session: {e}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool via MCP client
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool execution result
        """
        if not self.client_session:
            raise RuntimeError("MCP client session not initialized. Use 'async with' or call start_session()")
        
        try:
            result = await self.client_session.call_tool(tool_name, arguments)
            
            if result.isError:
                self.logger.error(f"Tool '{tool_name}' returned error: {result.content}")
                return {"error": f"Tool error: {result.content}"}
            
            # Parse the result content
            if hasattr(result, 'content') and result.content:
                # MCP results are typically TextContent objects
                if hasattr(result.content[0], 'text'):
                    import json
                    return json.loads(result.content[0].text)
                else:
                    return {"result": str(result.content[0])}
            
            return {"result": "Tool executed successfully but returned no content"}
            
        except Exception as e:
            self.logger.error(f"Error calling tool '{tool_name}': {e}")
            return {"error": f"Tool execution failed: {str(e)}"}
    
    async def verify_image(self, b64_image: str) -> Dict[str, Any]:
        """
        Verify if the image contains a vehicle
        
        Args:
            b64_image: Base64 encoded image data
            
        Returns:
            Verification result
        """
        return await self.call_tool("verify_image", {"b64_image": b64_image})
    
    async def get_vehicle_type(self, b64_image: str, is_vehicle: bool) -> Dict[str, Any]:
        """
        Get the type of vehicle in the image
        
        Args:
            b64_image: Base64 encoded image data
            is_vehicle: Whether the image contains a vehicle
            
        Returns:
            Vehicle type result
        """
        return await self.call_tool("get_vehicle_type", {
            "b64_image": b64_image,
            "is_vehicle": is_vehicle
        })
    
    async def review_image(self, b64_image: str, vehicle_type: str, is_vehicle: bool) -> Dict[str, Any]:
        """
        Get a detailed review of the vehicle
        
        Args:
            b64_image: Base64 encoded image data
            vehicle_type: Type of vehicle
            is_vehicle: Whether the image contains a vehicle
            
        Returns:
            Review result
        """
        return await self.call_tool("review_image", {
            "b64_image": b64_image,
            "vehicle_type": vehicle_type,
            "is_vehicle": is_vehicle
        })
    
    async def process_vehicle_image_workflow(self, b64_image: str) -> Dict[str, Any]:
        """
        Complete vehicle detection workflow using individual tools
        This mimics the original LangGraph workflow
        
        Args:
            b64_image: Base64 encoded image data
            
        Returns:
            Complete analysis result
        """
        try:
            # Step 1: Verify image
            self.logger.info("Step 1: Verifying if image contains a vehicle")
            verify_result = await self.verify_image(b64_image)
            
            if verify_result.get("error"):
                return verify_result
            
            is_vehicle = verify_result.get("is_vehicle", False)
            self.logger.info(f"Vehicle verification result: {is_vehicle}")
            
            if not is_vehicle:
                return {
                    "content": ReviewImageResponse(review="Not a vehicle").model_dump()
                }
            
            # Step 2: Get vehicle type
            self.logger.info("Step 2: Determining vehicle type")
            vehicle_type_result = await self.get_vehicle_type(b64_image, is_vehicle)
            
            if vehicle_type_result.get("error"):
                return vehicle_type_result
            
            vehicle_type = vehicle_type_result.get("vehicle_type")
            self.logger.info(f"Vehicle type: {vehicle_type}")
            
            if not vehicle_type:
                return {
                    "content": ReviewImageResponse(review="Vehicle detected but type could not be determined").model_dump()
                }
            
            # Step 3: Review vehicle
            self.logger.info("Step 3: Generating vehicle review")
            review_result = await self.review_image(b64_image, vehicle_type, is_vehicle)
            
            if review_result.get("error"):
                return review_result
            
            review = review_result.get("review", "")
            self.logger.info("Vehicle review completed")
            
            return {
                "content": ReviewImageResponse(review=review).model_dump()
            }
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            return {
                "error": f"Workflow execution failed: {str(e)}"
            }
    
    async def process_vehicle_image_single_tool(self, b64_image: str) -> Dict[str, Any]:
        """
        Complete vehicle detection workflow using the single composite tool
        
        Args:
            b64_image: Base64 encoded image data
            
        Returns:
            Complete analysis result
        """
        result = await self.call_tool("process_vehicle_image", {"b64_image": b64_image})
        
        if result.get("error"):
            return result
        
        if not result.get("is_vehicle"):
            review = result.get("review", "Not a vehicle")
        else:
            review = result.get("review", "")
        
        return {
            "content": ReviewImageResponse(review=review).model_dump()
        }
    
    async def aexecute(self, b64_image: str, use_single_tool: bool = False) -> Dict[str, Any]:
        """
        Execute the vehicle detection workflow
        
        Args:
            b64_image: Base64 encoded image data
            use_single_tool: Whether to use the single composite tool or individual tools
            
        Returns:
            Analysis result compatible with original agent interface
        """
        if use_single_tool:
            return await self.process_vehicle_image_single_tool(b64_image)
        else:
            return await self.process_vehicle_image_workflow(b64_image)


# Factory function for backward compatibility
async def create_mcp_agent(logger: Optional[logging.Logger] = None) -> MCPVehicleAgent:
    """
    Create and initialize an MCP vehicle agent
    
    Args:
        logger: Optional logger instance
        
    Returns:
        Initialized MCP agent
    """
    agent = MCPVehicleAgent(logger)
    await agent.start_session()
    return agent


if __name__ == "__main__":
    import sys
    import os
    
    # Add src to path for imports
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    from utils.utils import preprocess_image
    
    async def test_agent():
        """Test the MCP agent"""
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("MCPAgent")
        
        # Test with an image
        try:
            b64_image = preprocess_image("./images/image1.jpeg")
            
            async with MCPVehicleAgent(logger) as agent:
                # Test individual workflow
                print("Testing individual workflow...")
                result1 = await agent.aexecute(b64_image, use_single_tool=False)
                print("Individual workflow result:", result1)
                
                # Test single tool workflow
                print("\nTesting single tool workflow...")
                result2 = await agent.aexecute(b64_image, use_single_tool=True)
                print("Single tool workflow result:", result2)
                
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Run the test
    asyncio.run(test_agent())
