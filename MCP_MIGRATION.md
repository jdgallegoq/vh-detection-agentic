# MCP Migration Guide

This document explains how to migrate from LangGraph to Model Context Protocol (MCP) for the vehicle detection agent.

## Overview

The original agent used LangGraph to orchestrate a workflow with three steps:
1. **verify_image**: Check if image contains a vehicle
2. **get_vehicle_type**: Determine vehicle type (car, motorcycle, bicycle, other)  
3. **review_image**: Generate detailed vehicle review

The MCP implementation provides the same functionality but uses MCP tools instead of LangGraph nodes.

## Files Created for MCP Migration

### 1. MCP Server (`src/mcp/vehicle_detection_server.py`)
- Implements MCP tools for each agent function
- Uses FastMCP for easy tool creation
- Provides both individual tools and a composite workflow tool
- Maintains the same LLM integration as the original

### 2. MCP Agent (`src/mcp/mcp_agent.py`)
- Replaces the LangGraph agent with MCP client
- Supports both individual tool calls and composite workflow
- Maintains the same interface as the original agent
- Provides async context manager for proper resource management

### 3. Main Demo (`src/main_mcp.py`)
- Demonstrates both MCP and original implementations
- Allows comparison of results
- Shows different MCP usage patterns

### 4. Configuration (`mcp_config.json`)
- MCP server configuration
- Tool definitions and transport settings

## Key Differences

| Aspect | LangGraph | MCP |
|--------|-----------|-----|
| **Orchestration** | Graph-based state machine | Tool-based function calls |
| **State Management** | Built-in state passing | Manual parameter passing |
| **Workflow Definition** | Declarative graph edges | Imperative function calls |
| **Tool Isolation** | Methods in single class | Separate MCP server process |
| **Scalability** | Single process | Distributed tools possible |
| **Protocol** | Internal LangChain | Standard MCP protocol |

## Installation

1. Install MCP dependencies:
```bash
pip install -r src/requirements.txt
```

2. Ensure your environment variables are set (same as original):
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_MODEL="gpt-4-vision-preview"  # or your preferred model
export PROMPT_DIR="./llm/prompts"
```

## Usage

### Using the MCP Agent

```python
import asyncio
from src.mcp.mcp_agent import MCPVehicleAgent
from src.utils.utils import preprocess_image

async def analyze_vehicle():
    # Prepare image
    b64_image = preprocess_image("path/to/vehicle/image.jpg")
    
    # Use MCP agent
    async with MCPVehicleAgent() as agent:
        result = await agent.aexecute(b64_image)
        print(result)

# Run the analysis
asyncio.run(analyze_vehicle())
```

### Using Individual Tools

```python
async def analyze_with_individual_tools():
    b64_image = preprocess_image("path/to/vehicle/image.jpg")
    
    async with MCPVehicleAgent() as agent:
        # Step 1: Verify
        verify_result = await agent.verify_image(b64_image)
        
        if verify_result.get("is_vehicle"):
            # Step 2: Get type
            type_result = await agent.get_vehicle_type(
                b64_image, 
                verify_result["is_vehicle"]
            )
            
            # Step 3: Review
            review_result = await agent.review_image(
                b64_image,
                type_result["vehicle_type"],
                verify_result["is_vehicle"]
            )
```

### Running the Demo

```bash
cd /Users/juandiegogallegoquiceno/Desktop/PersonalProjects/vh-detection-agentic
python src/main_mcp.py
```

This will:
1. Test the MCP agent with both workflow approaches
2. Test the original LangGraph agent
3. Compare the results to verify migration accuracy

## MCP Tools Available

### Individual Tools

1. **verify_image(b64_image, format_instructions)**
   - Checks if image contains a vehicle
   - Returns: `{"is_vehicle": bool, "raw_response": str}`

2. **get_vehicle_type(b64_image, is_vehicle, format_instructions)**  
   - Determines vehicle type
   - Returns: `{"vehicle_type": str, "raw_response": str}`

3. **review_image(b64_image, vehicle_type, is_vehicle, format_instructions)**
   - Generates detailed review
   - Returns: `{"review": str, "raw_response": str}`

### Composite Tool

4. **process_vehicle_image(b64_image)**
   - Complete workflow in single tool call
   - Returns: `{"is_vehicle": bool, "vehicle_type": str, "review": str, "workflow_completed": bool}`

## Advantages of MCP Approach

1. **Standardized Protocol**: MCP is an industry standard for AI tool integration
2. **Tool Reusability**: Tools can be used by any MCP-compatible client
3. **Language Agnostic**: MCP tools can be written in any language
4. **Distributed Architecture**: Tools can run on different servers/processes
5. **Better Isolation**: Each tool runs in its own context
6. **Composability**: Tools can be easily combined and reused

## Migration Checklist

- [x] ✅ Create MCP server with equivalent tools
- [x] ✅ Implement MCP client agent
- [x] ✅ Maintain same interface compatibility
- [x] ✅ Update dependencies
- [x] ✅ Create configuration files
- [x] ✅ Add comprehensive testing
- [x] ✅ Document migration process

## Testing

The migration maintains full compatibility with the original interface. Run the demo to verify:

```bash
python src/main_mcp.py
```

Expected output should show equivalent results between LangGraph and MCP implementations.

## Next Steps

1. **Performance Testing**: Compare execution times and resource usage
2. **Error Handling**: Enhance error recovery and retry logic  
3. **Tool Distribution**: Consider running tools on separate servers
4. **Monitoring**: Add tool usage analytics and monitoring
5. **Additional Tools**: Extend with new MCP tools for enhanced functionality

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all paths are correctly set in `PYTHONPATH`
2. **MCP Server Startup**: Check that the server process starts correctly
3. **Tool Communication**: Verify stdio transport is working
4. **API Keys**: Ensure OpenAI API key is properly configured

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show detailed MCP communication and tool execution logs.
