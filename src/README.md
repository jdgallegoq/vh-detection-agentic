# Vehicle Detection Agentic System

A sophisticated AI agent system for vehicle detection and analysis in images, featuring both traditional LangGraph implementation and modern Model Context Protocol (MCP) architecture with support for multiple LLM providers.

## 🚀 Overview

This project demonstrates an intelligent vehicle detection system that can:
- ✅ **Verify** if an image contains a vehicle
- 🚗 **Classify** vehicle types (car, motorcycle, bicycle, other)
- 📝 **Generate** detailed reviews and analysis of detected vehicles

The system showcases two architectural approaches with multiple LLM provider support:
1. **LangGraph Implementation**: Traditional state machine-based workflow
2. **MCP Implementation**: Modern tool-based architecture using Model Context Protocol

**Supported LLM Providers:**
- 🤖 **OpenAI**: GPT-4 Vision models (gpt-4-vision-preview, gpt-4o)
- 🌐 **Amazon Bedrock**: Claude and other vision-capable models

## 📁 Project Structure

```
src/
├── core/
│   └── settings.py              # Configuration management
├── dto/
│   ├── agent_response.py        # Response models (Pydantic)
│   └── agent_state.py          # State management models
├── llm/
│   ├── agent.py                # LangGraph-based agent
│   ├── client/
│   │   └── llm_client.py       # Multi-provider LLM client (OpenAI, Bedrock)
│   ├── prompt_manager.py       # Jinja2 prompt template manager
│   └── prompts/
│       ├── verify_image.j2     # Vehicle verification prompt
│       ├── vehicle_type.j2     # Vehicle classification prompt
│       └── review_image.j2     # Vehicle review prompt
├── mcp/
│   ├── mcp_agent.py           # MCP client agent
│   └── vehicle_detection_server.py  # MCP server with tools
├── utils/
│   └── utils.py               # Image preprocessing utilities
├── images/                    # Test images directory
├── main.py                    # Basic entry point
├── main_mcp.py               # MCP demonstration and comparison
├── simple_mcp_demo.py        # Simplified MCP example
├── requirements.txt          # Python dependencies
└── pyproject.toml           # Project configuration
```

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- **LLM Provider Access** (choose one or both):
  - OpenAI API key (for GPT-4 Vision)
  - AWS credentials with Bedrock access (for Claude Vision)
- PIL/Pillow for image processing

### Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
# OR using uv
uv sync
```

2. **Configure environment:**

**For OpenAI (GPT-4 Vision):**
```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_API_MODEL="gpt-4-vision-preview"  # or gpt-4o
export DEFAULT_LLM_CLIENT="openai"
```

**For Amazon Bedrock (Claude Vision):**
```bash
export AWS_ACCESS_KEY_ID="your-aws-access-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
export AWS_REGION="us-east-1"  # or your preferred region
export AWS_BEDROCK_MODEL_ID="amazon.nova-pro-v1:0"
export DEFAULT_LLM_CLIENT="bedrock"
```

**Common settings:**
```bash
export PROMPT_DIR="./llm/prompts"
```

3. **Add test images:**
```bash
# Place vehicle images in src/images/ directory
# Example: src/images/image1.jpeg
```

## 🚀 Usage

### Quick Start - MCP Agent

```python
import asyncio
from vehicle_mcp.mcp_agent import MCPVehicleAgent
from utils.utils import preprocess_image

async def analyze_vehicle():
    # Preprocess image
    b64_image = preprocess_image("images/car.jpg")
    
    # Use MCP agent with context manager (uses DEFAULT_LLM_CLIENT)
    async with MCPVehicleAgent() as agent:
        result = await agent.aexecute(b64_image)
        print(f"Analysis: {result}")

# Run analysis
asyncio.run(analyze_vehicle())
```

### Choosing LLM Provider

You can specify the LLM provider in multiple ways:

1. **Environment variable (global default):**
   ```bash
   export DEFAULT_LLM_CLIENT="bedrock"  # or "openai"
   ```

2. **Tool configuration (runtime):**
   ```python
   # Configure for current session
   await agent.call_tool("configure_llm_client", {"client_type": "bedrock"})
   ```

3. **Per-tool basis:**
   ```python
   # Use Bedrock for this specific call
   result = await agent.call_tool("verify_image", {
       "b64_image": b64_image,
       "client_type": "bedrock"
   })
   ```

### LangGraph Agent (Traditional)

```python
import asyncio
import logging
from llm.agent import Agent
from llm.client.llm_client import OpenAILLMClient, BedrockLLMClient
from llm.prompt_manager import PromptManager
from utils.utils import preprocess_image

def analyze_with_langgraph(use_bedrock=False):
    # Setup components with chosen LLM provider
    if use_bedrock:
        client = BedrockLLMClient()
    else:
        client = OpenAILLMClient()
        
    prompt_manager = PromptManager()
    logger = logging.getLogger("Agent")
    agent = Agent(client, prompt_manager, logger)
    
    # Process image
    b64_image = preprocess_image("images/car.jpg")
    result = asyncio.run(agent.aexecute(b64_image))
    print(f"Analysis: {result}")

# Use OpenAI
analyze_with_langgraph(use_bedrock=False)

# Use Bedrock
analyze_with_langgraph(use_bedrock=True)
```

### Complete Demo

```bash
# Run comprehensive comparison demo
python main_mcp.py

# Run simplified MCP demo
python simple_mcp_demo.py
```

## 🔧 Architecture Details

### LangGraph Implementation

Uses LangChain's StateGraph for workflow orchestration:

```python
# Workflow: verify_image → get_vehicle_type → review_image
graph = StateGraph(AgentState)
graph.add_node("verify_image", self._verify_image)
graph.add_node("get_vehicle_type", self._get_vehicle_type)
graph.add_node("review_image", self._review_image)
```

**Features:**
- State machine-based execution
- Conditional branching
- Built-in state management
- LangChain integration

### MCP Implementation

Uses Model Context Protocol for tool-based architecture:

```python
# Individual tools
await agent.verify_image(b64_image)
await agent.get_vehicle_type(b64_image, is_vehicle)
await agent.review_image(b64_image, vehicle_type, is_vehicle)

# Or composite workflow
await agent.process_vehicle_image_workflow(b64_image)
```

**Features:**
- Standardized tool protocol
- Distributed tool execution
- Language-agnostic tools
- Better composability
- Tool reusability

## 🔧 Available Tools (MCP)

### LLM Provider Configuration

0. **`configure_llm_client`**
   ```python
   configure_llm_client(client_type: str = None) -> {
       "success": bool,
       "client_type": str,
       "message": str
   }
   ```
   Configure which LLM provider to use ("openai" or "bedrock")

### Individual Tools

1. **`verify_image`**
   ```python
   verify_image(
       b64_image: str,
       format_instructions: str = None,
       client_type: str = None
   ) -> {"is_vehicle": bool}
   ```
   Determines if image contains a vehicle

2. **`get_vehicle_type`**
   ```python
   get_vehicle_type(
       b64_image: str,
       is_vehicle: bool,
       format_instructions: str = None,
       client_type: str = None
   ) -> {"vehicle_type": str}
   ```
   Classifies vehicle type (car, motorcycle, bicycle, other)

3. **`review_image`**
   ```python
   review_image(
       b64_image: str,
       vehicle_type: str,
       is_vehicle: bool,
       format_instructions: str = None,
       client_type: str = None
   ) -> {"review": str}
   ```
   Generates detailed vehicle analysis

### Composite Tool

4. **`process_vehicle_image`**
   ```python
   process_vehicle_image(
       b64_image: str,
       client_type: str = None
   ) -> {
       "is_vehicle": bool,
       "vehicle_type": str,
       "review": str,
       "workflow_completed": bool
   }
   ```
   Complete workflow in a single tool call

**Note**: All tools accept an optional `client_type` parameter to override the default LLM provider for that specific call.

## 📊 Response Models

### VerifyImageResponse
```python
class VerifyImageResponse(BaseModel):
    is_vehicle: bool
```

### VehicleTypeResponse
```python
class VehicleTypeEnum(str, Enum):
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    OTHER = "other"

class VehicleTypeResponse(BaseModel):
    vehicle_type: VehicleTypeEnum
```

### ReviewImageResponse
```python
class ReviewImageResponse(BaseModel):
    review: str
```

## 🎯 Key Features

### Image Preprocessing
- Automatic image resizing (300x300)
- RGB conversion
- Base64 encoding for LLM compatibility

### Prompt Management
- Jinja2 templating system
- Structured prompts for each task
- Format instructions for structured output

### Error Handling
- Comprehensive exception handling
- Graceful fallbacks
- Detailed error reporting

### Logging
- Structured logging throughout
- Debug mode support
- Performance tracking

## 🔄 Migration Guide

This project demonstrates migrating from LangGraph to MCP and supporting multiple LLM providers. See `../MCP_MIGRATION.md` for detailed comparison:

### Architecture Migration

| Aspect | LangGraph | MCP |
|--------|-----------|-----|
| **Orchestration** | Graph-based state machine | Tool-based function calls |
| **State Management** | Built-in state passing | Manual parameter passing |
| **Tool Isolation** | Methods in single class | Separate MCP server process |
| **Scalability** | Single process | Distributed tools possible |
| **Protocol** | Internal LangChain | Standard MCP protocol |

### LLM Provider Migration

To migrate from single-provider to multi-provider setup:

1. **Update imports:**
   ```python
   # Before
   from llm.client.llm_client import OpenAILLMClient
   
   # After
   from llm.client.llm_client import OpenAILLMClient, BedrockLLMClient
   ```

2. **Add provider selection logic:**
   ```python
   # Before
   client = OpenAILLMClient()
   
   # After
   if use_bedrock:
       client = BedrockLLMClient()
   else:
       client = OpenAILLMClient()
   ```

3. **Update environment configuration:**
   ```bash
   # Add to your .env
   DEFAULT_LLM_CLIENT=openai  # or bedrock
   
   # Add Bedrock credentials if using
   AWS_ACCESS_KEY_ID=...
   AWS_SECRET_ACCESS_KEY=...
   AWS_REGION=...
   AWS_BEDROCK_MODEL_ID=...
   ```

## 🧪 Testing

### Running Tests
```bash
# Test both implementations and compare results
python main_mcp.py

# Test individual MCP workflow (includes both LLM providers)
python simple_mcp_demo.py

# Test specific image with OpenAI
python -c "
import asyncio
from vehicle_mcp.mcp_agent import MCPVehicleAgent
from utils.utils import preprocess_image

async def test():
    async with MCPVehicleAgent() as agent:
        img = preprocess_image('images/your_image.jpg')
        result = await agent.aexecute(img)
        print('OpenAI Result:', result)

asyncio.run(test())
"

# Test specific image with Bedrock
python -c "
import asyncio
from vehicle_mcp.mcp_agent import MCPVehicleAgent
from utils.utils import preprocess_image

async def test():
    async with MCPVehicleAgent() as agent:
        # Configure Bedrock client
        await agent.call_tool('configure_llm_client', {'client_type': 'bedrock'})
        
        img = preprocess_image('images/your_image.jpg')
        result = await agent.aexecute(img)
        print('Bedrock Result:', result)

asyncio.run(test())
"
```

### Expected Output Format
```json
{
  "content": {
    "review": "This is a [vehicle_type] with [detailed_analysis]..."
  }
}
```

## 🔧 Configuration

### Environment Variables

**Core Settings:**
```bash
# LLM Provider Selection (choose one)
DEFAULT_LLM_CLIENT=openai        # or "bedrock"
PROMPT_DIR=./llm/prompts
```

**OpenAI Configuration:**
```bash
# Required for OpenAI
OPENAI_API_KEY=your-api-key-here

# Optional (with defaults)
OPENAI_API_MODEL=gpt-4-vision-preview  # or gpt-4o
OPENAI_API_VERSION=latest
```

**Amazon Bedrock Configuration:**
```bash
# Required for Bedrock
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1

# Model selection
AWS_BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
```

### MCP Configuration
See `../mcp_config.json` for MCP server settings.

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the src directory
   cd src
   python main_mcp.py
   ```

2. **Missing API Credentials**
   ```bash
   # For OpenAI
   export OPENAI_API_KEY="your-key"
   
   # For Bedrock
   export AWS_ACCESS_KEY_ID="your-key"
   export AWS_SECRET_ACCESS_KEY="your-secret"
   export AWS_REGION="us-east-1"
   ```

3. **LLM Client Configuration Issues**
   ```bash
   # Check which client is configured
   echo $DEFAULT_LLM_CLIENT
   
   # Test client initialization
   python -c "from llm.client.llm_client import OpenAILLMClient; print('OpenAI OK')"
   python -c "from llm.client.llm_client import BedrockLLMClient; print('Bedrock OK')"
   ```

4. **MCP Server Issues**
   ```bash
   # Check server logs
   python mcp/vehicle_detection_server.py
   ```

5. **Image Processing Errors**
   ```bash
   # Ensure PIL is installed
   pip install Pillow
   ```

6. **AWS/Bedrock Specific Issues**
   ```bash
   # Check AWS credentials
   aws sts get-caller-identity
   
   # List available Bedrock models
   aws bedrock list-foundation-models --region us-east-1
   
   # Test Bedrock access
   aws bedrock-runtime invoke-model --model-id anthropic.claude-3-sonnet-20240229-v1:0 --body '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":10}' output.json --region us-east-1
   ```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit a pull request

## 📝 License

This project is for educational and demonstration purposes.

## 🔗 Related Files

- `../README.md` - Project overview
- `../MCP_MIGRATION.md` - Detailed migration guide
- `../mcp_config.json` - MCP configuration
- `requirements.txt` - Dependencies
- `pyproject.toml` - Project metadata

---

**Note**: This system requires access to an LLM provider with vision capabilities:
- **OpenAI**: GPT-4 Vision models (ensure appropriate API credits and rate limits)
- **Amazon Bedrock**: Claude 3 models with vision support (ensure IAM permissions for Bedrock access)

Choose the provider that best fits your requirements for cost, privacy, and availability.
