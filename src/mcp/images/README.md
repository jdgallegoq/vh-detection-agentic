# MCP Vehicle Detection - Diagrams

This folder contains visual documentation for the MCP (Model Context Protocol) implementation of the vehicle detection system.

## 📊 Available Diagrams

### 1. [Sequence Diagram](mcp_sequence_diagram.md)
**File**: `mcp_sequence_diagram.md`
**Purpose**: Shows the detailed step-by-step interaction flow between components
**Best for**: Understanding the communication protocol and timing of operations

### 2. [Architecture Diagram](mcp_architecture_diagram.md)
**File**: `mcp_architecture_diagram.md`
**Purpose**: Illustrates the overall system architecture and component relationships
**Best for**: Understanding system structure and data flow

### 3. [Workflow Patterns](mcp_workflow_patterns.md)
**File**: `mcp_workflow_patterns.md`
**Purpose**: Shows different usage patterns and implementation benefits
**Best for**: Understanding how to use the system and its advantages

## 🔧 How to View the Diagrams

### Option 1: Markdown Viewers (Recommended)
Most modern markdown viewers (GitHub, GitLab, VS Code, etc.) can render Mermaid diagrams directly:

1. Open any `.md` file in your preferred markdown viewer
2. The diagrams will render automatically if Mermaid support is enabled

### Option 2: Mermaid Live Editor
1. Copy the Mermaid code from any diagram file
2. Paste it into [Mermaid Live Editor](https://mermaid.live/)
3. View, edit, and export as needed

### Option 3: Generate Images
Use the Mermaid CLI to generate static images:

```bash
# Install Mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# Generate PNG images
mmdc -i mcp_sequence_diagram.md -o sequence_diagram.png
mmdc -i mcp_architecture_diagram.md -o architecture_diagram.png
mmdc -i mcp_workflow_patterns.md -o workflow_patterns.png

# Generate SVG images (scalable)
mmdc -i mcp_sequence_diagram.md -o sequence_diagram.svg
mmdc -i mcp_architecture_diagram.md -o architecture_diagram.svg
mmdc -i mcp_workflow_patterns.md -o workflow_patterns.svg
```

## 📖 Understanding the MCP Architecture

These diagrams document the migration from LangGraph to MCP (Model Context Protocol) for the vehicle detection system. Key concepts:

- **MCP Client**: Orchestrates workflows and manages server communication
- **MCP Server**: Hosts tools and handles requests via JSON-RPC
- **Tools**: Individual functions (verify_image, get_vehicle_type, review_image, process_vehicle_image)
- **STDIO Transport**: Communication channel between client and server
- **Workflow Patterns**: Two approaches for different use cases

## 🔗 Related Files

- `../mcp_agent.py` - MCP client implementation
- `../vehicle_detection_server.py` - MCP server with tools
- `../../main_mcp.py` - Demo and testing script
- `../../mcp_config.json` - MCP server configuration

## 💡 Usage Examples

### Individual Tools Pattern
```python
async with MCPVehicleAgent() as agent:
    verify_result = await agent.verify_image(b64_image)
    if verify_result["is_vehicle"]:
        type_result = await agent.get_vehicle_type(b64_image, True)
        review_result = await agent.review_image(b64_image, type_result["vehicle_type"], True)
```

### Single Tool Pattern
```python
async with MCPVehicleAgent() as agent:
    result = await agent.aexecute(b64_image, use_single_tool=True)
```

## 🎯 Next Steps

1. **Performance Analysis**: Compare MCP vs LangGraph execution times
2. **Tool Distribution**: Consider running tools on separate servers
3. **Monitoring**: Add analytics and performance monitoring
4. **Extension**: Create additional MCP tools for enhanced functionality

For more details, see the main project documentation and MCP migration guide.

