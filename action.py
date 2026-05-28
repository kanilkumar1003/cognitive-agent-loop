import json
from typing import Any
from mcp import ClientSession

from schemas import ToolCall, ActionOutput
import artifacts


def _has_art_handle(val: Any) -> bool:
    """Recursively search for any string argument containing an 'art:' handle prefix."""
    if isinstance(val, str):
        return val.startswith("art:")
    if isinstance(val, dict):
        return any(_has_art_handle(v) for v in val.values())
    if isinstance(val, list):
        return any(_has_art_handle(v) for v in val)
    return False


async def execute(
    session: ClientSession,
    tool_call: ToolCall,
) -> ActionOutput:
    """Dispatch an MCP tool execution. Handles artifact handle verification and large payload storage."""
    
    # 1. Guard against model passing artifact ID strings to real file/web tools
    if _has_art_handle(tool_call.arguments):
        error_msg = (
            "Error: Artifact handles (like 'art:...') are internal database identifiers. "
            "You must read their contents from the prompt's 'ATTACHED ARTIFACTS:' section "
            "rather than passing the handle string directly to path or url arguments of tools."
        )
        return ActionOutput(result_descriptor=error_msg, artifact_id=None)

    # 2. Real MCP tool execution
    try:
        result = await session.call_tool(name=tool_call.name, arguments=tool_call.arguments)
    except Exception as e:
        return ActionOutput(result_descriptor=f"Error: MCP tool call failed: {e}", artifact_id=None)

    # 3. Collapse content blocks into a single string
    text_blocks = []
    content_list = getattr(result, "content", []) or []
    for block in content_list:
        # support both Pydantic models (TextContent) and dict formats
        b_type = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else "")
        if b_type == "text":
            b_text = getattr(block, "text", "") or (block.get("text", "") if isinstance(block, dict) else "")
            text_blocks.append(b_text)
            
    collapsed_text = "".join(text_blocks)
    
    # 4. Check payload size threshold
    blob = collapsed_text.encode("utf-8")
    if len(blob) > artifacts.ARTIFACT_THRESHOLD_BYTES:
        # Write large payload to the content-addressable artifact store
        art_id = artifacts.put(
            blob,
            content_type="text/plain",
            source=f"action_execute:{tool_call.name}",
            descriptor=f"Large output payload from tool {tool_call.name}"
        )
        descriptor = f"[artifact {art_id}, {len(blob)} bytes] preview: {collapsed_text[:200]}..."
        return ActionOutput(result_descriptor=descriptor, artifact_id=art_id)
    else:
        # Small payload: return text directly, no artifact
        return ActionOutput(result_descriptor=collapsed_text, artifact_id=None)
