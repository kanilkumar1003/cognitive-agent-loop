import os
import sys
import time
import subprocess
from pathlib import Path
from contextlib import asynccontextmanager
import httpx

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Add llm_gatewayV3 directory to sys.path so we can import client
GATEWAY_DIR = Path(__file__).parent / "llm_gatewayV3"
if str(GATEWAY_DIR) not in sys.path:
    sys.path.append(str(GATEWAY_DIR))

try:
    from client import LLM, ask
except ImportError:
    # Inline fallback definition of LLM client just in case
    DEFAULT_URL = os.getenv("LLM_GATEWAY_V3_URL", "http://localhost:8101")
    class LLM:
        def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 600):
            self.base_url = base_url.rstrip("/")
            self.timeout = timeout

        def chat(self, prompt: str = None, *, messages: list = None, system=None,
                 provider: str = None, model: str = None, max_tokens: int = 2048,
                 temperature: float = 0.7, tools: list = None, tool_choice=None,
                 cache_system: bool = None, reasoning: str = None,
                 response_format=None, auto_route: str = None) -> dict:
            body = {
                "prompt": prompt, "messages": messages, "system": system,
                "provider": provider, "model": model, "max_tokens": max_tokens,
                "temperature": temperature, "stream": False, "tools": tools,
                "tool_choice": tool_choice, "cache_system": cache_system,
                "reasoning": reasoning, "response_format": response_format,
                "auto_route": auto_route,
            }
            body = {k: v for k, v in body.items() if v is not None}
            r = httpx.post(f"{self.base_url}/v1/chat", json=body, timeout=self.timeout)
            r.raise_for_status()
            return r.json()

    def ask(prompt: str, provider: str = None, **kw) -> str:
        return LLM().chat(prompt, provider=provider, **kw)["text"]


def ensure_gateway():
    """Verify that the LLM Gateway V3 is running on port 8101. Starts it if not."""
    port = 8101
    url = f"http://localhost:{port}/v1/status"
    try:
        r = httpx.get(url, timeout=1.0)
        if r.status_code == 200:
            return  # already running
    except Exception:
        pass

    # Start the server as a background process
    log_dir = Path(__file__).parent / "misc"
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / "gateway.log"
    
    log_file = open(log_file_path, "a", encoding="utf-8")
    
    # We execute run.sh which handles venv setup and starts the FastAPI server
    env = os.environ.copy()
    subprocess.Popen(
        ["/bin/bash", "run.sh"],
        cwd=GATEWAY_DIR,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env
    )
    
    # Poll until it starts responding
    for _ in range(30):
        try:
            r = httpx.get(url, timeout=1.0)
            if r.status_code == 200:
                return
        except Exception:
            time.sleep(0.5)
            
    raise RuntimeError(f"LLM Gateway V3 failed to start on port {port}")


@asynccontextmanager
async def mcp_session():
    """Asynchronous context manager that starts the local MCP server and yields its ClientSession."""
    server_script = Path(__file__).parent / "mcp_server.py"
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script)],
        env=os.environ.copy()
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def load_tools(session: ClientSession):
    """Fetch registered tools from the MCP session."""
    res = await session.list_tools()
    return res.tools


def mcp_tools_for_decision(mcp_tools: list) -> list[dict]:
    """Convert a list of MCP Tools into the schema expected by the LLM Gateway."""
    decision_tools = []
    for tool in mcp_tools:
        # Extract properties, handling both object attribute access and dictionary access
        name = getattr(tool, "name", None) or tool.get("name", "")
        description = getattr(tool, "description", None) or tool.get("description", "")
        
        input_schema = None
        if hasattr(tool, "inputSchema"):
            input_schema = tool.inputSchema
        elif isinstance(tool, dict) and "inputSchema" in tool:
            input_schema = tool["inputSchema"]
        else:
            input_schema = {}

        decision_tools.append({
            "name": name,
            "description": description,
            "input_schema": input_schema
        })
    return decision_tools
