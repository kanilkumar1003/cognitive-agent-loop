import json
import sys
from typing import Any, Optional

from schemas import Goal, DecisionOutput, ToolCall, MemoryItem, DecisionInput, HistoryItem, AttachedArtifact
from gateway import LLM


def next_step(data: DecisionInput) -> DecisionOutput:
    """Analyze current goal, history, memory, and attachments to select the next tool call or emit a final answer."""
    goal = data.goal
    hits = data.hits
    attached = data.attached
    history = data.history
    mcp_tools = data.mcp_tools
    
    # 1. Format attached artifacts contents
    attached_text = ""
    if attached:
        attached_text = "ATTACHED ARTIFACTS:\n"
        for art in attached:
            try:
                content = art.content.decode("utf-8", errors="replace")
            except Exception:
                content = f"[Binary data: {len(art.content)} bytes]"
            attached_text += f"---\nID: {art.artifact_id}\nContents:\n{content}\n---\n"
    else:
        attached_text = "No artifacts attached for this goal."

    # 2. Format memory hits
    hits_text = "RELEVANT MEMORIES:\n"
    if hits:
        for item in hits:
            hits_text += f"- [{item.kind}] descriptor: '{item.descriptor}', value: {json.dumps(item.value)}\n"
    else:
        hits_text += "No relevant memories found."

    # 3. Format run history
    history_text = "EXECUTION HISTORY:\n"
    if history:
        for item in history:
            kind = item.kind
            it = item.iter
            if kind == "action":
                history_text += f"Iteration {it}: Action '{item.tool}' with args {item.arguments} -> Result: {item.result_descriptor}\n"
            elif kind == "answer":
                history_text += f"Iteration {it}: Answer emitted -> \"{item.text}\"\n"
    else:
        history_text += "No history accumulated yet for this run."

    system_prompt = """You are the Decision Layer of an AI agent. Your job is to select the next action to achieve the current goal.

Instructions:
1. Respond with EXACTLY ONE of two outputs:
   - Call one or more tools: select the most appropriate tools from the available tools list to proceed. You may call multiple tools in parallel if their inputs do not depend on each other. Do not add any text explanation or prose outside of the tool call.
   - Return a final answer: if the current goal has been fully met, respond in plain text with the answer. Do not output any tool call.
   Do not do both. Do not narrate your thinking.
2. Artifact handles:
   - Any string starting with 'art:' (e.g. art:3406aa6d) is an internal artifact handle.
   - MCP tools accept only real paths/URLs and WILL fail if you pass an 'art:' handle to them.
   - If you need to read the contents of an artifact, read them from the prompt under 'ATTACHED ARTIFACTS:' where they are provided. Do not invoke read_file/fetch_url on an 'art:' string!
3. Substantive answers:
   - If the goal requires you to output a comparison, extraction, list, or selection, your final answer must be substantive (at least three sentences or a detailed list of items). Avoid short meta-replies (e.g. 'I fetched the page, what next?').
4. Iteration Efficiency:
   - Perform your tasks in as few iterations as possible. Favor parallel tool calls when appropriate.
   - If the task requires processing multiple items (e.g., reading multiple search results or URLs), look at the execution history to identify which items have already been processed. Proceed to the next unprocessed item immediately. NEVER repeat the exact same tool call (e.g., fetching the same URL) unless the prior attempt failed or returned empty content.
   - Short-circuiting: If you have already gathered all the information needed to fully answer the overall user query (even if there are subsequent goals left), you may output the final answer directly. Do not emit intermediate summaries or status updates; go straight to the final answer as soon as you have the data.
"""

    prompt = f"""Current Goal:
{goal.text}

{attached_text}

{hits_text}

{history_text}
"""

    llm = LLM()
    response = None
    
    # Try 1: Auto-routed Decision
    try:
        response = llm.chat(
            prompt=prompt,
            system=system_prompt,
            auto_route="decision",
            tools=mcp_tools,
            tool_choice="auto"
        )
    except Exception as e:
        sys.stderr.write(f"Decision auto-route failed: {e}. Trying Gemini (g) fallback...\n")
        # Try 2: Pinned to Gemini (supports up to 1M context and 250K TPM)
        try:
            response = llm.chat(
                prompt=prompt,
                system=system_prompt,
                provider="g",
                tools=mcp_tools,
                tool_choice="auto"
            )
        except Exception as e2:
            sys.stderr.write(f"Gemini decision fallback failed: {e2}. Trying Groq (gr) fallback...\n")
            # Try 3: Pinned to Groq
            try:
                response = llm.chat(
                    prompt=prompt,
                    system=system_prompt,
                    provider="gr",
                    tools=mcp_tools,
                    tool_choice="auto"
                )
            except Exception as e3:
                sys.stderr.write(f"All decision LLM calls failed: {e3}. Returning offline fallback answer.\n")
                return DecisionOutput(
                    answer="Error: Unable to make a decision due to LLM Gateway connection issues.",
                    tool_calls=[]
                )

    # 4. Parse response into DecisionOutput
    tc_list = response.get("tool_calls") or []
    if tc_list:
        tool_calls = [
            ToolCall(
                name=tc.get("name", ""),
                arguments=tc.get("arguments") or {}
            ) for tc in tc_list
        ]
        return DecisionOutput(answer=None, tool_calls=tool_calls)
    else:
        text = response.get("text", "")
        return DecisionOutput(answer=text, tool_calls=[])
