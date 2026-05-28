import uuid
import asyncio
import json
import datetime
import sys
from pathlib import Path

from schemas import (
    Goal, HistoryItem, AttachedArtifact, PerceptionInput, DecisionInput,
    RememberInput, RecordOutcomeInput, ReadInput
)
from gateway import ensure_gateway, mcp_session, load_tools, mcp_tools_for_decision

import memory
import perception
import artifacts
import decision
import action

# ANSI styles for terminal output
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"

MAX_ITERATIONS = 10


def final_answer_from(history: list[HistoryItem]) -> str:
    """Scan history from the end to find the final answer returned by the agent."""
    for item in reversed(history):
        if item.kind == "answer":
            return item.text or "Empty answer text."
    return "No final answer was provided by the decision layer."


def save_run_log(run_id: str, query: str, history: list[HistoryItem], goals: list[Goal], final_answer: str):
    """Save a structured execution log in Markdown format under misc/ directory."""
    log_dir = Path(__file__).parent / "misc"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{timestamp}_{run_id}.md"
    
    lines = []
    lines.append(f"# Agent Run Log - {run_id}")
    lines.append(f"- **Timestamp**: {datetime.datetime.now().isoformat()}")
    lines.append(f"- **Query**: {query}")
    lines.append("")
    lines.append("## Decomposed Goals")
    for g in goals:
        lines.append(f"- [{ 'x' if g.done else ' ' }] Goal {g.id}: {g.text}")
    lines.append("")
    lines.append("## Execution History")
    for item in history:
        kind = item.kind
        it = item.iter
        goal_id = item.goal_id
        if kind == "action":
            lines.append(f"### Iteration {it} - Action (Goal {goal_id})")
            lines.append(f"- **Tool**: `{item.tool}`")
            lines.append("- **Arguments**:")
            lines.append(f"  ```json\n  {json.dumps(item.arguments, indent=2)}\n  ```")
            lines.append(f"- **Outcome Summary**: {item.result_descriptor}")
            if item.artifact_id:
                lines.append(f"- **Artifact Saved**: `{item.artifact_id}`")
        elif kind == "answer":
            lines.append(f"### Iteration {it} - Intermediate Answer (Goal {goal_id})")
            lines.append(f"```text\n{item.text}\n```")
        lines.append("")
        
    lines.append("## Final Answer")
    lines.append(f"```text\n{final_answer}\n```")
    
    log_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{BOLD}{CYAN}Structured log saved to {log_file}{RESET}")


async def run(query: str, run_id: str = None) -> str:
    # 1. Ensure gateway is up
    ensure_gateway()

    if not run_id:
        run_id = uuid.uuid4().hex[:8]

    print(f"\n{BOLD}{BLUE}==================================================")
    print(f"🚀 AGENT RUN START (ID: {run_id})")
    print(f"==================================================")
    print(f"Query: {query}{RESET}\n")

    history: list[HistoryItem] = []
    prior_goals: list[Goal] = []

    # Durable memory: classify the user's query so facts/preferences
    # in it survive into future runs.
    print(f"{CYAN}Initializing user query in memory...{RESET}")
    memory.remember(RememberInput(raw_text=query, source="user_query", run_id=run_id))

    async with mcp_session() as session:
        mcp_tools = await load_tools(session)
        tools = mcp_tools_for_decision(mcp_tools)

        for it in range(1, MAX_ITERATIONS + 1):
            print(f"\n{BOLD}{BLUE}-------------------- ITERATION {it} --------------------{RESET}\n")
            hits = memory.read(ReadInput(query=query, history=history))
            print(f"{CYAN}Memory Read: Found {len(hits)} hits.{RESET}\n")

            obs = perception.observe(
                PerceptionInput(
                    query=query,
                    hits=hits,
                    history=history,
                    prior_goals=prior_goals,
                    run_id=run_id
                )
            )
            prior_goals = obs.goals

            print(f"{BOLD}{MAGENTA}--- Goals Status ---{RESET}")
            for g in obs.goals:
                status_icon = f"{GREEN}[✓]{RESET}" if g.done else f"{YELLOW}[ ]{RESET}"
                attach_info = f" {CYAN}(Attached: {g.attach_artifact_id}){RESET}" if g.attach_artifact_id else ""
                print(f" {status_icon} Goal {g.id}: {g.text}{attach_info}")

            if obs.all_done:
                print(f"\n{BOLD}{GREEN}✔ Perception reports all goals are DONE! Breaking loop.{RESET}")
                break

            goal = obs.next_unfinished()
            print(f"\n{BOLD}{CYAN}➔ Executing Goal {goal.id}:{RESET} {goal.text}")
            
            attached = []
            if goal.attach_artifact_id and artifacts.exists(goal.attach_artifact_id):
                blob = artifacts.get_bytes(goal.attach_artifact_id)
                attached.append(AttachedArtifact(artifact_id=goal.attach_artifact_id, content=blob))
                print(f"{CYAN}Loaded Attached Artifact: {goal.attach_artifact_id} ({len(blob)} bytes){RESET}")

            print(f"\n{CYAN}Invoking Decision...{RESET}")
            out = decision.next_step(
                DecisionInput(
                    goal=goal,
                    hits=hits,
                    attached=attached,
                    history=history,
                    mcp_tools=tools
                )
            )

            if out.is_answer:
                print(f"\n{BOLD}{GREEN}💡 Decision: EMIT ANSWER{RESET}")
                print(f"{GREEN}Answer Text:{RESET}\n{out.answer}")
                history.append(
                    HistoryItem(
                        iter=it,
                        kind="answer",
                        goal_id=goal.id,
                        text=out.answer
                    )
                )
                for g in prior_goals:
                    g.done = True
                break

            print(f"\n{BOLD}{YELLOW}🛠 Decision: CALL TOOLS ({len(out.tool_calls)} tool(s)){RESET}")
            
            # Concurrently execute tools
            tasks = [action.execute(session, tc) for tc in out.tool_calls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for tc, res in zip(out.tool_calls, results):
                print(f" - {BOLD}Tool{RESET}: {tc.name}")
                print(f" - {BOLD}Args{RESET}: {json.dumps(tc.arguments, indent=2)}")
                
                print(f"\n{CYAN}Executing Action...{RESET}")
                if isinstance(res, Exception):
                    result_text = f"Error executing tool: {res}"
                    art_id = None
                else:
                    result_text = res.result_descriptor
                    art_id = res.artifact_id

                print(f"{BOLD}{GREEN}✔ Action Outcome:{RESET}")
                print(f" - Summary: {result_text[:200]}...")
                if art_id:
                    print(f" - Artifact Saved: {BOLD}{art_id}{RESET}")

                memory.record_outcome(
                    RecordOutcomeInput(
                        tool_call=tc,
                        result_text=result_text,
                        artifact_id=art_id,
                        run_id=run_id,
                        goal_id=goal.id
                    )
                )
                history.append(
                    HistoryItem(
                        iter=it,
                        kind="action",
                        goal_id=goal.id,
                        tool=tc.name,
                        arguments=tc.arguments,
                        result_descriptor=result_text[:300],
                        artifact_id=art_id
                    )
                )
                print(f"\n")

    final_ans = final_answer_from(history)
    
    # Save structured run log in misc/
    save_run_log(run_id, query, history, prior_goals, final_ans)

    return final_ans


if __name__ == "__main__":
    # Check if a query is provided as a command-line argument
    if len(sys.argv) > 1:
        if sys.argv[1] == "--clean":
            print("Cleaning memory and artifacts...")
            memory.clear()
            artifacts.clear()
            # Also clean sandbox directory files except mom_birthday.txt / may_first_reminder.txt if needed
            # (but state/ directory is the primary location for memory persistence)
            print("Clean complete.")
            sys.exit(0)
        query = " ".join(sys.argv[1:])
    else:
        # Prompt the user for input if no argument was passed
        print(f"{BOLD}{BLUE}==================================================")
        print("ANTIGRAVITY AI AGENT SHELL")
        print(f"=================================================={RESET}")
        try:
            query = input(f"{BOLD}Enter your query/task or '--clean':{RESET}\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)
            
    if not query:
        print(f"{RED}Error: Empty query. Exiting.{RESET}")
        sys.exit(1)
        
    if query == "--clean":
        print("Cleaning memory and artifacts...")
        memory.clear()
        artifacts.clear()
        print("Clean complete.")
        sys.exit(0)

    final_answer = asyncio.run(run(query))
    print(f"\n{BOLD}{GREEN}==================================================")
    print("FINAL ANSWER")
    print(f"=================================================={RESET}")
    print(final_answer)
    print(f"{BOLD}{GREEN}=================================================={RESET}")