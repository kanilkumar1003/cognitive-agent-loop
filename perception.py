import json
import sys
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

from schemas import Goal, Observation, MemoryItem, PerceptionInput, HistoryItem
from gateway import LLM


class PerceptionGoalOutput(BaseModel):
    text: str
    done: bool
    artifact_index: Optional[int] = Field(
        default=None,
        description="0-based integer index of the artifact in the provided artifacts list, if this goal needs it attached."
    )


class PerceptionObservationOutput(BaseModel):
    goals: list[PerceptionGoalOutput]


def observe(data: PerceptionInput) -> Observation:
    """Analyze query, history, memory hits, and prior goals to produce an updated Observation."""
    query = data.query
    hits = data.hits
    history = data.history
    prior_goals = data.prior_goals
    run_id = data.run_id
    
    # 1. Filter memory hits that carry artifacts
    artifact_hits = [h for h in hits if h.artifact_id is not None]
    
    # 2. Format artifacts list with indices for the prompt
    artifacts_text = ""
    if artifact_hits:
        artifacts_text = "Available Artifacts in Memory:\n"
        for idx, art_hit in enumerate(artifact_hits):
            artifacts_text += f"[Index {idx}]: Descriptor: '{art_hit.descriptor}', Handle: '{art_hit.artifact_id}'\n"
    else:
        artifacts_text = "No artifacts available in memory."
        
    # 3. Format prior goals list status
    goals_text = ""
    if prior_goals:
        goals_text = "Current Goals List (Preserve order, length, and text):\n"
        for idx, g in enumerate(prior_goals):
            status = "Done" if g.done else "Unfinished"
            goals_text += f"{idx+1}. Text: \"{g.text}\" | Status: {status}\n"
    else:
        goals_text = "Prior goals list is empty. This is the first turn. You must decompose the user query into a list of short, imperative goals."

    # 4. Format run history
    history_text = "Execution History:\n"
    if history:
        for item in history:
            kind = item.kind
            it = item.iter
            if kind == "action":
                history_text += f"Iteration {it}: Action '{item.tool}' with args {item.arguments} -> Result summary: {item.result_descriptor}\n"
            elif kind == "answer":
                history_text += f"Iteration {it}: Answer emitted -> \"{item.text}\"\n"
    else:
        history_text += "No history accumulated yet."

    system_prompt = """You are the Perception Layer of an AI agent. Your job is to analyze the agent's query, execution history, memory hits, and prior goals to produce an updated list of goals. Output your response as a valid JSON object matching the JSON schema.

Obligations:
1. If the prior goal list is empty (first turn), decompose the user query into a minimal set of sequential goals (ideally 1 to 2 goals, max 3). Avoid over-decomposing into highly granular sub-steps, as this increases execution overhead. Combine related actions into a single goal where possible (e.g., searching and reading can be one goal, or writing a file and confirming it can be one goal).
2. If prior goals are provided, examine the history to check if each goal is now satisfied. If a goal was completed in the history, set done to true. A goal that was already marked done MUST remain done. Note: If the decision layer has emitted a final answer (even if it was for an earlier goal), you should mark all goals as done to terminate the loop immediately. CRITICAL: The final goal in the list must NOT be marked done until the decision layer has explicitly emitted a final answer or acknowledgment of type 'answer' in the execution history. If the last history event is a tool action and no answer text has been emitted yet, keep the final goal's done status as false.
3. For the first unfinished goal, decide if it needs raw content bytes from one of the available artifacts in memory (e.g. a scraped web page, a search outcome). If yes, set artifact_index to the index of that artifact (0, 1, 2, ...). If no, leave it null.
4. You MUST preserve the exact number, text, and order of the goals. Do not add, remove, or reorder the goals.
"""

    prompt = f"""User Query: {query}

{goals_text}

{artifacts_text}

{history_text}
"""

    extraction = None
    llm = LLM()
    
    # Try 1: Pinned to Gemini
    try:
        response = llm.chat(
            prompt=prompt,
            system=system_prompt,
            provider="g",
            auto_route="perception",
            response_format={"type": "json_schema", "schema": PerceptionObservationOutput.model_json_schema()}
        )
        parsed_data = response.get("parsed") or json.loads(response["text"])
        extraction = PerceptionObservationOutput.model_validate(parsed_data)
    except Exception as e:
        sys.stderr.write(f"Gemini perception failed: {e}. Trying Groq (gr) fallback...\n")
        # Try 2: Pinned to Groq
        try:
            response = llm.chat(
                prompt=prompt,
                system=system_prompt,
                provider="gr",
                response_format={"type": "json_schema", "schema": PerceptionObservationOutput.model_json_schema()}
            )
            parsed_data = response.get("parsed") or json.loads(response["text"])
            extraction = PerceptionObservationOutput.model_validate(parsed_data)
        except Exception as e2:
            sys.stderr.write(f"Groq perception failed: {e2}. Using rule-based fallback...\n")

    goals = []
    
    if extraction is not None:
        if not prior_goals:
            # First turn: create new goals from extraction
            for idx, eg in enumerate(extraction.goals):
                attach_id = None
                if eg.artifact_index is not None and 0 <= eg.artifact_index < len(artifact_hits):
                    attach_id = artifact_hits[eg.artifact_index].artifact_id
                    
                goals.append(Goal(
                    id=idx + 1,  # Use integer ID
                    text=eg.text,
                    done=eg.done,
                    attach_artifact_id=attach_id
                ))
        else:
            # Subsequent turns: preserve structure and only update flags
            first_unfinished_idx = next((i for i, g in enumerate(prior_goals) if not g.done), -1)
            
            for idx, prior_g in enumerate(prior_goals):
                done = prior_g.done
                attach_id = prior_g.attach_artifact_id
                
                if idx < len(extraction.goals):
                    eg = extraction.goals[idx]
                    # Enforce that once done, a goal remains done
                    done = prior_g.done or eg.done
                    if idx == len(prior_goals) - 1 and not prior_g.done:
                        last_is_answer = history and history[-1].kind == "answer"
                        if not last_is_answer:
                            done = False
                    
                    # Update attach_artifact_id only for the next unfinished goal
                    if idx == first_unfinished_idx:
                        if eg.artifact_index is not None and 0 <= eg.artifact_index < len(artifact_hits):
                            attach_id = artifact_hits[eg.artifact_index].artifact_id
                        else:
                            attach_id = None
                            
                goals.append(Goal(
                    id=prior_g.id,
                    text=prior_g.text,
                    done=done,
                    attach_artifact_id=attach_id
                ))
    else:
        # Fallback if both LLM calls failed: preserve goals exactly, or initialize one default goal
        if not prior_goals:
            goals = [Goal(id=1, text=f"Satisfy query: {query}", done=False, attach_artifact_id=None)]
        else:
            for prior_g in prior_goals:
                goals.append(prior_g.model_copy())
                
    return Observation(goals=goals)
