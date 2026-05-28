import os
import sys
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional
from pydantic import BaseModel

from schemas import (
    MemoryItem, RememberInput, RecordOutcomeInput, ReadInput, HistoryItem, ToolCall
)
from gateway import LLM

MEMORY_FILE = Path(__file__).parent / "state" / "memory.json"

STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", 
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
}


class MemoryExtraction(BaseModel):
    kind: Literal["fact", "preference", "scratchpad"]
    keywords: list[str]
    descriptor: str
    value: dict
    confidence: float


class MemoryStore:
    def __init__(self, filepath: Path = MEMORY_FILE):
        self.filepath = filepath
        self.items: list[MemoryItem] = []
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        if not self.filepath.exists():
            self.items = []
            self._loaded = True
            return
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.items = [MemoryItem.model_validate(item) for item in data]
        except Exception:
            self.items = []
        self._loaded = True

    def save(self):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        # Serialize MemoryItem objects to dict with json mode for datetime serialization
        data = [item.model_dump(mode="json") for item in self.items]
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def clear(self):
        self.items = []
        if self.filepath.exists():
            try:
                self.filepath.unlink()
            except OSError:
                pass
        self._loaded = True


# Global store instance
_store = MemoryStore()


def tokenize(text: str) -> set[str]:
    """Lowercase tokenizer that filters out punctuation and stopwords."""
    words = text.lower().split()
    tokens = set()
    for w in words:
        cleaned = w.strip(".,?!()\"';:-_/")
        if cleaned and cleaned not in STOPWORDS:
            tokens.add(cleaned)
    return tokens


def remember(data: RememberInput) -> MemoryItem:
    """Classify and extract structured memory items from raw text via Gemini, with Groq fallback."""
    system_prompt = """You are a Memory Extractor. Analyze the input text and extract a memory item.

Classify it into one of the following kinds:
- fact: A durable observed truth (e.g., birthday, name, location, API key info, config details).
- preference: A user-stated or inferred preference (e.g., meeting time, styling choices, formatting preference).
- scratchpad: Run-scoped temporary working notes or logic context.

Extract:
1. kind: The memory type (fact, preference, or scratchpad).
2. keywords: List of key lowercase, singular search keywords identifying the content (excluding common stopwords).
3. descriptor: A short, clear, single-line human-readable summary of the memory.
4. value: A structured JSON dictionary representing the details (e.g., {"entity": "John", "attribute": "birthday", "value": "2026-05-15"}).
5. confidence: A float from 0.0 to 1.0 representing classification confidence.
"""
    
    extraction = None
    llm = LLM()
    prompt = f"Analyze and extract memory item from: '{data.raw_text}'"
    
    # Try 1: Pinned to Gemini
    try:
        response = llm.chat(
            prompt=prompt,
            system=system_prompt,
            provider="g",
            auto_route="memory",
            response_format={"type": "json_schema", "schema": MemoryExtraction.model_json_schema()}
        )
        parsed_data = response.get("parsed") or json.loads(response["text"])
        extraction = MemoryExtraction.model_validate(parsed_data)
    except Exception as e:
        sys.stderr.write(f"Gemini remember failed: {e}. Trying Groq (gr) fallback...\n")
        # Try 2: Pinned to Groq
        try:
            response = llm.chat(
                prompt=prompt,
                system=system_prompt,
                provider="gr",
                response_format={"type": "json_schema", "schema": MemoryExtraction.model_json_schema()}
            )
            parsed_data = response.get("parsed") or json.loads(response["text"])
            extraction = MemoryExtraction.model_validate(parsed_data)
        except Exception as e2:
            sys.stderr.write(f"Groq remember failed: {e2}. Using offline fallback...\n")

    if extraction is not None:
        item = MemoryItem(
            id="mem_" + uuid.uuid4().hex[:8],
            kind=extraction.kind,
            keywords=[kw.lower() for kw in extraction.keywords],
            descriptor=extraction.descriptor,
            value=extraction.value,
            artifact_id=None,
            source=data.source,
            run_id=data.run_id,
            goal_id=data.goal_id,
            confidence=extraction.confidence,
            created_at=datetime.now()
        )
    else:
        # Fallback in case of all LLM failures
        words = [w.strip(".,?!()\"'").lower() for w in data.raw_text.split()]
        kws = [w for w in words if w and w not in STOPWORDS]
        item = MemoryItem(
            id="mem_fb_" + uuid.uuid4().hex[:8],
            kind="scratchpad",
            keywords=kws[:5],
            descriptor=f"Fallback: {data.raw_text[:50]}",
            value={"raw_text": data.raw_text},
            artifact_id=None,
            source=data.source,
            run_id=data.run_id,
            goal_id=data.goal_id,
            confidence=0.5,
            created_at=datetime.now()
        )

    _store.load()
    _store.items.append(item)
    _store.save()
    return item


def record_outcome(data: RecordOutcomeInput) -> MemoryItem:
    """Record an MCP tool execution outcome as a tool_outcome kind memory item."""
    tool_name = data.tool_call.name
    arguments = data.tool_call.arguments
    
    # Generate keywords from tool name and argument values
    kws = {tool_name.lower()}
    kws.update(tool_name.lower().replace("_", " ").split())
    for k, v in arguments.items():
        kws.add(k.lower())
        if isinstance(v, str):
            for word in v.lower().split():
                cleaned = word.strip(".,?!()\"'")
                if cleaned and cleaned not in STOPWORDS:
                    kws.add(cleaned)
                    
    keywords_list = sorted(list(kws))
    descriptor = f"Executed tool {tool_name} -> {data.result_text[:100]}..."
    
    item = MemoryItem(
        id="mem_" + uuid.uuid4().hex[:8],
        kind="tool_outcome",
        keywords=keywords_list,
        descriptor=descriptor,
        value={
            "tool_name": tool_name,
            "arguments": arguments,
            "result_summary": data.result_text[:500]
        },
        artifact_id=data.artifact_id,
        source="tool_execution",
        run_id=data.run_id,
        goal_id=data.goal_id,
        confidence=1.0,
        created_at=datetime.now()
    )
    
    _store.load()
    _store.items.append(item)
    _store.save()
    return item


def read(data: ReadInput) -> list[MemoryItem]:
    """Pure Python keyword matching. Score is based on intersection size with minor recency bias."""
    _store.load()
    
    combined_text = data.query
    if data.history:
        for turn in data.history:
            if turn.text:
                combined_text += " " + str(turn.text)
            if turn.result_descriptor:
                combined_text += " " + str(turn.result_descriptor)
            if turn.tool:
                combined_text += " " + str(turn.tool)
            if turn.arguments:
                for val in turn.arguments.values():
                    combined_text += " " + str(val)

    query_tokens = tokenize(combined_text)
    
    scored_items = []
    for item in _store.items:
        if data.kinds and item.kind not in data.kinds:
            continue
            
        desc_tokens = tokenize(item.descriptor)
        item_keywords = set(item.keywords)
        candidate_tokens = item_keywords.union(desc_tokens)
        
        score = len(query_tokens.intersection(candidate_tokens))
        
        # Tiny recency bias (time decay)
        age_in_seconds = (datetime.now() - item.created_at).total_seconds()
        recency_bonus = 1.0 / (1.0 + age_in_seconds / 3600.0)  # decays over hours
        
        # Only retain matches, or match everything if query is empty
        if score > 0 or not query_tokens:
            final_score = score + recency_bonus * 0.01
            scored_items.append((final_score, item))
            
    scored_items.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored_items[:data.top_k]]


def filter(kinds: list[str] | None = None, goal_id: str | None = None, recent: int | None = None) -> list[MemoryItem]:
    """Structured filter by kind, goal, and recency."""
    _store.load()
    filtered = []
    for item in _store.items:
        if kinds and item.kind not in kinds:
            continue
        if goal_id is not None and item.goal_id != goal_id:
            continue
        filtered.append(item)
        
    # Most recent first
    filtered.sort(key=lambda x: x.created_at, reverse=True)
    
    if recent is not None:
        filtered = filtered[:recent]
        
    return filtered


class RelevanceItemScore(BaseModel):
    item_id: str
    relevance_score: float
    reason: str


class RelevanceRanking(BaseModel):
    ranked_items: list[RelevanceItemScore]


def relevant(query: str, kinds: list[str] | None = None, top_k: int = 5) -> list[MemoryItem]:
    """Use LLM auto_route='memory' to score relevance over kind-filtered candidates."""
    _store.load()
    candidates = []
    for item in _store.items:
        if kinds and item.kind not in kinds:
            continue
        candidates.append(item)
        
    if not candidates:
        return []
        
    # Serialize candidate pool
    candidate_list_str = []
    for c in candidates:
        candidate_list_str.append(
            f"ID: {c.id}\nKind: {c.kind}\nDescriptor: {c.descriptor}\nValue: {json.dumps(c.value)}\n---"
        )
    candidates_text = "\n".join(candidate_list_str)
    
    system_prompt = """You are a Memory Relevance Scorer. You will be given a query and a list of memory items. Output your response as a valid JSON object matching the JSON schema.
Your job is to analyze each item and score its relevance to the query on a scale of 0.0 (completely irrelevant) to 1.0 (highly relevant).
Output a sorted list of items by relevance score descending.
"""
    prompt = f"Query: {query}\n\nCandidate Memory Items:\n{candidates_text}"
    
    try:
        llm = LLM()
        response = llm.chat(
            prompt=prompt,
            system=system_prompt,
            auto_route="memory",
            response_format={"type": "json_schema", "schema": RelevanceRanking.model_json_schema()}
        )
        
        parsed_data = response.get("parsed") or json.loads(response["text"])
        ranking = RelevanceRanking.model_validate(parsed_data)
        
        scores = {item.item_id: item.relevance_score for item in ranking.ranked_items}
        
        scored_candidates = []
        for c in candidates:
            score = scores.get(c.id, 0.0)
            if score > 0.0:
                scored_candidates.append((score, c))
                
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored_candidates[:top_k]]
    except Exception as e:
        # Fall back to standard keyword read in case of failure
        sys.stderr.write(f"LLM relevance check failed: {e}. Falling back to keyword search.\n")
        return read(ReadInput(query=query, kinds=kinds, top_k=top_k))


def clear():
    """Clear memory store and delete backing JSON file."""
    _store.clear()
