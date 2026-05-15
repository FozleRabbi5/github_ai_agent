"""
Research agent — tool-calling loop powered by OpenAI function calling.

The agent iteratively calls tools to explore a repository and build up
an answer. It stops when the LLM produces a response with no tool calls
or when it hits the iteration/token cap.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from django.conf import settings
from openai import OpenAI

from app.agents.tools import build_tools
from app.models import ResearchSession
from app.prompts.coding_assistant import SYSTEM_PROMPT, build_research_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 15        # Hard cap on tool-call rounds
MAX_TOTAL_TOKENS = 100_000  # Abort if cumulative tokens exceed this


# ---------------------------------------------------------------------------
# Tool schema builder — converts our plain functions into OpenAI tool specs
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories at the given path relative to the repository root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to list. Use '.' for the repo root.",
                        "default": ".",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's contents. Lines are 1-indexed. Max 500 lines per call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to repo root."},
                    "start_line": {"type": "integer", "description": "Start line (1-indexed).", "default": 1},
                    "end_line": {"type": "integer", "description": "End line (1-indexed, inclusive).", "default": 500},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for a text pattern across all repository files. Returns up to 20 matches with context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text pattern to search for."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_summary",
            "description": "Get file metadata: size, language, line count. For Python: classes, functions, imports.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to repo root."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_finding",
            "description": "Save an important finding during research. These persist across sessions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "File the finding relates to (or empty for general)."},
                    "note": {"type": "string", "description": "The observation or conclusion."},
                    "finding_type": {
                        "type": "string",
                        "enum": ["observation", "pattern", "issue", "dependency", "architecture"],
                        "description": "Type of finding.",
                        "default": "observation",
                    },
                },
                "required": ["file_path", "note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_previous_findings",
            "description": "Retrieve findings from prior research sessions on this repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_url": {"type": "string", "description": "Repository URL (defaults to current repo).", "default": ""},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_past_sessions",
            "description": "List past research sessions for this repository with questions and answers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_url": {"type": "string", "description": "Repository URL (defaults to current repo).", "default": ""},
                },
                "required": [],
            },
        },
    },
]


class ResearchAgent:
    """
    Tool-calling research agent. Uses OpenAI's function calling to let the
    LLM decide which tools to invoke, then executes them and feeds results
    back until the LLM produces a final text answer or hits limits.
    """

    def __init__(
        self,
        repo_local_path: str,
        repo_url: str,
        session: ResearchSession,
    ) -> None:
        self.repo_local_path = repo_local_path
        self.repo_url = repo_url
        self.session = session
        self.step_counter: list[int] = [0]  # mutable counter shared with tools

        # Build tool functions
        tool_fns = build_tools(repo_local_path, repo_url, session, self.step_counter)
        self._tool_map: dict[str, Any] = {fn.__name__: fn for fn in tool_fns}

        # OpenAI client
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")
        self._client = OpenAI(api_key=api_key)
        self._model = settings.OPENAI_CHAT_MODEL

    def run(self, question: str) -> dict[str, Any]:
        """
        Execute the agent loop. Returns a dict with:
        - answer: str
        - prompt_tokens, completion_tokens, total_tokens: int
        - source_references: list
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_research_prompt(question, self.repo_url)},
        ]

        total_prompt = 0
        total_completion = 0
        iteration = 0

        while iteration < MAX_ITERATIONS:
            iteration += 1
            logger.info(
                "Agent iteration %d/%d for session %s",
                iteration, MAX_ITERATIONS, self.session.session_id,
            )

            # Call the LLM
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0,
            )

            # Track tokens
            usage = response.usage
            if usage:
                total_prompt += usage.prompt_tokens
                total_completion += usage.completion_tokens

            # Check token budget
            total = total_prompt + total_completion
            if total > MAX_TOTAL_TOKENS:
                logger.warning(
                    "Token budget exceeded (%d > %d), stopping agent",
                    total, MAX_TOTAL_TOKENS,
                )
                break

            choice = response.choices[0]
            message = choice.message

            # If the LLM produced a final answer (no tool calls), we're done
            if not message.tool_calls:
                final_answer = message.content or ""
                return {
                    "answer": final_answer,
                    "prompt_tokens": total_prompt,
                    "completion_tokens": total_completion,
                    "total_tokens": total_prompt + total_completion,
                    "source_references": [],
                }

            # Append the assistant message with tool calls
            messages.append(message.model_dump())

            # Execute each tool call
            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                tool_fn = self._tool_map.get(fn_name)
                if tool_fn is None:
                    result = f"Error: Unknown tool '{fn_name}'."
                else:
                    try:
                        result = tool_fn(**fn_args)
                    except Exception as exc:
                        logger.exception("Tool %s failed", fn_name)
                        result = f"Error executing {fn_name}: {exc}"

                # Append tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)[:8000],  # Cap tool output in messages
                })

            # Context window management: keep last 30 messages + system
            if len(messages) > 32:
                messages = [messages[0]] + messages[-30:]

        # If we exit the loop without a final answer, ask for one
        messages.append({
            "role": "user",
            "content": (
                "You've reached the maximum number of tool calls. "
                "Please provide your final answer now based on what you've found so far."
            ),
        })
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0,
        )
        if response.usage:
            total_prompt += response.usage.prompt_tokens
            total_completion += response.usage.completion_tokens

        final_answer = response.choices[0].message.content or "Unable to produce an answer."
        return {
            "answer": final_answer,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "source_references": [],
        }
