"""
Research agent — LangGraph implementation with nodes and edges.

The agent iteratively calls tools to explore a repository and build up
an answer. It uses a StateGraph with a ToolNode to execute tools and 
routes dynamically based on LLM outputs.
"""
from __future__ import annotations

import logging
from typing import Annotated, Any, TypedDict

from django.conf import settings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

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
# State Definition
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration: int


# ---------------------------------------------------------------------------
# Agent Class
# ---------------------------------------------------------------------------
class ResearchAgent:
    """
    Tool-calling research agent using LangGraph.
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

        # Build tools (now wrapped with @tool)
        self.tools = build_tools(repo_local_path, repo_url, session, self.step_counter)

        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")
        
        self.llm = ChatOpenAI(
            model=settings.OPENAI_CHAT_MODEL,
            api_key=api_key,
            temperature=0,
        )
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build Graph
        workflow = StateGraph(AgentState)
        
        # Add Nodes
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Add Edges
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")
        
        self.graph = workflow.compile()

    def call_model(self, state: AgentState) -> dict[str, Any]:
        """Node function to invoke the LLM with the current state."""
        messages = state.get("messages", [])
        iteration = state.get("iteration", 0) + 1
        
        logger.info(
            "Agent iteration %d/%d for session %s",
            iteration, MAX_ITERATIONS, self.session.session_id,
        )

        # Context window management: keep SystemMessage, first HumanMessage, and last 30 messages
        if len(messages) > 32:
            messages = [messages[0], messages[1]] + messages[-30:]

        # Token budget management
        total_tokens = sum(
            msg.usage_metadata.get("total_tokens", 0)
            for msg in state["messages"]
            if hasattr(msg, "usage_metadata") and msg.usage_metadata
        )

        # Handle limits by removing tools and asking for a final answer
        if iteration > MAX_ITERATIONS or total_tokens > MAX_TOTAL_TOKENS:
            reason = "maximum tool calls" if iteration > MAX_ITERATIONS else "token budget"
            logger.warning(f"Reached {reason} limit. Forcing final answer.")
            
            prompt = HumanMessage(
                content=f"You've reached the {reason} limit. Please provide your final answer now based on what you've found so far."
            )
            messages.append(prompt)
            # Invoke LLM without tools to force text response
            response = self.llm.invoke(messages)
            return {"messages": [prompt, response], "iteration": iteration}

        # Normal execution
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response], "iteration": iteration}

    def run(self, question: str) -> dict[str, Any]:
        """
        Execute the agent graph. Returns a dict with:
        - answer: str
        - prompt_tokens, completion_tokens, total_tokens: int
        - source_references: list
        """
        initial_state = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=build_research_prompt(question, self.repo_url))
            ],
            "iteration": 0,
        }

        # Run the graph
        final_state = self.graph.invoke(initial_state, {"recursion_limit": MAX_ITERATIONS * 2 + 5})
        
        # Extract the final answer from the last message
        last_message = final_state["messages"][-1]
        final_answer = last_message.content if isinstance(last_message, AIMessage) else "Unable to produce an answer."

        # Calculate token usage
        total_prompt = 0
        total_completion = 0
        for msg in final_state["messages"]:
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                total_prompt += msg.usage_metadata.get("input_tokens", 0)
                total_completion += msg.usage_metadata.get("output_tokens", 0)

        return {
            "answer": final_answer,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "source_references": [],
        }
