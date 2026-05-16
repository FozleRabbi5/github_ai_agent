"""
Research agent — LangGraph implementation with nodes and edges.

The agent iteratively calls tools to explore a repository and build up
an answer. It uses a StateGraph with a ToolNode to execute tools and 
routes dynamically based on LLM outputs.
"""
from __future__ import annotations

import logging
from typing import Annotated, Any, TypedDict, Literal

from django.conf import settings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app.agents.tools import AGENT_TOOLS
from app.models import ResearchSession
from app.prompts.coding_assistant import SYSTEM_PROMPT, build_research_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 15        # Hard cap on tool-call rounds
MAX_TOTAL_TOKENS = 100_000  # Abort if cumulative tokens exceed this

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class FinalAnswer(BaseModel):
    """The final comprehensive answer to the user's research question."""
    answer: str = Field(description="The detailed, evidence-backed answer.")
    source_references: list[str] = Field(description="List of file paths explicitly used to derive the answer.")

# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration: int
    final_answer: dict[str, Any] | None

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
        
        # Tools are now imported directly as module-level functions
        self.tools = AGENT_TOOLS

        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")
        
        self.llm = ChatOpenAI(
            model=settings.OPENAI_CHAT_MODEL,
            api_key=api_key,
            temperature=0,
        )
        
        # We bind the final answer tool alongside regular tools.
        # This allows the LLM to signal it has finished by calling FinalAnswer.
        self.llm_with_tools = self.llm.bind_tools(self.tools + [FinalAnswer])

        # Build Graph
        workflow = StateGraph(AgentState)
        
        # Add Nodes
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Add Edges
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self.should_continue, ["tools", END])
        workflow.add_edge("tools", "agent")
        
        self.graph = workflow.compile()

    def should_continue(self, state: AgentState) -> Literal["tools", END]:
        """Determine whether to continue routing to tools or end."""
        messages = state.get("messages", [])
        last_message = messages[-1]

        # If there are no tool calls, or if the LLM called the FinalAnswer "tool", we're done.
        if not getattr(last_message, "tool_calls", None):
            return END
            
        # Check if the tool call is our special FinalAnswer schema
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "FinalAnswer":
                return END
                
        return "tools"

    def call_model(self, state: AgentState) -> dict[str, Any]:
        """Node function to invoke the LLM with the current state."""
        messages = state.get("messages", [])
        iteration = state.get("iteration", 0) + 1
        
        logger.info(
            "Agent iteration %d/%d for session %s",
            iteration, MAX_ITERATIONS, self.session.session_id,
        )

        # Context window management
        # We prune older tool calls and responses to save context, while keeping the system prompt and original question
        from langchain_core.messages import ToolMessage
        if len(messages) > 20:
            head = messages[:2]
            tail = messages[-16:]
            # Ensure tail does not start with a ToolMessage, as its corresponding AIMessage would be lost
            while tail and isinstance(tail[0], ToolMessage):
                tail.pop(0)
            messages = head + tail
            
        # Debug logging to see exactly what we are sending
        debug_msg_types = []
        for i, m in enumerate(messages):
            role = getattr(m, "type", type(m).__name__)
            has_tool_calls = bool(getattr(m, "tool_calls", None))
            debug_msg_types.append(f"[{i}]:{role}(has_tool_calls={has_tool_calls})")
        logger.info(f"LLM INVOKE MESSAGES: {', '.join(debug_msg_types)}")

        total_tokens = sum(
            msg.usage_metadata.get("total_tokens", 0)
            for msg in messages
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
            # Invoke LLM specifically bound to only FinalAnswer to force structured output
            response = self.llm.with_structured_output(FinalAnswer).invoke(messages)
            
            return {
                "messages": [prompt, AIMessage(content="Final answer generated due to limit.")],
                "iteration": iteration,
                "final_answer": response.model_dump() if hasattr(response, "model_dump") else response
            }

        # Normal execution
        response = self.llm_with_tools.invoke(messages)
        
        state_updates = {"messages": [response], "iteration": iteration}
        
        # Check if the response was the FinalAnswer tool call
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                if tc["name"] == "FinalAnswer":
                    state_updates["final_answer"] = tc["args"]
                    break
                    
        return state_updates

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
            "final_answer": None,
        }

        # Run the graph, injecting config with the repo path and session ID so tools can use them
        config = {
            "recursion_limit": MAX_ITERATIONS * 2 + 5,
            "configurable": {
                "repo_local_path": self.repo_local_path,
                "session_id": self.session.pk,
            }
        }
        
        final_state = self.graph.invoke(initial_state, config)
        
        final_answer_data = final_state.get("final_answer")
        
        # If the LLM didn't use the structured tool and just output text (fallback)
        if not final_answer_data:
            last_message = final_state["messages"][-1]
            content = last_message.content if isinstance(last_message, AIMessage) else "Unable to produce an answer."
            final_answer_data = {
                "answer": content,
                "source_references": []
            }

        # Calculate token usage
        total_prompt = 0
        total_completion = 0
        for msg in final_state["messages"]:
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                total_prompt += msg.usage_metadata.get("input_tokens", 0)
                total_completion += msg.usage_metadata.get("output_tokens", 0)

        return {
            "answer": final_answer_data.get("answer", "No answer provided."),
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "source_references": final_answer_data.get("source_references", []),
        }
