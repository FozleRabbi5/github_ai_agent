from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import RepositoryAgentGraphNodes
from app.graph.state import AgentGraphState
from app.memory.checkpointer import get_checkpointer


class RepositoryAgentGraphBuilder:
    def __init__(self, nodes: RepositoryAgentGraphNodes) -> None:
        self._nodes = nodes

    def build(self):
        graph = StateGraph(AgentGraphState)
        graph.add_node("clone_repository", self._nodes.clone_repository)
        graph.add_node("scan_repository", self._nodes.scan_repository)
        graph.add_node("chunk_codebase", self._nodes.chunk_codebase)
        graph.add_node("generate_embeddings", self._nodes.generate_embeddings)
        graph.add_node("retrieve_relevant_context", self._nodes.retrieve_relevant_context)
        graph.add_node("analyze_code_relationships", self._nodes.analyze_code_relationships)
        graph.add_node("generate_answer", self._nodes.generate_answer)
        graph.add_node("generate_source_references", self._nodes.generate_source_references)
        graph.add_node("persist_conversation", self._nodes.persist_conversation)

        graph.add_edge(START, "clone_repository")
        graph.add_edge("clone_repository", "scan_repository")
        graph.add_edge("scan_repository", "chunk_codebase")
        graph.add_edge("chunk_codebase", "generate_embeddings")
        graph.add_edge("generate_embeddings", "retrieve_relevant_context")
        graph.add_edge("retrieve_relevant_context", "analyze_code_relationships")
        graph.add_edge("analyze_code_relationships", "generate_answer")
        graph.add_edge("generate_answer", "generate_source_references")
        graph.add_edge("generate_source_references", "persist_conversation")
        graph.add_edge("persist_conversation", END)

        return graph.compile(checkpointer=get_checkpointer())
