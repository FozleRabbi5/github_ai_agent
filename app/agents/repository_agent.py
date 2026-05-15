from __future__ import annotations

import uuid

from app.graph.builder import RepositoryAgentGraphBuilder
from app.graph.nodes import RepositoryAgentGraphNodes


class RepositoryQuestionAnsweringAgent:
    def __init__(self, service: "RepositoryAIService") -> None:
        nodes = RepositoryAgentGraphNodes(service)
        self._graph = RepositoryAgentGraphBuilder(nodes).build()

    def invoke(self, *, repository_url: str, question: str) -> dict:
        thread_id = str(uuid.uuid4())
        result = self._graph.invoke(
            {
                "repository_url": repository_url,
                "user_question": question,
                "thread_id": thread_id,
                "errors": [],
                "execution_metadata": {"thread_id": thread_id},
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        return result
