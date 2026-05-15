from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable

from app.api.serializers import RepositoryIndexModelSerializer
from app.graph.state import AgentGraphState
from app.models import QuestionHistory, RepositoryIndex
from app.schemas.dto import ExecutionStep, RetrievedCodeChunk
from app.utils.logging import get_logger, log_event


logger = get_logger(__name__)


def timed_node(node_name: str):
    def decorator(func: Callable[..., dict[str, Any]]):
        def wrapper(self, state: AgentGraphState) -> dict[str, Any]:
            started_at = datetime.now(timezone.utc)
            start_perf = time.perf_counter()
            log_event(logger, "graph.node.start", node=node_name, thread_id=state.get("thread_id"))
            updates = func(self, state)
            completed_at = datetime.now(timezone.utc)
            duration_ms = round((time.perf_counter() - start_perf) * 1000, 3)

            steps = list(state.get("execution_metadata", {}).get("steps", []))
            steps.append(
                ExecutionStep(
                    node=node_name,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_ms=duration_ms,
                    details={
                        k: v
                        for k, v in updates.items()
                        if k not in {"code_chunks", "repository_index_id", "repository_index_payload"}
                    },
                ).to_dict()
            )
            metadata = dict(state.get("execution_metadata", {}))
            metadata["steps"] = steps
            updates["execution_metadata"] = metadata
            log_event(logger, "graph.node.end", node=node_name, duration_ms=duration_ms)
            return updates

        return wrapper

    return decorator


class RepositoryAgentGraphNodes:
    def __init__(self, agent_service: "RepositoryAIService") -> None:
        self._agent_service = agent_service

    @timed_node("clone_repository")
    def clone_repository(self, state: AgentGraphState) -> dict[str, Any]:
        cloned = self._agent_service.git_service.clone_or_update(state["repository_url"])
        return {
            "local_repository_path": cloned.local_path,
            "repository_name": cloned.repository_name,
            "repository_branch": cloned.default_branch,
            "repository_commit_hash": cloned.commit_hash,
        }

    @timed_node("scan_repository")
    def scan_repository(self, state: AgentGraphState) -> dict[str, Any]:
        scan_summary, documents = self._agent_service.indexing_service.scan_and_chunk(
            state["local_repository_path"]
        )
        analyzed_files = sorted({doc.metadata["file_path"] for doc in documents})
        return {
            "scan_summary": scan_summary.to_dict(),
            "code_chunks": documents,
            "analyzed_files": analyzed_files,
        }

    @timed_node("chunk_codebase")
    def chunk_codebase(self, state: AgentGraphState) -> dict[str, Any]:
        chunks = state.get("code_chunks", [])
        return {
            "embeddings_metadata": {
                **state.get("embeddings_metadata", {}),
                "chunk_count": len(chunks),
            }
        }

    @timed_node("generate_embeddings")
    def generate_embeddings(self, state: AgentGraphState) -> dict[str, Any]:
        repository_index = self._agent_service.repository_index_repository.get_by_url(
            state["repository_url"]
        )
        current_commit = state["repository_commit_hash"]
        code_chunks = state.get("code_chunks", [])

        if (
            repository_index
            and repository_index.status == RepositoryIndex.Status.INDEXED
            and repository_index.last_commit_hash == current_commit
        ):
            metadata = dict(repository_index.metadata)
            metadata["cache_hit"] = True
            return {
                "repository_index_id": repository_index.id,
                "repository_index_payload": RepositoryIndexModelSerializer(repository_index).data,
                "embeddings_metadata": {**state.get("embeddings_metadata", {}), **metadata},
            }

        vector_metadata = self._agent_service.vector_service.reindex_repository(
            state["repository_url"],
            code_chunks,
        )
        metadata = {
            **state.get("scan_summary", {}),
            **state.get("embeddings_metadata", {}),
            **vector_metadata,
            "cache_hit": False,
        }
        index = self._agent_service.repository_index_repository.upsert_index(
            repository_url=state["repository_url"],
            repository_name=state["repository_name"],
            local_path=state["local_repository_path"],
            default_branch=state["repository_branch"],
            last_commit_hash=current_commit,
            status=RepositoryIndex.Status.INDEXED,
            metadata=metadata,
        )
        return {
            "repository_index_id": index.id,
            "repository_index_payload": RepositoryIndexModelSerializer(index).data,
            "embeddings_metadata": metadata,
        }

    @timed_node("retrieve_relevant_context")
    def retrieve_relevant_context(self, state: AgentGraphState) -> dict[str, Any]:
        results = self._agent_service.vector_service.similarity_search(
            state["repository_url"],
            state["user_question"],
        )
        documents: list[dict[str, Any]] = []
        for document, score in results:
            payload = RetrievedCodeChunk(
                content=document.page_content,
                file_path=document.metadata["file_path"],
                language=document.metadata["language"],
                symbol_name=document.metadata["symbol_name"],
                symbol_type=document.metadata["symbol_type"],
                line_start=document.metadata["line_start"],
                line_end=document.metadata["line_end"],
                chunk_index=document.metadata["chunk_index"],
                score=round(float(score), 4) if score is not None else None,
            )
            documents.append(payload.to_dict())

        return {"retrieved_documents": documents}

    @timed_node("analyze_code_relationships")
    def analyze_code_relationships(self, state: AgentGraphState) -> dict[str, Any]:
        retrieved = state.get("retrieved_documents", [])
        if not retrieved:
            return {"relationship_summary": "No repository chunks were retrieved for the question."}

        lines = []
        for item in retrieved:
            lines.append(
                (
                    f"{item['file_path']}::{item['symbol_name']} "
                    f"({item['symbol_type']}, lines {item['line_start']}-{item['line_end']})"
                )
            )
        return {
            "relationship_summary": (
                "The following repository symbols are the strongest candidates for the answer:\n"
                + "\n".join(lines)
            )
        }

    @timed_node("generate_answer")
    def generate_answer(self, state: AgentGraphState) -> dict[str, Any]:
        retrieved = state.get("retrieved_documents", [])
        context = "\n\n".join(
            [
                (
                    f"FILE: {item['file_path']}\n"
                    f"SYMBOL: {item['symbol_name']} ({item['symbol_type']})\n"
                    f"LINES: {item['line_start']}-{item['line_end']}\n"
                    f"CODE:\n{item['content']}"
                )
                for item in retrieved
            ]
        )
        answer = self._agent_service.llm_service.answer_question(
            repository_url=state["repository_url"],
            question=state["user_question"],
            context=context,
            relationship_summary=state.get("relationship_summary", ""),
        )
        return {"generated_answer": answer}

    @timed_node("generate_source_references")
    def generate_source_references(self, state: AgentGraphState) -> dict[str, Any]:
        references = []
        seen = set()
        for item in state.get("retrieved_documents", []):
            reference = RetrievedCodeChunk(**item).to_reference().to_dict()
            unique_key = (
                reference["file_path"],
                reference["symbol_name"],
                reference["line_start"],
                reference["line_end"],
            )
            if unique_key in seen:
                continue
            seen.add(unique_key)
            references.append(reference)
        return {"source_references": references}

    @timed_node("persist_conversation")
    def persist_conversation(self, state: AgentGraphState) -> dict[str, Any]:
        repository_index = self._agent_service.repository_index_repository.get_by_id(
            state["repository_index_id"]
        )
        if repository_index is None:
            raise ValueError(
                f"RepositoryIndex with id {state['repository_index_id']} was not found."
            )
        question = self._agent_service.question_history_repository.create(
            repository=repository_index,
            thread_id=state["thread_id"],
            question=state["user_question"],
            answer=state.get("generated_answer", ""),
            source_references=state.get("source_references", []),
            execution_metadata=state.get("execution_metadata", {}),
            status=QuestionHistory.Status.COMPLETED,
        )
        return {"persisted_question_id": question.id}
