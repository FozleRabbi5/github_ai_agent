from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.documents import Document

class AgentGraphState(TypedDict, total=False):
    repository_url: str
    user_question: str
    thread_id: str
    local_repository_path: str
    repository_name: str
    repository_branch: str
    repository_commit_hash: str
    repository_index_id: int
    repository_index_payload: dict[str, Any]
    scan_summary: dict[str, Any]
    analyzed_files: list[str]
    code_chunks: list[Document]
    embeddings_metadata: dict[str, Any]
    retrieved_documents: list[dict[str, Any]]
    relationship_summary: str
    generated_answer: str
    source_references: list[dict[str, Any]]
    persisted_question_id: int
    execution_metadata: dict[str, Any]
    errors: list[str]
