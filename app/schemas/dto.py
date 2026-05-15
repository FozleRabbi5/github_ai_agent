from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class ClonedRepository:
    repository_url: str
    repository_name: str
    local_path: str
    default_branch: str
    commit_hash: str
    was_cloned: bool


@dataclass(slots=True)
class RepositoryScanResult:
    total_files: int
    indexed_files: int
    skipped_files: int
    detected_languages: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SourceReference:
    file_path: str
    language: str
    symbol_name: str
    symbol_type: str
    line_start: int
    line_end: int
    chunk_index: int
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievedCodeChunk:
    content: str
    file_path: str
    language: str
    symbol_name: str
    symbol_type: str
    line_start: int
    line_end: int
    chunk_index: int
    score: float | None = None

    def to_reference(self) -> SourceReference:
        return SourceReference(
            file_path=self.file_path,
            language=self.language,
            symbol_name=self.symbol_name,
            symbol_type=self.symbol_type,
            line_start=self.line_start,
            line_end=self.line_end,
            chunk_index=self.chunk_index,
            score=self.score,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExecutionStep:
    node: str
    started_at: datetime
    completed_at: datetime
    duration_ms: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["started_at"] = self.started_at.isoformat()
        payload["completed_at"] = self.completed_at.isoformat()
        return payload
