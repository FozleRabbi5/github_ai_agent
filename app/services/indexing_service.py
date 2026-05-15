from __future__ import annotations

from langchain_core.documents import Document

from app.schemas.dto import RepositoryScanResult
from app.utils.code_parser import RepositoryCodeChunker


class RepositoryIndexingService:
    def __init__(self, chunker: RepositoryCodeChunker) -> None:
        self._chunker = chunker

    def scan_and_chunk(self, repository_path: str) -> tuple[RepositoryScanResult, list[Document]]:
        parsed = self._chunker.parse_repository(repository_path)
        return (
            RepositoryScanResult(
                total_files=parsed.total_files,
                indexed_files=parsed.indexed_files,
                skipped_files=parsed.skipped_files,
                detected_languages=parsed.detected_languages,
            ),
            parsed.documents,
        )
