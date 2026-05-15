from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
}

SKIP_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
}


@dataclass(slots=True)
class ParsedRepository:
    documents: list[Document]
    total_files: int
    indexed_files: int
    skipped_files: int
    detected_languages: list[str]


class RepositoryCodeChunker:
    def parse_repository(self, repository_path: str) -> ParsedRepository:
        root = Path(repository_path)
        documents: list[Document] = []
        total_files = 0
        indexed_files = 0
        skipped_files = 0
        languages: set[str] = set()

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            total_files += 1

            if any(part in SKIP_DIRECTORIES for part in path.parts):
                skipped_files += 1
                continue

            language = SUPPORTED_EXTENSIONS.get(path.suffix.lower())
            if language is None:
                skipped_files += 1
                continue

            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                skipped_files += 1
                continue

            relative_path = path.relative_to(root).as_posix()
            file_documents = self._build_documents(relative_path, language, content)
            if not file_documents:
                skipped_files += 1
                continue

            documents.extend(file_documents)
            indexed_files += 1
            languages.add(language)

        return ParsedRepository(
            documents=documents,
            total_files=total_files,
            indexed_files=indexed_files,
            skipped_files=skipped_files,
            detected_languages=sorted(languages),
        )

    def _build_documents(self, relative_path: str, language: str, content: str) -> list[Document]:
        if language == "python":
            return self._build_python_documents(relative_path, content)
        return self._build_generic_documents(relative_path, language, content)

    def _build_python_documents(self, relative_path: str, content: str) -> list[Document]:
        lines = content.splitlines()
        documents: list[Document] = []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._build_generic_documents(relative_path, "python", content)

        module_doc = self._create_document(
            content=content,
            file_path=relative_path,
            language="python",
            symbol_name="module",
            symbol_type="module",
            line_start=1,
            line_end=max(len(lines), 1),
            chunk_index=0,
        )
        documents.append(module_doc)

        chunk_index = 1
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.FunctionDef)):
                continue
            end_lineno = getattr(node, "end_lineno", node.lineno)
            snippet = "\n".join(lines[node.lineno - 1 : end_lineno]).strip()
            if not snippet:
                continue
            symbol_type = "class" if isinstance(node, ast.ClassDef) else "function"
            documents.append(
                self._create_document(
                    content=snippet,
                    file_path=relative_path,
                    language="python",
                    symbol_name=node.name,
                    symbol_type=symbol_type,
                    line_start=node.lineno,
                    line_end=end_lineno,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

        return documents

    def _build_generic_documents(
        self,
        relative_path: str,
        language: str,
        content: str,
        lines_per_chunk: int = 120,
    ) -> list[Document]:
        lines = content.splitlines()
        documents: list[Document] = []
        for chunk_index, start in enumerate(range(0, len(lines) or 1, lines_per_chunk)):
            end = min(start + lines_per_chunk, len(lines) or 1)
            snippet = "\n".join(lines[start:end]).strip() or content[:4000]
            documents.append(
                self._create_document(
                    content=snippet,
                    file_path=relative_path,
                    language=language,
                    symbol_name=f"chunk_{chunk_index}",
                    symbol_type="module_chunk",
                    line_start=start + 1,
                    line_end=end,
                    chunk_index=chunk_index,
                )
            )
        return documents

    def _create_document(
        self,
        *,
        content: str,
        file_path: str,
        language: str,
        symbol_name: str,
        symbol_type: str,
        line_start: int,
        line_end: int,
        chunk_index: int,
    ) -> Document:
        return Document(
            page_content=content,
            metadata={
                "file_path": file_path,
                "language": language,
                "symbol_name": symbol_name,
                "symbol_type": symbol_type,
                "line_start": line_start,
                "line_end": line_end,
                "chunk_index": chunk_index,
            },
        )
