from __future__ import annotations

from functools import lru_cache

from app.agents.repository_agent import RepositoryQuestionAnsweringAgent
from app.api.serializers import RepositoryIndexModelSerializer
from app.models import RepositoryIndex
from app.repositories.question_history_repository import QuestionHistoryRepository
from app.repositories.repository_index_repository import RepositoryIndexRepository
from app.services.embedding_service import OpenAIEmbeddingService
from app.services.github_service import GitHubRepositoryService
from app.services.indexing_service import RepositoryIndexingService
from app.services.llm_service import OpenAILLMService
from app.services.vector_service import VectorIndexService
from app.utils.code_parser import RepositoryCodeChunker
from app.vectorstores.chroma_store import ChromaStoreFactory


class RepositoryAIService:
    def __init__(self) -> None:
        self.git_service = GitHubRepositoryService()
        self.embedding_service = OpenAIEmbeddingService()
        self.vector_service = VectorIndexService(
            ChromaStoreFactory(self.embedding_service.client)
        )
        self.indexing_service = RepositoryIndexingService(RepositoryCodeChunker())
        self.llm_service = OpenAILLMService()
        self.repository_index_repository = RepositoryIndexRepository()
        self.question_history_repository = QuestionHistoryRepository()
        self.agent = RepositoryQuestionAnsweringAgent(self)

    def index_repository(self, *, repository_url: str) -> dict:
        cloned = self.git_service.clone_or_update(repository_url)
        existing_index = self.repository_index_repository.get_by_url(repository_url)
        if (
            existing_index
            and existing_index.status == RepositoryIndex.Status.INDEXED
            and existing_index.last_commit_hash == cloned.commit_hash
        ):
            payload = dict(existing_index.metadata)
            payload["cache_hit"] = True
            existing_index.metadata = payload
            existing_index.save(update_fields=["metadata", "updated_at"])
            return RepositoryIndexModelSerializer(existing_index).data

        scan_summary, documents = self.indexing_service.scan_and_chunk(cloned.local_path)
        vector_metadata = self.vector_service.reindex_repository(repository_url, documents)
        index = self.repository_index_repository.upsert_index(
            repository_url=repository_url,
            repository_name=cloned.repository_name,
            local_path=cloned.local_path,
            default_branch=cloned.default_branch,
            last_commit_hash=cloned.commit_hash,
            status=RepositoryIndex.Status.INDEXED,
            metadata={
                **scan_summary.to_dict(),
                **vector_metadata,
                "cache_hit": False,
            },
        )
        return RepositoryIndexModelSerializer(index).data

    def ask_question(self, *, repository_url: str, question: str) -> dict:
        result = self.agent.invoke(repository_url=repository_url, question=question)
        return {
            "repository": result["repository_index_payload"],
            "thread_id": result["thread_id"],
            "question": question,
            "answer": result["generated_answer"],
            "source_references": result.get("source_references", []),
            "execution_metadata": result.get("execution_metadata", {}),
        }

    def get_history(self, repository_url: str | None = None):
        return self.question_history_repository.list_history(repository_url=repository_url)


@lru_cache(maxsize=1)
def get_repository_ai_service() -> RepositoryAIService:
    return RepositoryAIService()
