from __future__ import annotations

from app.models import QuestionHistory, RepositoryIndex


class QuestionHistoryRepository:
    def create(
        self,
        *,
        repository: RepositoryIndex,
        thread_id: str,
        question: str,
        answer: str,
        source_references: list[dict],
        execution_metadata: dict,
        status: str,
    ) -> QuestionHistory:
        return QuestionHistory.objects.create(
            repository=repository,
            thread_id=thread_id,
            question=question,
            answer=answer,
            source_references=source_references,
            execution_metadata=execution_metadata,
            status=status,
        )

    def list_history(self, repository_url: str | None = None):
        queryset = QuestionHistory.objects.select_related("repository")
        if repository_url:
            queryset = queryset.filter(repository__repository_url=repository_url)
        return queryset
