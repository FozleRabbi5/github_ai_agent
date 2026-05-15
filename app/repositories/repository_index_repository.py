from __future__ import annotations

from django.utils import timezone

from app.models import RepositoryIndex


class RepositoryIndexRepository:
    def get_by_id(self, repository_id: int) -> RepositoryIndex | None:
        return RepositoryIndex.objects.filter(id=repository_id).first()

    def get_by_url(self, repository_url: str) -> RepositoryIndex | None:
        return RepositoryIndex.objects.filter(repository_url=repository_url).first()

    def list_all(self):
        return RepositoryIndex.objects.all()

    def upsert_index(
        self,
        *,
        repository_url: str,
        repository_name: str,
        local_path: str,
        default_branch: str,
        last_commit_hash: str,
        status: str,
        metadata: dict,
    ) -> RepositoryIndex:
        index, _ = RepositoryIndex.objects.update_or_create(
            repository_url=repository_url,
            defaults={
                "repository_name": repository_name,
                "local_path": local_path,
                "default_branch": default_branch,
                "last_commit_hash": last_commit_hash,
                "status": status,
                "metadata": metadata,
                "indexed_at": timezone.now() if status == RepositoryIndex.Status.INDEXED else None,
            },
        )
        return index
