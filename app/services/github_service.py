from __future__ import annotations

import re
from pathlib import Path

from django.conf import settings
from git import Repo

from app.schemas.dto import ClonedRepository


class GitHubRepositoryService:
    def __init__(self) -> None:
        self._base_dir = Path(settings.BASE_DIR) / "repositories"
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def clone_or_update(self, repository_url: str) -> ClonedRepository:
        repository_name = self._extract_repository_name(repository_url)
        local_path = self._base_dir / repository_name
        was_cloned = False

        if not local_path.exists():
            repo = Repo.clone_from(repository_url, local_path)
            was_cloned = True
        else:
            repo = Repo(local_path)

        branch = self._safe_branch_name(repo)
        commit_hash = repo.head.commit.hexsha

        return ClonedRepository(
            repository_url=repository_url,
            repository_name=repository_name,
            local_path=str(local_path),
            default_branch=branch,
            commit_hash=commit_hash,
            was_cloned=was_cloned,
        )

    def _extract_repository_name(self, repository_url: str) -> str:
        slug = repository_url.rstrip("/").split("/")[-1].replace(".git", "")
        sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "-", slug).strip("-")
        if not sanitized:
            raise ValueError("Unable to derive repository name from URL.")
        return sanitized

    def _safe_branch_name(self, repo: Repo) -> str:
        try:
            return repo.active_branch.name
        except TypeError:
            return ""
