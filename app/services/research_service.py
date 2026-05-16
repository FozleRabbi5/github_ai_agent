"""
Research service — orchestrates the agent lifecycle.
"""
from __future__ import annotations

import logging
import time
from functools import lru_cache

from django.utils import timezone

from app.models import Repository, ResearchSession
from app.services.github_service import GitHubRepositoryService

logger = logging.getLogger(__name__)


class ResearchService:
    def __init__(self) -> None:
        self.git_service = GitHubRepositoryService()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_session(self, *, repo_url: str, question: str) -> ResearchSession:
        """Clone/update a repo and start the research agent in the background."""
        
        # 1. Clone or update the repository (fast enough to be synchronous for metadata)
        # We might want to do this in the background too, but for now we'll do it to get the repo name immediately
        cloned = self.git_service.clone_or_update(repo_url)

        # 2. Upsert the Repository record
        repo, _ = Repository.objects.update_or_create(
            url=repo_url,
            defaults={
                "name": cloned.repository_name,
                "local_path": str(cloned.local_path),
                "default_branch": cloned.default_branch,
                "last_commit_hash": cloned.commit_hash,
                "status": Repository.Status.CLONED,
                "last_analyzed_at": timezone.now(),
            },
        )

        # 3. Create the session
        session = ResearchSession.objects.create(
            repository=repo,
            question=question,
            status=ResearchSession.Status.RUNNING,
        )

        # 4. Run the agent synchronously so the API blocks and returns the final answer
        self._run_agent_background(repo_url, str(cloned.local_path), session.id, question)
        
        # 5. Refresh from DB to get the final state
        session.refresh_from_db()

        return session

    def _run_agent_background(self, repo_url: str, local_path: str, session_id: int, question: str) -> None:
        """The background worker that executes the LangGraph agent."""
        start = time.perf_counter()
        
        # We need to re-fetch the session inside the thread
        session = ResearchSession.objects.get(pk=session_id)
        
        try:
            from app.agents.repository_agent import ResearchAgent

            agent = ResearchAgent(
                repo_local_path=local_path,
                repo_url=repo_url,
                session=session,
            )
            result = agent.run(question)

            # 5. Finalise session
            duration = time.perf_counter() - start
            session.answer = result.get("answer", "No answer provided.")
            session.source_references = result.get("source_references", [])
            session.prompt_tokens = result.get("prompt_tokens", 0)
            session.completion_tokens = result.get("completion_tokens", 0)
            session.total_tokens = result.get("total_tokens", 0)
            session.total_tool_calls = session.tool_calls.count()
            session.total_findings = session.findings.count()
            session.duration_seconds = round(duration, 2)
            session.status = ResearchSession.Status.COMPLETED
            session.completed_at = timezone.now()
            session.save()

        except Exception as exc:
            logger.exception("Research session %s failed", session.session_id)
            session.status = ResearchSession.Status.FAILED
            session.error_message = str(exc)[:2000]
            session.duration_seconds = round(time.perf_counter() - start, 2)
            session.completed_at = timezone.now()
            session.save()

    def get_session(self, session_id: str) -> ResearchSession | None:
        try:
            return (
                ResearchSession.objects.select_related("repository")
                .prefetch_related("tool_calls", "findings")
                .get(session_id=session_id)
            )
        except ResearchSession.DoesNotExist:
            return None

    def list_sessions(self, repo_url: str | None = None):
        qs = ResearchSession.objects.select_related("repository").order_by("-created_at")
        if repo_url:
            qs = qs.filter(repository__url=repo_url)
        return qs


@lru_cache(maxsize=1)
def get_research_service() -> ResearchService:
    return ResearchService()
