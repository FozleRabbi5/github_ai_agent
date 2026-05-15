"""
Management command to seed the database with sample research data.

Usage:
    python manage.py seed_sample_data          # Creates mock data (no API key needed)
    python manage.py seed_sample_data --live   # Runs the real agent (requires OPENAI_API_KEY)
"""
import uuid
from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from app.models import Finding, Repository, ResearchSession, ToolCall


class Command(BaseCommand):
    help = "Seed the database with sample research session data."

    def add_arguments(self, parser):
        parser.add_argument(
            "--live",
            action="store_true",
            help="Run a real agent session against a small repo (requires OPENAI_API_KEY).",
        )
        parser.add_argument(
            "--repo-url",
            type=str,
            default="https://github.com/pallets/flask",
            help="Repository URL for live mode.",
        )
        parser.add_argument(
            "--question",
            type=str,
            default="How does Flask handle routing and URL rule matching?",
            help="Question for live mode.",
        )

    def handle(self, *args, **options):
        if options["live"]:
            self._run_live(options["repo_url"], options["question"])
        else:
            self._create_mock_data()

    def _run_live(self, repo_url: str, question: str):
        self.stdout.write(f"Running live agent against {repo_url}...")
        self.stdout.write(f"Question: {question}")
        from app.services.research_service import get_research_service

        service = get_research_service()
        session = service.run_session(repo_url=repo_url, question=question)
        self.stdout.write(self.style.SUCCESS(
            f"Session {session.session_id} completed: "
            f"{session.total_tool_calls} tool calls, "
            f"{session.total_findings} findings, "
            f"{session.total_tokens} tokens"
        ))

    def _create_mock_data(self):
        self.stdout.write("Creating mock sample data...")
        now = timezone.now()

        # --- Repository 1: Flask ---
        repo1, _ = Repository.objects.update_or_create(
            url="https://github.com/pallets/flask",
            defaults={
                "name": "flask",
                "local_path": "/tmp/repos/flask",
                "default_branch": "main",
                "last_commit_hash": "a1b2c3d4e5f6",
                "status": Repository.Status.CLONED,
                "last_analyzed_at": now - timedelta(hours=2),
            },
        )

        # --- Repository 2: FastAPI ---
        repo2, _ = Repository.objects.update_or_create(
            url="https://github.com/tiangolo/fastapi",
            defaults={
                "name": "fastapi",
                "local_path": "/tmp/repos/fastapi",
                "default_branch": "master",
                "last_commit_hash": "f6e5d4c3b2a1",
                "status": Repository.Status.CLONED,
                "last_analyzed_at": now - timedelta(days=1),
            },
        )

        # --- Session 1: Flask routing ---
        s1 = ResearchSession.objects.create(
            repository=repo1,
            session_id=uuid.uuid4(),
            question="How does Flask handle routing and URL rule matching?",
            answer=(
                "Flask uses Werkzeug's URL routing system. Routes are registered via "
                "`@app.route()` decorator which calls `add_url_rule()` in `app.py` "
                "(line 1234). The `Map` class in `werkzeug/routing/map.py` manages "
                "all URL rules and performs matching via the `MapAdapter.match()` "
                "method. URL rules support converters (string, int, float, path, "
                "uuid) defined in `werkzeug/routing/converters.py`. When a request "
                "arrives, `Flask.full_dispatch_request()` calls the URL adapter to "
                "find the matching endpoint and view function."
            ),
            status=ResearchSession.Status.COMPLETED,
            total_tool_calls=8,
            total_findings=3,
            prompt_tokens=12500,
            completion_tokens=2800,
            total_tokens=15300,
            duration_seconds=18.5,
            completed_at=now - timedelta(hours=2),
        )

        # Tool calls for session 1
        _tools_s1 = [
            ("get_previous_findings", {"repo_url": ""}, "No previous findings for this repository.", 1),
            ("list_files", {"path": "."}, "[dir] src/flask\n[dir] tests\n[file] setup.py\n[file] README.md", 2),
            ("list_files", {"path": "src/flask"}, "[file] app.py\n[file] blueprints.py\n[file] config.py\n[file] ctx.py\n[file] wrappers.py", 3),
            ("get_file_summary", {"path": "src/flask/app.py"}, "File: src/flask/app.py\nLanguage: Python\nLines: 1500\nClasses: Flask\nFunctions: add_url_rule, route, dispatch_request", 4),
            ("read_file", {"path": "src/flask/app.py", "start_line": 1200, "end_line": 1300}, "def add_url_rule(self, rule, endpoint=None, view_func=None, ...):\n    ...", 5),
            ("search_code", {"query": "url_rule_class"}, "Found 3 matches:\nsrc/flask/app.py:89\nsrc/flask/scaffold.py:45", 6),
            ("save_finding", {"file_path": "src/flask/app.py", "note": "Flask.add_url_rule() delegates to Werkzeug's Map for URL matching", "finding_type": "architecture"}, "Finding saved.", 7),
            ("save_finding", {"file_path": "src/flask/app.py", "note": "Routes registered via @app.route() decorator which wraps add_url_rule()", "finding_type": "pattern"}, "Finding saved.", 8),
        ]
        for name, inp, out, step in _tools_s1:
            ToolCall.objects.create(
                session=s1, tool_name=name, tool_input=inp,
                tool_output=out, step_number=step, duration_ms=step * 15.0,
            )

        # Findings for session 1
        Finding.objects.create(
            session=s1, file_path="src/flask/app.py",
            note="Flask.add_url_rule() delegates to Werkzeug's Map for URL matching",
            finding_type=Finding.FindingType.ARCHITECTURE, step_number=7,
        )
        Finding.objects.create(
            session=s1, file_path="src/flask/app.py",
            note="Routes registered via @app.route() decorator which wraps add_url_rule()",
            finding_type=Finding.FindingType.PATTERN, step_number=8,
        )
        Finding.objects.create(
            session=s1, file_path="src/flask/wrappers.py",
            note="Request/Response classes extend Werkzeug wrappers",
            finding_type=Finding.FindingType.DEPENDENCY, step_number=6,
        )

        # --- Session 2: FastAPI dependency injection ---
        s2 = ResearchSession.objects.create(
            repository=repo2,
            session_id=uuid.uuid4(),
            question="How does FastAPI implement dependency injection?",
            answer=(
                "FastAPI's dependency injection is built on top of Starlette and "
                "uses Python's type hints. Dependencies are declared as function "
                "parameters with `Depends()` in `fastapi/dependencies/utils.py`. "
                "The `solve_dependencies()` function recursively resolves the "
                "dependency tree, handling sub-dependencies, caching (via "
                "`use_cache=True`), and generator-based dependencies for cleanup. "
                "Dependencies are resolved per-request and cached within the request scope."
            ),
            status=ResearchSession.Status.COMPLETED,
            total_tool_calls=6,
            total_findings=2,
            prompt_tokens=10200,
            completion_tokens=2100,
            total_tokens=12300,
            duration_seconds=14.2,
            completed_at=now - timedelta(days=1),
        )

        _tools_s2 = [
            ("get_previous_findings", {"repo_url": ""}, "No previous findings.", 1),
            ("list_files", {"path": "."}, "[dir] fastapi\n[dir] tests\n[file] setup.py", 2),
            ("list_files", {"path": "fastapi"}, "[file] applications.py\n[dir] dependencies\n[file] routing.py", 3),
            ("get_file_summary", {"path": "fastapi/dependencies/utils.py"}, "File: fastapi/dependencies/utils.py\nLines: 800\nFunctions: solve_dependencies, get_dependant", 4),
            ("read_file", {"path": "fastapi/dependencies/utils.py", "start_line": 1, "end_line": 100}, "...", 5),
            ("save_finding", {"file_path": "fastapi/dependencies/utils.py", "note": "solve_dependencies() recursively resolves DI tree with per-request caching", "finding_type": "architecture"}, "Finding saved.", 6),
        ]
        for name, inp, out, step in _tools_s2:
            ToolCall.objects.create(
                session=s2, tool_name=name, tool_input=inp,
                tool_output=out, step_number=step, duration_ms=step * 12.0,
            )

        Finding.objects.create(
            session=s2, file_path="fastapi/dependencies/utils.py",
            note="solve_dependencies() recursively resolves DI tree with per-request caching",
            finding_type=Finding.FindingType.ARCHITECTURE, step_number=6,
        )
        Finding.objects.create(
            session=s2, file_path="fastapi/params.py",
            note="Depends() is a simple dataclass that stores the dependency callable and use_cache flag",
            finding_type=Finding.FindingType.PATTERN, step_number=5,
        )

        # --- Session 3: Failed session example ---
        ResearchSession.objects.create(
            repository=repo1,
            session_id=uuid.uuid4(),
            question="Explain Flask's signal system in detail",
            answer="",
            status=ResearchSession.Status.FAILED,
            error_message="Token budget exceeded (102000 > 100000)",
            total_tool_calls=15,
            total_findings=0,
            prompt_tokens=85000,
            completion_tokens=17000,
            total_tokens=102000,
            duration_seconds=45.0,
            completed_at=now - timedelta(hours=5),
        )

        self.stdout.write(self.style.SUCCESS(
            f"Created {Repository.objects.count()} repositories, "
            f"{ResearchSession.objects.count()} sessions, "
            f"{ToolCall.objects.count()} tool calls, "
            f"{Finding.objects.count()} findings."
        ))
