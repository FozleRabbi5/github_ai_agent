"""
Tests for the GitHub AI Agent.

Covers: models, tools, API endpoints, and agent loop control.
"""
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient

from app.agents.tools import build_tools
from app.models import Finding, Repository, ResearchSession, ToolCall


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class RepositoryModelTests(TestCase):
    def test_create_repository(self):
        repo = Repository.objects.create(
            url="https://github.com/test/repo",
            name="repo",
            status=Repository.Status.CLONED,
        )
        self.assertEqual(str(repo), "repo (cloned)")
        self.assertIsNotNone(repo.created_at)

    def test_url_uniqueness(self):
        Repository.objects.create(url="https://github.com/test/repo", name="repo")
        with self.assertRaises(Exception):
            Repository.objects.create(url="https://github.com/test/repo", name="repo2")


class ResearchSessionModelTests(TestCase):
    def setUp(self):
        self.repo = Repository.objects.create(
            url="https://github.com/test/repo", name="repo",
        )

    def test_create_session(self):
        session = ResearchSession.objects.create(
            repository=self.repo,
            question="How does X work?",
            status=ResearchSession.Status.RUNNING,
        )
        self.assertIsNotNone(session.session_id)
        self.assertEqual(session.total_tool_calls, 0)
        self.assertEqual(session.total_tokens, 0)

    def test_session_relationships(self):
        session = ResearchSession.objects.create(
            repository=self.repo, question="test",
        )
        ToolCall.objects.create(
            session=session, tool_name="list_files",
            tool_input={"path": "."}, tool_output="files...",
            step_number=1,
        )
        Finding.objects.create(
            session=session, file_path="main.py",
            note="Entry point", step_number=1,
        )
        self.assertEqual(session.tool_calls.count(), 1)
        self.assertEqual(session.findings.count(), 1)

    def test_cascade_delete(self):
        session = ResearchSession.objects.create(
            repository=self.repo, question="test",
        )
        ToolCall.objects.create(
            session=session, tool_name="test", tool_input={},
            tool_output="", step_number=1,
        )
        Finding.objects.create(
            session=session, file_path="", note="test", step_number=1,
        )
        session.delete()
        self.assertEqual(ToolCall.objects.count(), 0)
        self.assertEqual(Finding.objects.count(), 0)


class ToolCallModelTests(TestCase):
    def test_output_truncation(self):
        repo = Repository.objects.create(
            url="https://github.com/test/repo", name="repo",
        )
        session = ResearchSession.objects.create(
            repository=repo, question="test",
        )
        long_output = "x" * 15_000
        tc = ToolCall.objects.create(
            session=session, tool_name="read_file",
            tool_input={"path": "big.py"}, tool_output=long_output,
            step_number=1,
        )
        tc.refresh_from_db()
        self.assertLessEqual(len(tc.tool_output), 10_100)
        self.assertIn("truncated", tc.tool_output)


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------


class ToolFunctionTests(TestCase):
    def setUp(self):
        self.repo = Repository.objects.create(
            url="https://github.com/test/repo", name="repo",
        )
        self.session = ResearchSession.objects.create(
            repository=self.repo, question="test",
        )
        self.tmpdir = tempfile.mkdtemp()
        # Create a small test repo structure
        root = Path(self.tmpdir)
        (root / "src").mkdir()
        (root / "src" / "main.py").write_text(
            "class App:\n    def run(self):\n        return 'hello'\n\ndef helper():\n    pass\n"
        )
        (root / "README.md").write_text("# Test Project\nA test repo.\n")
        (root / ".git").mkdir()  # Should be skipped

        self.step_counter = [0]
        self.tools = build_tools(
            self.tmpdir, "https://github.com/test/repo",
            self.session, self.step_counter,
        )
        self.tool_map = {fn.__name__: fn for fn in self.tools}

    def test_list_files(self):
        result = self.tool_map["list_files"](".")
        self.assertIn("README.md", result)
        self.assertIn("src", result)
        self.assertNotIn(".git", result)  # Should be skipped
        self.assertEqual(ToolCall.objects.count(), 1)

    def test_read_file(self):
        result = self.tool_map["read_file"]("src/main.py")
        self.assertIn("class App", result)
        self.assertIn("def run", result)

    def test_read_file_not_found(self):
        result = self.tool_map["read_file"]("nonexistent.py")
        self.assertIn("Error", result)

    def test_search_code(self):
        result = self.tool_map["search_code"]("class App")
        self.assertIn("main.py", result)
        self.assertIn("class App", result)

    def test_search_code_no_matches(self):
        result = self.tool_map["search_code"]("nonexistent_pattern_xyz")
        self.assertIn("No matches", result)

    def test_get_file_summary(self):
        result = self.tool_map["get_file_summary"]("src/main.py")
        self.assertIn("Python", result)
        self.assertIn("App", result)
        self.assertIn("helper", result)

    def test_save_finding(self):
        result = self.tool_map["save_finding"]("src/main.py", "Entry point class", "architecture")
        self.assertIn("Finding saved", result)
        self.assertEqual(Finding.objects.count(), 1)
        finding = Finding.objects.first()
        self.assertEqual(finding.finding_type, "architecture")
        self.assertEqual(finding.file_path, "src/main.py")

    def test_get_previous_findings_empty(self):
        result = self.tool_map["get_previous_findings"]()
        self.assertIn("No previous findings", result)

    def test_list_past_sessions_empty(self):
        result = self.tool_map["list_past_sessions"]()
        self.assertIn("No previous research sessions", result)

    def test_tool_calls_logged_to_db(self):
        self.tool_map["list_files"](".")
        self.tool_map["read_file"]("README.md")
        self.tool_map["search_code"]("test")
        # save_finding creates 2 records (1 Finding + 1 ToolCall)
        expected_tool_calls = ToolCall.objects.count()
        self.assertGreaterEqual(expected_tool_calls, 3)


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------


class APIEndpointTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.repo = Repository.objects.create(
            url="https://github.com/test/repo",
            name="repo",
            status=Repository.Status.CLONED,
        )
        self.session = ResearchSession.objects.create(
            repository=self.repo,
            question="How does X work?",
            answer="X works by doing Y.",
            status=ResearchSession.Status.COMPLETED,
            total_tool_calls=5,
            total_findings=2,
            prompt_tokens=1000,
            completion_tokens=200,
            total_tokens=1200,
            duration_seconds=5.0,
            completed_at=timezone.now(),
        )
        ToolCall.objects.create(
            session=self.session, tool_name="list_files",
            tool_input={"path": "."}, tool_output="...",
            step_number=1, duration_ms=10.0,
        )
        Finding.objects.create(
            session=self.session, file_path="main.py",
            note="Entry point", step_number=1,
        )

    def test_list_repositories(self):
        resp = self.client.get("/api/repositories/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()), 1)
        self.assertEqual(resp.json()[0]["name"], "repo")

    def test_get_repository_detail(self):
        resp = self.client.get(f"/api/repositories/{self.repo.pk}/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["url"], "https://github.com/test/repo")

    def test_get_repository_not_found(self):
        resp = self.client.get("/api/repositories/9999/")
        self.assertEqual(resp.status_code, 404)

    def test_list_sessions(self):
        resp = self.client.get("/api/research/sessions/list/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()), 1)

    def test_list_sessions_filter_by_repo(self):
        resp = self.client.get(
            "/api/research/sessions/list/",
            {"repo_url": "https://github.com/test/repo"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()), 1)

    def test_list_sessions_filter_no_match(self):
        resp = self.client.get(
            "/api/research/sessions/list/",
            {"repo_url": "https://github.com/other/repo"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()), 0)

    def test_get_session_detail(self):
        resp = self.client.get(
            f"/api/research/sessions/{self.session.session_id}/"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["question"], "How does X work?")
        self.assertEqual(data["answer"], "X works by doing Y.")
        self.assertEqual(len(data["tool_calls"]), 1)
        self.assertEqual(len(data["findings"]), 1)
        self.assertIn("repository", data)

    def test_get_session_not_found(self):
        fake_id = uuid.uuid4()
        resp = self.client.get(f"/api/research/sessions/{fake_id}/")
        self.assertEqual(resp.status_code, 404)

    def test_start_session_validation(self):
        resp = self.client.post(
            "/api/research/sessions/",
            {"repo_url": "not-a-url", "question": "test"},
            format="json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_start_session_missing_question(self):
        resp = self.client.post(
            "/api/research/sessions/",
            {"repo_url": "https://github.com/test/repo"},
            format="json",
        )
        self.assertEqual(resp.status_code, 400)


# ---------------------------------------------------------------------------
# Agent loop tests
# ---------------------------------------------------------------------------


class AgentLoopTests(TestCase):
    """Test the agent respects iteration limits and handles tool errors."""

    def setUp(self):
        self.repo = Repository.objects.create(
            url="https://github.com/test/repo",
            name="repo",
            status=Repository.Status.CLONED,
        )
        self.session = ResearchSession.objects.create(
            repository=self.repo,
            question="test",
            status=ResearchSession.Status.RUNNING,
        )

    @patch("app.agents.repository_agent.ChatOpenAI")
    def test_agent_stops_on_no_tool_calls(self, mock_chat_openai_cls):
        """Agent should return when LLM produces a response with no tool calls."""
        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()
        
        mock_chat_openai_cls.return_value = mock_llm
        mock_llm.bind_tools.return_value = mock_llm_with_tools

        from langchain_core.messages import AIMessage
        
        # Mock a response with no tool calls (final answer)
        mock_message = AIMessage(
            content="The answer is 42.", 
            usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        )
        
        mock_llm_with_tools.invoke.return_value = mock_message

        from app.agents.repository_agent import ResearchAgent

        agent = ResearchAgent(
            repo_local_path="/tmp/test",
            repo_url="https://github.com/test/repo",
            session=self.session,
        )
        result = agent.run("What is the answer?")

        self.assertEqual(result["answer"], "The answer is 42.")
        self.assertEqual(result["prompt_tokens"], 100)
        self.assertEqual(result["completion_tokens"], 50)
        # LLM should have been called exactly once
        self.assertEqual(mock_llm_with_tools.invoke.call_count, 1)
