from django.test import TestCase

from app.api.serializers import QuestionAskRequestSerializer
from app.api.views import (
    _remove_extracted_url_from_text,
    _resolve_repository_url_and_question,
    extract_github_url_with_regex,
)
from app.utils.code_parser import RepositoryCodeChunker


class QuestionAskRequestSerializerTests(TestCase):
    def test_accepts_embedded_repository_url_in_question(self):
        serializer = QuestionAskRequestSerializer(
            data={"question": "https://github.com/example/repo What does this do?"}
        )
        self.assertTrue(serializer.is_valid(), serializer.errors)


class GitHubUrlParsingTests(TestCase):
    def test_extract_github_url_with_regex(self):
        extracted_url = extract_github_url_with_regex(
            "Please inspect github.com/tiangolo/fastapi for me."
        )
        self.assertEqual(extracted_url, "https://github.com/tiangolo/fastapi")

    def test_remove_extracted_url_from_text(self):
        cleaned_question = _remove_extracted_url_from_text(
            'https://github.com/tiangolo/fastapi - "How does FastAPI handle dependency injection internally?"'
        )
        self.assertEqual(
            cleaned_question,
            "How does FastAPI handle dependency injection internally?",
        )

    def test_resolve_repository_url_and_question(self):
        repository_url, question = _resolve_repository_url_and_question(
            question='https://github.com/tiangolo/fastapi - "How does FastAPI handle dependency injection internally?"',
        )
        self.assertEqual(repository_url, "https://github.com/tiangolo/fastapi")
        self.assertEqual(
            question,
            "How does FastAPI handle dependency injection internally?",
        )


class RepositoryCodeChunkerTests(TestCase):
    def test_python_chunking_extracts_symbols(self):
        content = """
class Service:
    def run(self):
        return "ok"

def helper():
    return 1
""".strip()
        documents = RepositoryCodeChunker()._build_python_documents("service.py", content)
        symbols = {document.metadata["symbol_name"] for document in documents}
        self.assertIn("module", symbols)
        self.assertIn("Service", symbols)
        self.assertIn("run", symbols)
        self.assertIn("helper", symbols)
