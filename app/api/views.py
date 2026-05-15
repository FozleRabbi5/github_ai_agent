import re
from typing import Final

from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import status
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response
from rest_framework.views import APIView

from app.api.serializers import (
    QuestionAskRequestSerializer,
    QuestionAskResponseSerializer,
    QuestionHistoryItemSerializer,
    RepositoryIndexRequestSerializer,
    RepositoryIndexResponseSerializer,
)
from app.services.rag_service import get_repository_ai_service
from app.utils.logging import get_logger, log_event


logger = get_logger(__name__)

GITHUB_REPOSITORY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?P<full>(?:https?://)?github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+))/?",
    flags=re.IGNORECASE,
)


def extract_github_url_with_regex(text: str) -> str | None:
    match = GITHUB_REPOSITORY_PATTERN.search(text)
    if match is None:
        return None

    owner = match.group("owner")
    repo = match.group("repo")
    return f"https://github.com/{owner}/{repo}".rstrip("/")


def extract_github_url_with_llm(text: str) -> str | None:
    try:
        return get_repository_ai_service().llm_service.extract_github_repository_url(text)
    except Exception as exc:
        log_event(
            logger,
            "question_ask.github_url_llm_extraction_failed",
            error=str(exc),
        )
        return None


def _remove_extracted_url_from_text(text: str) -> str:
    cleaned_text = GITHUB_REPOSITORY_PATTERN.sub("", text, count=1)
    cleaned_text = re.sub(r'^[\s"\'`.,:;|/-]+', "", cleaned_text)
    cleaned_text = re.sub(r'[\s"\'`.,:;|/-]+$', "", cleaned_text)
    return cleaned_text.strip()


def _resolve_repository_url_and_question(
    question: str,
) -> tuple[str, str]:
    """
    Resolve the repository URL from free-form user input and return the cleaned
    technical question without the repository URL.
    """
    extracted_repository_url = extract_github_url_with_regex(question)
    if extracted_repository_url is None:
        extracted_repository_url = extract_github_url_with_llm(question)

    if extracted_repository_url is None:
        raise ValidationError(
            {"message": "No GitHub repository URL found in the input text."}
        )

    cleaned_question = _remove_extracted_url_from_text(question)
    return extracted_repository_url, cleaned_question or question.strip()


class RepositoryIndexView(APIView):
    """
    Clone a GitHub repository if needed and build or refresh its vector index.
    """

    @extend_schema(
        tags=["Repositories"],
        operation_id="indexRepository",
        request=RepositoryIndexRequestSerializer,
        responses={200: RepositoryIndexResponseSerializer},
        description="Clone a GitHub repository if needed and build or refresh its vector index.",
    )
    def post(self, request, *args, **kwargs):
        serializer = RepositoryIndexRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        service = get_repository_ai_service()

        index = service.index_repository(
            repository_url=serializer.validated_data["repository_url"]
        )

        response_serializer = RepositoryIndexResponseSerializer(index)
        return Response(response_serializer.data, status=status.HTTP_200_OK)


class QuestionAskView(APIView):
    """
    Ask a technical question about an indexed repository and receive a grounded answer.
    """

    @extend_schema(
        tags=["Questions"],
        operation_id="askRepositoryQuestion",
        request=QuestionAskRequestSerializer,
        responses={200: QuestionAskResponseSerializer},
        description="Ask a technical question about an indexed repository and receive a grounded answer with source references.",
    )
    def post(self, request, *args, **kwargs):
        serializer = QuestionAskRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        service = get_repository_ai_service()

        # Resolve the repository URL from either an explicit field or the
        # free-form question text, then remove the URL so the remaining text
        # becomes the grounded technical question for the agent.
        repository_url, question = _resolve_repository_url_and_question(
            question=serializer.validated_data["question"],
        )
        log_event(
            logger,
            "question_ask.input_resolved",
            repository_url=repository_url,
            question_length=len(question),
        )

        result = service.ask_question(
            repository_url=repository_url,
            question=question,
        )

        response_serializer = QuestionAskResponseSerializer(result)
        return Response(response_serializer.data, status=status.HTTP_200_OK)


class QuestionHistoryView(APIView):
    """
    List persisted repository question-answer history, optionally filtered by repository URL.
    """

    @extend_schema(
        tags=["Questions"],
        operation_id="listQuestionHistory",
        parameters=[
            OpenApiParameter(
                name="repository_url",
                type=str,
                location=OpenApiParameter.QUERY,
                required=False,
                description="Optional repository URL used to filter question history.",
            )
        ],
        responses={200: QuestionHistoryItemSerializer(many=True)},
        description="List persisted repository question-answer history.",
    )
    def get(self, request, *args, **kwargs):
        repository_url = request.query_params.get("repository_url")
        history = get_repository_ai_service().get_history(repository_url=repository_url)
        serializer = QuestionHistoryItemSerializer(history, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
