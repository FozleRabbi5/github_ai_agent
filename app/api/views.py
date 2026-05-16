import logging
import re

from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import status
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response
from rest_framework.views import APIView

from app.api.serializers import (
    RepositorySerializer,
    ResearchSessionDetailSerializer,
    ResearchSessionListSerializer,
    StartSessionRequestSerializer,
)
from app.models import Repository, ResearchSession

logger = logging.getLogger(__name__)


def extract_github_url_from_text(text: str) -> str | None:
    # 1. Regex extraction
    match = re.search(r"(https?://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)", text)
    if match:
        return match.group(1).rstrip('.')
    
    # 2. OpenAI Fallback
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        messages = [
            SystemMessage(content="Extract the GitHub repository URL from the following text. Return ONLY the URL. If no GitHub URL is found, return the exact word 'NONE'."),
            HumanMessage(content=text)
        ]
        response = llm.invoke(messages)
        content = response.content.strip()
        if content != "NONE" and "github.com" in content:
            return content
    except Exception as e:
        logger.error(f"OpenAI extraction failed: {e}")
        
    return None

def clean_question_text(question: str, url: str) -> str:
    cleaned = question.replace(url, "")
    # Remove separators like '-' and extra spaces
    cleaned = re.sub(r'^\s*-\s*', '', cleaned)
    cleaned = re.sub(r'\s*-\s*$', '', cleaned)
    # Strip quotes and extra spaces
    cleaned = cleaned.strip(" '\"- \t\n\r")
    return cleaned


class StartResearchSessionView(APIView):
    """Start a new research session: clone/update a repo and run the AI agent."""

    @extend_schema(
        tags=["Research Sessions"],
        operation_id="startResearchSession",
        request=StartSessionRequestSerializer,
        responses={201: ResearchSessionDetailSerializer},
        description="Start a new research session against a GitHub repository.",
    )
    def post(self, request, *args, **kwargs):
        serializer = StartSessionRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        question_text = serializer.validated_data["question"]
        
        repo_url = extract_github_url_from_text(question_text)
        if not repo_url:
            raise ValidationError("No GitHub repository URL found in the question.")
            
        cleaned_question = clean_question_text(question_text, repo_url)

        from app.services.research_service import get_research_service

        service = get_research_service()
        session = service.run_session(
            repo_url=repo_url,
            question=cleaned_question,
        )

        response_serializer = ResearchSessionDetailSerializer(session)
        return Response(response_serializer.data, status=status.HTTP_201_CREATED)


class ResearchSessionListView(APIView):
    """List research sessions, optionally filtered by repo_url."""

    @extend_schema(
        tags=["Research Sessions"],
        operation_id="listResearchSessions",
        parameters=[
            OpenApiParameter(
                name="repo_url",
                type=str,
                location=OpenApiParameter.QUERY,
                required=False,
                description="Filter sessions by repository URL.",
            ),
        ],
        responses={200: ResearchSessionListSerializer(many=True)},
        description="List past research sessions, optionally filtered by repository URL.",
    )
    def get(self, request, *args, **kwargs):
        repo_url = request.query_params.get("repo_url")
        queryset = ResearchSession.objects.select_related("repository").order_by("-created_at")
        if repo_url:
            queryset = queryset.filter(repository__url=repo_url)
        serializer = ResearchSessionListSerializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class ResearchSessionDetailView(APIView):
    """Retrieve a single research session with all tool calls and findings."""

    @extend_schema(
        tags=["Research Sessions"],
        operation_id="getResearchSession",
        responses={200: ResearchSessionDetailSerializer},
        description="Retrieve a research session by its session_id, including tool calls and findings.",
    )
    def get(self, request, session_id, *args, **kwargs):
        try:
            session = (
                ResearchSession.objects.select_related("repository")
                .prefetch_related("tool_calls", "findings")
                .get(session_id=session_id)
            )
        except ResearchSession.DoesNotExist:
            return Response(
                {"detail": f"Session {session_id} not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
        serializer = ResearchSessionDetailSerializer(session)
        return Response(serializer.data, status=status.HTTP_200_OK)


class RepositoryListView(APIView):
    """List all researched repositories."""

    @extend_schema(
        tags=["Repositories"],
        operation_id="listRepositories",
        responses={200: RepositorySerializer(many=True)},
        description="List all GitHub repositories that have been researched.",
    )
    def get(self, request, *args, **kwargs):
        repos = Repository.objects.all()
        serializer = RepositorySerializer(repos, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class RepositoryDetailView(APIView):
    """Retrieve a single repository."""

    @extend_schema(
        tags=["Repositories"],
        operation_id="getRepository",
        responses={200: RepositorySerializer},
        description="Retrieve details of a single researched repository.",
    )
    def get(self, request, pk, *args, **kwargs):
        try:
            repo = Repository.objects.get(pk=pk)
        except Repository.DoesNotExist:
            return Response(
                {"detail": "Repository not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
        serializer = RepositorySerializer(repo)
        return Response(serializer.data, status=status.HTTP_200_OK)
