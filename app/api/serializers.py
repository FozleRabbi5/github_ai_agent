from rest_framework import serializers

from app.models import Finding, Repository, ResearchSession, ToolCall


# ---------------------------------------------------------------------------
# Request serializers
# ---------------------------------------------------------------------------


class StartSessionRequestSerializer(serializers.Serializer):
    repo_url = serializers.URLField(help_text="GitHub repository URL to research")
    question = serializers.CharField(
        help_text="The research question to investigate",
        allow_blank=False,
        trim_whitespace=True,
    )


# ---------------------------------------------------------------------------
# Model serializers
# ---------------------------------------------------------------------------


class RepositorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Repository
        fields = [
            "id",
            "url",
            "name",
            "status",
            "default_branch",
            "last_commit_hash",
            "last_analyzed_at",
            "created_at",
            "updated_at",
        ]


class ToolCallSerializer(serializers.ModelSerializer):
    class Meta:
        model = ToolCall
        fields = [
            "id",
            "tool_name",
            "tool_input",
            "tool_output",
            "step_number",
            "duration_ms",
            "tokens_used",
            "created_at",
        ]


class FindingSerializer(serializers.ModelSerializer):
    class Meta:
        model = Finding
        fields = [
            "id",
            "file_path",
            "note",
            "finding_type",
            "step_number",
            "created_at",
        ]


class ResearchSessionListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for listing sessions (no nested tool calls)."""

    repository_url = serializers.URLField(source="repository.url", read_only=True)
    repository_name = serializers.CharField(source="repository.name", read_only=True)

    class Meta:
        model = ResearchSession
        fields = [
            "id",
            "session_id",
            "repository_url",
            "repository_name",
            "question",
            "answer",
            "status",
            "total_tool_calls",
            "total_findings",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "duration_seconds",
            "created_at",
            "completed_at",
        ]


class ResearchSessionDetailSerializer(serializers.ModelSerializer):
    """Full serializer with nested tool calls and findings."""

    repository = RepositorySerializer(read_only=True)
    tool_calls = ToolCallSerializer(many=True, read_only=True)
    findings = FindingSerializer(many=True, read_only=True)

    class Meta:
        model = ResearchSession
        fields = [
            "id",
            "session_id",
            "repository",
            "question",
            "answer",
            "status",
            "source_references",
            "total_tool_calls",
            "total_findings",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "duration_seconds",
            "error_message",
            "created_at",
            "completed_at",
            "tool_calls",
            "findings",
        ]
