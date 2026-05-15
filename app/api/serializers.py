from rest_framework import serializers

from app.models import QuestionHistory, RepositoryIndex


class RepositoryIndexRequestSerializer(serializers.Serializer):
    repository_url = serializers.URLField()


class RepositoryIndexResponseSerializer(serializers.Serializer):
    repository_url = serializers.URLField()
    repository_name = serializers.CharField()
    status = serializers.CharField()
    last_commit_hash = serializers.CharField(allow_blank=True)
    indexed_at = serializers.DateTimeField(allow_null=True)
    metadata = serializers.JSONField()


class QuestionAskRequestSerializer(serializers.Serializer):
    question = serializers.CharField(allow_blank=False, trim_whitespace=True)


class QuestionHistoryQuerySerializer(serializers.Serializer):
    repository_url = serializers.URLField(required=False)


class QuestionHistoryItemSerializer(serializers.ModelSerializer):
    repository_url = serializers.URLField(source="repository.repository_url", read_only=True)

    class Meta:
        model = QuestionHistory
        fields = [
            "id",
            "repository_url",
            "thread_id",
            "question",
            "answer",
            "source_references",
            "execution_metadata",
            "status",
            "created_at",
        ]


class QuestionAskResponseSerializer(serializers.Serializer):
    repository = RepositoryIndexResponseSerializer()
    thread_id = serializers.CharField()
    question = serializers.CharField()
    answer = serializers.CharField()
    source_references = serializers.JSONField()
    execution_metadata = serializers.JSONField()


class RepositoryIndexModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = RepositoryIndex
        fields = [
            "repository_url",
            "repository_name",
            "status",
            "last_commit_hash",
            "indexed_at",
            "metadata",
        ]
