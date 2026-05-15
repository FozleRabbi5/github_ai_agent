from django.db import models


class RepositoryIndex(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        INDEXED = "indexed", "Indexed"
        FAILED = "failed", "Failed"

    repository_url = models.URLField(unique=True)
    repository_name = models.CharField(max_length=255)
    local_path = models.CharField(max_length=1024)
    default_branch = models.CharField(max_length=255, blank=True)
    last_commit_hash = models.CharField(max_length=64, blank=True)
    status = models.CharField(
        max_length=32,
        choices=Status.choices,
        default=Status.PENDING,
    )
    metadata = models.JSONField(default=dict, blank=True)
    indexed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return f"{self.repository_name} ({self.status})"


class QuestionHistory(models.Model):
    class Status(models.TextChoices):
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    repository = models.ForeignKey(
        RepositoryIndex,
        on_delete=models.CASCADE,
        related_name="questions",
    )
    thread_id = models.CharField(max_length=128, db_index=True)
    question = models.TextField()
    answer = models.TextField()
    source_references = models.JSONField(default=list, blank=True)
    execution_metadata = models.JSONField(default=dict, blank=True)
    status = models.CharField(
        max_length=32,
        choices=Status.choices,
        default=Status.COMPLETED,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Q&A {self.id} [{self.status}]"
