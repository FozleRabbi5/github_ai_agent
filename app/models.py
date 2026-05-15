import uuid

from django.db import models


class Repository(models.Model):
    """A GitHub repository that has been researched."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        CLONED = "cloned", "Cloned"
        FAILED = "failed", "Failed"

    url = models.URLField(unique=True, help_text="GitHub repository URL")
    name = models.CharField(max_length=255, help_text="Derived repo name (e.g. 'fastapi')")
    local_path = models.CharField(max_length=1024, blank=True)
    default_branch = models.CharField(max_length=255, blank=True)
    last_commit_hash = models.CharField(max_length=64, blank=True)
    status = models.CharField(
        max_length=32,
        choices=Status.choices,
        default=Status.PENDING,
    )
    metadata = models.JSONField(default=dict, blank=True)
    last_analyzed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        verbose_name_plural = "repositories"

    def __str__(self) -> str:
        return f"{self.name} ({self.status})"


class ResearchSession(models.Model):
    """A single research investigation: one question against one repo."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    repository = models.ForeignKey(
        Repository,
        on_delete=models.CASCADE,
        related_name="sessions",
    )
    session_id = models.UUIDField(
        default=uuid.uuid4,
        unique=True,
        db_index=True,
        editable=False,
    )
    question = models.TextField(help_text="The user's research question")
    answer = models.TextField(blank=True, help_text="Final agent answer")
    status = models.CharField(
        max_length=32,
        choices=Status.choices,
        default=Status.PENDING,
    )
    source_references = models.JSONField(default=list, blank=True)

    # Aggregate counters (denormalized for quick access)
    total_tool_calls = models.PositiveIntegerField(default=0)
    total_findings = models.PositiveIntegerField(default=0)

    # Token usage tracking
    prompt_tokens = models.PositiveIntegerField(default=0)
    completion_tokens = models.PositiveIntegerField(default=0)
    total_tokens = models.PositiveIntegerField(default=0)

    duration_seconds = models.FloatField(null=True, blank=True)
    error_message = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Session {self.session_id} [{self.status}]"


class ToolCall(models.Model):
    """Log of a single tool invocation during a research session."""

    session = models.ForeignKey(
        ResearchSession,
        on_delete=models.CASCADE,
        related_name="tool_calls",
    )
    tool_name = models.CharField(max_length=64, db_index=True)
    tool_input = models.JSONField(default=dict, help_text="Arguments passed to the tool")
    tool_output = models.TextField(blank=True, help_text="Tool result (truncated to 10KB)")
    step_number = models.PositiveIntegerField(help_text="Order within the session")
    duration_ms = models.FloatField(null=True, blank=True)
    tokens_used = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["step_number"]

    def __str__(self) -> str:
        return f"Step {self.step_number}: {self.tool_name}"

    def save(self, *args, **kwargs):
        # Truncate tool_output to 10KB to prevent DB bloat
        if self.tool_output and len(self.tool_output) > 10_000:
            self.tool_output = self.tool_output[:10_000] + "\n...[truncated]"
        super().save(*args, **kwargs)


class Finding(models.Model):
    """A semantic discovery the agent explicitly saved during research."""

    class FindingType(models.TextChoices):
        OBSERVATION = "observation", "Observation"
        PATTERN = "pattern", "Pattern"
        ISSUE = "issue", "Issue"
        DEPENDENCY = "dependency", "Dependency"
        ARCHITECTURE = "architecture", "Architecture"

    session = models.ForeignKey(
        ResearchSession,
        on_delete=models.CASCADE,
        related_name="findings",
    )
    file_path = models.CharField(
        max_length=1024,
        blank=True,
        help_text="File path the finding relates to",
    )
    note = models.TextField(help_text="The agent's observation or conclusion")
    finding_type = models.CharField(
        max_length=32,
        choices=FindingType.choices,
        default=FindingType.OBSERVATION,
    )
    step_number = models.PositiveIntegerField(
        help_text="The tool-call step at which this finding was recorded",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["step_number"]

    def __str__(self) -> str:
        label = self.file_path or "general"
        return f"Finding@{label}: {self.note[:60]}"
