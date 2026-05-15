from django.contrib import admin

from .models import Finding, Repository, ResearchSession, ToolCall


@admin.register(Repository)
class RepositoryAdmin(admin.ModelAdmin):
    list_display = ("name", "url", "status", "last_analyzed_at", "updated_at")
    list_filter = ("status",)
    search_fields = ("name", "url")
    readonly_fields = ("created_at", "updated_at")


@admin.register(ResearchSession)
class ResearchSessionAdmin(admin.ModelAdmin):
    list_display = (
        "session_id",
        "repository",
        "status",
        "total_tool_calls",
        "total_findings",
        "total_tokens",
        "duration_seconds",
        "created_at",
    )
    list_filter = ("status", "repository")
    search_fields = ("session_id", "question")
    readonly_fields = ("session_id", "created_at", "completed_at")


@admin.register(ToolCall)
class ToolCallAdmin(admin.ModelAdmin):
    list_display = ("session", "step_number", "tool_name", "duration_ms", "tokens_used", "created_at")
    list_filter = ("tool_name",)
    readonly_fields = ("created_at",)


@admin.register(Finding)
class FindingAdmin(admin.ModelAdmin):
    list_display = ("session", "step_number", "finding_type", "file_path", "created_at")
    list_filter = ("finding_type",)
    search_fields = ("note", "file_path")
    readonly_fields = ("created_at",)
