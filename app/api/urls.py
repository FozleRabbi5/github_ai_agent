from django.urls import path

from app.api.views import (
    RepositoryDetailView,
    RepositoryListView,
    ResearchSessionDetailView,
    ResearchSessionListView,
    StartResearchSessionView,
)

urlpatterns = [
    # Research sessions
    path(
        "research/sessions/",
        StartResearchSessionView.as_view(),
        name="start-research-session",
    ),
    path(
        "research/sessions/list/",
        ResearchSessionListView.as_view(),
        name="list-research-sessions",
    ),
    path(
        "research/sessions/<uuid:session_id>/",
        ResearchSessionDetailView.as_view(),
        name="research-session-detail",
    ),
    # Repositories
    path(
        "repositories/",
        RepositoryListView.as_view(),
        name="repository-list",
    ),
    path(
        "repositories/<int:pk>/",
        RepositoryDetailView.as_view(),
        name="repository-detail",
    ),
]
