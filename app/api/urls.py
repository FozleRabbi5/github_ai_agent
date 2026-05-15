from django.urls import path

from app.api.views import QuestionAskView, QuestionHistoryView, RepositoryIndexView


urlpatterns = [
    # path("repositories/index/", RepositoryIndexView.as_view(), name="repository-index"),
    path("questions/ask/", QuestionAskView.as_view(), name="question-ask"),
    # path("questions/history/", QuestionHistoryView.as_view(), name="question-history"),
]
