from django.conf import settings
from langchain_openai import OpenAIEmbeddings


class OpenAIEmbeddingService:
    def __init__(self, model: str | None = None) -> None:
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured. Set it in the environment or "
                "in github_ai_agent/.env before starting the app."
            )

        self._embeddings = OpenAIEmbeddings(
            model=model or settings.OPENAI_EMBEDDING_MODEL,
            api_key=api_key,
        )

    @property
    def client(self) -> OpenAIEmbeddings:
        return self._embeddings
