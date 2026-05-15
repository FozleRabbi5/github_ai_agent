from __future__ import annotations

import re

from django.conf import settings
from langchain_openai import ChatOpenAI

from app.prompts.coding_assistant import SYSTEM_PROMPT, build_answer_prompt


class OpenAILLMService:
    def __init__(self, model: str | None = None, temperature: float = 0) -> None:
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured. Set it in the environment or "
                "in github_ai_agent/.env before starting the app."
            )

        self._llm = ChatOpenAI(
            model=model or settings.OPENAI_CHAT_MODEL,
            temperature=temperature,
            api_key=api_key,
        )
        self._extractor_llm = ChatOpenAI(
            model=settings.OPENAI_EXTRACTOR_MODEL,
            temperature=0,
            api_key=api_key,
        )

    def answer_question(
        self,
        *,
        repository_url: str,
        question: str,
        context: str,
        relationship_summary: str,
    ) -> str:
        prompt = build_answer_prompt(
            repository_url=repository_url,
            question=question,
            context=context,
            relationship_summary=relationship_summary,
        )
        response = self._llm.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        return str(response.content)

    def extract_github_repository_url(self, text: str) -> str | None:
        response = self._extractor_llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Extract exactly one GitHub repository URL from the input text. "
                        "Return only the repository URL in the format "
                        "https://github.com/owner/repo. "
                        "If no GitHub repository URL is present, return an empty string."
                    ),
                },
                {"role": "user", "content": text},
            ]
        )
        content = str(response.content).strip()
        if not content:
            return None

        match = re.search(
            r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?",
            content,
            flags=re.IGNORECASE,
        )
        if match is None:
            return None
        return match.group(0).rstrip("/")
