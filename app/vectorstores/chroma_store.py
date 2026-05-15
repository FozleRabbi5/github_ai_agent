from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from django.conf import settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class ChromaStoreFactory:
    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings
        self._persist_directory = Path(settings.BASE_DIR) / "chroma_db"
        self._persist_directory.mkdir(parents=True, exist_ok=True)

    def collection_name_for_repository(self, repository_url: str) -> str:
        return f"repo_{hashlib.sha256(repository_url.encode('utf-8')).hexdigest()[:24]}"

    def get_store(self, repository_url: str) -> Chroma:
        return Chroma(
            collection_name=self.collection_name_for_repository(repository_url),
            embedding_function=self._embeddings,
            persist_directory=str(self._persist_directory),
        )

    def max_batch_size_for_store(self, store: Chroma) -> int:
        client = getattr(store._collection, "_client", None)
        get_max_batch_size = getattr(client, "get_max_batch_size", None)
        if callable(get_max_batch_size):
            return max(1, int(get_max_batch_size()))
        return 1000

    def iter_batches(self, items: Sequence, batch_size: int):
        for start in range(0, len(items), batch_size):
            yield items[start : start + batch_size]

    def reset_store(self, repository_url: str) -> Chroma:
        store = self.get_store(repository_url)
        ids = store.get().get("ids", [])
        if ids:
            batch_size = self.max_batch_size_for_store(store)
            for batch_ids in self.iter_batches(ids, batch_size):
                store.delete(ids=list(batch_ids))
        return store
