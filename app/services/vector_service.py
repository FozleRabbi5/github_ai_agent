from __future__ import annotations

from langchain_core.documents import Document

from app.vectorstores.chroma_store import ChromaStoreFactory


class VectorIndexService:
    def __init__(self, store_factory: ChromaStoreFactory) -> None:
        self._store_factory = store_factory

    def reindex_repository(self, repository_url: str, documents: list[Document]) -> dict:
        store = self._store_factory.reset_store(repository_url)
        batch_size = self._store_factory.max_batch_size_for_store(store)
        batch_count = 0
        if documents:
            for batch in self._store_factory.iter_batches(documents, batch_size):
                store.add_documents(list(batch))
                batch_count += 1
        return {
            "collection_name": self._store_factory.collection_name_for_repository(repository_url),
            "document_count": len(documents),
            "batch_size": batch_size,
            "batch_count": batch_count,
        }

    def similarity_search(self, repository_url: str, query: str, limit: int = 8):
        store = self._store_factory.get_store(repository_url)
        return store.similarity_search_with_relevance_scores(query, k=limit)
