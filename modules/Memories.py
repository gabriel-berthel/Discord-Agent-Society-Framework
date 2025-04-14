import chromadb
import time
import uuid
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional
import logging
import os
from sentence_transformers import SentenceTransformer

class Memories:
    def __init__(self, collection_name: str, model_name: str = 'all-MiniLM-L6-v2'): 
        os.makedirs('chroma_db', exist_ok=True)
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(collection_name)
        self.model = SentenceTransformer(model_name)

        self._documents = []
        self._embeddings = []
        self._metadatas = []

        self._update_cache()

    def _update_cache(self):
        try:
            all_ids = self.collection.get(include=["documents"])["ids"]
            result = self.collection.get(ids=all_ids, include=["documents", "embeddings", "metadatas"])
            self._documents = result["documents"]
            self._embeddings = result["embeddings"]
            self._metadatas = result["metadatas"]
        except Exception as e:
            pass

    def add_document(self, document: str, doc_type: str, timestamp: Optional[float] = None):
        valid_types = ['FORMER-PLAN', 'MEMORY', 'KNOWLEDGE']
        if doc_type not in valid_types:
            return

        embedding = self.model.encode(document, show_progress_bar=False)

        metadatas = {"type": doc_type}
        if doc_type in ['MEMORY', 'FORMER-PLAN']:
            metadatas["timestamp"] = timestamp if timestamp else time.time()

        doc_id = str(uuid.uuid4())
        self.collection.add(
            ids=[doc_id],
            documents=[document],
            metadatas=[metadatas],
            embeddings=[embedding]
        )

        self._documents.append(document)
        self._embeddings.append(embedding)
        self._metadatas.append(metadatas)

    def get_all_documents(self):
        return self._documents, self._embeddings, self._metadatas

    def query_multiple(self, queries: list, n_results: int = 5):
        if not self._documents:
            return []

        results = []
        for query in queries:
            query_embedding = self.model.encode(query)
            similarities = cosine_similarity([query_embedding], self._embeddings)[0]

            docs_with_metadata = [
                {'doc': doc, 'metadata': metadata, 'similarity': similarity}
                for doc, metadata, similarity in zip(self._documents, self._metadatas, similarities)
            ]

            sorted_docs = sorted(
                docs_with_metadata,
                key=lambda x: (x['similarity'], x['metadata'].get('timestamp', 0)),
                reverse=True
            )

            results.extend([result['doc'] for result in sorted_docs[:n_results] if result['doc']])

        return results

    def get_last_n_memories(self, n: int = 1):
        memory_docs = [
            {'doc': doc, 'metadata': metadata}
            for doc, metadata in zip(self._documents, self._metadatas)
            if metadata.get('type') == 'MEMORY'
        ]

        if not memory_docs:
            return []

        sorted_memories = sorted(memory_docs, key=lambda x: x['metadata'].get('timestamp', 0), reverse=True)
        
        return [memory['doc'] for memory in sorted_memories[:n] if memory['doc']]
