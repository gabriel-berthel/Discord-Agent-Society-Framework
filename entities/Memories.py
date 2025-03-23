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
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(collection_name)
        self.model = SentenceTransformer(model_name)

    def add_document(self, document: str, doc_type: str, timestamp: Optional[float] = None):
        valid_types = ['FORMER-PLAN', 'MEMORY', 'SELF-KNOWLEDGE', 'KNOWLEDGE']
        if doc_type not in valid_types:
            return
        
        embedding = self.model.encode(document, show_progress_bar=False)

        metadatas = {
            "type": doc_type
        }

        if doc_type in ['MEMORY', 'FORMER-PLAN']:
            timestamp = timestamp if timestamp else time.time()
            metadatas["timestamp"] = timestamp


        self.collection.add(
            ids=[str(uuid.uuid4())],
            documents=[document],
            metadatas=[metadatas],
            embeddings=[embedding]
        )

    def query(self, query: str, n_results: int = 1):
        query_embedding = self.model.encode(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        return results['documents']

    def get_all_documents(self):
        all_docs = self.collection.get()
        return all_docs['documents'], all_docs['embeddings'], all_docs['metadatas']

    def query_multiple(self, queries: list, n_results: int = 5):
        documents, embeddings, metadatas = self.get_all_documents()

        results = []
        for query in queries:
            query_embedding = self.model.encode(query)
            similarities = cosine_similarity([query_embedding], embeddings)[0]

            docs_with_metadata = [
                {'doc': doc, 'metadata': metadata, 'similarity': similarity}
                for doc, metadata, similarity in zip(documents, metadatas, similarities)
            ]

            sorted_docs = sorted(
                docs_with_metadata,
                key=lambda x: (x['similarity'], x['metadata'].get('timestamp', 0)),
                reverse=True
            )

            results.extend([result['doc'] for result in sorted_docs[:n_results] if result['doc']])

        print("Mem:", results)
        return results


    def get_last_n_memories(self, n: int = 1):
        documents, embeddings, metadatas = self.get_all_documents()

        memory_docs = [
            {'doc': doc, 'metadata': metadata}
            for doc, metadata in zip(documents, metadatas)
            if metadata.get('type') == 'MEMORY'
        ]

        if not memory_docs:
            return []

        sorted_memories = sorted(memory_docs, key=lambda x: x['metadata'].get('timestamp', 0), reverse=True)
        
        print("Mem:", sorted_memories)
        return [memory['doc'] for memory in sorted_memories[:n] if memory['doc']]
