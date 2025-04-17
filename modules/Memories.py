import pickle
import time
import uuid
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
from collections import deque

class Memories:
    def __init__(self, collection_name, base_folder='memories', model_name='all-MiniLM-L6-v2', max_documents=1000):
        os.makedirs(base_folder, exist_ok=True)
        self.file_path = os.path.join(base_folder, collection_name)
        self.model = SentenceTransformer(model_name)
        self.max_documents = max_documents
        self._documents = deque(maxlen=self.max_documents)
        self._embeddings = deque(maxlen=self.max_documents)
        self._metadatas = deque(maxlen=self.max_documents)
        self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'rb') as f:
                data = pickle.load(f)
                self._documents.extend(data.get('documents', []))
                self._embeddings.extend(data.get('embeddings', []))
                self._metadatas.extend(data.get('metadatas', []))

    def _save_memory(self):
        with open(self.file_path, 'wb') as f:
            pickle.dump({
                'documents': list(self._documents),
                'embeddings': list(self._embeddings),
                'metadatas': list(self._metadatas)
            }, f)

    def add_document(self, document, doc_type, timestamp=None):
        embedding = self.model.encode(document, show_progress_bar=False)
        metadatas = {"type": doc_type}
        metadatas["timestamp"] = timestamp if timestamp else time.time()

        self._documents.append(document + '\n')
        self._embeddings.append(embedding)
        self._metadatas.append(metadatas)

        self._save_memory()

    def get_all_documents(self):
        return list(self._documents), list(self._embeddings), list(self._metadatas)

    def query_multiple(self, queries, n_results=5):
        if not self._embeddings:
            return []

        results = []
        for query in queries:
            query_embedding = self.model.encode(query)
            similarities = cosine_similarity([query_embedding], list(self._embeddings))[0]

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

    def get_last_n_memories(self, n=1):
        memory_docs = [
            {'doc': doc, 'metadata': metadata}
            for doc, metadata in zip(self._documents, self._metadatas)
            if metadata.get('type') == 'MEMORY'
        ]

        if not memory_docs:
            return []

        sorted_memories = sorted(memory_docs, key=lambda x: x['metadata'].get('timestamp', 0), reverse=True)
        
        return [memory['doc'] for memory in sorted_memories[:n] if memory['doc']]
