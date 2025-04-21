import pickle
import time
import uuid
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
from collections import deque

class Memories:
    """
    A class for managing a collection of documents, storing embeddings, and querying the collection using semantic similarity.

    Attributes:
        collection_name (str): The name of the memory collection.
        base_folder (str): The base folder where memory files are stored.
        model_name (str): The name of the SentenceTransformer model used for embedding generation.
        max_documents (int): The maximum number of documents to store in memory.
        _documents (deque): A deque storing the documents.
        _embeddings (deque): A deque storing the embeddings of the documents.
        _metadatas (deque): A deque storing metadata associated with the documents.
    """

    def __init__(self, collection_name, base_folder='memories', model_name='all-MiniLM-L6-v2', max_documents=500):
        """
        Initializes the Memories class, setting up the necessary directories and loading previous data if available.

        Args:
            collection_name (str): The name of the memory collection.
            base_folder (str): The base folder where memory files are stored (default is 'memories').
            model_name (str): The model name to use for SentenceTransformer (default is 'all-MiniLM-L6-v2').
            max_documents (int): The maximum number of documents to store in memory (default is 500).
        """
        os.makedirs(base_folder, exist_ok=True)
        self.file_path = os.path.join(base_folder, collection_name)
        self.model = SentenceTransformer(model_name)
        self.max_documents = max_documents
        self._documents = deque(maxlen=self.max_documents)
        self._embeddings = deque(maxlen=self.max_documents)
        self._metadatas = deque(maxlen=self.max_documents)
        self._load_memory()

    def _load_memory(self):
        """
        Loads the memory from a previously saved file, if available.

        This method checks if the memory file exists and loads the documents, embeddings, and metadata.
        """
        if os.path.exists(self.file_path):
            with open(self.file_path, 'rb') as f:
                data = pickle.load(f)
                self._documents.extend(data.get('documents', []))
                self._embeddings.extend(data.get('embeddings', []))
                self._metadatas.extend(data.get('metadatas', []))

    def _save_memory(self):
        """
        Saves the current state of the memory (documents, embeddings, and metadata) to a file.
        
        This method is called after adding a new document to ensure the memory is persistent.
        """
        with open(self.file_path, 'wb') as f:
            pickle.dump({
                'documents': list(self._documents),
                'embeddings': list(self._embeddings),
                'metadatas': list(self._metadatas)
            }, f)

    def add_document(self, document, doc_type, timestamp=None):
        """
        Adds a new document to the memory along with its embedding and metadata.

        Args:
            document (str): The document content to be added.
            doc_type (str): The type of document (e.g., "text", "note").
            timestamp (float, optional): The timestamp of when the document was added. If None, the current time is used.
        """
        embedding = self.model.encode(document, show_progress_bar=False)
        metadatas = {"type": doc_type}
        metadatas["timestamp"] = timestamp if timestamp else time.time()

        self._documents.append(document + '\n')
        self._embeddings.append(embedding)
        self._metadatas.append(metadatas)

        self._save_memory()

    def get_all_documents(self):
        """
        Retrieves all documents, embeddings, and metadata stored in memory.

        Returns:
            tuple: A tuple containing three lists:
                - List of all documents.
                - List of all document embeddings.
                - List of all document metadata.
        """
        return list(self._documents), list(self._embeddings), list(self._metadatas)

    def query_multiple(self, queries, n_results=5):
        """
        Queries the memory for the most similar documents to the given queries.

        Args:
            queries (list): A list of query strings.
            n_results (int): The number of top results to return for each query (default is 5).

        Returns:
            list: A list of the top `n_results` most similar documents for each query.
        """
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
