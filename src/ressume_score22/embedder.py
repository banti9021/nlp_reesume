# src/embedder.py

from sentence_transformers import SentenceTransformer
import numpy as np

class TextEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        """
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")

        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding

    def generate_batch_embeddings(self, texts: list) -> np.ndarray:
        """
        Generate embeddings for multiple texts
        """
        if not texts or not isinstance(texts, list):
            raise ValueError("Input must be a list of strings")

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=16,
            show_progress_bar=True
        )
        return embeddings
