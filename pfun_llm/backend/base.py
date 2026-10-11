"""Abstract embedding backend interface."""
from abc import ABC, abstractmethod
from typing import List, Sequence, Union

import numpy as np


class EmbeddingBackend(ABC):
    """Base class for all embedding backends."""

    @abstractmethod
    def embed(self, texts: Union[str, Sequence[str]]) -> np.ndarray:
        """Return an array of shape (n_texts, dim)."""

    def embed_query(self, text: str) -> List[float]:
        return self.embed(text)[0].tolist()

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self.embed(texts).tolist()

    def similarity(self, a: Union[str, Sequence[str]],
                   b: Union[str, Sequence[str]]) -> np.ndarray:
        """Cosine similarity matrix between texts in `a` and `b`."""
        ea, eb = self.embed(a), self.embed(b)
        ea = ea / np.linalg.norm(ea, axis=1, keepdims=True)
        eb = eb / np.linalg.norm(eb, axis=1, keepdims=True)
        return ea @ eb.T
