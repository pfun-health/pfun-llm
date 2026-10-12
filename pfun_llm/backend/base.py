"""Abstract embedding backend interface."""
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Union

import numpy as np


class BaseGenerativeModel:
    """Base class for all generative model backends."""

    _default_model: Optional[str] = None

    def __init__(self, model: Optional[str] = None, **kwargs):
        self._model = model if model is not None else self._default_model
        self._extra_kwds = kwargs
        self._client = self.setup_genai_client()

    def __call__(self, *args, **kwargs) -> Any:
        return self.call_genai_client(*args, **kwargs)

    def generate_content(self, contents: Any, **kwargs) -> Any:
        return self.call_genai_client(contents=contents, **kwargs)

    def call_genai_client(
        self, model: Optional[str] = None, contents: Any = None, **kwargs
    ) -> Optional[str]:
        if contents is None:
            raise ValueError("Contents must be provided")
        return model if model is not None else self._model

    def setup_genai_client(self) -> Any:
        raise NotImplementedError


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
