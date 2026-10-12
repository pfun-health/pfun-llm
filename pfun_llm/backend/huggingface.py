"""HuggingFace embedding backends (generic + EmbeddingGemma2)."""
from typing import Optional, Sequence, Union

import numpy as np

from pfun_llm.backend.base import EmbeddingBackend


class HuggingFaceEmbeddingBackend(EmbeddingBackend):
    """Generic HuggingFace backend using mean pooling over token states."""

    default_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    query_prefix: str = ""
    document_prefix: str = ""

    def __init__(self, model_name: Optional[str] = None,
                 device: Optional[str] = None, max_length: int = 512,
                 normalize: bool = True, **model_kwargs):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name or self.default_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.normalize = normalize
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name, **model_kwargs).to(self.device).eval()

    def _prepare(self, texts: Sequence[str]) -> Sequence[str]:
        return [self.document_prefix + t for t in texts]

    def _pool(self, outputs, attention_mask):
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def embed(self, texts: Union[str, Sequence[str]]) -> np.ndarray:
        import torch

        if isinstance(texts, str):
            texts = [texts]
        batch = self.tokenizer(
            self._prepare(list(texts)), padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**batch)
            emb = self._pool(out, batch["attention_mask"])
            if self.normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.float().cpu().numpy()

    def embed_query(self, text: str):
        return self.embed(self.query_prefix + text
                          if self.query_prefix else text)[0].tolist()


class EmbeddingGemma2Backend(HuggingFaceEmbeddingBackend):
    """EmbeddingGemma backend (google/embeddinggemma-300m).

    Uses EmbeddingGemma's task prompts for queries and documents.
    The model is gated on the HuggingFace Hub; authenticate first
    (e.g. `huggingface-cli login`).
    """

    default_model_name = "google/embeddinggemma-300m"
    query_prefix = "task: search result | query: "
    document_prefix = "title: none | text: "

    def _prepare(self, texts):
        return [self.document_prefix + t for t in texts]

    def embed_query(self, text: str):
        # queries use the query prompt; bypass the document prefix
        return self.embed_raw([self.query_prefix + text])[0].tolist()

    def embed_raw(self, texts: Sequence[str]) -> np.ndarray:
        saved, self.document_prefix = self.document_prefix, ""
        try:
            return self.embed(list(texts))
        finally:
            self.document_prefix = saved
