"""Q3: embedding-based candidate generation with an ANN index.

EB-NeRD ships four pre-computed article embeddings (word2vec 300d, a contrastive
vector, multilingual BERT, XLM-RoBERTa), already aligned to the dense index by
the build. MIND ships none, so its article vectors are encoded here on the GPU.

The user representation is a mean-pooled bag of their recently clicked article
vectors -- the semantic counterpart of the BM25 query built from the same
history, so lexical and semantic are compared on identical inputs.

Index choice is deliberately boring at this scale: a 22K-article corpus is a
brute-force `IndexFlatIP` on normalised vectors (i.e. exact cosine), which is
both faster and exact compared to an approximate index. HNSW is available for
the scale analysis, where the honest question is what approximation costs.
"""
from __future__ import annotations

import numpy as np

from .config import N_RECENT_DEFAULT


def l2_normalise(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def encode_texts(texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 512, device: str = "cuda") -> np.ndarray:
    """Encode article text (MIND, which ships no vectors of its own)."""
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name, device=device)
    v = m.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                 show_progress_bar=True, normalize_embeddings=True)
    return v.astype(np.float32)


def user_vectors(histories: list[np.ndarray], emb: np.ndarray,
                 n_recent: int = N_RECENT_DEFAULT,
                 recency_weighted: bool = False, halflife: float = 10.0) -> np.ndarray:
    """Mean-pool the embeddings of each user's recent clicks.

    `n_recent` truncates to the most recent clicks; 0 (the default, see
    N_RECENT_DEFAULT) reads the whole history, which the scripts/q3_userrep.py
    sweep found better than every window it tried.

    `recency_weighted` applies an exponential decay over history position, which
    is the only recency signal available on MIND (its history carries no
    timestamps, only order). It is off by default: the same sweep found decay
    hurt in 11 of the 12 settings it was tried in.
    """
    dim = emb.shape[1]
    out = np.zeros((len(histories), dim), dtype=np.float32)
    for i, h in enumerate(histories):
        if len(h) == 0:
            continue
        h = np.asarray(h)[-n_recent:] if n_recent else np.asarray(h)
        h = h[h < emb.shape[0]]
        if h.size == 0:
            continue
        vecs = emb[h]
        if recency_weighted:
            age = np.arange(len(h))[::-1]
            w = (0.5 ** (age / halflife)).astype(np.float32)[:, None]
            out[i] = (vecs * w).sum(0) / max(w.sum(), 1e-9)
        else:
            out[i] = vecs.mean(0)
    return l2_normalise(out)


class ANNIndex:
    """FAISS wrapper: exact inner product by default, HNSW on request."""

    def __init__(self, emb: np.ndarray, ids: np.ndarray | None = None,
                 kind: str = "flat", hnsw_m: int = 32, ef_search: int = 128):
        import faiss
        self.ids = np.arange(emb.shape[0]) if ids is None else np.asarray(ids)
        v = np.ascontiguousarray(l2_normalise(emb.astype(np.float32)))
        self.dim = v.shape[1]
        self.kind = kind
        if kind == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = ef_search
        else:
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(v)

    def search(self, queries: np.ndarray, top_k: int = 200) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(l2_normalise(queries.astype(np.float32)))
        scores, idx = self.index.search(q, min(top_k, self.index.ntotal))
        return scores, self.ids[idx]
