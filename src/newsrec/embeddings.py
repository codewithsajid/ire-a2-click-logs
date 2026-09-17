"""Align pre-computed article embeddings to the store's dense index.

EB-NeRD ships four embedding artifacts keyed by article_id; retrieval wants a
dense (n_articles, dim) matrix whose row i is article idx i. Articles with no
provided vector (in demo/small subsets, or MIND) stay zero and are recorded in
`coverage`, because a silently-zero row would otherwise look like a legitimate
"no similarity" answer at query time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from .config import Config

EBNERD_ARTIFACTS = {
    "word2vec": ("Ekstra_Bladet_word2vec/document_vector.parquet", "document_vector"),
    "contrastive": ("Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet", "contrastive_vector"),
    "bert_multilingual": ("google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet",
                          "google-bert/bert-base-multilingual-cased"),
    "xlm_roberta": ("FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet",
                    "FacebookAI/xlm-roberta-base"),
}


# Short aliases accepted on the command line, so `--embedding all-MiniLM-L6-v2`
# and the full hub id name the same file.
ALIASES = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}


def canonical(name: str) -> str:
    return ALIASES.get(name, name)


def filename(name: str) -> str:
    """One file name per model, whatever alias was typed."""
    return canonical(name).replace("/", "_") + ".npy"


def article_embeddings(fs, name: str) -> np.ndarray:
    """The (n_articles, dim) matrix for `name`, encoding it once if needed.

    Every embedding a run uses lands in the store next to the tables it was
    derived from. Before this, EB-NeRD's four shipped sets lived in the store
    while MIND's were written to reports/q3 under a name hardcoded in six call
    sites -- which meant `--embedding` was silently ignored on MIND and any model
    other than MiniLM was impossible to ask for.
    """
    from .semantic import encode_texts

    out_dir = fs.root / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    for candidate in (name, filename(name)[:-4]):
        try:
            return np.asarray(fs.embeddings(candidate))
        except FileNotFoundError:
            pass
    path = out_dir / filename(name)
    if path.exists():
        return np.load(path, mmap_mode="r")
    model = canonical(name)
    print(f"  encoding {fs.n_articles:,} articles with {model} -> {path}", flush=True)
    v = encode_texts(fs.texts(), model_name=model)
    np.save(path, v)
    return v


def align_ebnerd(cfg: Config) -> dict[str, dict]:
    out_dir = cfg.out / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = (
        pl.scan_parquet(cfg.out / "articles.parquet")
        .select("src_id").with_row_index("idx")
        .with_columns(pl.col("src_id").cast(pl.Int32).alias("article_id"))
        .collect()
    )
    n = idx.height
    report = {}
    for name, (rel, col) in EBNERD_ARTIFACTS.items():
        src = cfg.raw / rel
        if not src.exists():
            continue
        emb = pl.scan_parquet(src).collect()
        joined = idx.join(emb, on="article_id", how="left")
        vectors = joined[col].to_list()
        dim = len(next(v for v in vectors if v is not None))
        mat = np.zeros((n, dim), dtype=np.float32)
        hit = 0
        for i, v in enumerate(vectors):
            if v is not None:
                mat[i] = v
                hit += 1
        np.save(out_dir / f"{name}.npy", mat)
        report[name] = {"dim": dim, "rows": n, "covered": hit, "coverage": round(hit / n, 4)}
    (out_dir / "coverage.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import argparse, json
    from .config import Config

    ap = argparse.ArgumentParser(description="Align shipped article embeddings to the dense index")
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    print(json.dumps(align_ebnerd(Config.load(a.config)), indent=2))
