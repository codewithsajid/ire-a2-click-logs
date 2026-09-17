"""Part 0 sanity check: every raw bundle opens, and the headline counts match
what the assignment/paper claims."""
import os, polars as pl
from pathlib import Path

RAW = Path(os.environ.get("DATA_ROOT", Path.home() / "ire_a1" / "data")) / "raw"
EB, MI = RAW / "ebnerd", RAW / "mind"

def rows(p):  # cheap row count without loading the file
    return pl.scan_parquet(p).select(pl.len()).collect().item()

def lines(p):
    with open(p, encoding="utf8") as f:
        return sum(1 for _ in f)

print("=== EB-NeRD ===")
for b in ["ebnerd_demo", "ebnerd_small", "ebnerd_large", "ebnerd_testset"]:
    d = EB / b
    if not d.is_dir():
        print(f"  {b:16s} MISSING"); continue
    parts = [f"{s}: {rows(d/s/'behaviors.parquet'):,} impressions / {rows(d/s/'history.parquet'):,} users"
             for s in ["train", "validation", "test"] if (d / s / "behaviors.parquet").exists()]
    print(f"  {b:16s} {rows(d/'articles.parquet'):>9,} articles | " + "  ".join(parts))

print("\n=== EB-NeRD article schema (large) ===")
s = pl.scan_parquet(EB / "ebnerd_large" / "articles.parquet").collect_schema()
print("  " + ", ".join(f"{k}:{v}" for k, v in list(s.items())[:14]))
print("  ... total", len(s), "columns")

print("\n=== EB-NeRD behaviour schema (large/train) ===")
print("  " + ", ".join(pl.scan_parquet(EB / "ebnerd_large" / "train" / "behaviors.parquet").collect_schema().names()))

print("\n=== embedding artifacts ===")
for name, f in [("word2vec", "Ekstra_Bladet_word2vec/document_vector.parquet"),
                ("contrastive", "Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet"),
                ("bert-multilingual", "google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet"),
                ("xlm-roberta", "FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet")]:
    p = EB / f
    if not p.exists(): print(f"  {name:18s} MISSING"); continue
    lf = pl.scan_parquet(p); col = lf.collect_schema().names()[-1]
    dim = len(lf.select(pl.col(col).first()).collect().item())
    print(f"  {name:18s} {rows(p):>9,} rows, dim {dim}")
alo = EB / "articles_large_only"
print(f"  {'articles_large_only':18s} {rows(alo/'articles.parquet'):>9,} articles" if alo.is_dir() else "  articles_large_only MISSING")

print("\n=== MIND ===")
for d in sorted(p for p in MI.glob("MIND*") if p.is_dir()):
    b = lines(d / "behaviors.tsv") if (d / "behaviors.tsv").exists() else 0
    n = lines(d / "news.tsv") if (d / "news.tsv").exists() else 0
    extra = sorted(p.name for p in d.iterdir() if p.suffix == ".vec")
    print(f"  {d.name:18s} {b:>9,} impressions | {n:>7,} news | {' '.join(extra) or 'no .vec files'}")

print("\n=== MIND test set is unlabelled (Codabench) ===")
t = MI / "MINDlarge_test" / "behaviors.tsv"
if t.exists():
    with open(t, encoding="utf8") as f:
        row = f.readline().rstrip("\n").split("\t")
    print(f"  columns: {len(row)} -> {[c[:28] for c in row]}")
