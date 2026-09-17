"""Q3.1-3.4: reproduce NRMS, improve it once, ablate the improvement, ship a CI.

Baseline is the starter's NRMS-docvec (`newsrec.nrms`), trained on the same
article vectors, the same temporal splits and the same metric harness as
everything else in this assignment, so the comparison is about the model rather
than the plumbing.

The improvement (Q3.2) is **freshness weighting**, which the spec names
explicitly and which this codebase has independent reason to expect. NRMS sees
only document vectors: it has no way to know that an article is four days old on
a corpus whose median clicked article is 80 hours young, and no way to know that
nobody has clicked it. The GBDT, given those columns, spent 40% of its gain on
`age_hours` alone and 46% on the rolling counters. So the improvement adds a
small head over exactly those scalars, whose output is *added* to the dot-product
score.

Additive, and initialised to zero, so that the improved model starts as the
baseline bit for bit. The ablation is then an isolation rather than a second
training run that differs in a dozen unmeasured ways.

Writes reports/q3/nrms_<dataset>_<variant>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from newsrec.embeddings import article_embeddings
from newsrec.evaluate import METRICS, paired_report, rank_metrics, summarise_ranking
from newsrec.nrms import NRMS, l2_groups, sample_training_groups
from newsrec.semantic import l2_normalise
from newsrec.store import FeatureStore

# Serving-safe freshness / popularity scalars. Everything here is in the
# `shipped` families -- no label-derived rolling click counts -- so the improved
# model remains submittable.
EXTRA = ("age_hours", "roll_age_hours", "roll_inview", "clicks_decayed", "ctr_smoothed")


def histories(fs: FeatureStore, split: str, n_articles: int, h: int) -> dict[int, np.ndarray]:
    """Most recent `h` clicks per user, as article indices (0 = pad)."""
    d = fs.history(split).select("user_idx", "article_idx").collect()
    out = {}
    for u, arts in zip(d["user_idx"].to_numpy(), d["article_idx"].to_list()):
        a = np.asarray(arts or [], dtype=np.int64)
        a = a[a < n_articles][-h:]
        out[int(u)] = a
    return out


def build_hist_matrix(user_idx: np.ndarray, hist: dict, h: int) -> tuple[np.ndarray, np.ndarray]:
    """(N, h) article ids, left-padded, plus the mask that keeps pads out of the
    attention softmax."""
    ids = np.zeros((len(user_idx), h), dtype=np.int64)
    mask = np.zeros((len(user_idx), h), dtype=np.float32)
    for i, u in enumerate(user_idx):
        a = hist.get(int(u))
        if a is None or a.size == 0:
            continue
        ids[i, -a.size:] = a
        mask[i, -a.size:] = 1.0
    return ids, mask


def extra_matrix(df: pl.DataFrame, cols, stats=None):
    """Standardised scalars, with counts log-compressed first.

    Popularity counts are heavy-tailed enough that a raw z-score is dominated by
    a handful of viral articles; log1p first so the feature separates 0 from 3
    as much as 300 from 3000. Statistics come from train and are reused, which is
    the whole point of computing them once.
    """
    X = np.zeros((df.height, len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        v = df[c].to_numpy().astype(np.float64) if c in df.columns else np.zeros(df.height)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if c in ("roll_inview", "clicks_decayed"):
            v = np.log1p(np.maximum(v, 0.0))
        elif c in ("age_hours", "roll_age_hours"):
            v = np.log1p(np.clip(v, 0.0, 24 * 60))
        X[:, j] = v
    if stats is None:
        stats = (X.mean(0), X.std(0) + 1e-6)
    return (X - stats[0]) / stats[1], stats


@torch.no_grad()
def score_all(model, emb_t, hist_ids, hist_mask, art, extra, device, batch=4096):
    """Score flat (impression, candidate) rows, one impression's user vector at a
    time reused across its candidates."""
    model.eval()
    # Fail on the host with a readable message rather than as an out-of-bounds
    # gather on the device, which reports itself as an unrelated cuBLAS error.
    if len(art) and (art.max() >= emb_t.shape[0] or art.min() < 0):
        raise ValueError(f"article id out of range for {emb_t.shape[0]} embeddings")
    out = np.empty(len(art), dtype=np.float32)
    for s in range(0, len(art), batch):
        e = min(s + batch, len(art))
        h = torch.from_numpy(hist_ids[s:e]).to(device)
        m = torch.from_numpy(hist_mask[s:e]).to(device)
        c = emb_t[torch.from_numpy(art[s:e].astype(np.int64)).to(device)].unsqueeze(1)
        # (B, extra_dim) -> (B, 1, extra_dim): the head is per candidate, and a
        # 2-D tensor here would broadcast the bias across the batch instead of
        # across the (single) candidate.
        x = (torch.from_numpy(extra[s:e]).to(device).unsqueeze(1)
             if extra is not None else None)
        sc = model(emb_t[h], m, c, x)
        out[s:e] = sc.squeeze(1).float().cpu().numpy()
    return out


def train_nrms(tr, va, emb_t, hist_tr, hist_va, args, device, extra_dim, stats):
    model = NRMS(doc_dim=emb_t.shape[1], extra_dim=extra_dim).to(device)
    opt = torch.optim.Adam(l2_groups(model, 1e-4), lr=args.lr)

    imp = tr["imp"].to_numpy()
    art = tr["article_idx"].to_numpy()
    lab = tr["label"].to_numpy().astype(bool)
    slates, _ = sample_training_groups(imp, art, lab, npratio=args.npratio, seed=args.seed)
    if slates.size == 0:
        raise SystemExit("no usable training slates")
    print(f"   {len(slates):,} slates of 1+{args.npratio}")

    uidx = tr["user_idx"].to_numpy()
    hid, hmk = build_hist_matrix(uidx[slates[:, 0]], hist_tr, args.history)
    # `slates` holds ROW indices into the frame, not article ids. Indexing the
    # embedding table with them directly is an out-of-bounds gather that corrupts
    # the CUDA context and then surfaces, misleadingly, as a cuBLAS alloc
    # failure several calls later. Map through article_idx once, here.
    slate_art = art[slates].astype(np.int64)
    n_art = emb_t.shape[0]
    if slate_art.max(initial=0) >= n_art:
        raise SystemExit(f"article id {slate_art.max()} >= {n_art} embeddings")
    Xtr = None
    if extra_dim:
        Xtr, _ = extra_matrix(tr, EXTRA, stats)

    best, best_state, bad = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        perm = np.random.default_rng(args.seed + ep).permutation(len(slates))
        tot, nb = 0.0, 0
        for s in range(0, len(perm), args.batch):
            b = perm[s:s + args.batch]
            rows = slates[b]
            h = torch.from_numpy(hid[b]).to(device)
            m = torch.from_numpy(hmk[b]).to(device)
            c = emb_t[torch.from_numpy(slate_art[b]).to(device)]
            x = (torch.from_numpy(Xtr[rows.reshape(-1)]).to(device)
                 .reshape(len(b), rows.shape[1], -1) if extra_dim else None)
            logits = model(emb_t[h], m, c, x)
            # the positive is always slot 0 by construction
            loss = torch.nn.functional.cross_entropy(
                logits, torch.zeros(len(b), dtype=torch.long, device=device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1

        # --- validation AUC, on the real candidate lists
        va_ids, va_mask = build_hist_matrix(va["user_idx"].to_numpy(), hist_va, args.history)
        Xva = extra_matrix(va, EXTRA, stats)[0] if extra_dim else None
        sv = score_all(model, emb_t, va_ids, va_mask, va["article_idx"].to_numpy(),
                       Xva, device)
        m_va = summarise_ranking(rank_metrics(
            va.select("imp", "article_idx", "label").with_columns(pl.Series("score", sv)),
            ks=(5, 10)), n_boot=50)
        print(f"     epoch {ep + 1}: loss {tot / max(nb, 1):.4f}  "
              f"val AUC {m_va['auc']:.4f}  nDCG@10 {m_va['ndcg@10']:.4f}")
        if m_va["auc"] > best:
            best, bad = m_va["auc"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"     early stop (no val gain for {bad} epochs)")
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ebnerd", "mind"])
    ap.add_argument("--variant", default="small")
    ap.add_argument("--embedding", default=None)
    ap.add_argument("--history", type=int, default=20)     # starter hparam
    ap.add_argument("--npratio", type=int, default=4)      # starter hparam
    ap.add_argument("--lr", type=float, default=1e-4)      # starter hparam
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("reports/q3"))
    a = ap.parse_args()

    emb_name = a.embedding or ("contrastive" if a.dataset == "ebnerd"
                               else "sentence-transformers/all-MiniLM-L6-v2")
    fs = FeatureStore(a.dataset, a.variant)
    root = Path("artifacts/features") / a.dataset / a.variant
    need = [root / f"{s}.parquet" for s in ("train", "val", "test")]
    if not all(p.exists() for p in need):
        raise SystemExit(f"run scripts/q2_rerank.py --dataset {a.dataset} first")
    tr, va, te = (pl.read_parquet(p).sort("imp") for p in need)

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    emb = l2_normalise(np.asarray(article_embeddings(fs, emb_name), dtype=np.float32))
    emb_t = torch.from_numpy(emb).to(device)
    print(f"[{a.dataset}/{a.variant}] NRMS-docvec on {device}; "
          f"{emb.shape[0]:,} articles x {emb.shape[1]}d ({emb_name})")

    hist_tr = histories(fs, "train", emb.shape[0], a.history)
    hist_va = histories(fs, "val", emb.shape[0], a.history)
    hist_te = histories(fs, "test", emb.shape[0], a.history)
    _, stats = extra_matrix(tr, EXTRA)

    te_ids, te_mask = build_hist_matrix(te["user_idx"].to_numpy(), hist_te, a.history)
    te_art = te["article_idx"].to_numpy()
    Xte = extra_matrix(te, EXTRA, stats)[0]

    results, per_imp, timing = {}, {}, {}
    for name, extra_dim in (("nrms", 0), ("nrms+freshness", len(EXTRA))):
        print(f"\n   === {name} ===")
        t0 = time.perf_counter()
        model, best_val = train_nrms(tr, va, emb_t, hist_tr, hist_va, a, device,
                                     extra_dim, stats)
        timing[name] = round(time.perf_counter() - t0, 1)
        s = score_all(model, emb_t, te_ids, te_mask, te_art,
                      Xte if extra_dim else None, device)
        pairs = te.select("imp", "article_idx", "label").with_columns(pl.Series("score", s))
        per_imp[name] = rank_metrics(pairs, ks=(5, 10))
        results[name] = summarise_ranking(per_imp[name], n_boot=300)
        print(f"   {name}: AUC {results[name]['auc']:.4f}  "
              f"nDCG@10 {results[name]['ndcg@10']:.4f}  ({timing[name]}s)")

    paired = paired_report(per_imp["nrms"], per_imp["nrms+freshness"], n_boot=a.n_boot)
    print(f"\n   Q3.4 paired bootstrap -- nrms+freshness vs nrms")
    print(f"   {'metric':<10} {'delta':>9} {'CI95':>24} {'excludes 0':>11}")
    for m in METRICS:
        r = paired.get(m)
        if r:
            print(f"   {m:<10} {r['delta']:>+9.4f} "
                  f"[{r['ci95'][0]:>+9.4f}, {r['ci95'][1]:>+9.4f}] "
                  f"{'yes' if r['excludes_zero'] else 'NO':>11}")

    out = {"dataset": a.dataset, "variant": a.variant, "embedding": emb_name,
           "hparams": {"history": a.history, "npratio": a.npratio, "lr": a.lr,
                       "batch": a.batch, "epochs": a.epochs, "heads": 16,
                       "head_dim": 16, "attn_hidden": 200, "dropout": 0.2,
                       "units": [512, 512, 512], "l2_newsencoder": 1e-4},
           "extra_features": list(EXTRA),
           "train_seconds": timing, "results": results,
           "paired_improvement": paired}
    a.out.mkdir(parents=True, exist_ok=True)
    f = a.out / f"nrms_{a.dataset}_{a.variant}.json"
    f.write_text(json.dumps(out, indent=2))
    print(f"\n   wrote {f}")


if __name__ == "__main__":
    main()
