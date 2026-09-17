"""Q3.1: NRMS-docvec, the starter baseline, reproduced in PyTorch.

**Why a reimplementation rather than the starter itself.** `ebnerd-benchmark`'s
NRMS is Keras, and `ebrec` pins `tensorflow`, `polars==0.20.8`,
`numpy<1.26.1`, `torch<2.3` and `scikit-learn==1.4.0`. That set does not
co-install with this project's stack (polars 1.43, torch 2.11, transformers 5.15,
numpy 2.x), and TensorFlow's CUDA builds do not cover the Blackwell card this
runs on. Standing up a second frozen environment would reproduce the *code*
while making every number incomparable with the rest of the assignment, because
the feature store, the split definitions and the metric implementation would all
be different objects.

So the architecture is ported layer for layer and the *inputs and metrics are
shared with everything else here*: the same article vectors the GBDT sees, the
same temporal splits, and A1's metric harness -- which `tests/test_official_metrics.py`
already validates against `ebrec.evaluation.metrics` itself. That keeps the
comparison about the model instead of about the plumbing.

**The architecture** (Wu et al., 2019, as specialised by the starter's
`NRMSDocVec`). Faithful to `nrms_docvec.py` and `layers.py`:

    news encoder   docvec(D) -> [Linear(512) ReLU, BatchNorm, Dropout(0.2)] x3
                             -> Linear(head_num*head_dim) ReLU
    user encoder   history(H docvecs) -> news encoder per item
                             -> multi-head self-attention (16 heads x 16 dims)
                             -> additive attention pooling (hidden 200)
    score          dot(news vector, user vector)

Details that are easy to get silently wrong and are matched deliberately:

  * the self-attention has **no output projection, no residual and no layer
    norm** -- it is the 2019 formulation, not a transformer block;
  * the additive pooling is `softmax(q^T tanh(Wx + b))`, with the same epsilon
    guard in the denominator;
  * the news encoder's ReLU on the final layer is kept, unusual as it is for an
    embedding that then goes into a dot product;
  * L2 on the news-encoder dense layers (1e-4) is applied as decoupled weight
    decay on exactly those parameters, not globally;
  * training is softmax cross-entropy over one positive against `npratio`
    sampled negatives, which is what makes the loss a *ranking* loss rather than
    a click classifier.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """`AttLayer2`: softmax(q^T tanh(W x + b)) pooling over a sequence.

    Masked because histories are padded: a user with 3 clicks and H=20 must not
    have 17 zero vectors voting in the softmax. The starter pads without masking
    here, which quietly dilutes every short history towards the pad embedding --
    keeping the mask is the one deliberate deviation, and `--no-mask` reproduces
    the starter's behaviour for the comparison.
    """

    def __init__(self, dim: int, hidden: int = 200):
        super().__init__()
        self.W = nn.Linear(dim, hidden)
        self.q = nn.Linear(hidden, 1, bias=False)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        nn.init.xavier_uniform_(self.q.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, L, D) -> (B, D)
        a = self.q(torch.tanh(self.W(x))).squeeze(-1)          # (B, L)
        a = torch.exp(a - a.max(dim=-1, keepdim=True).values)   # stabilised
        if mask is not None:
            a = a * mask.to(a.dtype)
        a = a / (a.sum(dim=-1, keepdim=True) + 1e-8)
        return (x * a.unsqueeze(-1)).sum(dim=1)


class MultiHeadSelfAttention(nn.Module):
    """`SelfAttention`: scaled dot-product over heads, concatenated.

    Deliberately not `nn.MultiheadAttention`: that adds an output projection and
    defaults to a different initialisation, so it is a different model wearing
    the same name.
    """

    def __init__(self, dim: int, heads: int = 16, head_dim: int = 16):
        super().__init__()
        self.h, self.d = heads, head_dim
        out = heads * head_dim
        self.WQ = nn.Linear(dim, out, bias=False)
        self.WK = nn.Linear(dim, out, bias=False)
        self.WV = nn.Linear(dim, out, bias=False)
        for w in (self.WQ, self.WK, self.WV):
            nn.init.xavier_uniform_(w.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.WQ(x).view(B, L, self.h, self.d).transpose(1, 2)   # (B,h,L,d)
        k = self.WK(x).view(B, L, self.h, self.d).transpose(1, 2)
        v = self.WV(x).view(B, L, self.h, self.d).transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) / (self.d ** 0.5)             # (B,h,L,L)
        if mask is not None:
            a = a.masked_fill(~mask[:, None, None, :].bool(), float("-inf"))
        a = torch.softmax(a, dim=-1)
        a = torch.nan_to_num(a)          # a fully-masked row is all -inf
        o = (a @ v).transpose(1, 2).reshape(B, L, self.h * self.d)
        return o


class NewsEncoder(nn.Module):
    """docvec -> [Linear/ReLU/BN/Dropout] x N -> Linear/ReLU."""

    def __init__(self, in_dim: int, units=(512, 512, 512), out_dim: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        layers, d = [], in_dim
        for u in units:
            layers += [nn.Linear(d, u), nn.ReLU(), nn.BatchNorm1d(u), nn.Dropout(dropout)]
            d = u
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(d, out_dim)
        self.l2_params = [m.weight for m in self.body if isinstance(m, nn.Linear)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # accepts (B, D) or (B, L, D); BatchNorm1d needs a flat batch either way
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        h = F.relu(self.out(self.body(flat)))
        return h.reshape(*shape[:-1], h.shape[-1])


class NRMS(nn.Module):
    """NRMS-docvec, plus an optional freshness/popularity head (Q3.2).

    `extra_dim > 0` turns on the improvement: a small MLP over per-candidate
    scalars -- article age and the rolling popularity counters -- whose output is
    *added* to the dot-product score. Additive rather than concatenated so that
    setting the branch to zero recovers the baseline exactly, which is what makes
    the ablation an isolation rather than a retrain.
    """

    def __init__(self, doc_dim: int, heads: int = 16, head_dim: int = 16,
                 units=(512, 512, 512), attn_hidden: int = 200,
                 dropout: float = 0.2, extra_dim: int = 0):
        super().__init__()
        out = heads * head_dim
        self.news = NewsEncoder(doc_dim, units, out, dropout)
        self.self_attn = MultiHeadSelfAttention(out, heads, head_dim)
        self.pool = AdditiveAttention(out, attn_hidden)
        self.extra_dim = extra_dim
        if extra_dim:
            self.extra = nn.Sequential(
                nn.Linear(extra_dim, 64), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(64, 1))
            # start at exactly zero so the improved model *begins* as the
            # baseline and any gain is attributable to what it learns here
            nn.init.zeros_(self.extra[-1].weight)
            nn.init.zeros_(self.extra[-1].bias)

    def user_vector(self, hist: torch.Tensor, hist_mask: torch.Tensor) -> torch.Tensor:
        h = self.news(hist)                       # (B, H, out)
        h = self.self_attn(h, hist_mask)
        return self.pool(h, hist_mask)            # (B, out)

    def forward(self, hist: torch.Tensor, hist_mask: torch.Tensor,
                cand: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        u = self.user_vector(hist, hist_mask)     # (B, out)
        c = self.news(cand)                       # (B, C, out)
        s = torch.einsum("bcd,bd->bc", c, u)
        if self.extra_dim and extra is not None:
            s = s + self.extra(extra).squeeze(-1)
        return s


def l2_groups(model: NRMS, weight_decay: float = 1e-4):
    """Parameter groups applying the starter's L2 to the news encoder only.

    The starter regularises `newsencoder`'s Dense kernels and nothing else.
    Applying the same decay globally would also shrink the attention and pooling
    weights, which is a different model.
    """
    l2_ids = {id(p) for p in model.news.l2_params}
    decay = [p for p in model.parameters() if id(p) in l2_ids]
    rest = [p for p in model.parameters() if id(p) not in l2_ids]
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": rest, "weight_decay": 0.0}]


def sample_training_groups(imp_ids: np.ndarray, article: np.ndarray,
                           label: np.ndarray, npratio: int = 4,
                           seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """One positive against `npratio` negatives, per positive.

    This is what makes NRMS's loss a ranking loss: the softmax runs over a slate
    containing exactly one correct answer. Impressions with no negatives are
    dropped (the slate would be degenerate) and short ones are sampled with
    replacement, so every positive contributes a full slate.

    Returns (rows, group_id) indexing into the input arrays.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(imp_ids, kind="stable")
    imp_s, art_s, lab_s = imp_ids[order], article[order], label[order]
    bounds = np.flatnonzero(np.diff(imp_s)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(imp_s)]])

    rows, gid, g = [], [], 0
    for s, e in zip(starts, ends):
        idx = order[s:e]
        lab = lab_s[s:e]
        pos, neg = idx[lab], idx[~lab]
        if pos.size == 0 or neg.size == 0:
            continue
        for p in pos:
            take = rng.choice(neg, size=npratio, replace=neg.size < npratio)
            rows.append(np.concatenate([[p], take]))
            gid.append(np.full(npratio + 1, g))
            g += 1
    if not rows:
        return np.zeros((0, npratio + 1), dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.stack(rows), np.concatenate(gid)
