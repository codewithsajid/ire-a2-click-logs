"""The user query reads the whole history, and both retrievers read the same one.

scripts/q3_userrep.py swept n_recent over {5, 10, 20, 30, 50, 100, all} crossed with
exponential recency decay, on both datasets and both retrievers. Reading the whole
history won every comparison; decay lost 11 of 12. These tests pin that outcome so
the window cannot drift back in silently, and pin the property the Q3 lexical-vs-
semantic comparison rests on: BM25 and the embedding user vector are fed identical
input, so any difference between them is representation and not history.
"""
from __future__ import annotations

import numpy as np
import pytest

from newsrec.config import N_RECENT_DEFAULT
from newsrec.lexical import BM25Index
from newsrec.semantic import user_vectors

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def test_default_reads_the_whole_history():
    assert N_RECENT_DEFAULT == 0, "0 means untruncated; see scripts/q3_userrep.py"


def test_user_vectors_default_uses_every_click():
    emb = np.eye(6, dtype=np.float32)
    hist = [np.array([0, 1, 2, 3, 4, 5])]
    whole = user_vectors(hist, emb)
    last_two = user_vectors(hist, emb, n_recent=2)
    # the truncated vector puts all its mass on the final two rows; the default does not
    assert np.count_nonzero(whole) == 6
    assert np.count_nonzero(last_two) == 2
    assert not np.allclose(whole, last_two)


def test_user_vectors_no_decay_by_default():
    emb = np.eye(4, dtype=np.float32)
    hist = [np.array([0, 1, 2, 3])]
    flat = user_vectors(hist, emb)
    decayed = user_vectors(hist, emb, recency_weighted=True, halflife=1.0)
    assert np.allclose(flat, flat[0, 0])          # every click weighted alike
    assert decayed[0, 3] > decayed[0, 0]          # decay tilts towards the newest


def test_bm25_query_default_uses_every_click():
    texts = ["alpha alpha", "beta", "gamma", "delta"]
    idx = BM25Index.build(texts, lang="en")
    hist = [np.array([0, 1, 2, 3])]
    whole = idx.queries_from_history(hist)
    last_one = idx.queries_from_history(hist, n_recent=1)
    assert whole.nnz > last_one.nnz
    assert whole.nnz == idx.queries_from_history(hist, n_recent=0).nnz
