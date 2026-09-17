"""BM25 lexical retrieval with a polars-built inverted index.

The index is built with dataframe operations, not a Python `Counter`: tokens are
exploded into a (doc_id, term_id, tf) postings table and aggregated in polars, so
the build is vectorised and streams. The only Python-level loop is over the
*vocabulary* when stemming -- PyStemmer is called once per distinct token rather
than once per token occurrence, which is what makes stemming affordable at
125K documents.

Scoring keeps the postings as a scipy CSR matrix of precomputed document
weights, so a query batch is one sparse matmul. Because k1/b only affect the
document weights, the (k1, b) ablation re-weights the same postings instead of
re-tokenising.

    idx = BM25Index.build(texts, lang="da")
    idx.reweight(k1=1.2, b=0.75)
    scores, docs = idx.search(queries, top_k=200)
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import polars as pl
import scipy.sparse as sp

from .config import N_RECENT_DEFAULT

# Compact stopword lists. The obvious argument is that BM25's IDF already
# discounts frequent terms, so this only shrinks the postings table -- but
# scripts/q2_index.py measured it and removing stopwords is worth ~13% recall,
# not just 44% fewer postings. The reason is the query side: a query is the summed
# term counts of a user's whole history and carries no saturation (k3 = inf, which
# scripts/q2_query.py shows is the right choice), so a stopword left in the index
# accumulates an enormous query weight and drowns the topical terms. The two
# choices are coupled: unbounded query term frequency is only safe because the
# highest-frequency terms are gone.
STOPWORDS = {
    "da": {
        "og", "i", "på", "er", "til", "det", "at", "en", "den", "af", "for", "som", "med",
        "de", "der", "har", "var", "et", "han", "hun", "sig", "men", "om", "så", "fra",
        "ikke", "blev", "vi", "du", "jeg", "kan", "vil", "efter", "ved", "nu", "over",
        "være", "hans", "hendes", "man", "mig", "dig", "sin", "sit", "under", "mod",
        "eller", "når", "hvad", "hvor", "hvis", "skal", "bliver", "have", "her", "får",
    },
    "en": {
        "the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "for", "on", "with",
        "as", "at", "by", "from", "that", "this", "was", "were", "be", "been", "are",
        "he", "she", "his", "her", "they", "them", "you", "your", "i", "we", "our",
        "but", "not", "have", "has", "had", "will", "would", "can", "could", "s", "t",
        "what", "when", "where", "who", "how", "why", "all", "out", "up", "about",
    },
}

_TOKEN_RE = r"[^\w\såæøÅÆØ]"          # strip punctuation, keep Danish letters
MIN_TOKEN_LEN = 2


def tokenize(texts: list[str], lang: str = "en") -> pl.DataFrame:
    """Text -> (doc_id, token) rows, entirely in polars.

    Returns the exploded token table; callers aggregate it into postings. Kept
    separate so the notebook can show the intermediate representation.
    """
    stop = STOPWORDS.get(lang, STOPWORDS["en"])
    return (
        pl.DataFrame({"text": texts})
        .with_row_index("doc_id")
        .with_columns(
            pl.col("text").fill_null("").str.to_lowercase()
              .str.replace_all(_TOKEN_RE, " ")
              .str.split(" ").alias("token")
        )
        .explode("token")
        .filter(
            (pl.col("token").str.len_chars() >= MIN_TOKEN_LEN)
            & ~pl.col("token").is_in(list(stop))
            & ~pl.col("token").str.contains(r"^\d+$")     # bare numbers carry no topic signal
        )
        .select("doc_id", "token")
    )


# Languages that are stemmed. Not a universal win: scripts/q2_index.py measured
# it on both corpora and the sign flips with the language.
#
#   Danish  (EB-NeRD)  stemming ON vs OFF: -15.8 / -11.2 / -8.3% at recall@50/100/200
#   English (MIND)     stemming ON vs OFF: +6.0 / +5.9 / +5.2%
#
# Danish is heavily inflected, so conflating forms recovers matches that are
# really there. English news text is entity-dense -- names, places, organisations
# -- and Snowball English over-conflates them, merging distinct headlines onto one
# stem. So EB-NeRD is stemmed and MIND is not, which is a bigger effect on MIND
# than the entire k1 axis of the BM25 grid.
STEM_LANGS = {"da"}


def _stem_vocabulary(tokens: pl.Series, lang: str) -> pl.DataFrame:
    """Stem each *distinct* token once, not each occurrence -- where it helps."""
    vocab = tokens.unique().sort()
    if lang not in STEM_LANGS:
        return pl.DataFrame({"token": vocab, "stem": vocab})
    try:
        import Stemmer
        stemmer = Stemmer.Stemmer({"da": "danish", "en": "english"}.get(lang, "english"))
        stems = stemmer.stemWords(vocab.to_list())
    except Exception:                     # stemming is an optimisation, not a requirement
        stems = vocab.to_list()
    return pl.DataFrame({"token": vocab, "stem": stems})


@dataclass
class BM25Index:
    """Inverted index plus the arrays needed to re-score under new (k1, b)."""
    vocab: pl.DataFrame          # stem, term_id, df, idf
    doc_ids: np.ndarray          # postings, parallel arrays
    term_ids: np.ndarray
    tf: np.ndarray
    doc_len: np.ndarray
    n_docs: int
    lang: str
    k1: float = 1.2
    b: float = 0.75
    variant: str = "bm25"        # "bm25" | "bm25l"
    delta: float = 0.5           # BM25L lower bound
    _W: sp.csr_matrix | None = None   # document weights, depend on (k1, b)
    _TF: sp.csr_matrix | None = None  # raw tf, used to build queries from history

    # ------------------------------------------------------------- build

    @classmethod
    def build(cls, texts: list[str], lang: str = "en", **kw) -> "BM25Index":
        toks = tokenize(texts, lang)
        stem_map = _stem_vocabulary(toks["token"], lang)
        toks = toks.join(stem_map, on="token", how="left")

        vocab = (
            toks.select("doc_id", "stem").unique()
            .group_by("stem").agg(pl.len().alias("df"))
            .sort("stem").with_row_index("term_id")
        )
        n_docs = len(texts)
        # Lucene/Robertson IDF: the +1 inside the log keeps it positive, unlike the
        # textbook form which crosses zero at 50% document frequency and would
        # actively penalise a document for matching a very common term.
        vocab = vocab.with_columns(
            (((pl.lit(n_docs) - pl.col("df") + 0.5) / (pl.col("df") + 0.5)) + 1).log().alias("idf")
        )

        postings = (
            toks.join(vocab.select("stem", "term_id"), on="stem", how="inner")
            .group_by("doc_id", "term_id").agg(pl.len().alias("tf"))
        )
        doc_len = np.zeros(n_docs, dtype=np.float32)
        dl = toks.group_by("doc_id").agg(pl.len().alias("n"))
        doc_len[dl["doc_id"].to_numpy()] = dl["n"].to_numpy()

        idx = cls(
            vocab=vocab,
            doc_ids=postings["doc_id"].to_numpy().astype(np.int32),
            term_ids=postings["term_id"].to_numpy().astype(np.int32),
            tf=postings["tf"].to_numpy().astype(np.float32),
            doc_len=doc_len,
            n_docs=n_docs,
            lang=lang,
            **kw,
        )
        idx._TF = sp.csr_matrix(
            (idx.tf, (idx.doc_ids, idx.term_ids)),
            shape=(n_docs, idx.vocab_size), dtype=np.float32,
        )
        idx.reweight(idx.k1, idx.b, idx.variant, idx.delta)
        return idx

    @property
    def vocab_size(self) -> int:
        return self.vocab.height

    @property
    def avgdl(self) -> float:
        return float(self.doc_len.mean())

    # ------------------------------------------------------------ score

    def reweight(self, k1: float | None = None, b: float | None = None,
                 variant: str | None = None, delta: float | None = None) -> "BM25Index":
        """Recompute document weights for new parameters.

        Only the weights depend on (k1, b), so the (k1, b) grid search reuses the
        tokenisation and the postings -- the expensive parts.
        """
        self.k1 = self.k1 if k1 is None else k1
        self.b = self.b if b is None else b
        self.variant = self.variant if variant is None else variant
        self.delta = self.delta if delta is None else delta

        idf = self.vocab["idf"].to_numpy().astype(np.float32)
        norm = 1.0 - self.b + self.b * (self.doc_len / max(self.avgdl, 1e-6))
        norm_p = norm[self.doc_ids]

        if self.variant == "bm25l":
            # BM25L (Lv & Zhai 2011): normalise tf first, then add delta so a single
            # occurrence always contributes something, even in a very long document.
            c = self.tf / norm_p
            sat = (self.k1 + 1.0) * (c + self.delta) / (self.k1 + c + self.delta)
        else:
            sat = self.tf * (self.k1 + 1.0) / (self.tf + self.k1 * norm_p)

        w = sat * idf[self.term_ids]
        self._W = sp.csr_matrix(
            (w, (self.doc_ids, self.term_ids)), shape=(self.n_docs, self.vocab_size), dtype=np.float32
        )
        return self

    def query_matrix(self, queries: list[list[int]]) -> sp.csr_matrix:
        """Term-id bags -> sparse query matrix (counts as query-side weights)."""
        indptr, indices, data = [0], [], []
        for q in queries:
            if q:
                terms, counts = np.unique(np.asarray(q, dtype=np.int32), return_counts=True)
                indices.append(terms)
                data.append(counts.astype(np.float32))
            indptr.append(indptr[-1] + (len(np.unique(q)) if q else 0))
        indices = np.concatenate(indices) if indices else np.array([], dtype=np.int32)
        data = np.concatenate(data) if data else np.array([], dtype=np.float32)
        return sp.csr_matrix((data, indices, indptr), shape=(len(queries), self.vocab_size), dtype=np.float32)

    def search(self, queries: list[list[int]], top_k: int = 200, batch: int | None = None,
               universe: np.ndarray | None = None, max_cells: int = 2 ** 26
               ) -> tuple[np.ndarray, np.ndarray]:
        """Top-k document ids and scores per query.

        `universe` restricts scoring to a subset of documents -- the "articles
        live in the last N days" filter, without which recall@K is measured
        against a catalogue full of articles nobody could have been shown.

        The scored block is (batch x documents) and dense, so `batch` is derived
        from the corpus rather than fixed: a constant 2048 is 240 MB against MIND's
        live universe and 10 GB against a 1.3M-document catalogue. `max_cells`
        caps the intermediate instead.
        """
        W = self._W
        if universe is not None:
            sub = W[universe]
        else:
            sub = W
        n_docs_eff = sub.shape[0]
        k = min(top_k, n_docs_eff)
        out_idx = np.zeros((len(queries), k), dtype=np.int32)
        out_scr = np.zeros((len(queries), k), dtype=np.float32)
        batch = batch or max(1, min(len(queries) or 1, max_cells // max(n_docs_eff, 1)))

        for start in range(0, len(queries), batch):
            Q = self.query_matrix(queries[start:start + batch])
            S = (Q @ sub.T).toarray()                      # (batch, n_docs_eff)
            part = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
            rows = np.arange(S.shape[0])[:, None]
            order = np.argsort(-S[rows, part], axis=1)
            top = part[rows, order]
            out_idx[start:start + S.shape[0]] = top if universe is None else universe[top]
            out_scr[start:start + S.shape[0]] = S[rows, top]
        return out_scr, out_idx

    # ------------------------------------------------------- query build

    def queries_from_history(
        self,
        histories: list[np.ndarray],
        n_recent: int = N_RECENT_DEFAULT,
        weights: list[np.ndarray] | None = None,
    ) -> sp.csr_matrix:
        """Build one query vector per user from the articles they clicked.

        The query for a user is the bag of terms of their recent articles, which
        is exactly `H @ TF` for a user-by-document history matrix H. Doing it as a
        matmul rather than per-user gathering is what makes 50K users tractable:
        one sparse product instead of 50K passes over the postings.

        `n_recent` truncates to the most recent clicks; 0 reads the whole history.
        Truncating is the intuitive choice -- news interest decays, so an unbounded
        query should drift towards a user's whole reading career -- but the sweep
        in scripts/q3_userrep.py measured the opposite on both datasets, so the
        default is 0. See N_RECENT_DEFAULT.
        """
        rows, cols, vals = [], [], []
        for u, docs in enumerate(histories):
            if len(docs) == 0:
                continue
            docs = np.asarray(docs)[-n_recent:] if n_recent else np.asarray(docs)
            w = np.ones(len(docs), dtype=np.float32)
            if weights is not None:
                w = np.asarray(weights[u], dtype=np.float32)[-len(docs):]
            rows.append(np.full(len(docs), u, dtype=np.int32))
            cols.append(docs.astype(np.int32))
            vals.append(w)
        if not rows:
            return sp.csr_matrix((len(histories), self.vocab_size), dtype=np.float32)
        H = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(len(histories), self.n_docs), dtype=np.float32,
        )
        return (H @ self._TF).tocsr()

    def search_sparse(self, Q: sp.csr_matrix, top_k: int = 200, batch: int | None = None,
                      universe: np.ndarray | None = None, max_cells: int = 2 ** 26
                      ) -> tuple[np.ndarray, np.ndarray]:
        """Same as `search`, for queries already in sparse-matrix form.

        The block is sized from the corpus for the same reason as `search`: the
        (batch x documents) intermediate is dense, so a fixed batch silently
        becomes a multi-gigabyte allocation once the candidate universe grows.
        """
        sub = self._W[universe] if universe is not None else self._W
        k = min(top_k, sub.shape[0])
        n = Q.shape[0]
        out_idx = np.zeros((n, k), dtype=np.int32)
        out_scr = np.zeros((n, k), dtype=np.float32)
        batch = batch or max(1, min(n or 1, max_cells // max(sub.shape[0], 1)))
        for start in range(0, n, batch):
            S = (Q[start:start + batch] @ sub.T).toarray()
            part = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
            r = np.arange(S.shape[0])[:, None]
            order = np.argsort(-S[r, part], axis=1)
            top = part[r, order]
            out_idx[start:start + S.shape[0]] = top if universe is None else universe[top]
            out_scr[start:start + S.shape[0]] = S[r, top]
        return out_scr, out_idx

    def score_pairs(self, Q: sp.csr_matrix, user_rows: np.ndarray, doc_ids: np.ndarray,
                    max_pairs: int = 2_000_000) -> np.ndarray:
        """BM25 for specific (query, document) pairs -- the Q4 ranking case.

        Retrieval scores a query against the whole corpus; ranking scores it
        against the handful of candidates one impression actually showed. Scoring
        a batch of users densely against the corpus and gathering the wanted cells
        costs O(users x corpus) to keep O(pairs) of it -- on MIND, 50K users x 65K
        documents to read out 2.7M numbers.

        Instead each pair is summed over the *document's* postings, which is what
        the score is: sum over the terms the document actually contains of
        q[user, term] * w[doc, term]. Cost is O(pairs x terms-per-document), with
        no dependence on corpus size. Measured: 11.1s -> 3.5s on EB-NeRD small
        (3.1x) and 102.1s -> 6.9s on MIND small (14.8x), same scores.
        """
        W, Qc = self._W.tocsr(), Q.tocsr()
        U = np.asarray(user_rows)
        A = np.asarray(doc_ids)
        out = np.zeros(len(U), dtype=np.float32)
        for s in range(0, len(U), max_pairs):
            e = min(s + max_pairs, len(U))
            u, a = U[s:e], A[s:e]
            # one dense query row per distinct user in the batch, not per pair
            uniq, uinv = np.unique(u, return_inverse=True)
            Qd = np.asarray(Qc[uniq].todense(), dtype=np.float32)
            starts = W.indptr[a].astype(np.int64)
            nnz = (W.indptr[a + 1] - W.indptr[a]).astype(np.int64)
            rep = np.repeat(np.arange(len(a)), nnz)
            # index into each document's postings run without a Python-level loop
            off = np.arange(int(nnz.sum()), dtype=np.int64) - np.repeat(np.cumsum(nnz) - nnz, nnz)
            flat = np.repeat(starts, nnz) + off
            contrib = Qd[uinv[rep], W.indices[flat]] * W.data[flat]
            out[s:e] = np.bincount(rep, weights=contrib, minlength=len(a)).astype(np.float32)
        return out

    def explain(self, q: sp.csr_matrix, doc_id: int, top_n: int = 12) -> pl.DataFrame:
        """Per-term decomposition of one (query, document) score.

        score(q, d) = sum over shared terms of  query_weight * idf * saturation,
        which is the table this returns -- the worked example behind the ranking.
        """
        qd = q.tocoo()
        wd = self._W[doc_id].tocoo()
        doc_w = dict(zip(wd.col.tolist(), wd.data.tolist()))
        rows = []
        for term, qw in zip(qd.col.tolist(), qd.data.tolist()):
            if term in doc_w:
                rows.append((term, qw, doc_w[term], qw * doc_w[term]))
        if not rows:
            return pl.DataFrame(schema={"term_id": pl.Int64, "stem": pl.Utf8, "df": pl.UInt32,
                                        "idf": pl.Float64, "q_weight": pl.Float64,
                                        "doc_weight": pl.Float64, "contribution": pl.Float64})
        df = pl.DataFrame(rows, schema=["term_id", "q_weight", "doc_weight", "contribution"],
                          orient="row")
        return (
            df.join(self.vocab.select(pl.col("term_id").cast(pl.Int64), "stem", "df", "idf"),
                    on="term_id", how="left")
            .sort("contribution", descending=True)
            .head(top_n)
        )
