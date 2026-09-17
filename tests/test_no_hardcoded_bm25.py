"""No script may hardcode BM25's operating point.

Q2 tunes (k1, b) per corpus and the design note reports those values. Any script
that scores BM25 with a different pair makes its table quietly disagree with the
table above it -- and when the untuned pair is the weaker one, it flatters
whatever BM25 is being compared against.

That is not hypothetical. `scripts/q3_semantic.py` produces the headline
lexical-vs-semantic answer and scored BM25 at k1=1.5 while Q2 had selected
k1=2.0 for MIND, understating BM25 by 6% on the exact comparison the assignment
asks about. Four other call sites had the same bug. This test is a lint, not a
behaviour check: it exists because grepping for the literal string missed two
files that built the arguments dynamically.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

SCRIPTS = sorted(Path("scripts").glob("*.py"))
# a numeric literal passed to k1= or b=, e.g. reweight(k1=1.5, b=1.0)
LITERAL = re.compile(r"\b(?:k1|b)\s*=\s*\d")
# q2_bm25 owns the grid, and q2_query/q2_index deliberately vary the parameters
ALLOWED = {"q2_bm25.py", "q2_query.py", "q2_index.py"}

pytestmark = pytest.mark.skipif(not SCRIPTS, reason="run from the repo root")


@pytest.mark.parametrize("path", SCRIPTS, ids=[p.name for p in SCRIPTS])
def test_bm25_parameters_come_from_the_q2_selection(path: Path):
    if path.name in ALLOWED:
        pytest.skip("owns or deliberately varies the (k1, b) grid")
    src = path.read_text()
    calls = [m for m in re.finditer(r"reweight\s*\(([^)]*)\)", src, re.S)]
    offenders = [c.group(0).replace("\n", " ") for c in calls if LITERAL.search(c.group(1))]
    assert not offenders, (
        f"{path} pins BM25's operating point instead of reading Q2's: {offenders}. "
        "Use tuned_bm25(dataset, variant, None, None).")
