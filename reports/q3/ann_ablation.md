# Q3 ablation: ANN index and its parameters

Every row searches the *same* vectors; the exact `IndexFlatIP` row is the
reference, so any difference is attributable to the index alone. `ANN r@100`
is overlap with the exact top-100 (the index's own fidelity); `task r@100` is
the assignment's recall against the clicked articles. Timings are
single-threaded on purpose — this box has 48 cores, and letting BLAS use all
of them makes brute force look like an ANN index.

### ebnerd_small_contrastive — 1,677 candidates × 768d, 15,342 queries, 1 thread

| index | build (s) | size (MB) | q/s | ANN r@100 | task r@100 | Δ task |
|---|--:|--:|--:|--:|--:|--:|
| numpy brute force | 0.00 | 4.9 | 20,394 | 1.0000 | 0.1065 | +0.0% |
| faiss flat (exact) | 0.00 | 4.9 | 20,463 | 1.0000 | 0.1065 | +0.0% |
| faiss SQ8 (exact scan, 8-bit) | 0.00 | 1.2 | 5,764 | 0.9975 | 0.1064 | -0.1% |
| hnsw M=8 efC=200 efS=16 | 0.21 | 5.0 | 34,020 | 0.6350 | 0.1040 | -2.3% |
| hnsw M=8 efC=200 efS=64 | 0.17 | 5.0 | 16,626 | 0.9534 | 0.1074 | +0.9% |
| hnsw M=8 efC=200 efS=256 | 0.20 | 5.0 | 7,338 | 0.9998 | 0.1065 | +0.0% |
| hnsw M=16 efC=200 efS=16 | 0.18 | 5.1 | 31,246 | 0.7231 | 0.1056 | -0.9% |
| hnsw M=16 efC=200 efS=64 | 0.21 | 5.1 | 13,858 | 0.9800 | 0.1067 | +0.2% |
| hnsw M=16 efC=200 efS=256 | 0.18 | 5.1 | 7,316 | 1.0000 | 0.1065 | +0.0% |
| hnsw M=32 efC=200 efS=16 | 0.21 | 5.3 | 27,697 | 0.7422 | 0.1057 | -0.7% |
| hnsw M=32 efC=200 efS=64 | 0.16 | 5.3 | 14,981 | 0.9851 | 0.1070 | +0.5% |
| hnsw M=32 efC=200 efS=256 | 0.19 | 5.3 | 6,939 | 1.0000 | 0.1065 | +0.0% |
| hnsw M=64 efC=200 efS=16 | 0.17 | 5.8 | 29,974 | 0.7426 | 0.1042 | -2.1% |
| hnsw M=64 efC=200 efS=64 | 0.20 | 5.8 | 13,638 | 0.9838 | 0.1066 | +0.1% |
| hnsw M=64 efC=200 efS=256 | 0.18 | 5.8 | 7,324 | 1.0000 | 0.1065 | +0.0% |
| hnsw M=32 efC=40 efS=64 | 0.07 | 5.3 | 11,901 | 0.9885 | 0.1068 | +0.3% |
| hnsw M=32 efC=40 efS=256 | 0.07 | 5.3 | 6,858 | 1.0000 | 0.1065 | +0.0% |
| hnsw M=32 efC=500 efS=64 | 0.36 | 5.3 | 13,157 | 0.9867 | 0.1069 | +0.4% |
| hnsw M=32 efC=500 efS=256 | 0.34 | 5.3 | 6,650 | 1.0000 | 0.1065 | +0.0% |
| hnsw M=32 efC=200 efS=256/L2 | 0.18 | 5.3 | 7,055 | 1.0000 | 0.1065 | +0.0% |
| ivf nlist=64 nprobe=1 | 0.02 | 5.1 | 120,190 | 0.1758 | 0.0334 | -68.6% |
| ivf nlist=64 nprobe=8 | 0.02 | 5.1 | 30,220 | 0.7379 | 0.1046 | -1.7% |
| ivf nlist=64 nprobe=32 | 0.03 | 5.1 | 12,104 | 0.9967 | 0.1065 | -0.0% |
| ivf nlist=256 nprobe=1 | 0.09 | 5.7 | 103,019 | 0.0652 | 0.0099 | -90.7% |
| ivf nlist=256 nprobe=8 | 0.09 | 5.7 | 58,262 | 0.3874 | 0.0677 | -36.5% |
| ivf nlist=256 nprobe=32 | 0.09 | 5.7 | 23,286 | 0.8556 | 0.1068 | +0.3% |
| ivf nlist=1024 nprobe=1 | 0.20 | 7.9 | 51,915 | 0.0231 | 0.0022 | -97.9% |
| ivf nlist=1024 nprobe=8 | 0.18 | 7.9 | 44,865 | 0.1591 | 0.0200 | -81.2% |
| ivf nlist=1024 nprobe=32 | 0.20 | 7.9 | 28,117 | 0.5144 | 0.0689 | -35.3% |
| ivfpq nlist=256 nprobe=32 m=8 | 0.22 | 1.5 | 1,925 | 0.6742 | 0.1070 | +0.5% |

### mind_small_all-MiniLM-L6-v2 — 22,771 candidates × 384d, 10,000 queries, 1 thread

| index | build (s) | size (MB) | q/s | ANN r@100 | task r@100 | Δ task |
|---|--:|--:|--:|--:|--:|--:|
| numpy brute force | 0.00 | 33.4 | 2,197 | 1.0000 | 0.0353 | +0.0% |
| faiss flat (exact) | 0.00 | 33.4 | 4,227 | 1.0000 | 0.0353 | +0.0% |
| faiss SQ8 (exact scan, 8-bit) | 0.03 | 8.3 | 996 | 0.9939 | 0.0352 | -0.3% |
| hnsw M=8 efC=200 efS=16 | 4.10 | 35.1 | 22,619 | 0.4330 | 0.0353 | +0.0% |
| hnsw M=8 efC=200 efS=64 | 4.06 | 35.1 | 10,230 | 0.7501 | 0.0384 | +8.7% |
| hnsw M=8 efC=200 efS=256 | 4.27 | 35.1 | 4,032 | 0.9549 | 0.0359 | +1.7% |
| hnsw M=16 efC=200 efS=16 | 5.39 | 36.5 | 14,092 | 0.5469 | 0.0364 | +3.2% |
| hnsw M=16 efC=200 efS=64 | 5.71 | 36.5 | 6,345 | 0.8584 | 0.0365 | +3.5% |
| hnsw M=16 efC=200 efS=256 | 5.29 | 36.5 | 2,854 | 0.9885 | 0.0352 | -0.3% |
| hnsw M=32 efC=200 efS=16 | 6.19 | 39.3 | 11,315 | 0.6249 | 0.0366 | +3.7% |
| hnsw M=32 efC=200 efS=64 | 6.00 | 39.3 | 5,691 | 0.9071 | 0.0364 | +3.2% |
| hnsw M=32 efC=200 efS=256 | 6.79 | 39.3 | 2,064 | 0.9947 | 0.0355 | +0.7% |
| hnsw M=64 efC=200 efS=16 | 7.06 | 44.8 | 8,880 | 0.6636 | 0.0358 | +1.4% |
| hnsw M=64 efC=200 efS=64 | 6.83 | 44.8 | 4,306 | 0.9253 | 0.0355 | +0.7% |
| hnsw M=64 efC=200 efS=256 | 6.65 | 44.8 | 2,006 | 0.9959 | 0.0354 | +0.4% |
| hnsw M=32 efC=40 efS=64 | 1.84 | 39.3 | 5,764 | 0.8925 | 0.0360 | +2.2% |
| hnsw M=32 efC=40 efS=256 | 1.84 | 39.3 | 2,443 | 0.9892 | 0.0354 | +0.4% |
| hnsw M=32 efC=500 efS=64 | 14.63 | 39.3 | 4,476 | 0.9154 | 0.0364 | +3.2% |
| hnsw M=32 efC=500 efS=256 | 14.93 | 39.3 | 1,901 | 0.9967 | 0.0354 | +0.4% |
| hnsw M=32 efC=200 efS=256/L2 | 6.97 | 39.3 | 2,038 | 0.9947 | 0.0356 | +1.0% |
| ivf nlist=64 nprobe=1 | 0.17 | 33.6 | 26,968 | 0.3198 | 0.0359 | +1.8% |
| ivf nlist=64 nprobe=8 | 0.16 | 33.6 | 7,432 | 0.8356 | 0.0380 | +7.6% |
| ivf nlist=64 nprobe=32 | 0.15 | 33.6 | 2,497 | 0.9861 | 0.0354 | +0.5% |
| ivf nlist=256 nprobe=1 | 0.58 | 33.9 | 62,744 | 0.1779 | 0.0297 | -15.8% |
| ivf nlist=256 nprobe=8 | 0.58 | 33.9 | 16,603 | 0.5952 | 0.0362 | +2.6% |
| ivf nlist=256 nprobe=32 | 0.55 | 33.9 | 7,615 | 0.8813 | 0.0365 | +3.5% |
| ivf nlist=1024 nprobe=1 | 1.98 | 35.0 | 72,320 | 0.1014 | 0.0250 | -29.1% |
| ivf nlist=1024 nprobe=8 | 1.98 | 35.0 | 30,195 | 0.4008 | 0.0471 | +33.6% |
| ivf nlist=1024 nprobe=32 | 2.03 | 35.0 | 14,108 | 0.7128 | 0.0393 | +11.3% |
| ivfpq nlist=256 nprobe=32 m=8 | 2.69 | 1.1 | 2,251 | 0.3398 | 0.0403 | +14.4% |

### scale sweep — ebnerd_large_contrastive, 768d, 2,000 queries, 1 thread

| index | N=2,000 | N=8,000 | N=32,000 | N=125,000 |
|---|--:|--:|--:|--:|
| **throughput (q/s)** | | | | |
| numpy brute force | 16,585 | 4,641 | 1,164 | 288 |
| faiss flat (exact) | 17,416 | 6,496 | 1,930 | 526 |
| hnsw M=16 efC=200 efS=64 | 15,568 | 12,520 | 6,558 | 4,953 |
| hnsw M=32 efC=200 efS=64 | 14,321 | 11,175 | 6,026 | 4,502 |
| hnsw M=32 efC=200 efS=256 | 7,229 | 5,571 | 2,403 | 1,713 |
| ivf nlist=50 nprobe=8 | 24,478 | — | — | — |
| ivf nlist=200 nprobe=8 | — | 21,729 | — | — |
| ivf nlist=800 nprobe=8 | — | — | 13,186 | — |
| ivf nlist=1024 nprobe=8 | — | — | — | 5,728 |
| **build time (s)** | | | | |
| numpy brute force | 0.00 | 0.00 | 0.00 | 0.00 |
| faiss flat (exact) | 0.00 | 0.01 | 0.05 | 0.19 |
| hnsw M=16 efC=200 efS=64 | 0.22 | 1.14 | 8.44 | 50.02 |
| hnsw M=32 efC=200 efS=64 | 0.23 | 1.23 | 9.07 | 53.83 |
| hnsw M=32 efC=200 efS=256 | 0.21 | 1.13 | 8.91 | 54.01 |
| ivf nlist=50 nprobe=8 | 0.03 | — | — | — |
| ivf nlist=200 nprobe=8 | — | 0.29 | — | — |
| ivf nlist=800 nprobe=8 | — | — | 4.38 | — |
| ivf nlist=1024 nprobe=8 | — | — | — | 21.40 |
| **ANN recall@100** | | | | |
| numpy brute force | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| faiss flat (exact) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hnsw M=16 efC=200 efS=64 | 0.9830 | 0.9804 | 0.9742 | 0.9646 |
| hnsw M=32 efC=200 efS=64 | 0.9858 | 0.9844 | 0.9799 | 0.9732 |
| hnsw M=32 efC=200 efS=256 | 1.0000 | 1.0000 | 0.9997 | 0.9995 |
| ivf nlist=50 nprobe=8 | 0.8960 | — | — | — |
| ivf nlist=200 nprobe=8 | — | 0.8680 | — | — |
| ivf nlist=800 nprobe=8 | — | — | 0.8117 | — |
| ivf nlist=1024 nprobe=8 | — | — | — | 0.8781 |
