# Q4 — offline evaluation harness

Ranking quality *within* the impression, which is what both leaderboards score.
Q2/Q3 measured retrieval from the whole candidate universe; a retriever that never
surfaces an article can still rank it correctly once the impression puts it in front
of the user, so the two tables answer different questions and disagree on purpose.

`random` is the calibration check: a correct AUC implementation must put it at 0.5000.

### ebnerd/large — test split, history=shipped

600,000 labelled impressions · 7,185,103 candidate rows · 602,391 clicks · catalogue 6,024 · BM25 k1=2.0 b=0.3 · 200 bootstrap resamples

| ranker | AUC | MRR | nDCG@5 | nDCG@10 (95% CI) | ILD@10 | novelty@10 | coverage@10 |
|---|--:|--:|--:|--:|--:|--:|--:|
| random | 0.4999 | 0.3127 | 0.3450 | 0.4298 (0.4289–0.4303) | 0.7720 | 19.922 | 0.9095 |
| pop_prior | 0.4274 | 0.2591 | 0.2804 | 0.3804 (0.3797–0.3809) | 0.7564 | 19.446 | 0.8810 |
| ctr_prior | 0.4244 | 0.2587 | 0.2785 | 0.3795 (0.3788–0.3800) | 0.7549 | 19.470 | 0.9314 |
| recency | 0.5009 | 0.3070 | 0.3359 | 0.4208 (0.4199–0.4214) | 0.7777 | 20.280 | 0.8357 |
| bm25 | 0.5275 | 0.3371 | 0.3708 | 0.4520 (0.4512–0.4526) | 0.7617 | 19.828 | 0.8966 |
| emb | 0.5462 | 0.3527 | 0.3878 | 0.4666 (0.4658–0.4672) | 0.6728 | 19.929 | 0.8913 |
| hybrid_rrf | 0.5456 | 0.3491 | 0.3863 | 0.4636 (0.4628–0.4642) | 0.7072 | 19.863 | 0.8936 |
| **pop_oracle*** | 0.6547 | 0.4185 | 0.4726 | 0.5331 (0.5323–0.5337) | 0.7739 | 20.214 | 0.8206 |

**Slices** (nDCG@10)

| ranker | cold | warm | head | tail |
|---|--:|--:|--:|--:|
| random | 0.4160 | 0.4300 | 0.3728 | 0.4323 |
| pop_prior | 0.3667 | 0.3807 | 0.6873 | 0.3669 |
| ctr_prior | 0.3652 | 0.3798 | 0.6367 | 0.3681 |
| recency | 0.3845 | 0.4215 | 0.2846 | 0.4268 |
| bm25 | 0.4234 | 0.4525 | 0.4246 | 0.4532 |
| emb | 0.4475 | 0.4670 | 0.3872 | 0.4701 |
| hybrid_rrf | 0.4387 | 0.4640 | 0.4034 | 0.4662 |
| pop_oracle* | 0.5228 | 0.5333 | 0.3916 | 0.5394 |

Slice sizes: cold 10,970 / warm 589,030 (history < 10), head 25,402 / tail 574,598 (clicked article in the top 20% by prior decayed clicks).

### ebnerd/small — test split, history=augmented

244,647 labelled impressions · 2,928,942 candidate rows · 245,622 clicks · catalogue 4,701 · BM25 k1=2.0 b=1.0 · 500 bootstrap resamples

| ranker | AUC | MRR | nDCG@5 | nDCG@10 (95% CI) | ILD@10 | novelty@10 | coverage@10 |
|---|--:|--:|--:|--:|--:|--:|--:|
| random | 0.4993 | 0.3125 | 0.3447 | 0.4297 (0.4284–0.4308) | 0.7722 | 15.523 | 0.9258 |
| pop_prior | 0.4437 | 0.2665 | 0.2918 | 0.3883 (0.3872–0.3893) | 0.7616 | 15.323 | 0.8983 |
| ctr_prior | 0.4245 | 0.2569 | 0.2771 | 0.3782 (0.3771–0.3792) | 0.7546 | 15.337 | 0.9221 |
| recency | 0.5015 | 0.3080 | 0.3368 | 0.4219 (0.4205–0.4230) | 0.7779 | 15.643 | 0.8371 |
| bm25 | 0.5377 | 0.3480 | 0.3831 | 0.4617 (0.4604–0.4628) | 0.7618 | 15.495 | 0.9081 |
| emb | 0.5453 | 0.3522 | 0.3870 | 0.4660 (0.4646–0.4671) | 0.6728 | 15.533 | 0.9013 |
| hybrid_rrf | 0.5518 | 0.3541 | 0.3918 | 0.4688 (0.4674–0.4699) | 0.7076 | 15.511 | 0.9064 |
| **pop_oracle*** | 0.6540 | 0.4177 | 0.4721 | 0.5326 (0.5314–0.5336) | 0.7742 | 15.614 | 0.8396 |

**Slices** (nDCG@10)

| ranker | cold | warm | head | tail |
|---|--:|--:|--:|--:|
| random | 0.4162 | 0.4299 | 0.3597 | 0.4319 |
| pop_prior | 0.3750 | 0.3885 | 0.6997 | 0.3783 |
| ctr_prior | 0.3668 | 0.3784 | 0.6476 | 0.3696 |
| recency | 0.3823 | 0.4225 | 0.2981 | 0.4258 |
| bm25 | 0.4419 | 0.4621 | 0.4303 | 0.4627 |
| emb | 0.4526 | 0.4662 | 0.3758 | 0.4689 |
| hybrid_rrf | 0.4563 | 0.4690 | 0.4048 | 0.4708 |
| pop_oracle* | 0.5228 | 0.5328 | 0.4331 | 0.5358 |

Slice sizes: cold 4,276 / warm 240,371 (history < 10), head 7,621 / tail 237,026 (clicked article in the top 20% by prior decayed clicks).

### ebnerd/small — test split, history=shipped

244,647 labelled impressions · 2,928,942 candidate rows · 245,622 clicks · catalogue 4,701 · BM25 k1=2.0 b=1.0 · 500 bootstrap resamples

| ranker | AUC | MRR | nDCG@5 | nDCG@10 (95% CI) | ILD@10 | novelty@10 | coverage@10 |
|---|--:|--:|--:|--:|--:|--:|--:|
| random | 0.4993 | 0.3125 | 0.3447 | 0.4297 (0.4284–0.4308) | 0.7722 | 15.523 | 0.9258 |
| pop_prior | 0.4437 | 0.2665 | 0.2918 | 0.3883 (0.3872–0.3893) | 0.7616 | 15.323 | 0.8983 |
| ctr_prior | 0.4245 | 0.2569 | 0.2771 | 0.3782 (0.3771–0.3792) | 0.7546 | 15.337 | 0.9221 |
| recency | 0.5015 | 0.3080 | 0.3368 | 0.4219 (0.4205–0.4230) | 0.7779 | 15.643 | 0.8371 |
| bm25 | 0.5377 | 0.3479 | 0.3831 | 0.4617 (0.4604–0.4628) | 0.7618 | 15.495 | 0.9081 |
| emb | 0.5453 | 0.3522 | 0.3870 | 0.4660 (0.4646–0.4671) | 0.6728 | 15.533 | 0.9013 |
| hybrid_rrf | 0.5518 | 0.3541 | 0.3918 | 0.4688 (0.4675–0.4699) | 0.7076 | 15.512 | 0.9066 |
| **pop_oracle*** | 0.6540 | 0.4177 | 0.4721 | 0.5326 (0.5314–0.5336) | 0.7742 | 15.614 | 0.8396 |

**Slices** (nDCG@10)

| ranker | cold | warm | head | tail |
|---|--:|--:|--:|--:|
| random | 0.4162 | 0.4299 | 0.3597 | 0.4319 |
| pop_prior | 0.3750 | 0.3885 | 0.6997 | 0.3783 |
| ctr_prior | 0.3668 | 0.3784 | 0.6476 | 0.3696 |
| recency | 0.3823 | 0.4225 | 0.2981 | 0.4258 |
| bm25 | 0.4419 | 0.4620 | 0.4304 | 0.4627 |
| emb | 0.4528 | 0.4662 | 0.3758 | 0.4689 |
| hybrid_rrf | 0.4563 | 0.4690 | 0.4048 | 0.4708 |
| pop_oracle* | 0.5228 | 0.5328 | 0.4331 | 0.5358 |

Slice sizes: cold 4,276 / warm 240,371 (history < 10), head 7,621 / tail 237,026 (clicked article in the top 20% by prior decayed clicks).

### mind/large — test split, history=shipped

200,000 labelled impressions · 7,505,610 candidate rows · 306,063 clicks · catalogue 6,533 · BM25 k1=2.0 b=1.0 · 200 bootstrap resamples

| ranker | AUC | MRR | nDCG@5 | nDCG@10 (95% CI) | ILD@10 | novelty@10 | coverage@10 |
|---|--:|--:|--:|--:|--:|--:|--:|
| random | 0.4992 | 0.2185 | 0.2233 | 0.2859 (0.2847–0.2872) | 0.9448 | 16.662 | 0.6839 |
| pop_prior | 0.5450 | 0.2379 | 0.2501 | 0.3132 (0.3119–0.3144) | 0.9183 | 13.692 | 0.3443 |
| ctr_prior | 0.6044 | 0.2708 | 0.2853 | 0.3546 (0.3532–0.3559) | 0.9628 | 14.945 | 0.3334 |
| recency | 0.5140 | 0.2190 | 0.2332 | 0.2957 (0.2945–0.2969) | 0.9387 | 18.306 | 0.5844 |
| bm25 | 0.5751 | 0.2733 | 0.2925 | 0.3535 (0.3521–0.3549) | 0.8954 | 16.565 | 0.6819 |
| emb | 0.6360 | 0.3057 | 0.3332 | 0.3935 (0.3921–0.3947) | 0.8287 | 16.232 | 0.6603 |
| hybrid_rrf | 0.6259 | 0.2982 | 0.3244 | 0.3851 (0.3837–0.3864) | 0.8531 | 16.360 | 0.6792 |
| **pop_oracle*** | 0.5934 | 0.2880 | 0.3110 | 0.3700 (0.3686–0.3713) | 0.9195 | 16.242 | 0.2362 |

**Slices** (nDCG@10)

| ranker | cold | warm | head | tail |
|---|--:|--:|--:|--:|
| random | 0.3183 | 0.2807 | 0.2657 | 0.3047 |
| pop_prior | 0.3549 | 0.3064 | 0.3808 | 0.2505 |
| ctr_prior | 0.3941 | 0.3482 | 0.4201 | 0.2939 |
| recency | 0.3118 | 0.2931 | 0.2100 | 0.3751 |
| bm25 | 0.3411 | 0.3556 | 0.3371 | 0.3688 |
| emb | 0.3744 | 0.3966 | 0.3906 | 0.3962 |
| hybrid_rrf | 0.3636 | 0.3886 | 0.3763 | 0.3933 |
| pop_oracle* | 0.3999 | 0.3652 | 0.3540 | 0.3849 |

Slice sizes: cold 27,928 / warm 172,072 (history < 5), head 96,184 / tail 103,816 (clicked article in the top 20% by prior decayed clicks).

### mind/small — test split, history=augmented

73,152 labelled impressions · 2,740,998 candidate rows · 111,383 clicks · catalogue 5,369 · BM25 k1=2.0 b=1.0 · 500 bootstrap resamples

| ranker | AUC | MRR | nDCG@5 | nDCG@10 (95% CI) | ILD@10 | novelty@10 | coverage@10 |
|---|--:|--:|--:|--:|--:|--:|--:|
| random | 0.5008 | 0.2189 | 0.2240 | 0.2868 (0.2846–0.2888) | 0.9453 | 15.452 | 0.6633 |
| pop_prior | 0.5331 | 0.2375 | 0.2466 | 0.3107 (0.3084–0.3127) | 0.9209 | 13.513 | 0.3563 |
| ctr_prior | 0.5582 | 0.2564 | 0.2731 | 0.3331 (0.3306–0.3353) | 0.9583 | 14.250 | 0.3172 |
| recency | 0.5141 | 0.2141 | 0.2286 | 0.2913 (0.2892–0.2933) | 0.9521 | 16.304 | 0.6243 |
| bm25 | 0.5755 | 0.2748 | 0.2935 | 0.3547 (0.3522–0.3569) | 0.8955 | 15.375 | 0.6655 |
| emb | 0.6381 | 0.3069 | 0.3352 | 0.3947 (0.3923–0.3971) | 0.8283 | 15.203 | 0.6364 |
| hybrid_rrf | 0.6276 | 0.2989 | 0.3247 | 0.3857 (0.3834–0.3880) | 0.8524 | 15.258 | 0.6595 |
| **pop_oracle*** | 0.5976 | 0.2892 | 0.3126 | 0.3722 (0.3697–0.3745) | 0.9194 | 15.245 | 0.1980 |

**Slices** (nDCG@10)

| ranker | cold | warm | head | tail |
|---|--:|--:|--:|--:|
| random | 0.3264 | 0.2803 | 0.2773 | 0.2925 |
| pop_prior | 0.3529 | 0.3038 | 0.4298 | 0.2396 |
| ctr_prior | 0.3724 | 0.3267 | 0.4433 | 0.2674 |
| recency | 0.3123 | 0.2879 | 0.2001 | 0.3458 |
| bm25 | 0.3507 | 0.3553 | 0.3558 | 0.3541 |
| emb | 0.3841 | 0.3965 | 0.4067 | 0.3875 |
| hybrid_rrf | 0.3720 | 0.3880 | 0.3923 | 0.3818 |
| pop_oracle* | 0.3977 | 0.3680 | 0.3580 | 0.3806 |

Slice sizes: cold 10,306 / warm 62,846 (history < 5), head 27,347 / tail 45,805 (clicked article in the top 20% by prior decayed clicks).

### mind/small — test split, history=shipped

73,152 labelled impressions · 2,740,998 candidate rows · 111,383 clicks · catalogue 5,369 · BM25 k1=2.0 b=1.0 · 500 bootstrap resamples

| ranker | AUC | MRR | nDCG@5 | nDCG@10 (95% CI) | ILD@10 | novelty@10 | coverage@10 |
|---|--:|--:|--:|--:|--:|--:|--:|
| random | 0.5008 | 0.2189 | 0.2240 | 0.2868 (0.2846–0.2888) | 0.9453 | 15.452 | 0.6633 |
| pop_prior | 0.5331 | 0.2375 | 0.2466 | 0.3107 (0.3084–0.3127) | 0.9209 | 13.513 | 0.3563 |
| ctr_prior | 0.5582 | 0.2564 | 0.2731 | 0.3331 (0.3306–0.3353) | 0.9583 | 14.250 | 0.3172 |
| recency | 0.5141 | 0.2141 | 0.2286 | 0.2913 (0.2892–0.2933) | 0.9521 | 16.304 | 0.6243 |
| bm25 | 0.5744 | 0.2741 | 0.2924 | 0.3536 (0.3511–0.3559) | 0.8963 | 15.380 | 0.6651 |
| emb | 0.6368 | 0.3063 | 0.3344 | 0.3938 (0.3914–0.3962) | 0.8291 | 15.207 | 0.6344 |
| hybrid_rrf | 0.6262 | 0.2978 | 0.3233 | 0.3845 (0.3822–0.3868) | 0.8531 | 15.264 | 0.6593 |
| **pop_oracle*** | 0.5976 | 0.2892 | 0.3126 | 0.3722 (0.3697–0.3745) | 0.9194 | 15.245 | 0.1980 |

**Slices** (nDCG@10)

| ranker | cold | warm | head | tail |
|---|--:|--:|--:|--:|
| random | 0.3264 | 0.2803 | 0.2773 | 0.2925 |
| pop_prior | 0.3529 | 0.3038 | 0.4298 | 0.2396 |
| ctr_prior | 0.3724 | 0.3267 | 0.4433 | 0.2674 |
| recency | 0.3123 | 0.2879 | 0.2001 | 0.3458 |
| bm25 | 0.3480 | 0.3545 | 0.3544 | 0.3531 |
| emb | 0.3794 | 0.3962 | 0.4049 | 0.3871 |
| hybrid_rrf | 0.3685 | 0.3871 | 0.3906 | 0.3809 |
| pop_oracle* | 0.3977 | 0.3680 | 0.3580 | 0.3806 |

Slice sizes: cold 10,306 / warm 62,846 (history < 5), head 27,347 / tail 45,805 (clicked article in the top 20% by prior decayed clicks).

## Q9 — what a serving-time-unavailable feature buys

`pop_oracle*` is the same popularity ranker as `pop_prior`, with one change: it counts
clicks from *inside* the scored split instead of strictly before it. Nothing else
differs — same candidates, same impressions, same metric code. The gap is the size of
the illusion that a future-leaking feature creates.

| dataset | metric | serving-safe | with future clicks | inflation |
|---|---|--:|--:|--:|
| ebnerd/large | AUC | 0.4274 | 0.6547 | +53.2% |
| ebnerd/large | NDCG@10 | 0.3804 | 0.5331 | +40.1% |
| ebnerd/small | AUC | 0.4437 | 0.6540 | +47.4% |
| ebnerd/small | NDCG@10 | 0.3883 | 0.5326 | +37.2% |
| mind/large | AUC | 0.5450 | 0.5934 | +8.9% |
| mind/large | NDCG@10 | 0.3132 | 0.3700 | +18.1% |
| mind/small | AUC | 0.5331 | 0.5976 | +12.1% |
| mind/small | NDCG@10 | 0.3107 | 0.3722 | +19.8% |

## History-window ablation (shipped vs augmented)

`augmented` appends every click observed in earlier splits to the history the dataset
ships, cut strictly at the target split's start. It is legal at serving time — a real
system remembers what it logged — but it is not what the leaderboard hands you.

| dataset | ranker | nDCG@10 shipped | nDCG@10 augmented | Δ |
|---|---|--:|--:|--:|
| ebnerd/small | bm25 | 0.4617 | 0.4617 | +0.0% |
| ebnerd/small | emb | 0.4660 | 0.4660 | +0.0% |
| ebnerd/small | hybrid_rrf | 0.4688 | 0.4688 | -0.0% |
| mind/small | bm25 | 0.3536 | 0.3547 | +0.3% |
| mind/small | emb | 0.3938 | 0.3947 | +0.2% |
| mind/small | hybrid_rrf | 0.3845 | 0.3857 | +0.3% |
