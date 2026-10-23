# VQAv2 Rescoring Findings

## Research Question

Can the existing CPU VQAv2 rescoring path run faster without changing official scores, validation, or input data? This local infrastructure study does not test the repository's multimodal expansion hypothesis.

## Current Understanding

Repeated official answer normalization dominated the fixed synthetic workload. A bounded cache of normalized strings removed repeated regex work without caching scores or bypassing validation. Exact consensus arithmetic, native vote counting and direct whitespace cleanup reduced the remaining overhead. The official exception for unanimous references remains essential: those references must not receive digit/article/punctuation normalization.

## Key Results

- Registered baseline: **612.717439 ms** per 3,520 predictions.
- Final unchanged-code replication: **13.978255 ms**, **251,819.70 predictions/s**, approximately **43.83× faster** (97.72% less time).
- Best observed kept run: **13.875030 ms**. Use the final replication, not this minimum, for the headline comparison.
- Every canonical run retained **46.090909090909 accuracy points**, with individual/aggregate parity tolerance `1e-12` and reference immutability checks.
- Separate cold-cache comparison: 6,000 unique-answer cases, three repeats, baseline median **1,273.078 ms**, optimized **371.549 ms**, **3.426×**. Each row still repeats annotator answers; this is a synthetic stress case, not representative deployment evidence.
- Verification: 5,000 seeded varied differential cases against the pre-change implementation; tuple/list parity, malformed-reference errors and immutability; 6,000 unseen-string cases with cache eviction and revisits; **10 existing VQA/GQA regressions passed**. Caption scoring was not exercised.

## Patterns and Insights

The largest gain came from bounded normalization reuse, not changing the metric. Per-call deduplication helped before caching but became redundant afterward. Building a normalized list and using native `list.count` beat a fused Python generator despite its extra list allocation. Small incremental timing differences are single-run observations, not isolated causal estimates.

## Lessons and Constraints

- Keep the workload, oracle, dependencies, seed and timing policy fixed. No score or whole-row cache.
- The normalization LRU holds at most 4,096 strings; this is not a bound on the archived processor's existing identity-map state or on total process memory.
- Preserve ten-reference validation and conditional normalization, including unanimous gold answers.
- Do not infer model quality, caption performance, or expansion effectiveness from scoring throughput.
- The benchmark intentionally repeats a small vocabulary and excludes import/startup time. Its warm-cache speedup does not generalize to arbitrary data.
- Fused generator reduction was slower (14.701181 ms vs 13.875030 ms) and was discarded.
- No wall-clock continuation scheduler is available. The configured ten-run segment is complete; remote sync/push was not performed.

## Open Questions

How much of the improvement transfers to real prediction corpora, other Python versions, concurrent callers and vocabularies larger than the cache? These questions remain unmeasured; no workload expansion was performed in this frozen segment. The original multimodal research question remains unresolved by this infrastructure study.

## Optimization Trajectory

Each value is the median of seven full-corpus passes after warmup. Sequential changes are cumulative; rows are not independent ablations.

| Run | Change | scoring_ms | Decision |
|---|---|---:|---|
| 1 | Unmodified baseline | 612.717439 | Baseline |
| 2 | Normalize distinct answers per call | 398.711319 | Keep |
| 3 | Equivalent count-based consensus | 392.961949 | Keep |
| 4 | Bounded string-normalization LRU | 18.641178 | Keep |
| 5 | Remove redundant per-call dictionary | 17.931396 | Keep |
| 6 | Native vote counting | 15.901323 | Keep |
| 7 | Direct identical whitespace cleanup | 14.199462 | Keep |
| 8 | Allocation-free unanimity check | 13.875030 | Keep |
| 9 | Fuse normalization and counting | 14.701181 | Discard |
| 10 | Replicate retained implementation | 13.978255 | Keep |

Harness: `autoresearch.sh`; production implementation: `sema_multimodal/metrics.py`; [visual report](to_human/vqa-scoring-progress.html). Raw canonical run output is retained by the autoresearch session manager.
