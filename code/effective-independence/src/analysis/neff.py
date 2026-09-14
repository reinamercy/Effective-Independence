"""N_eff = N / (1 + (N-1) * mean_pairwise_error_correlation), with bootstrap
(1000 resamples over questions) 95% confidence intervals.

Error correlation is computed as the phi coefficient (Pearson correlation on
binary 0/1 vectors) between each pair of configs' error indicators (1 =
wrong, 0 = correct), using only questions where both configs in the pair
have a graded (non-null) result. With only ~9-30 questions per dataset,
individual configs can easily be constant (all-correct or all-wrong) over
the graded subset, which makes the correlation undefined for that pair
(zero variance) -- these pairs are excluded from the mean and counted
separately rather than silently treated as zero, per the project's
"fail loud on data-quality issues" rule (a large undefined-pair count is a
data-quality signal worth surfacing, not hiding).
"""
from __future__ import annotations

import random

import numpy as np


def pairwise_phi(vec_a: list, vec_b: list) -> float | None:
    """Phi coefficient between two error vectors (1.0=wrong, 0.0=correct, None=ungraded).

    Returns None if fewer than 2 overlapping graded questions, or either
    vector is constant over the overlap (correlation undefined).
    """
    pairs = [(a, b) for a, b in zip(vec_a, vec_b) if a is not None and b is not None]
    if len(pairs) < 2:
        return None
    a = np.array([p[0] for p in pairs], dtype=float)
    b = np.array([p[1] for p in pairs], dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def max_attainable_phi(vec_a: list, vec_b: list) -> float | None:
    """Largest phi two error vectors COULD have, given their error rates.

    Phi between two binary variables is bounded by their marginals: if config
    A is wrong on 12% of questions and B on 50%, then even when A's errors are
    a strict subset of B's (the most concordant arrangement possible) phi
    cannot exceed ~0.38. For marginals p <= q the bound is
    sqrt(p(1-q) / (q(1-p))), and it equals 1 only when p == q.

    This matters for interpreting the axis comparison. A pair can show low
    measured correlation either because its members genuinely fail on
    different questions, or merely because they have different accuracy and
    the coefficient cannot go any higher. Comparing the observed phi against
    this ceiling separates those two cases; a ratio near 1.0 means the pair
    is already as correlated as its marginals permit, so the low phi carries
    no evidence of real error decorrelation.

    Returns None when the bound is undefined (no overlap, or a config that is
    all-correct or all-wrong over the overlap).
    """
    pairs = [(a, b) for a, b in zip(vec_a, vec_b) if a is not None and b is not None]
    if len(pairs) < 2:
        return None
    n = len(pairs)
    p = sum(a for a, _ in pairs) / n
    q = sum(b for _, b in pairs) / n
    lo, hi = min(p, q), max(p, q)
    if lo in (0.0, 1.0) or hi in (0.0, 1.0):
        return None
    return float(((lo * (1 - hi)) / (hi * (1 - lo))) ** 0.5)


def _contingency(vec_a: list, vec_b: list) -> tuple[int, int, int, int] | None:
    """2x2 counts (both_wrong, a_wrong_only, b_wrong_only, both_correct) over
    the questions where both configs are graded. None if the overlap is empty."""
    pairs = [(a, b) for a, b in zip(vec_a, vec_b) if a is not None and b is not None]
    if not pairs:
        return None
    n11 = sum(1 for a, b in pairs if a == 1 and b == 1)
    n10 = sum(1 for a, b in pairs if a == 1 and b == 0)
    n01 = sum(1 for a, b in pairs if a == 0 and b == 1)
    n00 = sum(1 for a, b in pairs if a == 0 and b == 0)
    return n11, n10, n01, n00


def pairwise_yules_q(vec_a: list, vec_b: list) -> float | None:
    """Yule's Q, an association measure NOT bounded by the marginals.

    Q = (ad - bc) / (ad + bc) on the 2x2 error table. Unlike phi (see
    max_attainable_phi), Q can reach +1 whenever either off-diagonal cell is
    empty, regardless of how different the two configs' error rates are. That
    makes it the right robustness check for this project's central question:
    if an axis's low phi is only an artifact of unequal member accuracy, its Q
    will be high even though its phi is not.

    Deliberately does NOT apply the Haldane-Anscombe (+0.5) correction that is
    standard for raw odds ratios. Q is already the bounded transform of the
    odds ratio, so it handles an empty off-diagonal cell cleanly by reaching
    exactly +-1; adding 0.5 everywhere would shrink those values toward zero
    and destroy precisely the marginal-robustness this function exists to
    provide. (Verified by unit test: for two identical error vectors, the
    corrected version returned Q < 1, which is wrong.) The only genuinely
    degenerate case is ad + bc == 0, which returns None.

    Returns None if the overlap is empty or either config is constant over it
    (no association is defined).
    """
    counts = _contingency(vec_a, vec_b)
    if counts is None:
        return None
    n11, n10, n01, n00 = counts
    n = n11 + n10 + n01 + n00
    if n < 2:
        return None
    # Constant vectors carry no association information, same exclusion rule
    # phi uses, so the two measures are computed on the same set of pairs.
    if (n11 + n10) in (0, n) or (n11 + n01) in (0, n):
        return None
    a, b, c, d = n11, n10, n01, n00
    denom = a * d + b * c
    if denom == 0:
        return None
    return float((a * d - b * c) / denom)


def pairwise_phi_ratio(vec_a: list, vec_b: list) -> float | None:
    """Observed phi as a fraction of the largest phi these marginals allow.

    1.0 means the pair is already as correlated as its differing accuracy
    permits, so its phi carries no evidence of genuine error decorrelation.
    """
    rho = pairwise_phi(vec_a, vec_b)
    mx = max_attainable_phi(vec_a, vec_b)
    if rho is None or mx is None or mx == 0:
        return None
    return float(rho / mx)


def mean_pairwise_correlation(vectors: dict[str, list], measure=None) -> tuple[float, int, int]:
    """vectors: config_id -> error vector (same question order across configs).

    `measure` is the pairwise association function, defaulting to phi. Pass
    pairwise_yules_q to get the marginal-robust version used as a robustness
    check on the axis ranking.

    Returns (mean_over_defined_pairs, n_pairs_defined, n_pairs_undefined).
    The mean defaults to 0.0 (independence) if every pair is undefined --
    this is a deliberate, documented fallback, not silent data loss, since
    n_pairs_undefined is always returned alongside it.
    """
    measure = measure or pairwise_phi
    config_ids = list(vectors.keys())
    rhos = []
    n_undefined = 0
    for i in range(len(config_ids)):
        for j in range(i + 1, len(config_ids)):
            rho = measure(vectors[config_ids[i]], vectors[config_ids[j]])
            if rho is None:
                n_undefined += 1
            else:
                rhos.append(rho)
    mean_rho = float(np.mean(rhos)) if rhos else 0.0
    return mean_rho, len(rhos), n_undefined


def n_eff(n_configs: int, mean_rho: float) -> float:
    """Classical design-effect N_eff. Floored at n_configs if mean_rho < 0
    enough to make the denominator <= 0 (can't have more effective voters
    than actual voters in this framing -- negative correlation still helps,
    but the formula's literal blow-up past N is not meaningful here)."""
    denom = 1 + (n_configs - 1) * mean_rho
    if denom <= 0:
        return float(n_configs)
    return n_configs / denom


def bootstrap_neff(question_ids: list[str], vectors: dict[str, list], n_resamples: int = 1000, seed: int = 0, measure=None) -> dict:
    """Resamples questions with replacement `n_resamples` times, recomputing
    the mean association and N_eff on each resample, and reports the point
    estimate (on the real, unresampled data) plus a percentile 95% CI.

    `measure` selects the pairwise association function (default phi); pass
    pairwise_yules_q for the marginal-robust variant.
    """
    rng = random.Random(seed)
    n_q = len(question_ids)
    n_configs = len(vectors)

    point_mean_rho, n_pairs, n_undef = mean_pairwise_correlation(vectors, measure=measure)
    point_neff = n_eff(n_configs, point_mean_rho)

    if n_q == 0 or n_configs < 2:
        return {
            "n_configs": n_configs,
            "n_questions": n_q,
            "n_pairs_used": n_pairs,
            "n_pairs_undefined": n_undef,
            "mean_rho": point_mean_rho,
            "mean_rho_ci95": (None, None),
            "n_eff": point_neff,
            "n_eff_ci95": (None, None),
        }

    all_idx = list(range(n_q))
    boot_rhos = []
    boot_neffs = []
    for _ in range(n_resamples):
        sample_idx = [rng.choice(all_idx) for _ in range(n_q)]
        resampled_vectors = {cfg: [vec[i] for i in sample_idx] for cfg, vec in vectors.items()}
        rho, _, _ = mean_pairwise_correlation(resampled_vectors, measure=measure)
        boot_rhos.append(rho)
        boot_neffs.append(n_eff(n_configs, rho))

    boot_rhos.sort()
    boot_neffs.sort()
    lo_i = max(0, int(0.025 * n_resamples))
    hi_i = min(n_resamples - 1, int(0.975 * n_resamples))

    return {
        "n_configs": n_configs,
        "n_questions": n_q,
        "n_pairs_used": n_pairs,
        "n_pairs_undefined": n_undef,
        "mean_rho": point_mean_rho,
        "mean_rho_ci95": (boot_rhos[lo_i], boot_rhos[hi_i]),
        "n_eff": point_neff,
        "n_eff_ci95": (boot_neffs[lo_i], boot_neffs[hi_i]),
    }
