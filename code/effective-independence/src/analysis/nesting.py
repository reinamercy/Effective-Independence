"""Error-nesting analysis: are two configurations' mistakes *complementary*
or merely *ordered*?

Motivation. The design-effect statistic N_eff (neff.py) summarizes how
correlated two members' errors are, but it cannot distinguish two very
different situations that matter to an ensemble builder:

  - Overlapping errors: each member gets some questions wrong that the other
    gets right. A majority vote has something to work with.
  - Nested errors: one member's mistakes are a strict subset of the other's.
    The pair is ordered by capability, and the weaker member contributes no
    question the stronger one has not already failed. A majority vote cannot
    recover anything, because there is nothing the weaker member knows that
    the stronger one does not.

Nesting is the structural reason a correlation coefficient can look moderate
while the ensemble is worth nothing: phi is additionally capped by the
members' differing error rates (see neff.max_attainable_phi), so a perfectly
nested pair with unequal accuracy reports a middling phi rather than the 1.0
its actual dependence deserves. This module measures the structure directly
instead of inferring it from a coefficient.

Pair types are derived from the model_id recorded in error_matrix.csv rather
than hardcoded, so the analysis stays correct when the model set is swapped
(which has happened twice on this project already).
"""
from __future__ import annotations

import itertools
from collections import defaultdict

from src.analysis.neff import _contingency

PAIR_TYPES = ("same exact model", "same lab, different model", "different lab")


def lab_of(model_id: str) -> str:
    """Coarse provider/lab bucket for a model_id string.

    Only needs to separate the labs actually present in this project's model
    set; unknown ids fall back to their own name, which makes them their own
    lab rather than silently grouping them with something else.
    """
    m = model_id.lower()
    if m.startswith("gemini") or m.startswith("google/"):
        return "Google"
    if m.startswith("z-ai/") or "glm" in m:
        return "Zhipu AI"
    if m.startswith("nvidia/") or "nemotron" in m:
        return "NVIDIA"
    if m.startswith("openai/") or m.startswith("gpt"):
        return "OpenAI"
    if m.startswith("anthropic/") or m.startswith("claude"):
        return "Anthropic"
    return model_id


def classify_pair(model_a: str, model_b: str) -> str:
    if model_a == model_b:
        return "same exact model"
    if lab_of(model_a) == lab_of(model_b):
        return "same lab, different model"
    return "different lab"


def config_model_map(rows: list[dict]) -> dict[str, str]:
    """config_id -> model_id, read straight from the graded error matrix."""
    out: dict[str, str] = {}
    for r in rows:
        mid = r.get("model_id")
        if mid:
            out.setdefault(r["config_id"], mid)
    return out


def pair_nesting(vec_a: list, vec_b: list) -> dict | None:
    """Classify one pair's error structure.

    Returns None when the pair carries no information: empty overlap, or
    either member has zero errors over the overlap (a member that never fails
    is trivially nested inside anything and would inflate the nesting rate).
    """
    counts = _contingency(vec_a, vec_b)
    if counts is None:
        return None
    n11, n10, n01, n00 = counts
    errs_a, errs_b = n11 + n10, n11 + n01
    if errs_a == 0 or errs_b == 0:
        return None
    nested = (n10 == 0) or (n01 == 0)
    return {
        "both_wrong": n11,
        "a_only": n10,
        "b_only": n01,
        "both_correct": n00,
        "n_overlap": n11 + n10 + n01 + n00,
        "nested": nested,
        # How many questions only the weaker member fails: the ensemble's
        # entire potential upside from adding it.
        "complementary_errors": min(n10, n01),
    }


def nesting_by_pair_type(
    config_ids: list[str], vectors: dict[str, list], model_map: dict[str, str]
) -> dict[str, dict]:
    """Nesting rate for every pair, bucketed by how related the two models are.

    The expected pattern, if ensembling from one base model buys nothing, is a
    monotone decrease in nesting as the pair becomes more genuinely different.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for a, b in itertools.combinations(config_ids, 2):
        if a not in vectors or b not in vectors:
            continue
        info = pair_nesting(vectors[a], vectors[b])
        if info is None:
            continue
        ma, mb = model_map.get(a), model_map.get(b)
        if ma is None or mb is None:
            continue
        info = {**info, "config_a": a, "config_b": b}
        buckets[classify_pair(ma, mb)].append(info)

    out: dict[str, dict] = {}
    for kind, items in buckets.items():
        n_nested = sum(1 for i in items if i["nested"])
        out[kind] = {
            "n_pairs": len(items),
            "n_nested": n_nested,
            "rate": n_nested / len(items) if items else 0.0,
            "pairs": items,
        }
    return out


def overall_nesting(by_type: dict[str, dict]) -> tuple[int, int, float]:
    n_nested = sum(v["n_nested"] for v in by_type.values())
    n_pairs = sum(v["n_pairs"] for v in by_type.values())
    return n_nested, n_pairs, (n_nested / n_pairs if n_pairs else 0.0)


def permutation_null(
    config_ids: list[str],
    vectors: dict[str, list],
    model_map: dict[str, str],
    n_permutations: int = 2000,
    seed: int = 0,
) -> dict:
    """How often would we see this much nesting by chance alone?

    This is the control the headline nesting result needs. With only six or
    seven errors per configuration, an empty off-diagonal cell could plausibly
    arise by luck, so the raw rate means nothing without a baseline.

    The null holds each configuration's error COUNT and its set of graded
    questions fixed, and shuffles only *which* of its graded questions are
    wrong. That destroys any genuine dependence between configurations while
    preserving the marginals and the coverage pattern exactly, so the
    resulting distribution isolates the contribution of real shared failure
    modes from the contribution of sparse tables.

    Returns observed rates, null means, and one-sided p-values, for all pairs
    and for same-model pairs separately.
    """
    import random

    rng = random.Random(seed)
    all_pairs = list(itertools.combinations(config_ids, 2))
    same_pairs = [
        (a, b) for a, b in all_pairs
        if model_map.get(a) is not None and model_map.get(a) == model_map.get(b)
    ]

    def rate(vecs: dict[str, list], pairs: list) -> tuple[int, int]:
        n = t = 0
        for a, b in pairs:
            if a not in vecs or b not in vecs:
                continue
            info = pair_nesting(vecs[a], vecs[b])
            if info is None:
                continue
            t += 1
            n += int(info["nested"])
        return n, t

    def shuffle_within_config(vecs: dict[str, list]) -> dict[str, list]:
        out = {}
        for c, vec in vecs.items():
            idx = [i for i, x in enumerate(vec) if x is not None]
            vals = [vec[i] for i in idx]
            rng.shuffle(vals)
            nv = list(vec)
            for i, val in zip(idx, vals):
                nv[i] = val
            out[c] = nv
        return out

    obs_all_n, obs_all_t = rate(vectors, all_pairs)
    obs_same_n, obs_same_t = rate(vectors, same_pairs)
    obs_all = obs_all_n / obs_all_t if obs_all_t else 0.0
    obs_same = obs_same_n / obs_same_t if obs_same_t else 0.0

    null_all, null_same = [], []
    for _ in range(n_permutations):
        sv = shuffle_within_config(vectors)
        n, t = rate(sv, all_pairs)
        null_all.append(n / t if t else 0.0)
        n, t = rate(sv, same_pairs)
        null_same.append(n / t if t else 0.0)

    null_all.sort()
    null_same.sort()

    def pct(sorted_vals: list[float], q: float) -> float:
        if not sorted_vals:
            return 0.0
        i = min(len(sorted_vals) - 1, max(0, int(q * len(sorted_vals))))
        return sorted_vals[i]

    return {
        "n_permutations": n_permutations,
        "observed_all": obs_all,
        "observed_all_counts": (obs_all_n, obs_all_t),
        "observed_same_model": obs_same,
        "observed_same_model_counts": (obs_same_n, obs_same_t),
        "null_all_mean": sum(null_all) / len(null_all),
        "null_all_ci": (pct(null_all, 0.025), pct(null_all, 0.975)),
        "null_same_mean": sum(null_same) / len(null_same),
        "null_same_ci": (pct(null_same, 0.025), pct(null_same, 0.975)),
        "p_all": sum(1 for x in null_all if x >= obs_all) / len(null_all),
        "p_same_model": sum(1 for x in null_same if x >= obs_same) / len(null_same),
    }


def bootstrap_nesting_rate(
    config_ids: list[str],
    vectors: dict[str, list],
    n_resamples: int = 1000,
    seed: int = 0,
    model_map: dict[str, str] | None = None,
    same_model_only: bool = False,
) -> tuple[float, float]:
    """Percentile 95% CI for the nesting rate, resampling questions.

    Reported so the rate is not mistaken for an exact population value at
    $N=30$ questions.
    """
    import random

    rng = random.Random(seed)
    pairs = list(itertools.combinations(config_ids, 2))
    if same_model_only and model_map:
        pairs = [(a, b) for a, b in pairs if model_map.get(a) == model_map.get(b)]
    n_q = len(next(iter(vectors.values()))) if vectors else 0
    if n_q == 0:
        return (0.0, 0.0)

    rates = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n_q) for _ in range(n_q)]
        rs = {c: [vec[i] for i in idx] for c, vec in vectors.items()}
        n = t = 0
        for a, b in pairs:
            info = pair_nesting(rs[a], rs[b])
            if info is None:
                continue
            t += 1
            n += int(info["nested"])
        if t:
            rates.append(n / t)
    if not rates:
        return (0.0, 0.0)
    rates.sort()
    lo = rates[max(0, int(0.025 * len(rates)))]
    hi = rates[min(len(rates) - 1, int(0.975 * len(rates)))]
    return (lo, hi)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation here because several buckets are
    small and at least one sits at exactly 1.0, where the normal interval
    degenerates to zero width and would misleadingly imply certainty.
    """
    import math

    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def trend_test(by_type: dict[str, dict]) -> dict:
    """Cochran-Armitage test for a monotone trend in nesting rate across the
    three pair types, scored 0/1/2 by increasing dissimilarity.

    The paper claims the nesting rate "falls monotonically" as members become
    genuinely more different. Three descending percentages are not evidence of
    a trend on their own, so this supplies the test behind that word.
    Returns the z statistic and a two-sided p-value; z < 0 means the rate
    decreases as members become more different.
    """
    import math

    scores, ks, ns = [], [], []
    for i, kind in enumerate(PAIR_TYPES):
        v = by_type.get(kind)
        if not v or v["n_pairs"] == 0:
            continue
        scores.append(i)
        ks.append(v["n_nested"])
        ns.append(v["n_pairs"])
    if len(scores) < 3:
        return {"z": float("nan"), "p_two_sided": float("nan"), "n_groups": len(scores)}

    total_n = sum(ns)
    p_bar = sum(ks) / total_n
    x_bar = sum(n * x for x, n in zip(scores, ns)) / total_n
    numerator = sum(n * x * (k / n - p_bar) for x, k, n in zip(scores, ks, ns))
    variance = p_bar * (1 - p_bar) * sum(n * (x - x_bar) ** 2 for x, n in zip(scores, ns))
    if variance <= 0:
        return {"z": float("nan"), "p_two_sided": float("nan"), "n_groups": len(scores)}
    z = numerator / math.sqrt(variance)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {"z": z, "p_two_sided": p, "n_groups": len(scores)}


def format_nesting_table(by_type: dict[str, dict]) -> str:
    lines = [f"{'pair type':28s} {'nested':>7s} {'pairs':>6s} {'rate':>7s}"]
    for kind in PAIR_TYPES:
        v = by_type.get(kind)
        if not v:
            continue
        lines.append(f"{kind:28s} {v['n_nested']:7d} {v['n_pairs']:6d} {v['rate']*100:6.0f}%")
    n_nested, n_pairs, rate = overall_nesting(by_type)
    lines.append(f"{'ALL PAIRS':28s} {n_nested:7d} {n_pairs:6d} {rate*100:6.0f}%")
    return "\n".join(lines)
