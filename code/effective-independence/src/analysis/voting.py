"""Does majority voting actually help?

Everything else in this project measures ensemble independence through a
statistic (N_eff) or a structure (nesting). Neither is the quantity a
practitioner cares about. They care whether aggregating several samples
produces a more accurate answer than just using one model.

This module answers that directly, from the cached error matrix and with no
new API calls:

  - majority-vote accuracy for an ensemble,
  - the accuracy of its best and mean individual member,
  - "recoverable" questions, where the majority is wrong but at least one
    member is right (the upside a perfect aggregator could have captured),
  - "unrecoverable" questions, where no member is right at all, which no
    aggregation rule of any kind can fix.

The nesting result predicts the outcome: if one member's errors are a subset
of another's, the ensemble is ordered by capability, and a vote cannot beat
the strongest member. Reporting the vote directly turns that prediction into
a measurement.

Ties (an even split on a binary correct/incorrect vote) are counted as
incorrect, which is the conservative choice for our thesis in the sense that
it cannot manufacture an apparent majority-vote win.
"""
from __future__ import annotations


def _graded_questions(question_ids: list[str], config_ids: list[str], matrix: dict) -> list[str]:
    """Questions where every listed config has a graded result.

    Voting is only meaningful on complete rows: a question where half the
    ensemble is ungraded would otherwise be scored against a smaller, and
    differently-composed, panel than the rest.
    """
    return [
        q for q in question_ids
        if all(matrix.get(q, {}).get(c) is not None for c in config_ids)
    ]


def majority_vote_report(
    question_ids: list[str], config_ids: list[str], matrix: dict
) -> dict | None:
    """Compare majority voting against the individual members it aggregates.

    `matrix[question_id][config_id]` is True when that config answered
    correctly. Returns None when no question has the full panel graded.
    """
    qs = _graded_questions(question_ids, config_ids, matrix)
    if not qs or not config_ids:
        return None

    k = len(config_ids)
    n_majority_correct = 0
    n_recoverable = 0
    n_unrecoverable = 0

    for q in qs:
        votes = [bool(matrix[q][c]) for c in config_ids]
        n_correct = sum(votes)
        majority_correct = n_correct > k / 2
        if majority_correct:
            n_majority_correct += 1
        elif any(votes):
            # The answer was present in the ensemble but the vote discarded it.
            n_recoverable += 1
        else:
            n_unrecoverable += 1

    per_config = {
        c: sum(1 for q in qs if matrix[q][c]) / len(qs) for c in config_ids
    }
    best_config = max(per_config, key=per_config.get)

    return {
        "n_questions": len(qs),
        "k": k,
        "majority_accuracy": n_majority_correct / len(qs),
        "best_member_accuracy": per_config[best_config],
        "best_member": best_config,
        "mean_member_accuracy": sum(per_config.values()) / len(per_config),
        "n_recoverable": n_recoverable,
        "n_unrecoverable": n_unrecoverable,
        # Positive means voting beat the strongest individual member. The
        # central claim of the paper predicts this is <= 0.
        "vote_minus_best": (n_majority_correct / len(qs)) - per_config[best_config],
        "per_config_accuracy": per_config,
    }


def format_voting_table(reports: dict[str, dict | None]) -> str:
    lines = [
        f"{'ensemble':22s} {'n_q':>4s} {'k':>3s} {'majority':>9s} {'best':>8s} "
        f"{'mean':>8s} {'vote-best':>10s} {'recov':>6s} {'unrec':>6s}"
    ]
    for name, r in reports.items():
        if r is None:
            lines.append(f"{name:22s}    -   -         -        -        -          -      -      -")
            continue
        lines.append(
            f"{name:22s} {r['n_questions']:4d} {r['k']:3d} "
            f"{r['majority_accuracy']*100:8.1f}% {r['best_member_accuracy']*100:7.1f}% "
            f"{r['mean_member_accuracy']*100:7.1f}% {r['vote_minus_best']*100:+9.1f}pp "
            f"{r['n_recoverable']:6d} {r['n_unrecoverable']:6d}"
        )
    lines.append(
        "\n'vote-best' <= 0 everywhere means majority voting never beat simply using the "
        "strongest single member, which is what the nesting structure predicts.\n"
        "'recov' = majority wrong but some member right (upside the vote threw away).\n"
        "'unrec' = no member right; no aggregation rule could have helped."
    )
    return "\n".join(lines)
