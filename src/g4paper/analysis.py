"""Statistical analysis utilities for g4paper.

Provides AUROC, Brier score, per‑judge metrics, ensemble aggregation, and agreement curves.
"""

from __future__ import annotations

from typing import List, Dict, Tuple


def auroc(scores: List[float], labels: List[int]) -> float | None:
    """Compute AUROC via the Mann‑Whitney U statistic.

    Ties contribute 0.5 to the count. Returns ``None`` if either class is empty.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    # Separate positive and negative scores
    pos_scores = [s for s, l in zip(scores, labels) if l == 1]
    neg_scores = [s for s, l in zip(scores, labels) if l == 0]

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    if n_pos == 0 or n_neg == 0:
        return None

    # Count concordant pairs, using 0.5 for ties
    concordant = 0.0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                concordant += 1.0
            elif ps == ns:
                concordant += 0.5
            # else contributes 0

    return concordant / (n_pos * n_neg)


def brier(probs: List[float], labels: List[int]) -> float:
    """Return the mean squared error between probability predictions and binary labels."""
    if len(probs) != len(labels):
        raise ValueError("probs and labels must have the same length")
    if not probs:
        return 0.0
    total = 0.0
    for p, y in zip(probs, labels):
        diff = p - float(y)
        total += diff * diff
    return total / len(probs)


def per_judge(
    rows: List[Dict],
    labels: Dict[Tuple[str, int], int],
) -> Dict[str, Dict]:
    """Compute metrics per judge, ignoring rows where ``p`` is ``None``.

    Returns a dict mapping each judge to a record containing ``auroc``, ``brier``,
    ``n`` (number of evaluated items) and ``coverage`` (fraction with non‑null ``p``).
    """
    # Group rows by judge
    judge_data: Dict[str, Dict] = {}
    for row in rows:
        judge = row.get("judge")
        if judge is None:
            continue
        entry = judge_data.setdefault(
            judge,
            {
                "scores": [],
                "true": [],
                "total": 0,
                "non_null": 0,
            },
        )
        entry["total"] += 1
        p = row.get("p")
        if p is None:
            continue
        entry["non_null"] += 1
        submission = row.get("submission")
        instance_id = row.get("instance_id")
        label = labels.get((submission, instance_id))
        # If label is missing we skip the sample; this situation should not arise in tests
        if label is None:
            continue
        entry["scores"].append(float(p))
        entry["true"].append(int(label))

    result: Dict[str, Dict] = {}
    for judge, data in judge_data.items():
        n = data["non_null"]
        coverage = data["non_null"] / data["total"] if data["total"] > 0 else 0.0
        au = None
        if n > 0:
            au = auroc(data["scores"], data["true"])
        brier_score = brier(data["scores"], data["true"]) if n > 0 else 0.0
        result[judge] = {
            "auroc": au,
            "brier": brier_score,
            "n": n,
            "coverage": coverage,
        }
    return result


def ensemble(
    rows: List[Dict],
    labels: Dict[Tuple[str, int], int],
    judges: List[str],
) -> Dict[str, float]:
    """Compute AUROC and Brier for the mean‑p ensemble over items with complete judge coverage.

    Returns a mapping with keys ``"auroc"`` and ``"brier"``.
    """
    # Map each item to its per‑judge probabilities
    item_to_probs: Dict[Tuple[str, int], Dict[str, float]] = {}
    for row in rows:
        p = row.get("p")
        if p is None:
            continue
        judge = row.get("judge")
        if judge not in judges:
            continue
        key = (row.get("submission"), row.get("instance_id"))
        probs = item_to_probs.setdefault(key, {})
        probs[judge] = float(p)

    # Keep only items for which every judge supplied a non‑null probability
    mean_scores: List[float] = []
    true_labels: List[int] = []
    required_set = set(judges)
    for key, probs in item_to_probs.items():
        if set(probs.keys()) == required_set:
            mean_scores.append(sum(probs.values()) / len(judges))
            label = labels.get(key)
            if label is not None:
                true_labels.append(int(label))

    # Compute metrics; if no items, return NaN‑like zeros
    if not mean_scores:
        return {"auroc": None, "brier": 0.0}
    return {
        "auroc": auroc(mean_scores, true_labels),
        "brier": brier(mean_scores, true_labels),
    }


def agreement_curve(
    rows: List[Dict],
    labels: Dict[Tuple[str, int], int],
    judges: List[str],
    threshold: float = 0.5,
) -> List[Dict[str, float]]:
    """Generate an agreement curve.

    For each k from 1 to ``len(judges)`` computes the coverage (fraction of items where at
    least k judges give ``p >= threshold``) and precision (fraction of those accepted items
    that are resolved). Returns a list of dicts ``{"k": k, "coverage": ..., "precision": ...}``.
    """
    # Build a mapping from item to list of probabilities per judge (None if missing)
    item_probs: Dict[Tuple[str, int], List[float | None]] = {}
    for row in rows:
        key = (row.get("submission"), row.get("instance_id"))
        judge = row.get("judge")
        if judge not in judges:
            continue
        prob = row.get("p")
        probs = item_probs.setdefault(key, [None] * len(judges))
        try:
            idx = judges.index(judge)
        except ValueError:
            continue
        probs[idx] = prob

    total_items = len(item_probs)
    if total_items == 0:
        # Return empty list if there are no items
        return []

    curve: List[Dict[str, float]] = []
    for k in range(1, len(judges) + 1):
        accepted = 0
        resolved = 0
        for key, probs in item_probs.items():
            # Count judges with p >= threshold (ignoring None)
            count = sum(1 for p in probs if p is not None and p >= threshold)
            if count >= k:
                accepted += 1
                if labels.get(key, 0) == 1:
                    resolved += 1
        coverage = accepted / total_items if total_items > 0 else 0.0
        precision = resolved / accepted if accepted > 0 else 0.0
        curve.append({"k": k, "coverage": coverage, "precision": precision})
    return curve
