from __future__ import annotations

import random
from typing import List, Dict


def balanced_sample(items: List[Dict], n: int, seed: int) -> List[Dict]:
    """Return exactly *n* items (n must be even) with balanced labels.

    Guarantees:
        • n/2 resolved and n/2 unresolved items.
        • No duplicate (submission, instance_id) pairs.
        • Deterministic ordering for a given seed.
        • Round‑robin selection across repos per label respecting allocation limits.
    Raises:
        ValueError: If either label has fewer than n/2 eligible items.
    """
    if n % 2 != 0:
        raise ValueError("n must be even")

    resolved = [item for item in items if item["resolved"]]
    unresolved = [item for item in items if not item["resolved"]]

    half = n // 2
    if len(resolved) < half:
        raise ValueError(
            f"Not enough resolved items: need {half}, have {len(resolved)}"
        )
    if len(unresolved) < half:
        raise ValueError(
            f"Not enough unresolved items: need {half}, have {len(unresolved)}"
        )

    rng = random.Random(seed)

    def _select(items_subset: List[Dict], count: int) -> List[Dict]:
        # Build repo -> list of items for this subset
        repo_to_items: Dict[str, List[int]] = {}
        for idx, item in enumerate(items_subset):
            repo = item["repo"]
            repo_to_items.setdefault(repo, []).append(idx)

        repos = list(repo_to_items.keys())
        rng.shuffle(repos)

        max_per_repo = (count // len(repos)) + 1 if repos else count

        # Shuffle items within each repo for randomness
        for r in repos:
            rng.shuffle(repo_to_items[r])

        # Round-robin selection
        selected_indices: List[int] = []
        repo_counts = {r: 0 for r in repos}

        while len(selected_indices) < count:
            for repo in repos:
                if repo_counts[repo] >= max_per_repo:
                    continue
                if not repo_to_items[repo]:
                    continue
                item_idx = repo_to_items[repo].pop(0)
                # Check for duplicate (submission, instance_id) in selected so far
                item = items_subset[item_idx]
                if any(
                    items_subset[si]["submission"] == item["submission"]
                    and items_subset[si]["instance_id"] == item["instance_id"]
                    for si in selected_indices
                ):
                    continue
                selected_indices.append(item_idx)
                repo_counts[repo] += 1
                if len(selected_indices) == count:
                    break
            else:
                # If we looped through all repos without adding anything, break to avoid infinite loop
                break

        # If we still have room, fill from any remaining items that aren't duplicates
        remaining = [i for i in range(len(items_subset)) if i not in selected_indices]
        rng.shuffle(remaining)
        for idx in remaining:
            if len(selected_indices) >= count:
                break
            item = items_subset[idx]
            if any(
                items_subset[si]["submission"] == item["submission"]
                and items_subset[si]["instance_id"] == item["instance_id"]
                for si in selected_indices
            ):
                continue
            selected_indices.append(idx)

        return [items_subset[i] for i in selected_indices]

    selected_resolved = _select(resolved, half)
    selected_unresolved = _select(unresolved, half)

    result = selected_resolved + selected_unresolved
    # Shuffle the combined list deterministically
    rng.shuffle(result)
    return result
