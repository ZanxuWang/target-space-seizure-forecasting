"""loo_splits.py

Leave-one-out enumeration over the RT mouse cohort.

Each fold holds out one mouse as the test subject and uses the remaining
mice for training (with an internal validation split drawn from them).
"""

from __future__ import annotations

from .config import RT_MICE


def loo_folds(mice: list = None) -> list:
    """Return a list of (held_out_mouse, train_mice_list) tuples."""
    mice = list(mice) if mice is not None else list(RT_MICE)
    folds = []
    for held_out in mice:
        train = [m for m in mice if m != held_out]
        folds.append((held_out, train))
    return folds


def fold_name(held_out: str) -> str:
    return f"fold_{held_out}"
