"""
URL-stratified splitting utilities for pattern-train embeddings.

Provides holdout and K-fold cross-validation splits that group clips by
their source URL so that the same song never appears in both training and
evaluation sets.  When ``tastes_ids`` (genre) is supplied the splits are
also stratified by genre for balanced representation.

Usage from training / evaluation scripts:

    from split_utils import load_embeddings, holdout_split, kfold_cv

    data = load_embeddings()                       # single .pt file
    train_idx, val_idx, test_idx = holdout_split(  # numpy index arrays
        data["urls"], data["labels"],
        tastes_ids=data.get("tastes_ids"))

    for train_idx, val_idx in kfold_cv(
            data["urls"], data["labels"],
            tastes_ids=data.get("tastes_ids"), n_folds=5):
        ...
"""

from pathlib import Path

import numpy as np
import torch
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
    MultilabelStratifiedShuffleSplit,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR        = REPO_ROOT / "data"
EMBEDDINGS_PATH = DATA_DIR / "embeddings_all.pt"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_embeddings(path: Path | str | None = None, mmap: bool = True) -> dict:
    """Load the unified embeddings file.

    Returns dict with keys: ``frames``, ``labels``, ``urls``, ``ids``.
    """
    path = Path(path) if path else EMBEDDINGS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found – run extract_embeddings.py first")
    try:
        return torch.load(path, weights_only=False, mmap=mmap)
    except TypeError:
        return torch.load(path, weights_only=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aggregate_labels_per_url(
    urls: np.ndarray,
    labels: np.ndarray,
    tastes_ids: np.ndarray | None = None,
):
    """Return (unique_urls, url_strat_labels).

    ``url_strat_labels[i]`` is the element-wise max of all clip labels
    sharing ``unique_urls[i]``.  When *tastes_ids* is provided, the
    genre is one-hot encoded and appended so that the stratifier
    balances both pattern labels and genre.
    """
    unique_urls = np.unique(urls)
    url_to_idx  = {u: i for i, u in enumerate(unique_urls)}
    url_labels  = np.zeros((len(unique_urls), labels.shape[1]), dtype=labels.dtype)
    for i, url in enumerate(urls):
        url_labels[url_to_idx[url]] = np.maximum(url_labels[url_to_idx[url]], labels[i])

    # Append one-hot genre columns so the stratifier considers genre too
    if tastes_ids is not None:
        url_tastes = np.full(len(unique_urls), -1, dtype=int)
        for i, url in enumerate(urls):
            url_tastes[url_to_idx[url]] = tastes_ids[i]
        # One-hot encode; ignore unknown (-1) values
        genre_ids = np.unique(url_tastes[url_tastes >= 0])
        genre_to_col = {g: j for j, g in enumerate(genre_ids)}
        ohe = np.zeros((len(unique_urls), len(genre_ids)), dtype=url_labels.dtype)
        for i, g in enumerate(url_tastes):
            if g in genre_to_col:
                ohe[i, genre_to_col[g]] = 1
        url_labels = np.hstack([url_labels, ohe])

    return unique_urls, url_labels


def _url_indices(urls: np.ndarray, url_set: set) -> np.ndarray:
    """Return clip-level indices whose URL belongs to *url_set*."""
    mask = np.array([u in url_set for u in urls])
    return np.where(mask)[0]


# ---------------------------------------------------------------------------
# Holdout split
# ---------------------------------------------------------------------------

def holdout_split(
    urls,
    labels,
    val_ratio: float  = 0.10,
    test_ratio: float = 0.10,
    seed: int         = 42,
    tastes_ids         = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """URL-stratified holdout split.

    Parameters
    ----------
    urls : list or array of URL strings (one per clip).
    labels : (N, C) tensor or array of multilabel targets.
    val_ratio, test_ratio : fractions of *songs* for val / test.
    seed : random state.
    tastes_ids : optional list/array of genre ids (one per clip).
        When provided the split is also stratified by genre.

    Returns
    -------
    (train_indices, val_indices, test_indices) — numpy int arrays.
    """
    urls = np.asarray(urls)
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    tastes_arr = np.asarray(tastes_ids) if tastes_ids is not None else None

    unique_urls, url_labels = _aggregate_labels_per_url(urls, labels, tastes_arr)

    # Split 1: train+val vs test
    msss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed)
    tv_idx, test_idx = next(msss1.split(unique_urls, url_labels))

    # Split 2: train vs val (from the train+val pool)
    tv_urls   = unique_urls[tv_idx]
    tv_labels = url_labels[tv_idx]
    val_adj   = val_ratio / (1.0 - test_ratio)

    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_adj, random_state=seed)
    train_sub, val_sub = next(msss2.split(tv_urls, tv_labels))

    train_urls = set(tv_urls[train_sub])
    val_urls   = set(tv_urls[val_sub])
    test_urls  = set(unique_urls[test_idx])

    return (
        _url_indices(urls, train_urls),
        _url_indices(urls, val_urls),
        _url_indices(urls, test_urls),
    )


# ---------------------------------------------------------------------------
# K-fold cross-validation
# ---------------------------------------------------------------------------

def kfold_cv(
    urls,
    labels,
    n_folds: int = 5,
    seed: int    = 42,
    tastes_ids    = None,
):
    """URL-stratified K-fold cross-validation.

    Yields ``(train_indices, val_indices)`` for each fold.
    When *tastes_ids* is provided the folds are also stratified by genre.
    """
    urls = np.asarray(urls)
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    tastes_arr = np.asarray(tastes_ids) if tastes_ids is not None else None

    unique_urls, url_labels = _aggregate_labels_per_url(urls, labels, tastes_arr)

    mskf = MultilabelStratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=seed)

    for train_url_idx, val_url_idx in mskf.split(unique_urls, url_labels):
        train_urls = set(unique_urls[train_url_idx])
        val_urls   = set(unique_urls[val_url_idx])
        yield _url_indices(urls, train_urls), _url_indices(urls, val_urls)
