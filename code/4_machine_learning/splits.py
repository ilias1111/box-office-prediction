"""Train/test and inner-CV split helpers.

Supports the existing random split and a chronological (temporal) alternative
that keeps all validation/test rows strictly after their training rows.
"""

from __future__ import annotations

from typing import Iterator, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split

SplitStrategy = Literal["random", "temporal"]


def build_time_key(frame: pd.DataFrame) -> pd.Series:
    """Ordinal time key from year/month columns (month defaults to 1)."""
    if "year" not in frame.columns:
        raise KeyError("Temporal splits require a 'year' column in X.")
    year = pd.to_numeric(frame["year"], errors="coerce")
    if year.isna().any():
        raise ValueError("Temporal splits require non-null year values.")
    if "month" in frame.columns:
        month = pd.to_numeric(frame["month"], errors="coerce").fillna(1).clip(1, 12)
    else:
        month = pd.Series(1, index=frame.index, dtype="float64")
    return (year.astype("int64") * 12 + month.astype("int64")).astype("int64")


def assert_temporal_order(
    time_key: Sequence[int] | np.ndarray | pd.Series,
    train_idx: Sequence[int] | np.ndarray,
    test_idx: Sequence[int] | np.ndarray,
    *,
    label: str,
) -> None:
    """Fail fast if any train row is not strictly before all test/val rows."""
    time_arr = np.asarray(time_key)
    train_times = time_arr[np.asarray(train_idx)]
    test_times = time_arr[np.asarray(test_idx)]
    if train_times.size == 0 or test_times.size == 0:
        raise ValueError(f"{label}: empty train or test indices.")
    if train_times.max() >= test_times.min():
        raise AssertionError(
            f"{label}: temporal leakage detected "
            f"(train_max={train_times.max()}, test_min={test_times.min()})."
        )


def _snap_cut_left(time_arr: np.ndarray, split_at: int) -> int:
    """Move a cut left until adjacent sides do not share a time key."""
    n = len(time_arr)
    while split_at > 0 and split_at < n and time_arr[split_at - 1] == time_arr[split_at]:
        split_at -= 1
    return split_at


def random_outer_split(X, y, *, test_size=0.2, random_state=42, stratify=None):
    try:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    except ValueError:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def temporal_outer_split(
    X: pd.DataFrame,
    y,
    *,
    test_size: float = 0.2,
    test_start_year: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.Series]:
    """Hold out the most recent rows (or year >= cutoff) as the test set."""
    time_key = build_time_key(X)
    order = np.argsort(time_key.to_numpy(), kind="mergesort")
    X_sorted = X.iloc[order].reset_index(drop=True)
    y_sorted = np.asarray(y)[order]
    time_sorted = time_key.iloc[order].reset_index(drop=True)

    n = len(X_sorted)
    time_arr = time_sorted.to_numpy()
    if test_start_year is not None:
        cutoff = int(test_start_year) * 12 + 1
        split_at = int(np.searchsorted(time_arr, cutoff, side="left"))
    else:
        split_at = int(np.floor(n * (1.0 - test_size)))

    # Keep a shared year-month entirely in the test set.
    split_at = _snap_cut_left(time_arr, split_at)
    if split_at <= 0 or split_at >= n:
        raise ValueError(
            f"Temporal outer split produced an empty side "
            f"(split_at={split_at}, n={n}, test_start_year={test_start_year})."
        )

    train_idx = np.arange(split_at)
    test_idx = np.arange(split_at, n)
    assert_temporal_order(time_sorted, train_idx, test_idx, label="outer holdout")
    return (
        X_sorted.iloc[train_idx],
        X_sorted.iloc[test_idx],
        y_sorted[train_idx],
        y_sorted[test_idx],
        time_sorted.iloc[train_idx].reset_index(drop=True),
    )


class TemporalExpandingSplit:
    """Expanding-window CV over rows already sorted by time.

    If ``time_key`` is provided, fold boundaries are snapped so a shared
    year-month cannot sit on both sides of a fold.
    """

    def __init__(
        self,
        n_splits: int = 3,
        gap: int = 0,
        time_key: Sequence[int] | np.ndarray | pd.Series | None = None,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2 for temporal CV.")
        if gap < 0:
            raise ValueError("gap must be >= 0.")
        self.n_splits = n_splits
        self.gap = gap
        self.time_key = None if time_key is None else np.asarray(time_key)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(
        self, X, y=None, groups=None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        if n < self.n_splits + 1:
            raise ValueError(
                f"Not enough samples ({n}) for {self.n_splits} temporal folds."
            )
        time_arr = self.time_key
        if time_arr is not None and len(time_arr) != n:
            raise ValueError("time_key length must match X.")

        tscv = TimeSeriesSplit(n_splits=self.n_splits, gap=self.gap)
        for fold_i, (train_idx, val_idx) in enumerate(tscv.split(np.arange(n))):
            if time_arr is None:
                yield train_idx, val_idx
                continue

            train_end = int(np.max(train_idx)) + 1
            val_start = int(np.min(val_idx))
            val_end = int(np.max(val_idx)) + 1
            while (
                train_end > 0
                and val_start < val_end
                and time_arr[train_end - 1] >= time_arr[val_start]
            ):
                train_end -= 1
            if train_end <= 0 or val_start >= val_end:
                raise ValueError(
                    f"inner CV fold {fold_i}: empty side after snapping time boundary."
                )
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)
            assert_temporal_order(
                time_arr, train_idx, val_idx, label=f"inner CV fold {fold_i}"
            )
            yield train_idx, val_idx


def make_temporal_cv(
    X_train: pd.DataFrame,
    *,
    n_splits: int,
    gap: int = 0,
    time_key: pd.Series | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialise temporal folds (with leakage asserts when a time key exists)."""
    if time_key is None:
        time_key = build_time_key(X_train)
    else:
        time_key = pd.Series(time_key).reset_index(drop=True)
        if len(time_key) != len(X_train):
            raise ValueError("time_key length must match X_train.")
    splitter = TemporalExpandingSplit(
        n_splits=n_splits, gap=gap, time_key=time_key.to_numpy()
    )
    return list(splitter.split(X_train))


def year_range(frame: pd.DataFrame) -> tuple[int | None, int | None]:
    if "year" not in frame.columns or frame.empty:
        return None, None
    years = pd.to_numeric(frame["year"], errors="coerce").dropna()
    if years.empty:
        return None, None
    return int(years.min()), int(years.max())
