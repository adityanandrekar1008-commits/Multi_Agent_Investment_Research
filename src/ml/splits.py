"""Train/test splits that respect time."""

from __future__ import annotations


def date_boundary_split(df, train_share=0.70, date_column="Date"):
    """Chronological split that never puts one reporting date in both sets.

    Splitting by row position (the original approach) put some companies'
    2025-12-31 rows in training and others in test. Training rows from the
    same date carry labels about the same future quarter as the test rows,
    which leaks information across companies.

    Training = all dates before the cutoff; test = the cutoff date and later.
    The cutoff is the first date at which the earlier dates hold at least
    `train_share` of the rows.

    Returns (train, test, cutoff_date).
    """
    if not 0 < train_share < 1:
        raise ValueError("train_share must be between 0 and 1.")

    dates = sorted(df[date_column].dropna().unique())
    if len(dates) < 2:
        raise ValueError("At least two distinct dates are needed to split.")

    counts = df[date_column].value_counts()
    needed = train_share * len(df)
    cutoff = dates[-1]
    cumulative = 0
    for date in dates:
        if cumulative >= needed:
            cutoff = date
            break
        cumulative += counts[date]

    train = df[df[date_column] < cutoff].copy()
    test = df[df[date_column] >= cutoff].copy()
    return train, test, cutoff
