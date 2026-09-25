"""Per-item prediction-probability calibration against finished sessions.

``GET /api/items/{item_id}/calibration`` helps item-bank maintainers spot
*systematic* deviations between the 2PL model's predicted probabilities and
the observed outcomes of one item.  The review is strictly read-only and
deliberately based on the ``p_correct``/``score`` values frozen into
``responses`` at answer time: the current ability estimate and the current
item parameters are never consulted, so later bank edits (changed difficulty,
discrimination or answer) cannot rewrite history.

Statistic
---------
Every response row written by a *finished* session for the target item is one
sample ``(p, u)`` with

* ``p`` -- the ``p_correct`` stored when the answer was scored (the 2PL
  prediction made *before* the answer was known);
* ``u`` -- the stored ``score`` (1 correct / 0 wrong).

Under a well-calibrated model the observed correct count behaves like a sum
of independent Bernoulli(``p``) trials, so the standardised deviation is::

    z = (sum(u) - sum(p)) / sqrt(sum(p * (1 - p)))

A positive ``z`` means the item is answered correctly *more* often than the
model predicted (the item is easier than its parameters suggest), a negative
``z`` the opposite.  ``|z| >= 1.96`` (the 5% two-sided level) flags the item
as miscalibrated.

``z`` and the deviation flag are reported as ``null`` unless there are at
least :data:`MIN_SAMPLES` samples **and** the denominator is positive (the
variance sum is zero only in degenerate cases, e.g. every stored probability
sat at 0 or 1).  Only ``finished`` sessions are counted; active/paused
sessions are excluded so in-progress quizzes cannot leak into the statistics.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass

# Below this many samples the normal approximation is not reported.
MIN_SAMPLES = 30

# Two-sided 5% critical value of the standard normal distribution.
Z_THRESHOLD = 1.96


@dataclass(frozen=True)
class ItemCalibration:
    """Aggregated calibration result for one item."""

    sample_count: int
    actual_correct_rate: float | None
    mean_predicted_probability: float | None
    z_score: float | None
    deviation_flag: bool | None


def calibrate_item(conn: sqlite3.Connection, item_id: str) -> ItemCalibration:
    """Compute the calibration payload for one item from finished sessions.

    Scores and predicted probabilities come straight from the ``responses``
    rows saved at answer time; neither the current theta of any session nor
    the current item parameters are ever consulted.
    """
    rows = conn.execute(
        """
        SELECT r.score     AS score,
               r.p_correct AS p_correct
          FROM responses r
          JOIN sessions s ON s.id = r.session_id
         WHERE s.status = 'finished' AND r.item_id = ?
        """,
        (item_id,),
    ).fetchall()
    n = len(rows)
    if n == 0:
        return ItemCalibration(
            sample_count=0,
            actual_correct_rate=None,
            mean_predicted_probability=None,
            z_score=None,
            deviation_flag=None,
        )

    score_sum = math.fsum(float(r["score"]) for r in rows)
    p_sum = math.fsum(float(r["p_correct"]) for r in rows)
    z_score: float | None = None
    deviation_flag: bool | None = None
    if n >= MIN_SAMPLES:
        variance = math.fsum(
            p * (1.0 - p) for p in (float(r["p_correct"]) for r in rows)
        )
        if variance > 0.0:
            z_score = (score_sum - p_sum) / math.sqrt(variance)
            deviation_flag = abs(z_score) >= Z_THRESHOLD
    return ItemCalibration(
        sample_count=n,
        actual_correct_rate=score_sum / n,
        mean_predicted_probability=p_sum / n,
        z_score=z_score,
        deviation_flag=deviation_flag,
    )
