"""Event scoring engine: filter DSL, base/multiplier resolution, rarity tracker.

This module implements a small, regex-based DSL used by MusiCue grammars to
decide whether an event matches a rule and how strongly it should fire.

Filter DSL (`evaluate_filter`)
------------------------------
The filter expression is a single string evaluated against an event dict. The
following forms are supported (any unrecognized form returns ``True`` so that
unknown filters degrade gracefully):

- ``None`` -> always matches (returns ``True``).
- ``field == 'value'`` -> string equality on ``event[field]``.
- ``field == true`` / ``field == false`` -> boolean check on ``event[field]``.
- ``field > value`` -> numeric strict-greater-than on ``event[field]``.
- ``any_label('label', min_score=X)`` -> ``True`` when any entry in
  ``event['labels']`` has matching ``label`` and ``score >= X``.
- ``near_downbeat(seconds)`` -> reads the precomputed ``event['near_downbeat']``
  flag. The compiler is responsible for setting that flag using the actual
  proximity value before invoking the filter (this function is a stub that
  trusts the boolean).

Scoring (`compute_score`)
-------------------------
A score config is a dict with a ``base`` and an optional list of
``multiplier`` rules::

    {"base": "strength", "multiplier": [{"when": "is_downbeat == true", "factor": 1.5}]}

The ``base`` may be:

- a numeric literal (``int``/``float``),
- a field name present on the event (e.g. ``"strength"``),
- ``"label_score('<label>')"`` -> the score of the matching label, or ``0.0``,
- ``"max(energy_curve)"`` -> max of ``event['energy_curve']['values']``.

Each multiplier with a matching ``when`` filter contributes its ``factor``.
The final score is::

    base * product(matching_factors) * timescale_weight * rarity_bonus

Rarity (`RarityTracker`)
------------------------
Maintains a rolling window of emission timestamps. ``bonus(t)`` returns
``exp(-count / decay)`` where ``count`` is the number of recorded events in
the last ``window_sec`` seconds. Recently-fired events therefore receive a
smaller bonus, encouraging the compiler to spread emissions over time.
"""

from __future__ import annotations

import math
import re
from typing import Any

# Precompiled regexes for the DSL forms.
_RE_ANY_LABEL = re.compile(r"any_label\('([^']+)',\s*min_score=([\d.]+)\)")
_RE_NEAR_DOWNBEAT = re.compile(r"near_downbeat\(([\d.]+)\)")
_RE_FIELD_EQ_STR = re.compile(r"(\w+)\s*==\s*'([^']*)'")
_RE_FIELD_EQ_BOOL = re.compile(r"(\w+)\s*==\s*(true|false)")
_RE_FIELD_GT = re.compile(r"(\w+)\s*>\s*([\d.]+)")
_RE_LABEL_SCORE = re.compile(r"label_score\('([^']+)'\)")


def evaluate_filter(expr: str | None, event: dict) -> bool:
    """Evaluate a filter expression against an event dict.

    Returns ``True`` when the event matches the filter. Unknown expressions
    return ``True`` so the rule fires; this is intentional so grammars can
    use forward-compatible DSL features without breaking older runtimes.
    """
    if expr is None:
        return True

    # any_label('label', min_score=X)
    m = _RE_ANY_LABEL.match(expr)
    if m:
        target_label, min_score = m.group(1), float(m.group(2))
        return any(
            lbl.get("label") == target_label and float(lbl.get("score", 0.0)) >= min_score
            for lbl in event.get("labels", [])
        )

    # near_downbeat(seconds) -- relies on the compiler having set the flag.
    m = _RE_NEAR_DOWNBEAT.match(expr)
    if m:
        return bool(event.get("near_downbeat", False))

    # field == 'value'
    m = _RE_FIELD_EQ_STR.match(expr)
    if m:
        field, value = m.group(1), m.group(2)
        return str(event.get(field, "")) == value

    # field == true/false
    m = _RE_FIELD_EQ_BOOL.match(expr)
    if m:
        field, want_true = m.group(1), m.group(2) == "true"
        return bool(event.get(field, False)) == want_true

    # field > value
    m = _RE_FIELD_GT.match(expr)
    if m:
        field, value = m.group(1), float(m.group(2))
        return float(event.get(field, 0)) > value

    # Unknown expression: pass through so unknown DSL doesn't silently drop events.
    return True


def _resolve_base(base: Any, event: dict) -> float:
    """Resolve a `base` spec to a numeric value against the event."""
    if isinstance(base, (int, float)):
        return float(base)
    if isinstance(base, str):
        # Direct field reference (e.g. "strength").
        if base in event:
            try:
                return float(event[base])
            except (TypeError, ValueError):
                return 0.0
        # label_score('<label>')
        m = _RE_LABEL_SCORE.match(base)
        if m:
            target = m.group(1)
            for lbl in event.get("labels", []):
                if lbl.get("label") == target:
                    return float(lbl.get("score", 0.0))
            return 0.0
        # max(energy_curve)
        if base == "max(energy_curve)":
            curve = event.get("energy_curve", {}) or {}
            vals = curve.get("values", [0.0])
            return float(max(vals)) if vals else 0.0
    return 1.0


def compute_score(
    score_cfg: dict,
    event: dict,
    timescale_weight: float = 1.0,
    rarity_bonus: float = 1.0,
) -> float:
    """Compute the final score for an event under a grammar rule.

    ``score_cfg`` is the rule's score configuration (``base`` + optional
    ``multiplier`` list). ``timescale_weight`` and ``rarity_bonus`` are
    supplied by the compiler.
    """
    base = _resolve_base(score_cfg.get("base", 1.0), event)
    multiplier = 1.0
    for rule in score_cfg.get("multiplier", []) or []:
        if evaluate_filter(rule.get("when"), event):
            multiplier *= float(rule.get("factor", 1.0))
    return base * multiplier * timescale_weight * rarity_bonus


class RarityTracker:
    """Rolling-window rarity bonus.

    Records emission timestamps and returns a multiplicative bonus that
    decays as ``exp(-count / decay)`` where ``count`` is the number of
    timestamps within the last ``window_sec`` seconds.
    """

    def __init__(self, window_sec: float = 1.0, decay: float = 4.0) -> None:
        self.window_sec = float(window_sec)
        self.decay = float(decay)
        self._history: list[float] = []

    def bonus(self, t: float) -> float:
        """Return the rarity bonus at time ``t`` (exp-decay over recent count)."""
        count = sum(1 for et in self._history if t - et <= self.window_sec)
        return math.exp(-count / self.decay)

    def record(self, t: float) -> None:
        """Record an emission at time ``t`` and prune old entries."""
        self._history.append(t)
        cutoff = t - self.window_sec * 2
        self._history = [et for et in self._history if et >= cutoff]
