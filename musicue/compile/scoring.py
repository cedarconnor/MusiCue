"""Event scoring engine: filter DSL, base/multiplier resolution, rarity tracker.

This module implements a small, regex-based DSL used by MusiCue grammars to
decide whether an event matches a rule and how strongly it should fire.

Filter DSL (`evaluate_filter`)
------------------------------
The filter expression is a single string evaluated against an event dict.
Each expression must match exactly one of the supported forms (we use
``re.fullmatch`` so partial-prefix matches are rejected). Unknown
expressions log a warning and return ``False`` so silently-broken filters
fail loud rather than letting every event pass.

- ``None`` -> always matches (returns ``True``).
- ``field == 'value'`` -> string equality on ``event[field]``.
- ``field != 'value'`` -> string inequality on ``event[field]``.
- ``field == true`` / ``field == false`` -> boolean check on ``event[field]``.
- ``field > value`` / ``field >= value`` -> numeric comparison on ``event[field]``.
- ``field < value`` / ``field <= value`` -> numeric comparison on ``event[field]``.
- ``any_label('label', min_score=X)`` -> ``True`` when any entry in
  ``event['labels']`` has matching ``label`` and ``score >= X``.
- ``near_downbeat(seconds)`` -> ``True`` when the event's
  ``downbeat_distance_sec`` (set by the compiler) is within ``seconds``.
  Returns ``False`` when no downbeat distance is recorded on the event.

Compound expressions (``and``, ``or``) are not supported: use multiple
multiplier rules with separate ``when`` clauses for AND, or define
separate tracks for OR.

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

import logging
import math
import re
from typing import Any

log = logging.getLogger(__name__)

# Precompiled regexes for the DSL forms.
# Field name patterns accept dotted paths (e.g. ``ramp_evidence.spectral_flux_rise``)
# and are walked via :func:`_get_dotted`.
#
# All filter regexes are matched via :func:`re.Pattern.fullmatch` so a partial
# prefix (e.g. ``drum_class == 'kick' and strength > 0.5``) does not silently
# match the simple ``field == 'value'`` shape.
_RE_ANY_LABEL = re.compile(r"any_label\('([^']+)',\s*min_score=([\d.]+)\)")
_RE_NEAR_DOWNBEAT = re.compile(r"near_downbeat\(([\d.]+)\)")
_RE_IS_FILL = re.compile(r"is_fill\(\)")
_RE_EVERY_NTH = re.compile(r"every_nth\((\d+)(?:,\s*offset=(\d+))?\)")
_RE_IS_PHRASE_START = re.compile(r"is_phrase_start\(\)")
_RE_IS_PHRASE_END = re.compile(r"is_phrase_end\(\)")
_RE_FIELD_EQ_STR = re.compile(r"([\w.]+)\s*==\s*'([^']*)'")
_RE_FIELD_NE_STR = re.compile(r"([\w.]+)\s*!=\s*'([^']*)'")
_RE_FIELD_EQ_BOOL = re.compile(r"([\w.]+)\s*==\s*(true|false)")
# Two-character comparison operators must be checked BEFORE their single-char
# cousins so ``>=`` doesn't get parsed as ``>`` (consuming the ``>`` and
# leaving ``= 0.5`` unmatched).
_RE_FIELD_GE = re.compile(r"([\w.]+)\s*>=\s*([\d.]+)")
_RE_FIELD_LE = re.compile(r"([\w.]+)\s*<=\s*([\d.]+)")
_RE_FIELD_GT = re.compile(r"([\w.]+)\s*>\s*([\d.]+)")
_RE_FIELD_LT = re.compile(r"([\w.]+)\s*<\s*([\d.]+)")
_RE_LABEL_SCORE = re.compile(r"label_score\('([^']+)'\)")


def _get_dotted(event: dict, path: str, default: Any = None) -> Any:
    """Walk a dotted ``path`` against ``event`` and return the leaf value.

    Returns ``default`` when any intermediate segment is missing or the
    current node is not a dict. This lets filter expressions reference
    nested fields like ``ramp_evidence.spectral_flux_rise`` without the
    grammar having to flatten event payloads beforehand.
    """
    cur: Any = event
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def evaluate_filter(expr: str | None, event: dict) -> bool:
    """Evaluate a filter expression against an event dict.

    Returns ``True`` when the event matches the filter. Unknown
    expressions log a warning and return ``False`` so silently-broken
    filters fail loud rather than passing every event.
    """
    if expr is None:
        return True

    # any_label('label', min_score=X)
    m = _RE_ANY_LABEL.fullmatch(expr)
    if m:
        target_label, min_score = m.group(1), float(m.group(2))
        return any(
            lbl.get("label") == target_label and float(lbl.get("score", 0.0)) >= min_score
            for lbl in event.get("labels", [])
        )

    # near_downbeat(seconds) -- compare the event's recorded downbeat
    # distance to the captured window. Falls closed when no distance is
    # available.
    m = _RE_NEAR_DOWNBEAT.fullmatch(expr)
    if m:
        window = float(m.group(1))
        distance = event.get("downbeat_distance_sec")
        if distance is None:
            return False
        try:
            return float(distance) <= window
        except (TypeError, ValueError):
            return False

    # field == 'value' (supports dotted paths)
    m = _RE_FIELD_EQ_STR.fullmatch(expr)
    if m:
        field, value = m.group(1), m.group(2)
        return str(_get_dotted(event, field, "")) == value

    # field != 'value' (supports dotted paths)
    m = _RE_FIELD_NE_STR.fullmatch(expr)
    if m:
        field, value = m.group(1), m.group(2)
        return str(_get_dotted(event, field, "")) != value

    # field == true/false (supports dotted paths)
    m = _RE_FIELD_EQ_BOOL.fullmatch(expr)
    if m:
        field, want_true = m.group(1), m.group(2) == "true"
        return bool(_get_dotted(event, field, False)) == want_true

    # field >= value (supports dotted paths) -- check before ``>``.
    m = _RE_FIELD_GE.fullmatch(expr)
    if m:
        field, value = m.group(1), float(m.group(2))
        try:
            return float(_get_dotted(event, field, 0)) >= value
        except (TypeError, ValueError):
            return False

    # field <= value (supports dotted paths) -- check before ``<``.
    m = _RE_FIELD_LE.fullmatch(expr)
    if m:
        field, value = m.group(1), float(m.group(2))
        try:
            return float(_get_dotted(event, field, 0)) <= value
        except (TypeError, ValueError):
            return False

    # field > value (supports dotted paths)
    m = _RE_FIELD_GT.fullmatch(expr)
    if m:
        field, value = m.group(1), float(m.group(2))
        try:
            return float(_get_dotted(event, field, 0)) > value
        except (TypeError, ValueError):
            return False

    # field < value (supports dotted paths)
    m = _RE_FIELD_LT.fullmatch(expr)
    if m:
        field, value = m.group(1), float(m.group(2))
        try:
            return float(_get_dotted(event, field, 0)) < value
        except (TypeError, ValueError):
            return False

    # is_fill() — pattern primitive (v0.2c).
    if _RE_IS_FILL.fullmatch(expr):
        return bool(event.get("is_fill", False))

    # is_phrase_start() / is_phrase_end() — convenience for phrase_position == 1
    # and phrase_position == phrase_length.
    if _RE_IS_PHRASE_START.fullmatch(expr):
        pos = event.get("phrase_position")
        return pos == 1
    if _RE_IS_PHRASE_END.fullmatch(expr):
        pos = event.get("phrase_position")
        length = event.get("phrase_length")
        return pos is not None and length is not None and pos == length

    # every_nth(N, offset=K) — bar-level periodic selector. Matches when the
    # event's bar (mod N) equals offset (default 0). Used for "every 4th beat",
    # "every 8 bars from bar 0", etc.
    m = _RE_EVERY_NTH.fullmatch(expr)
    if m:
        n = int(m.group(1))
        offset = int(m.group(2)) if m.group(2) else 0
        bar = event.get("bar")
        if bar is None:
            return False
        try:
            return (int(bar) - offset) % n == 0
        except (TypeError, ValueError):
            return False

    # Unknown expression: fail-loud so silently-broken filters get noticed.
    log.warning("Unrecognized filter expression: %r -- treating as False.", expr)
    return False


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
