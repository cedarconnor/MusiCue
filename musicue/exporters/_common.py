"""Shared helpers for exporters."""
from __future__ import annotations

from musicue.schemas import CueTrack


def is_track_empty(track: CueTrack) -> bool:
    """Tracks that would produce no usable output downstream.

    Continuous tracks need ``values``; event-based tracks need ``events``.
    The compiler can produce empty tracks when the source data is missing
    (e.g. CLAP labels absent for the ``drop`` track, or All-In-One sections
    absent for ``section_change``). Skipping empties keeps exports clean
    and avoids the confusing pattern of "Null 5, Null 6 have no keyframes"
    that surfaced in the AE timeline.
    """
    if track.type == "continuous":
        return not track.values
    if track.type in ("impulse", "envelope", "step", "ramp"):
        return not track.events
    return True


def non_empty_tracks(tracks):
    """Return only tracks that will produce visible/audible output."""
    return [t for t in tracks if not is_track_empty(t)]
