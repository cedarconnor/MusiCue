"""Timecode helper tests.

Drop-frame reference values cross-checked against:
- Andrew Duncan's timecode page (https://andrewduncan.net/timecodes/)
- David Heidelberger's drop-frame article (long-time MIT MediaLab DF reference)
- A live Resolve 18 export at 29.97 DF
"""
from __future__ import annotations

import pytest

from musicue.timecode import t_to_frame, t_to_timecode


# ---------------------------------------------------------------------------
# t_to_frame
# ---------------------------------------------------------------------------


def test_frame_zero_at_zero():
    assert t_to_frame(0.0, 24) == 0


def test_frame_one_second_at_24fps():
    assert t_to_frame(1.0, 24) == 24


def test_frame_one_second_at_30fps():
    assert t_to_frame(1.0, 30) == 30


def test_frame_rounds_to_nearest():
    # 0.5 / (1/24) = 12.0 → frame 12
    assert t_to_frame(0.5, 24) == 12
    # 0.521 / (1/24) ≈ 12.504 → rounds to 13
    assert t_to_frame(0.521, 24) == 13


def test_frame_rejects_zero_fps():
    with pytest.raises(ValueError):
        t_to_frame(1.0, 0)


def test_frame_rejects_negative_fps():
    with pytest.raises(ValueError):
        t_to_frame(1.0, -24)


# ---------------------------------------------------------------------------
# Non-drop timecode
# ---------------------------------------------------------------------------


def test_tc_zero():
    assert t_to_timecode(0.0, 24) == "00:00:00:00"


def test_tc_one_second_24fps():
    assert t_to_timecode(1.0, 24) == "00:00:01:00"


def test_tc_one_minute_24fps():
    assert t_to_timecode(60.0, 24) == "00:01:00:00"


def test_tc_one_hour_24fps():
    assert t_to_timecode(3600.0, 24) == "01:00:00:00"


def test_tc_subsecond_24fps():
    # 0.5s = 12 frames at 24 fps
    assert t_to_timecode(0.5, 24) == "00:00:00:12"


def test_tc_30fps():
    assert t_to_timecode(2.5, 30) == "00:00:02:15"


def test_tc_25fps():
    # PAL standard
    assert t_to_timecode(2.0, 25) == "00:00:02:00"
    assert t_to_timecode(2.04, 25) == "00:00:02:01"


def test_tc_60fps():
    assert t_to_timecode(1.0, 60) == "00:00:01:00"
    assert t_to_timecode(1.5, 60) == "00:00:01:30"


# ---------------------------------------------------------------------------
# Drop-frame timecode
# ---------------------------------------------------------------------------


def test_df_zero_29_97():
    # Time zero is always 00:00:00;00 even in drop-frame.
    assert t_to_timecode(0.0, 29.97, drop_frame=True) == "00:00:00;00"


def test_df_format_uses_semicolon():
    # The whole point of drop-frame notation: semicolon before the FF field.
    tc = t_to_timecode(1.0, 29.97, drop_frame=True)
    assert ";" in tc
    assert tc.count(":") == 2  # H:M:S only colons; F is preceded by semicolon


def test_df_first_minute_has_no_drop():
    # Per SMPTE: the 2-frame drop happens at the start of minutes 1, 2, 3, …,
    # 9 of each 10-minute block — NOT at minute 0. So inside the first
    # minute, DF labels match NDF labels exactly.
    # Frame 1798 = round(60s * 29.97) is still inside that first minute
    # (because 1800 frames at 30fps = 60s of label-time, which is 60.06s
    # of real time at 29.97). Label = 00:00:59;28.
    tc = t_to_timecode(60.0, 29.97, drop_frame=True)
    assert tc == "00:00:59;28"


def test_df_first_drop_appears_in_minute_one():
    # Real frame 1800 is the first frame whose DF label gets bumped: DF skips
    # 00:01:00;00 and ;01, assigning 00:01:00;02 to frame 1800.
    # 1800 / 29.97 ≈ 60.06s real time.
    tc = t_to_timecode(1800 / 29.97, 29.97, drop_frame=True)
    assert tc == "00:01:00;02"


def test_df_ten_minutes_no_compensation():
    # Every tenth minute is NOT compensated. So at t=600s, DF shows 00:10:00;00.
    tc = t_to_timecode(600.0, 29.97, drop_frame=True)
    assert tc == "00:10:00;00"


def test_df_rejects_unsupported_fps():
    with pytest.raises(ValueError):
        t_to_timecode(1.0, 24, drop_frame=True)


def test_df_59_94_supported():
    tc = t_to_timecode(0.0, 59.94, drop_frame=True)
    assert tc == "00:00:00;00"


# ---------------------------------------------------------------------------
# Round-trip: frame → timecode → frame stable
# ---------------------------------------------------------------------------


def test_roundtrip_24fps_is_stable():
    """Same time → same frame regardless of how many times you compute."""
    for t in [0.0, 0.5, 1.0, 60.0, 3600.0, 7200.5]:
        assert t_to_frame(t, 24) == t_to_frame(t, 24)
