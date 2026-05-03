from musicue.diff import diff_cuesheets
from musicue.schemas import CueSheet, CueTrack


def _cs(tracks) -> CueSheet:
    return CueSheet(
        source_sha256="x", grammar="g", duration_sec=10.0, tempo_map=[], tracks=tracks
    )


def _impulse_track(name, times) -> CueTrack:
    return CueTrack(
        name=name,
        type="impulse",
        timescale="micro",
        events=[
            {
                "t": t,
                "strength": 0.8,
                "envelope": {"a": 0.005, "d": 0.12, "s": 0, "r": 0},
                "tags": [],
            }
            for t in times
        ],
    )


def test_diff_identical_cuesheets():
    cs = _cs([_impulse_track("kick", [0.5, 1.0, 1.5])])
    result = diff_cuesheets(cs, cs)
    assert result["kick"]["added"] == 0
    assert result["kick"]["removed"] == 0
    assert result["kick"]["count_a"] == 3
    assert result["kick"]["count_b"] == 3


def test_diff_added_events():
    cs_a = _cs([_impulse_track("kick", [0.5, 1.0])])
    cs_b = _cs([_impulse_track("kick", [0.5, 1.0, 1.5, 2.0])])
    result = diff_cuesheets(cs_a, cs_b)
    assert result["kick"]["added"] == 2
    assert result["kick"]["removed"] == 0


def test_diff_removed_events():
    cs_a = _cs([_impulse_track("kick", [0.5, 1.0, 1.5])])
    cs_b = _cs([_impulse_track("kick", [0.5])])
    result = diff_cuesheets(cs_a, cs_b)
    assert result["kick"]["removed"] == 2
    assert result["kick"]["added"] == 0


def test_diff_new_track_in_b():
    cs_a = _cs([_impulse_track("kick", [0.5])])
    cs_b = _cs([_impulse_track("kick", [0.5]), _impulse_track("snare", [1.0])])
    result = diff_cuesheets(cs_a, cs_b)
    assert "snare" in result
    assert result["snare"]["added"] == 1
    assert result["snare"]["count_a"] == 0


def test_diff_missing_track_in_b():
    cs_a = _cs([_impulse_track("kick", [0.5]), _impulse_track("snare", [1.0])])
    cs_b = _cs([_impulse_track("kick", [0.5])])
    result = diff_cuesheets(cs_a, cs_b)
    assert "snare" in result
    assert result["snare"]["removed"] == 1
    assert result["snare"]["count_b"] == 0
