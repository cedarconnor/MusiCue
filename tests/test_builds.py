import pytest

from musicue.analysis.builds import (
    build_curve,
    build_window_start,
    build_windows,
    energy_ranks,
    is_build,
    section_lufs,
)

BPM = 120.0  # 0.5 s beats, 2 s bars in 4/4
HOP = 0.04


def _downbeats(until: float, bar: float = 2.0) -> list[float]:
    return [i * bar for i in range(int(until / bar) + 1)]


# ---- rank / qualification ----


def test_energy_ranks_minmax_with_unknowns():
    assert energy_ranks([-20.0, None, -10.0, -15.0]) == [0.0, 0.5, 1.0, 0.5]
    assert energy_ranks([-12.0, -12.0]) == [0.5, 0.5]
    assert energy_ranks([]) == []


def test_section_lufs_excludes_silence():
    values = [-70.0] * 10 + [-12.0] * 90
    assert section_lufs(values, HOP, 0.0, 4.0) == pytest.approx(-12.0)
    assert section_lufs([-70.0] * 10, HOP, 0.0, 0.4) == -70.0
    assert section_lufs([], HOP, 0.0, 4.0) is None


@pytest.mark.parametrize(
    "cur,nxt,expected",
    [
        (0.0, 1.0, True),     # quiet -> loud
        (0.3, 0.5, True),     # exactly +0.2
        (0.3, 0.45, False),   # small rise, not high
        (0.7, 0.8, True),     # small rise but next >= 0.75
        (0.8, 0.8, False),    # high but not louder
        (1.0, 0.0, False),    # loud -> quiet
    ],
)
def test_is_build(cur, nxt, expected):
    assert is_build(cur, nxt) is expected


# ---- window start ----


def test_window_start_uses_eighth_downbeat_before_boundary():
    # Boundary on the downbeat at 40 s; 8 bars back on a 2 s grid = 24 s.
    assert build_window_start(40.0, 0.0, _downbeats(60.0), BPM) == pytest.approx(24.0)


def test_window_start_boundary_slightly_after_its_downbeat():
    # Tracker put the downbeat 40 ms before the section boundary: still the
    # boundary's own bar, so the window is 8 whole bars (+40 ms).
    grid = [t - 0.04 for t in _downbeats(60.0)]
    assert build_window_start(40.0, 0.0, grid, BPM) == pytest.approx(24.0 - 0.04)


def test_window_start_follows_local_tempo():
    # Bars are 2.4 s here (100 bpm) even though bpm_global says 120.
    grid = [i * 2.4 for i in range(30)]
    tb = 20 * 2.4
    assert build_window_start(tb, 0.0, grid, BPM) == pytest.approx(tb - 8 * 2.4)


def test_window_start_extrapolates_short_grid_and_falls_back_to_bpm():
    # Only 3 downbeats before tb: extrapolate their spacing.
    assert build_window_start(6.0, -100.0, [0.0, 2.0, 4.0, 6.0], BPM) == pytest.approx(-10.0)
    # No grid at all: 8 * 4 * 60 / bpm = 16 s.
    assert build_window_start(40.0, 0.0, [], BPM) == pytest.approx(24.0)


def test_window_start_clamped_to_current_section_start():
    assert build_window_start(40.0, 34.0, _downbeats(60.0), BPM) == pytest.approx(34.0)


# ---- windows + curve ----


def test_quiet_verse_to_loud_chorus_ramp_ends_exactly_at_boundary():
    sections = [(0.0, 40.0, 0.0), (40.0, 60.0, 1.0)]
    windows = build_windows(sections, _downbeats(60.0), BPM)
    assert windows == [(pytest.approx(24.0), 40.0)]

    n = int(60.0 / HOP)
    curve = build_curve(windows, HOP, n)
    tb_idx = round(40.0 / HOP)
    t0_idx = round(24.0 / HOP)
    assert curve[t0_idx - 1] == 0.0
    assert curve[t0_idx] == pytest.approx(0.0)
    # Quadratic: halfway through the window -> 0.25.
    assert curve[round(32.0 / HOP)] == pytest.approx(0.25, abs=1e-6)
    # Last frame before the boundary is ~1; the boundary itself drops to 0.
    assert curve[tb_idx - 1] == pytest.approx(((39.96 - 24.0) / 16.0) ** 2)
    assert curve[tb_idx - 1] > 0.99
    assert curve[tb_idx] == 0.0
    assert max(curve[tb_idx:]) == 0.0
    # Monotonic rise inside the window.
    inside = curve[t0_idx:tb_idx]
    assert all(b >= a for a, b in zip(inside, inside[1:]))


def test_loud_to_quiet_gets_no_ramp():
    sections = [(0.0, 40.0, 1.0), (40.0, 60.0, 0.0)]
    assert build_windows(sections, _downbeats(60.0), BPM) == []
    assert max(build_curve([], HOP, 100)) == 0.0


def test_short_section_window_clamped_to_its_start():
    sections = [(0.0, 30.0, 0.0), (30.0, 36.0, 0.5), (36.0, 60.0, 1.0)]
    windows = build_windows(sections, _downbeats(60.0), BPM)
    assert windows[0] == (pytest.approx(14.0), 30.0)
    assert windows[1] == (pytest.approx(30.0), 36.0)


def test_overlapping_windows_take_max():
    windows = [(0.0, 10.0), (6.0, 8.0)]
    n = int(12.0 / HOP)
    curve = build_curve(windows, HOP, n)
    i = round(7.0 / HOP)
    a = (7.0 / 10.0) ** 2        # 0.49
    b = ((7.0 - 6.0) / 2.0) ** 2  # 0.25
    assert curve[i] == pytest.approx(max(a, b))
    j = 197  # t = 7.88
    assert curve[j] == pytest.approx(((j * HOP - 6.0) / 2.0) ** 2)  # second wins
    assert curve[round(8.0 / HOP)] == pytest.approx(0.64)       # first again


def test_build_curve_degenerate_inputs():
    assert build_curve([(1.0, 2.0)], 0.0, 10) == []
    assert build_curve([(1.0, 2.0)], HOP, 0) == []
    assert build_curve([(2.0, 2.0)], HOP, 100) == [0.0] * 100
