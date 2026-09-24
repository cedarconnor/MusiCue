import numpy as np
import pytest

from musicue.compile import controls as ctl

HOP = 0.04


def test_grid_length_avoids_float_truncation():
    assert ctl.grid_length(10.0, 0.04) == 250
    assert ctl.grid_length(10.0, 0.0) == 0
    assert ctl.grid_length(0.0, 0.04) == 0


def test_resample_same_and_coarser_hop():
    assert ctl.resample([0.0, 1.0, 2.0], HOP, HOP, 3).tolist() == [0.0, 1.0, 2.0]
    out = ctl.resample([0.0, 1.0], 0.08, HOP, 4)
    assert out.tolist() == pytest.approx([0.0, 0.5, 1.0, 1.0])  # ends held
    assert ctl.resample([], HOP, HOP, 3).size == 0


def test_energy_fast_silence_and_percentiles():
    db = np.array([-90.0] * 50 + [-30.0] * 100 + [-10.0] * 100)
    out = ctl.energy_fast(db)
    assert max(out[:50]) == 0.0
    assert max(out[50:150]) == 0.0      # at the 5th percentile
    assert min(out[150:]) == 1.0        # at the 95th percentile
    assert all(0.0 <= v <= 1.0 for v in out)


def test_brightness_orders_by_centroid_and_zeroes_silence():
    hz = np.array([500.0] * 50 + [1000.0] * 50 + [4000.0] * 50 + [9999.0] * 20)
    silent = np.zeros(hz.size, dtype=bool)
    silent[-20:] = True
    out = ctl.brightness(hz, silent, HOP)
    assert len(out) == hz.size
    assert out[25] == 0.0
    assert 0.0 < out[75] < out[125]
    assert out[125] == 1.0
    # Silent frames -> 0 and don't pollute their neighbours' smoothing.
    assert out[-20:] == [0.0] * 20
    assert out[-21] == 1.0


def test_brightness_is_log_frequency():
    # 250 -> 1000 -> 4000 Hz are equal log steps: the middle lands at 0.5.
    hz = np.array([250.0] * 100 + [1000.0] * 100 + [4000.0] * 100)
    out = ctl.brightness(hz, np.zeros(hz.size, dtype=bool), HOP)
    assert out[150] == pytest.approx(0.5)


def test_brightness_smoothing_is_centered_and_short():
    hz = np.array([500.0] * 50 + [8000.0] * 50)
    out = ctl.brightness(hz, np.zeros(hz.size, dtype=bool), HOP)
    # 3-frame (120 ms) centered box: only the frames adjacent to the step move.
    assert out[48] == 0.0
    assert 0.0 < out[49] < out[50] < 1.0
    assert out[51] == 1.0


def _beats(n: int, period: float = 0.5) -> list[float]:
    return [i * period for i in range(n)]


def test_onset_density_higher_where_drums_busier():
    beats = _beats(64)  # 32 s at 120 bpm
    sparse = [b for b in beats[:32] if (beats.index(b) % 4) == 0]      # 1 per bar
    busy = [b + k * 0.125 for b in beats[32:] for k in range(4)]       # 4 per beat
    n = int(32.0 / HOP)
    out = ctl.onset_density(sparse + busy, beats, HOP, n)
    assert len(out) == n
    assert all(0.0 <= v <= 1.0 for v in out)
    assert max(out[: round(14.0 / HOP)]) < 0.15
    assert out[round(24.0 / HOP)] == pytest.approx(1.0)


def test_onset_density_zero_outside_grid_and_without_onsets():
    beats = [2.0 + i * 0.5 for i in range(8)]  # grid covers 2.0 .. 6.0
    onsets = [2.0 + i * 0.25 for i in range(16)]
    n = int(10.0 / HOP)
    out = ctl.onset_density(onsets, beats, HOP, n)
    assert max(out[: round(1.9 / HOP)]) == 0.0
    assert max(out[round(6.0 / HOP):]) == 0.0
    assert max(out) == pytest.approx(1.0)
    assert ctl.onset_density([], beats, HOP, n) == [0.0] * n
    assert ctl.onset_density(onsets, [1.0], HOP, n) == [0.0] * n


def test_onset_density_smooths_over_one_bar():
    beats = _beats(32)
    # A single 8-hit burst in beat 16: smoothing spreads it over ~1 bar.
    onsets = [8.0 + k * 0.05 for k in range(8)]
    out = ctl.onset_density(onsets, beats, HOP, int(16.0 / HOP))
    at = lambda t: out[round(t / HOP)]  # noqa: E731
    assert at(8.25) > 0.0
    assert at(7.25) > 0.0 and at(9.25) > 0.0       # neighbours within a bar
    assert at(7.25) == pytest.approx(at(9.25), abs=0.03)  # centered
    assert at(6.25) == 0.0 and at(10.25) == 0.0     # beyond +-2 beats
