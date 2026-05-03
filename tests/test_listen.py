import numpy as np
import soundfile as sf

from musicue.schemas import CueSheet, CueTrack


def _make_cuesheet(duration=5.0) -> CueSheet:
    return CueSheet(
        source_sha256="abc",
        grammar="test",
        duration_sec=duration,
        tempo_map=[],
        tracks=[
            CueTrack(
                name="kick",
                type="impulse",
                timescale="micro",
                events=[
                    {
                        "t": 0.5,
                        "strength": 0.9,
                        "envelope": {"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0},
                        "tags": [],
                    },
                    {
                        "t": 1.0,
                        "strength": 0.8,
                        "envelope": {"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0},
                        "tags": [],
                    },
                ],
            ),
            CueTrack(
                name="energy",
                type="continuous",
                timescale="macro",
                hop_sec=1.0,
                values=[-20.0, -18.0, -22.0, -19.0, -21.0],
            ),
        ],
    )


def test_render_click_track_creates_wav(tmp_path):
    from musicue.listen import render_click_track
    cs = _make_cuesheet()
    out = tmp_path / "clicks.wav"
    render_click_track(cs, None, out)
    assert out.exists()


def test_render_click_track_duration(tmp_path):
    from musicue.listen import render_click_track
    cs = _make_cuesheet(duration=5.0)
    out = tmp_path / "clicks.wav"
    render_click_track(cs, None, out, sr=44100)
    data, sr = sf.read(str(out))
    assert abs(len(data) / sr - 5.0) < 0.1


def test_render_click_track_has_transients(tmp_path):
    from musicue.listen import render_click_track
    cs = _make_cuesheet()
    out = tmp_path / "clicks.wav"
    render_click_track(cs, None, out, sr=44100)
    data, _ = sf.read(str(out))
    if data.ndim > 1:
        data = data[:, 0]
    # There should be spikes near t=0.5 and t=1.0
    for burst_t in (0.5, 1.0):
        idx = int(burst_t * 44100)
        window = data[max(0, idx - 2205): idx + 2205]
        assert np.max(np.abs(window)) > 0.1, f"No click near t={burst_t}s"
