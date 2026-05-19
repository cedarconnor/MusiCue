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


# ---- Per-stem click routing ----


def test_route_track_to_stem_drum_names():
    from musicue.listen import route_track_to_stem

    for name in (
        "kick", "snare", "hihat", "hat", "cymbal", "tom",
        "downbeat_pulse", "phrase_pulse", "fill_flash",
    ):
        assert route_track_to_stem(name) == "drums", name


def test_route_track_to_stem_vocals():
    from musicue.listen import route_track_to_stem

    assert route_track_to_stem("vocal_phrase") == "vocals"
    assert route_track_to_stem("vocals_lead") == "vocals"


def test_route_track_to_stem_bass():
    from musicue.listen import route_track_to_stem

    assert route_track_to_stem("bass") == "bass"
    assert route_track_to_stem("sub_bass") == "bass"


def test_route_track_to_stem_falls_back_to_other():
    from musicue.listen import route_track_to_stem

    assert route_track_to_stem("drop") == "other"
    assert route_track_to_stem("section_change") == "other"
    assert route_track_to_stem("nonsense_track") == "other"


def _make_multi_stem_cuesheet(duration=4.0):
    """Cuesheet with one click-producing track per stem."""
    env = {"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0}
    return CueSheet(
        source_sha256="abc",
        grammar="test",
        duration_sec=duration,
        tempo_map=[],
        tracks=[
            CueTrack(
                name="kick", type="impulse", timescale="micro",
                events=[{"t": 0.5, "strength": 0.9, "envelope": env}],
            ),
            CueTrack(
                name="bass", type="impulse", timescale="micro",
                events=[{"t": 1.0, "strength": 0.7, "envelope": env}],
            ),
            CueTrack(
                name="vocal_phrase", type="impulse", timescale="meso",
                events=[{"t": 2.0, "strength": 0.8, "envelope": env}],
            ),
            CueTrack(
                name="drop", type="impulse", timescale="micro",
                events=[{"t": 3.0, "strength": 1.0, "envelope": env}],
            ),
        ],
    )


def test_render_stem_click_marks_writes_four_wavs(tmp_path):
    """Per-stem click_marks WAVs: one per stem, clicks only (no source)."""
    from musicue.listen import render_stem_click_marks

    cs = _make_multi_stem_cuesheet()
    render_stem_click_marks(cs, tmp_path, sr=44100)
    for stem in ("drums", "bass", "vocals", "other"):
        p = tmp_path / f"click_marks.{stem}.wav"
        assert p.exists(), f"missing {p.name}"


def test_render_stem_click_marks_routes_clicks_correctly(tmp_path):
    """Each stem file should only carry clicks routed to that stem."""
    from musicue.listen import render_stem_click_marks

    cs = _make_multi_stem_cuesheet(duration=4.0)
    render_stem_click_marks(cs, tmp_path, sr=44100)

    expected = {
        "drums": 0.5,   # kick
        "bass": 1.0,    # bass
        "vocals": 2.0,  # vocal_phrase
        "other": 3.0,   # drop
    }
    other_times = {
        "drums": (1.0, 2.0, 3.0),
        "bass": (0.5, 2.0, 3.0),
        "vocals": (0.5, 1.0, 3.0),
        "other": (0.5, 1.0, 2.0),
    }
    for stem, click_t in expected.items():
        data, _ = sf.read(str(tmp_path / f"click_marks.{stem}.wav"))
        if data.ndim > 1:
            data = data[:, 0]
        idx = int(click_t * 44100)
        window = data[max(0, idx - 2205): idx + 2205]
        assert np.max(np.abs(window)) > 0.1, (
            f"{stem}: no click near t={click_t}"
        )
        for other_t in other_times[stem]:
            idx = int(other_t * 44100)
            window = data[max(0, idx - 2205): idx + 2205]
            assert np.max(np.abs(window)) < 0.05, (
                f"{stem}: should not have a click at t={other_t} (belongs to another stem)"
            )


def test_render_stem_click_marks_no_source_audio_mixed_in(tmp_path):
    """click_marks files contain clicks only -- no song audio bled in."""
    from musicue.listen import render_stem_click_marks

    cs = _make_multi_stem_cuesheet()
    render_stem_click_marks(cs, tmp_path, sr=44100)
    # 'bass' has 1 click at t=1.0. Sample at t=0.1 (no clicks) and confirm silence.
    data, _ = sf.read(str(tmp_path / "click_marks.bass.wav"))
    if data.ndim > 1:
        data = data[:, 0]
    silence_window = data[int(0.05 * 44100): int(0.45 * 44100)]
    assert np.max(np.abs(silence_window)) < 0.01, "expected silence; got source audio bleed"
