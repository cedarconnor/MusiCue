from musicue.schemas import (
    BeatEvent,
    DrumOnset,
    MidiNoteBundle,
    MusiCueBundle,
    SectionBundleEntry,
    StemEnergyCurve,
    TempoInfo,
)


def _minimal_bundle_kwargs():
    return dict(
        source_sha256="x" * 64,
        duration_sec=10.0,
        fps=24.0,
        tempo=TempoInfo(bpm_global=120.0),
        beats=[],
        sections=[],
        drums={},
        midi={},
        midi_energy={},
        global_energy=StemEnergyCurve(hop_sec=0.04, values=[0.0]),
    )


def test_minimal_bundle_roundtrip():
    from musicue.schemas import CueSheet

    cs = CueSheet(source_sha256="x" * 64, grammar="concert_visuals", duration_sec=10.0)
    b = MusiCueBundle(cuesheet=cs, **_minimal_bundle_kwargs())

    assert b.schema_version == "1.1"
    assert b.stems_energy == {}
    roundtrip = MusiCueBundle.model_validate_json(b.model_dump_json())
    assert roundtrip.duration_sec == 10.0


def test_section_bundle_entry_required_fields():
    s = SectionBundleEntry(start=0.0, end=4.0, label="intro", confidence=0.9, energy_rank=0.5)
    assert s.lufs is None
    assert s.spectral_flux_rise is None


def test_drum_onset_and_midi_note_shapes():
    d = DrumOnset(t=1.5, strength=0.8)
    assert d.confidence is None

    n = MidiNoteBundle(t=0.5, duration=0.25, pitch=60, velocity=100)
    assert n.pitch == 60


def test_stem_energy_curve_shape():
    c = StemEnergyCurve(hop_sec=0.04, values=[0.1, 0.5, 0.9])
    assert len(c.values) == 3


def test_bundle_carries_decoded_audio_sha256_optional():
    """Schema 1.1 adds decoded_audio_sha256 as optional (None = legacy 1.0)."""
    from musicue.schemas import CueSheet

    cs = CueSheet(source_sha256="x" * 64, grammar="concert_visuals", duration_sec=10.0)
    b = MusiCueBundle(
        cuesheet=cs,
        decoded_audio_sha256="b" * 64,
        **_minimal_bundle_kwargs(),
    )
    assert b.decoded_audio_sha256 == "b" * 64
    assert b.schema_version == "1.1"


def test_bundle_legacy_schema_version_is_readable():
    """A bundle dict without decoded_audio_sha256 still parses (round-trip from 1.0)."""
    from musicue.schemas import CueSheet

    cs = CueSheet(source_sha256="x" * 64, grammar="concert_visuals", duration_sec=10.0)
    b = MusiCueBundle(
        cuesheet=cs,
        schema_version="1.0",
        **_minimal_bundle_kwargs(),
    )
    assert b.decoded_audio_sha256 is None
    assert b.schema_version == "1.0"
