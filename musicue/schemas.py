from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TimedCurve(BaseModel):
    hop_sec: float
    values: list[float]


class SourceInfo(BaseModel):
    path: str
    sha256: str
    duration_sec: float
    sample_rate: int


class AnalysisConfig(BaseModel):
    demucs_model: str = "htdemucs_ft"
    demucs_version: str = ""
    allin1_model: str = "harmonix-all"
    allin1_version: str = ""
    clap_model: str = "music_audioset_epoch_15_esc_90.14.pt"
    clap_version: str = ""
    basic_pitch_model: str = "icassp_2022"
    basic_pitch_version: str = ""
    drum_classifier_version: str = ""
    beat_backend: Literal["allin1", "librosa"] = "allin1"
    curve_hop_sec: float = 0.04
    fps: float = 24.0
    drop_frame: bool = False


class TempoInfo(BaseModel):
    bpm_global: float
    bpm_curve: list[dict[str, float]] = Field(default_factory=list)
    time_signature: list[int] = Field(default=[4, 4])


class Label(BaseModel):
    label: str
    score: float
    source: str


class BeatEvent(BaseModel):
    t: float
    beat_in_bar: int
    bar: int
    is_downbeat: bool
    confidence: float
    timescale: Literal["micro", "meso", "macro"] = "micro"
    frame: int | None = None
    timecode: str | None = None
    # Pattern-aware fields (populated by detect_patterns at end of analysis).
    phrase_id: int | None = None
    phrase_position: int | None = None
    phrase_length: int | None = None
    is_fill: bool = False
    syncopation: float | None = None


class SectionEvent(BaseModel):
    start: float
    end: float
    label: str
    confidence: float
    timescale: Literal["micro", "meso", "macro"] = "macro"
    frame_start: int | None = None
    frame_end: int | None = None
    timecode_start: str | None = None
    timecode_end: str | None = None


class RampEvidence(BaseModel):
    spectral_flux_rise: float
    lufs_rise_db: float


class SectionTransition(BaseModel):
    t: float
    from_section: str = Field(alias="from")
    to: str
    ramp: dict[str, Any]
    ramp_evidence: RampEvidence
    frame: int | None = None
    timecode: str | None = None

    model_config = {"populate_by_name": True}


class OnsetEvent(BaseModel):
    t: float
    strength: float
    timescale: Literal["micro", "meso", "macro"] = "micro"
    drum_class: str | None = None
    drum_class_conf: float | None = None
    labels: list[Label] = Field(default_factory=list)
    frame: int | None = None
    timecode: str | None = None


class MidiNote(BaseModel):
    t: float
    duration: float
    pitch: int
    velocity: int
    frame: int | None = None
    timecode: str | None = None


class PhraseEvent(BaseModel):
    t_start: float
    t_end: float
    timescale: Literal["micro", "meso", "macro"] = "meso"
    note_count: int
    pitch_peak: int
    pitch_low: int
    pitch_contour: list[int]
    energy_curve: TimedCurve
    labels: list[Label] = Field(default_factory=list)
    frame_start: int | None = None
    frame_end: int | None = None
    timecode_start: str | None = None
    timecode_end: str | None = None


class PhraseBlock(BaseModel):
    """A repeating phrase unit detected from beat-grid autocorrelation."""

    bar_start: int
    bar_end: int  # exclusive
    length: int  # 4, 8, 16, etc.
    section_label: str
    confidence: float


class FillEvent(BaseModel):
    """A drum-density fill, typically the bar before a section change."""

    bar: int
    t_start: float
    t_end: float
    density_zscore: float
    leads_into: str | None = None


class Patterns(BaseModel):
    phrases: list[PhraseBlock] = Field(default_factory=list)
    fills: list[FillEvent] = Field(default_factory=list)
    syncopation_per_bar: list[float] = Field(default_factory=list)
    bar_count: int = 0


class AnalysisResult(BaseModel):
    schema_version: str = "1.3"
    source: SourceInfo
    analysis_config: AnalysisConfig
    stems: dict[str, str]
    tempo: TempoInfo | None = None
    beats: list[BeatEvent] = Field(default_factory=list)
    sections: list[SectionEvent] = Field(default_factory=list)
    section_transitions: list[SectionTransition] = Field(default_factory=list)
    onsets: dict[str, list[OnsetEvent]] = Field(default_factory=dict)
    midi: dict[str, list[MidiNote]] = Field(default_factory=dict)
    phrases: dict[str, list[PhraseEvent]] = Field(default_factory=dict)
    curves: dict[str, TimedCurve] = Field(default_factory=dict)
    lufs_integrated: float | None = None
    patterns: Patterns | None = None


class ADSREnvelope(BaseModel):
    a: float
    d: float
    s: float
    r: float


class CueTrack(BaseModel):
    name: str
    type: Literal["impulse", "envelope", "step", "ramp", "continuous"]
    timescale: Literal["micro", "meso", "macro"]
    events: list[dict[str, Any]] = Field(default_factory=list)
    hop_sec: float | None = None
    values: list[float] | None = None


class CueSheet(BaseModel):
    schema_version: str = "1.2"
    source_sha256: str
    grammar: str
    duration_sec: float
    fps: float = 24.0
    drop_frame: bool = False
    tempo_map: list[dict[str, float]] = Field(default_factory=list)
    tracks: list[CueTrack] = Field(default_factory=list)


class SectionBundleEntry(BaseModel):
    start: float
    end: float
    label: str
    confidence: float
    lufs: float | None = None
    energy_rank: float
    spectral_flux_rise: float | None = None


class DrumOnset(BaseModel):
    t: float
    strength: float
    confidence: float | None = None


class MidiNoteBundle(BaseModel):
    t: float
    duration: float
    pitch: int
    velocity: int


class StemEnergyCurve(BaseModel):
    hop_sec: float
    values: list[float] = Field(default_factory=list)


class MusiCueBundle(BaseModel):
    schema_version: str = "1.1"
    source_sha256: str
    decoded_audio_sha256: str | None = None
    duration_sec: float
    fps: float = 24.0

    tempo: TempoInfo
    beats: list[BeatEvent] = Field(default_factory=list)

    sections: list[SectionBundleEntry] = Field(default_factory=list)

    drums: dict[str, list[DrumOnset]] = Field(default_factory=dict)
    midi: dict[str, list[MidiNoteBundle]] = Field(default_factory=dict)
    midi_energy: dict[str, StemEnergyCurve] = Field(default_factory=dict)

    stems_energy: dict[str, StemEnergyCurve] = Field(default_factory=dict)
    global_energy: StemEnergyCurve

    cuesheet: CueSheet
