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


class SectionEvent(BaseModel):
    start: float
    end: float
    label: str
    confidence: float
    timescale: Literal["micro", "meso", "macro"] = "macro"


class RampEvidence(BaseModel):
    spectral_flux_rise: float
    lufs_rise_db: float


class SectionTransition(BaseModel):
    t: float
    from_section: str = Field(alias="from")
    to: str
    ramp: dict[str, Any]
    ramp_evidence: RampEvidence

    model_config = {"populate_by_name": True}


class OnsetEvent(BaseModel):
    t: float
    strength: float
    timescale: Literal["micro", "meso", "macro"] = "micro"
    drum_class: str | None = None
    drum_class_conf: float | None = None
    labels: list[Label] = Field(default_factory=list)


class MidiNote(BaseModel):
    t: float
    duration: float
    pitch: int
    velocity: int


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


class AnalysisResult(BaseModel):
    schema_version: str = "1.1"
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
    schema_version: str = "1.1"
    source_sha256: str
    grammar: str
    duration_sec: float
    tempo_map: list[dict[str, float]] = Field(default_factory=list)
    tracks: list[CueTrack] = Field(default_factory=list)
