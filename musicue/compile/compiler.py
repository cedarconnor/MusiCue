from __future__ import annotations

from musicue.schemas import AnalysisResult, CueSheet, CueTrack


def compile_analysis(analysis: AnalysisResult, grammar: str = "concert_visuals") -> CueSheet:
    tracks: list[CueTrack] = []

    drum_onsets = analysis.onsets.get("drums", [])
    if drum_onsets:
        events = [
            {
                "t": o.t,
                "strength": o.strength,
                "envelope": {"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0},
                "tags": [o.drum_class] if o.drum_class else [],
            }
            for o in drum_onsets
        ]
        tracks.append(CueTrack(name="drums", type="impulse", timescale="micro", events=events))

    if "lufs" in analysis.curves:
        lufs = analysis.curves["lufs"]
        tracks.append(CueTrack(
            name="energy",
            type="continuous",
            timescale="macro",
            hop_sec=lufs.hop_sec,
            values=list(lufs.values),
        ))

    return CueSheet(
        source_sha256=analysis.source.sha256,
        grammar=grammar,
        duration_sec=analysis.source.duration_sec,
        tempo_map=analysis.tempo.bpm_curve if analysis.tempo else [],
        tracks=tracks,
    )
