"""disguise media-server cue table exporter.

disguise (d3) reads a tab-separated cue table with a hard-coded header line
and a fixed column layout. The file MUST be named ``<name>_cue_table.txt``
for d3 to pick it up.

File layout (tab-separated, NOT CSV):

    Cue table for <title>
    Beat<TAB>Tag<TAB>Note<TAB>Track_Time<TAB>TC_Time<TAB>Section_Break
    <beat_seconds><TAB><tag><TAB><note><TAB><track_time><TAB><tc_time><TAB><section_break>
    ...

Per disguise conventions:
- ``Beat`` is float seconds with 5 decimal places — the canonical cue time.
- ``Tag`` is left empty (free for the operator to fill in d3).
- ``Note`` carries any descriptive text — we put the cue name + label here.
- ``Track_Time`` is ``HH:MM:SS.FF`` at the export fps (frames, not centiseconds).
- ``TC_Time`` and ``Section_Break`` are left empty by convention; the operator
  fills these in d3 if they need them.

Default export fps is 30 (matches d3's default cue-table convention). The
caller can override via the ``fps`` kwarg.
"""
from __future__ import annotations

from pathlib import Path

from musicue.schemas import CueSheet

_HEADERS = ("Beat", "Tag", "Note", "Track_Time", "TC_Time", "Section_Break")


def _seconds_to_track_time(t: float, fps: float) -> str:
    """Format seconds as ``HH:MM:SS.FF`` at the given fps.

    Frames are floored (not rounded) to match d3's convention of treating
    ``Beat`` as the authoritative cue time and ``Track_Time`` as a display
    quantisation of it.
    """
    fps_int = max(1, int(round(fps)))
    total_frames = int(t * fps_int)
    ff = total_frames % fps_int
    total_seconds = total_frames // fps_int
    ss = total_seconds % 60
    mm = (total_seconds // 60) % 60
    hh = total_seconds // 3600
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ff:02d}"


def _note_for_impulse(track_name: str, idx: int, ev: dict) -> str:
    tags = ev.get("tags") or []
    tag_str = " ".join(str(t) for t in tags)
    note = f"{track_name}_{idx + 1:04d}"
    return f"{note} {tag_str}".strip()


def _note_for_step(idx: int, ev: dict) -> str:
    label = str(ev.get("label", "")).strip()
    if label:
        return f"section: {label}"
    return f"section_{idx + 1:04d}"


def _note_for_envelope(track_name: str, idx: int, ev: dict) -> str:
    tags = ev.get("tags") or []
    tag_str = " ".join(str(t) for t in tags)
    note = f"{track_name}_env_{idx + 1:04d}"
    return f"{note} {tag_str}".strip()


def _note_for_ramp(track_name: str, idx: int, ev: dict) -> str:
    label = str(ev.get("label", "")).strip()
    base = f"{track_name}_ramp_{idx + 1:04d}"
    return f"{base} {label}".strip()


def export(
    cuesheet: CueSheet,
    out_path: Path,
    fps: float | None = None,
    title: str | None = None,
    **opts,
) -> None:
    # disguise's cue table convention is frame-locked at 30 fps unless
    # the show is explicitly running at another rate.
    if fps is None:
        fps = float(cuesheet.fps) if cuesheet.fps else 30.0

    rows: list[tuple[float, str]] = []  # (beat_seconds, note) — Beat sorts the file.

    for track in cuesheet.tracks:
        if track.type == "impulse":
            for i, ev in enumerate(track.events):
                t = float(ev["t"])
                rows.append((t, _note_for_impulse(track.name, i, ev)))
        elif track.type == "step":
            for i, ev in enumerate(track.events):
                t = float(ev["t"])
                rows.append((t, _note_for_step(i, ev)))
        elif track.type == "envelope":
            for i, ev in enumerate(track.events):
                t = float(ev.get("t_start", 0.0))
                rows.append((t, _note_for_envelope(track.name, i, ev)))
        elif track.type == "ramp":
            for i, ev in enumerate(track.events):
                t = float(ev.get("t_start", 0.0))
                rows.append((t, _note_for_ramp(track.name, i, ev)))

    rows.sort(key=lambda r: r[0])

    if title is None:
        title = out_path.stem or "MusiCue export"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # disguise expects literal tabs as the delimiter, so write the file by
    # hand rather than going through csv.writer (which would quote the
    # \t-containing notes and break parsing). LF line endings keep the file
    # portable across the d3 Windows/macOS clients.
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"Cue table for {title}\n")
        f.write("\t".join(_HEADERS) + "\n")
        for beat_seconds, note in rows:
            track_time = _seconds_to_track_time(beat_seconds, fps)
            # Tag, TC_Time, Section_Break left empty per disguise convention;
            # the show operator fills these in d3 as needed.
            cells = [
                f"{beat_seconds:.5f}",
                "",
                note,
                track_time,
                "",
                "",
            ]
            f.write("\t".join(cells) + "\n")
