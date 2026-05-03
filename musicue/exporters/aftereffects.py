"""After Effects ExtendScript (.jsx) exporter for MusiCue cuesheets.

Emits an ExtendScript file that, when run inside After Effects (File > Scripts
> Run Script File...), populates the active composition with MusiCue data:

- One **Null layer per CueTrack**, named ``MusiCue_<trackname>``. Each null
  carries a single **Slider Control** effect whose slider value is animated
  with keyframes. Downstream layers reference the slider via expressions
  (``thisComp.layer("MusiCue_kick").effect("Slider Control")("Slider")``).
- **Composition markers** for every ``impulse`` (with strength label) and
  ``step`` event (with section label), shown on the comp's marker bar.
- The whole layer creation runs inside an ``app.beginUndoGroup`` so a single
  Cmd/Ctrl-Z reverts the import.

Track-type mapping:

- ``continuous`` -> dense slider keyframes at the track's hop rate. Value is
  remapped from a roughly LUFS-scale range (-70..0 dB) to 0..100.
- ``impulse``    -> three keyframes per event (0 -> peak at attack -> 0 after
  decay), forming a percussive spike, plus a comp marker.
- ``envelope``   -> three keyframes per event (0 -> peak after attack ->
  0 at end), forming a sustained envelope shape.
- ``step``       -> comp marker only (no slider keyframes).
- ``ramp``       -> not emitted as slider keyframes (visual ramps are
  expression-driven downstream); markers from neighboring step tracks
  carry the section change.

Pure stdlib. Output is UTF-8 text. Track names are sanitized for JS variable
identifiers (any non ``[A-Za-z0-9_]`` becomes ``_``; a leading digit is
prefixed with ``_``); the original name is JSX-escaped when interpolated into
layer-name string literals.
"""

from __future__ import annotations

import re
from pathlib import Path

from musicue.schemas import CueSheet

_VAR_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_]")


def _jsx_escape(s: str) -> str:
    """Escape a Python string for embedding in a JSX double-quoted string."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _rescale_to_unit(values: list[float]) -> list[float]:
    """Rescale values to [0, 1] based on observed min/max.

    Continuous tracks may arrive already-normalized (e.g. percentile-normalized
    in the grammar) or as raw signal (e.g. LUFS in [-70, 0]). Auto-rescaling per
    track handles both without an explicit range hint.
    """
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _var_name(track_name: str) -> str:
    """Translate a track name into a JS-safe identifier suffix.

    Replaces any character outside ``[A-Za-z0-9_]`` with ``_`` and prefixes a
    leading underscore if the result starts with a digit, so the suffix is
    always a valid ECMAScript identifier tail.
    """
    v = _VAR_NAME_PATTERN.sub("_", track_name)
    if v[:1].isdigit():
        v = "_" + v
    return v


def export(cuesheet: CueSheet, out_path: Path, fps: float = 24.0, **opts) -> None:
    lines: list[str] = []
    a = lines.append

    a("// MusiCue After Effects ExtendScript")
    a(f"// Grammar: {cuesheet.grammar}  Duration: {cuesheet.duration_sec:.3f}s")
    a("(function() {")
    a("  var comp = app.project.activeItem;")
    a("  if (!comp || !(comp instanceof CompItem)) {")
    a('    alert("MusiCue: No active composition found.");')
    a("    return;")
    a("  }")
    a(f"  var fps = {fps};")
    a("")

    # Composition-level markers for impulse + step tracks
    a("  // Composition markers for impulse and step events")
    a("  var compMarkers = comp.markerProperty;")
    for track in cuesheet.tracks:
        var = _var_name(track.name)
        if track.type == "impulse":
            for ev in track.events:
                t = float(ev["t"])
                strength = float(ev.get("strength", 1.0))
                label = f"{track.name} s={strength:.2f}"
                a(f'  var mk_{var} = new MarkerValue("{_jsx_escape(label)}");')
                a(f"  compMarkers.setValueAtTime({t:.4f}, mk_{var});")
        elif track.type == "step":
            for ev in track.events:
                t = float(ev["t"])
                label = str(ev.get("label", ""))
                a(f'  var mkS = new MarkerValue("section: {_jsx_escape(label)}");')
                a(f"  compMarkers.setValueAtTime({t:.4f}, mkS);")

    a("")
    a("  app.beginUndoGroup('MusiCue Import');")

    # Slider Control layers for continuous + impulse + envelope tracks
    for track in cuesheet.tracks:
        var = _var_name(track.name)
        a(f"  // Track: {track.name} ({track.type})")
        a(f"  var layer_{var} = comp.layers.addNull();")
        a(f'  layer_{var}.name = "MusiCue_{_jsx_escape(track.name)}";')
        a(
            f"  var effect_{var} = layer_{var}.Effects"
            f".addProperty('ADBE Slider Control');"
        )
        slider_ref = f"effect_{var}('ADBE Slider Control-0001')"

        if track.type == "continuous" and track.values and track.hop_sec:
            hop = track.hop_sec
            unit_values = _rescale_to_unit(track.values)
            for i, _val in enumerate(track.values):
                t = i * hop
                slider_val = max(0.0, min(100.0, unit_values[i] * 100))
                a(f"  {slider_ref}.setValueAtTime({t:.4f}, {slider_val:.2f});")

        elif track.type == "impulse":
            for ev in track.events:
                t = float(ev["t"])
                strength = float(ev.get("strength", 1.0))
                env = ev.get("envelope", {})
                a_time = float(env.get("a", 0.005))
                d_time = float(env.get("d", 0.1))
                a(f"  {slider_ref}.setValueAtTime({max(0.0, t - 0.001):.4f}, 0.0);")
                a(
                    f"  {slider_ref}.setValueAtTime("
                    f"{t + a_time:.4f}, {strength * 100:.2f});"
                )
                a(
                    f"  {slider_ref}.setValueAtTime("
                    f"{t + a_time + d_time:.4f}, 0.0);"
                )

        elif track.type == "envelope":
            for ev in track.events:
                t_start = float(ev.get("t_start", 0.0))
                t_end = float(ev.get("t_end", t_start + 1.0))
                strength = float(ev.get("strength", 0.8))
                env = ev.get("envelope", {})
                a_time = float(env.get("a", 0.3))
                a(
                    f"  {slider_ref}.setValueAtTime("
                    f"{max(0.0, t_start - 0.001):.4f}, 0.0);"
                )
                a(
                    f"  {slider_ref}.setValueAtTime("
                    f"{t_start + a_time:.4f}, {strength * 100:.2f});"
                )
                a(f"  {slider_ref}.setValueAtTime({t_end:.4f}, 0.0);")

    a("")
    a("  app.endUndoGroup();")
    a('  alert("MusiCue: Import complete. " + comp.layers.length + " layers added.");')
    a("})();")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
