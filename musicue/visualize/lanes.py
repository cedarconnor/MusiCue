"""Per-type lane renderers + fire-panel brightness."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from musicue.schemas import CueSheet, CueTrack
from musicue.visualize.colors import section_palette, track_color
from musicue.visualize.cue_video import events_in_window, spanning_events_in_window
from musicue.visualize.envelopes import ease, sample_adsr

_STEP_FLASH_SEC = 0.25
_SECTION_KEYWORDS = {"intro", "verse", "chorus", "bridge", "drop", "outro", "pre"}


def fire_brightness(track: CueTrack, t_now: float) -> float:
    """Return a 0..1 brightness value for the lane's fire cell at t_now."""
    t = track.type
    events = track.events or []

    if t == "impulse":
        best = 0.0
        for ev in events:
            t_event = float(ev.get("t", 0.0))
            env = ev.get("envelope") or {}
            v = sample_adsr(env, t_now - t_event)
            if v > best:
                best = v
        return min(1.0, best)

    if t == "envelope":
        best = 0.0
        for ev in events:
            t_start = float(ev.get("t_start", 0.0))
            env = ev.get("envelope") or {}
            v = sample_adsr(env, t_now - t_start)
            if v > best:
                best = v
        return min(1.0, best)

    if t == "step":
        last_change_t: float | None = None
        for ev in events:
            t_ev = float(ev.get("t", 0.0))
            if t_ev <= t_now:
                last_change_t = t_ev
            else:
                break
        if last_change_t is None:
            return 0.0
        delta = t_now - last_change_t
        if delta < 0 or delta > _STEP_FLASH_SEC:
            return 0.0
        return max(0.0, 1.0 - delta / _STEP_FLASH_SEC)

    if t == "ramp":
        for ev in events:
            t_start = float(ev.get("t_start", 0.0))
            t_end = float(ev.get("t_end", t_start))
            shape = str(ev.get("shape", "linear"))
            if t_start <= t_now <= t_end:
                if t_end <= t_start:
                    return 1.0
                x = (t_now - t_start) / (t_end - t_start)
                return ease(shape, x)
            if t_end < t_now <= t_end + 0.3:
                return max(0.0, 1.0 - (t_now - t_end) / 0.3)
        return 0.0

    if t == "continuous":
        if not track.values or not track.hop_sec or track.hop_sec <= 0:
            return 0.0
        idx_f = t_now / float(track.hop_sec)
        if idx_f < 0 or idx_f > len(track.values) - 1:
            return 0.0
        i = int(idx_f)
        frac = idx_f - i
        v0 = float(track.values[i])
        v1 = float(track.values[min(i + 1, len(track.values) - 1)])
        v = v0 + (v1 - v0) * frac
        lo = min(track.values)
        hi = max(track.values)
        if hi - lo < 1e-9:
            return 0.5
        return max(0.0, min(1.0, (v - lo) / (hi - lo)))

    return 0.0


def draw_impulse_lane(
    ax: Axes,
    track: CueTrack,
    t_now: float,
    window_sec: float,
) -> None:
    """Render an impulse track: thin ticks at event times + flash overlay."""
    color = track_color(track.name)
    events = events_in_window(track.events or [], t_now, window_sec, key="t")
    for ev in events:
        t = float(ev.get("t", 0.0))
        strength = float(ev.get("strength", ev.get("score", 1.0)))
        alpha = max(0.2, min(1.0, strength))
        ax.axvline(t, color=color, linewidth=1.6, alpha=alpha)

    brightness = fire_brightness(track, t_now)
    if brightness > 0.01:
        x0, x1 = ax.get_xlim()
        ax.add_patch(
            Rectangle(
                (x0, 0.0), x1 - x0, 1.0,
                facecolor="white",
                edgecolor="none",
                alpha=0.15 * brightness,
                zorder=0,
            )
        )


def draw_envelope_lane(
    ax: Axes,
    track: CueTrack,
    t_now: float,
    window_sec: float,
) -> None:
    """Render an envelope track: filled regions shaped by ADSR (or
    shape_curve_from if present in the event dict)."""
    color = track_color(track.name)
    events = spanning_events_in_window(
        track.events or [], t_now, window_sec,
        start_key="t_start", end_key="t_end",
    )
    for ev in events:
        t_start = float(ev.get("t_start", 0.0))
        t_end = float(ev.get("t_end", t_start))
        env = ev.get("envelope") or {}
        shape_curve = ev.get("shape_curve_from")

        n = 32
        xs = np.linspace(t_start, t_end, n)
        if shape_curve and isinstance(shape_curve, dict):
            hop = float(shape_curve.get("hop_sec", 0.04))
            values = list(shape_curve.get("values") or [])
            ys = []
            for x in xs:
                idx = (x - t_start) / hop
                if 0 <= idx < len(values) - 1:
                    i = int(idx)
                    frac = idx - i
                    ys.append(values[i] + (values[i + 1] - values[i]) * frac)
                elif idx >= len(values) - 1 and values:
                    ys.append(values[-1])
                else:
                    ys.append(0.0)
            mx = max(ys) if ys else 1.0
            ys = [y / mx if mx > 0 else 0.0 for y in ys]
        else:
            ys = [sample_adsr(env, x - t_start) for x in xs]

        if t_start <= t_now <= t_end:
            alpha = 0.85
        elif t_end < t_now:
            alpha = 0.30
        else:
            alpha = 0.55

        ax.fill_between(xs, 0.0, ys, color=color, alpha=alpha, linewidth=0)


def draw_step_lane(
    ax: Axes,
    track: CueTrack,
    t_now: float,
    window_sec: float,
) -> None:
    """Render a step track: segmented bar across the timeline, labeled."""
    events = track.events or []
    if not events:
        return
    x0, x1 = ax.get_xlim()
    seg_list: list[tuple[float, float, str]] = []
    for i, ev in enumerate(events):
        t = float(ev.get("t", 0.0))
        label = str(ev.get("label", ""))
        seg_end = float(events[i + 1].get("t", t)) if i + 1 < len(events) else x1 + 1.0
        seg_list.append((t, seg_end, label))

    fallback = track_color(track.name)
    for start, end, label in seg_list:
        if end < x0 or start > x1:
            continue
        color = section_palette(label) if label else fallback
        is_current = start <= t_now < end
        edgecolor = (1.0, 1.0, 1.0, 0.9) if is_current else (0.0, 0.0, 0.0, 0.0)
        ax.add_patch(
            Rectangle(
                (start, 0.1), end - start, 0.8,
                facecolor=(*color, 0.7),
                edgecolor=edgecolor,
                linewidth=1.5 if is_current else 0.0,
                zorder=1,
            )
        )
        vis_start = max(start, x0)
        vis_end = min(end, x1)
        if vis_end - vis_start > 0.1:
            ax.text(
                (vis_start + vis_end) / 2.0, 0.5, label,
                ha="center", va="center",
                color="white", fontsize=9,
                zorder=2,
            )


def draw_ramp_lane(
    ax: Axes,
    track: CueTrack,
    t_now: float,
    window_sec: float,
) -> None:
    """Render a ramp track: diagonal line shaped by easing."""
    color = track_color(track.name)
    events = spanning_events_in_window(
        track.events or [], t_now, window_sec,
        start_key="t_start", end_key="t_end",
    )
    for ev in events:
        t_start = float(ev.get("t_start", 0.0))
        t_end = float(ev.get("t_end", t_start))
        v_from = float(ev.get("from", 0.0))
        v_to = float(ev.get("to", 1.0))
        shape = str(ev.get("shape", "linear"))

        if t_end <= t_start:
            continue
        n = 24
        xs = np.linspace(t_start, t_end, n)
        ys = []
        for x in xs:
            frac = (x - t_start) / (t_end - t_start)
            ys.append(v_from + (v_to - v_from) * ease(shape, frac))

        if t_start <= t_now <= t_end:
            alpha = 0.95
            lw = 2.5
        elif t_end < t_now <= t_end + 0.3:
            alpha = 0.9 - (t_now - t_end) / 0.3 * 0.5
            lw = 2.0
        else:
            alpha = 0.4
            lw = 1.5
        ax.plot(xs, ys, color=color, alpha=alpha, linewidth=lw)


def draw_continuous_lane(
    ax: Axes,
    track: CueTrack,
    t_now: float,
    window_sec: float,
) -> None:
    """Render a continuous track: line + fill clipped to the window."""
    values = track.values or []
    hop = float(track.hop_sec or 0.0)
    if not values or hop <= 0:
        return
    color = track_color(track.name)
    x0, x1 = ax.get_xlim()
    i_lo = max(0, int(np.floor(x0 / hop)))
    i_hi = min(len(values), int(np.ceil(x1 / hop)) + 1)
    if i_hi <= i_lo:
        return
    xs = np.arange(i_lo, i_hi) * hop
    ys = np.asarray(values[i_lo:i_hi], dtype=float)

    lo = float(min(values))
    hi = float(max(values))
    if hi - lo > 1e-9:
        ys_norm = (ys - lo) / (hi - lo)
    else:
        ys_norm = np.full_like(ys, 0.5)

    ax.fill_between(xs, 0.0, ys_norm, color=color, alpha=0.4, linewidth=0)
    ax.plot(xs, ys_norm, color=color, alpha=0.9, linewidth=1.5)


def format_timecode(t_sec: float) -> str:
    """Decimal timecode HH:MM:SS.s."""
    if t_sec < 0:
        t_sec = 0.0
    h = int(t_sec // 3600)
    m = int((t_sec % 3600) // 60)
    s = t_sec % 60
    return f"{h:02d}:{m:02d}:{s:04.1f}"


def bpm_at(tempo_map: Sequence[dict], t_now: float) -> float | None:
    """Return the BPM in effect at t_now. Picks the latest entry whose
    `t` is <= t_now. None if the map is empty."""
    if not tempo_map:
        return None
    best: float | None = None
    for entry in tempo_map:
        if float(entry.get("t", 0.0)) <= t_now:
            best = float(entry.get("bpm", 0.0))
        else:
            break
    if best is None:
        return float(tempo_map[0].get("bpm", 0.0))
    return best


def _looks_like_section_track(track: CueTrack) -> bool:
    if track.type != "step":
        return False
    for ev in track.events or []:
        label = str(ev.get("label", "")).lower()
        if any(label.startswith(k) for k in _SECTION_KEYWORDS):
            return True
    return False


def current_section_label(cuesheet: CueSheet, t_now: float) -> str | None:
    """Walk every step track until we find one with section-like labels;
    return its currently active label at t_now."""
    for track in cuesheet.tracks:
        if not _looks_like_section_track(track):
            continue
        active: str | None = None
        for ev in track.events or []:
            if float(ev.get("t", 0.0)) <= t_now:
                active = str(ev.get("label", ""))
            else:
                break
        return active
    return None


def draw_header(
    ax: Axes,
    cuesheet: CueSheet,
    t_now: float,
) -> None:
    """Draw the header strip on a dedicated Axes."""
    ax.set_facecolor((0.05, 0.05, 0.05))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    tc = format_timecode(t_now)
    ax.text(0.02, 0.5, tc, color="white", fontsize=18,
            family="monospace", va="center", ha="left")

    bpm = bpm_at(cuesheet.tempo_map or [], t_now)
    if bpm is not None and bpm > 0:
        ax.text(0.32, 0.5, f"BPM {bpm:.0f}", color="white",
                fontsize=18, family="monospace", va="center", ha="left")

    section = current_section_label(cuesheet, t_now)
    if section:
        ax.text(0.62, 0.5, section.upper(),
                color=section_palette(section), fontsize=18,
                weight="bold", va="center", ha="left")
