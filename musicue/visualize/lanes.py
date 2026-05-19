"""Per-type lane renderers + fire-panel brightness."""
from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from musicue.schemas import CueTrack
from musicue.visualize.colors import track_color
from musicue.visualize.cue_video import events_in_window, spanning_events_in_window
from musicue.visualize.envelopes import ease, sample_adsr

_STEP_FLASH_SEC = 0.25


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
