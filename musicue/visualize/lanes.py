"""Per-type lane renderers + fire-panel brightness."""
from __future__ import annotations

from musicue.schemas import CueTrack
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
