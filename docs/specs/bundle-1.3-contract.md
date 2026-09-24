# MusiCue → CedarToy bundle contract, schema 1.3 (additive over 1.2)

All existing 1.2 fields keep their meaning. New/extended:

## beats[*] (already emitted by MusiCue from analysis.beats; CedarToy must now read them, all optional)
- phrase_id: int|null, phrase_position: int|null (bar index within phrase, 0-based), phrase_length: int|null (bars), is_fill: bool

## controls: { name: {"hop_sec": float, "values": [float 0..1]} }   (new, optional; missing key = absent)
Dense, already-normalized (0..1) control curves. MusiCue measures; CedarToy shapes. Names in 1.3:
- "energy_fast"  : mix loudness with a short window (~100 ms RMS → dBFS), percentile-normalized 5th→0 / 95th→1 over non-silent frames (> -60 dBFS). Centered, not causal.
- "brightness"   : spectral centroid of the mix, log-frequency, percentile-normalized (5/95) over non-silent frames, lightly smoothed (~150 ms centered). Silent frames → 0.
- "build"        : anticipation ramp. For each section boundary tb where the NEXT section's energy_rank exceeds the current section's by >= 0.2 (or next >= 0.75 and next > current): ramp over window [max(current_section.start, tb - 8 bars (use local beat grid; fallback 8*4*60/bpm s)), tb): value = ((t - t0)/(tb - t0))^2, drops to 0 at tb. 0 elsewhere. If the build window overlaps, take max.
- "onset_density": drum onsets (union of kick/snare/hat) per beat, smoothed over ~1 bar centered, normalized by 95th percentile, clip 0..1.

## sections[*]: unchanged (energy_rank 0..1 loudness rank).

## stems_energy: unchanged from 1.2 (drums, bass, vocals, other; per-stem percentile-normalized loudness 0..1).

Consumers MUST tolerate: controls missing entirely, any control missing, empty values, hop_sec <= 0, beats lacking phrase fields.
