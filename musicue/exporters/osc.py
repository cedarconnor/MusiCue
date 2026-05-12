"""OSC bundle exporter for MusiCue cuesheets.

Writes a JSON bundle of timestamped OSC messages plus a sibling ``play_osc.py``
player script. The bundle is *not* binary OSC -- it is a serialized intermediate
that downstream tools (or the player script) can deliver via UDP at playback
time.

Format
------
The bundle is a JSON object::

    {
      "grammar": str,
      "duration_sec": float,
      "target_host": str,
      "target_port": int,
      "message_count": int,
      "messages": [
        {"t": float, "address": "/musicue/<track>[/<sub>]", "args": [...]},
        ...
      ]
    }

Messages are sorted by ``t`` (seconds from playback start). Address pattern is
``/musicue/<track-name>`` for primary events, with ``/on``, ``/off``,
``/label``, ``/ramp_start``, ``/ramp_end`` sub-addresses where appropriate.

Track-type mapping
------------------
- ``impulse``    -> single message at ``t`` with ``[strength]``.
- ``envelope``   -> ``/on`` at ``t_start`` with ``[strength]`` and ``/off`` at
                    ``t_end`` with ``[0.0]``.
- ``step``       -> ``/label`` (string label) and main address (numeric value)
                    both at ``t``.
- ``ramp``       -> ``/ramp_start`` at ``t_start`` with ``[0.0]`` and
                    ``/ramp_end`` at ``t_end`` with ``[1.0]``.
- ``continuous`` -> downsampled to ~10 Hz; one message per sample at the main
                    address with ``[value]``.

Playback
--------
The companion ``play_osc.py`` script (written next to the bundle) requires
``python-osc`` (``pip install python-osc``) and replays messages over UDP to
``target_host:target_port``. The bundle file itself is plain JSON and has no
runtime dependencies beyond stdlib.
"""

from __future__ import annotations

import json
from pathlib import Path

from musicue.exporters._common import non_empty_tracks
from musicue.schemas import CueSheet

_TARGET_HZ = 10.0  # downsample rate for continuous tracks

_PLAYER_SCRIPT = '''\
#!/usr/bin/env python3
"""Play a MusiCue OSC bundle. Requires python-osc: pip install python-osc"""
import json
import sys
import time

from pythonosc.udp_client import SimpleUDPClient

bundle_path = sys.argv[1] if len(sys.argv) > 1 else "cuesheet_osc.json"
bundle = json.loads(open(bundle_path).read())
client = SimpleUDPClient(bundle["target_host"], bundle["target_port"])
messages = bundle["messages"]
start_time = time.monotonic()
for msg in messages:
    target = start_time + msg["t"]
    now = time.monotonic()
    if target > now:
        time.sleep(target - now)
    client.send_message(msg["address"], msg["args"])
print("Playback complete.")
'''


def export(
    cuesheet: CueSheet,
    out_path: Path,
    host: str = "127.0.0.1",
    port: int = 9000,
    **opts,
) -> None:
    """Write the OSC JSON bundle and a sibling ``play_osc.py`` player script."""
    messages: list[dict] = []

    for track in non_empty_tracks(cuesheet.tracks):
        address = f"/musicue/{track.name}"

        if track.type == "impulse":
            for ev in track.events:
                t = float(ev["t"])
                strength = float(ev.get("strength", 1.0))
                messages.append({"t": t, "address": address, "args": [strength]})

        elif track.type == "envelope":
            for ev in track.events:
                t_start = float(ev.get("t_start", 0.0))
                t_end = float(ev.get("t_end", t_start + 1.0))
                strength = float(ev.get("strength", 0.8))
                messages.append(
                    {"t": t_start, "address": f"{address}/on", "args": [strength]}
                )
                messages.append(
                    {"t": t_end, "address": f"{address}/off", "args": [0.0]}
                )

        elif track.type == "step":
            for ev in track.events:
                t = float(ev["t"])
                value = ev.get("value", 1)
                label = str(ev.get("label", ""))
                messages.append(
                    {"t": t, "address": f"{address}/label", "args": [label]}
                )
                messages.append(
                    {"t": t, "address": address, "args": [float(value)]}
                )

        elif track.type == "ramp":
            for ev in track.events:
                t_start = float(ev.get("t_start", 0.0))
                t_end = float(ev.get("t_end", t_start + 1.0))
                messages.append(
                    {"t": t_start, "address": f"{address}/ramp_start", "args": [0.0]}
                )
                messages.append(
                    {"t": t_end, "address": f"{address}/ramp_end", "args": [1.0]}
                )

        elif track.type == "continuous" and track.values and track.hop_sec:
            hop = track.hop_sec
            # Stride to keep _TARGET_HZ samples/sec from a 1/hop samples/sec source.
            # When source rate <= target rate (hop * _TARGET_HZ >= 1), emit every frame.
            step = max(1, int(round(1.0 / (_TARGET_HZ * hop))))
            for i in range(0, len(track.values), step):
                t = i * hop
                val = float(track.values[i])
                messages.append({"t": t, "address": address, "args": [val]})

    messages.sort(key=lambda m: m["t"])

    bundle = {
        "grammar": cuesheet.grammar,
        "duration_sec": cuesheet.duration_sec,
        "target_host": host,
        "target_port": port,
        "message_count": len(messages),
        "messages": messages,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2))

    player_path = out_path.parent / "play_osc.py"
    player_path.write_text(_PLAYER_SCRIPT)
