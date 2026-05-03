import json

from musicue.exporters.osc import export


def test_osc_export_creates_json_bundle(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_osc.json"
    export(full_cuesheet, out)
    assert out.exists()


def test_osc_bundle_is_valid_json(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_osc.json"
    export(full_cuesheet, out)
    data = json.loads(out.read_text())
    assert isinstance(data, dict)
    assert "messages" in data


def test_osc_bundle_message_fields(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_osc.json"
    export(full_cuesheet, out)
    data = json.loads(out.read_text())
    messages = data["messages"]
    assert len(messages) > 0
    msg = messages[0]
    assert "t" in msg
    assert "address" in msg
    assert "args" in msg


def test_osc_bundle_address_pattern(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_osc.json"
    export(full_cuesheet, out)
    data = json.loads(out.read_text())
    addresses = [m["address"] for m in data["messages"]]
    assert any(addr.startswith("/musicue/") for addr in addresses)


def test_osc_bundle_kick_events_present(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_osc.json"
    export(full_cuesheet, out)
    data = json.loads(out.read_text())
    kick_msgs = [m for m in data["messages"] if "kick" in m["address"]]
    assert len(kick_msgs) == 3


def test_osc_player_script_created(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_osc.json"
    export(full_cuesheet, out)
    player = tmp_path / "play_osc.py"
    assert player.exists()
    content = player.read_text()
    assert "pythonosc" in content or "python-osc" in content or "osc_message" in content.lower()


def test_osc_export_continuous_downsamples_to_10hz(tmp_path):
    """Continuous tracks finer than 10 Hz should NOT emit at source rate."""
    from musicue.exporters.osc import export
    from musicue.schemas import CueSheet, CueTrack

    cs = CueSheet(
        source_sha256="x",
        grammar="g",
        duration_sec=10.0,
        tempo_map=[],
        tracks=[
            CueTrack(
                name="energy",
                type="continuous",
                timescale="macro",
                hop_sec=0.01,
                values=[float(i) / 1000 - 0.5 for i in range(1000)],
            ),
        ],
    )
    out = tmp_path / "cuesheet_osc.json"
    export(cs, out)
    data = json.loads(out.read_text())
    energy_msgs = [m for m in data["messages"] if m["address"] == "/musicue/energy"]
    # 10 seconds at 10 Hz target should give ~100 messages, not 1000.
    assert 80 <= len(energy_msgs) <= 120, f"Expected ~100 messages, got {len(energy_msgs)}"
