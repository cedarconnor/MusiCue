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
