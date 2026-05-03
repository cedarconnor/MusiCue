import json

from musicue.exporters.unreal import export


def test_unreal_export_creates_json(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_unreal.json"
    export(full_cuesheet, out)
    assert out.exists()


def test_unreal_json_is_valid(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_unreal.json"
    export(full_cuesheet, out)
    data = json.loads(out.read_text())
    assert isinstance(data, dict)


def test_unreal_json_has_tracks(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_unreal.json"
    export(full_cuesheet, out)
    data = json.loads(out.read_text())
    assert "tracks" in data
    assert len(data["tracks"]) >= 1


def test_unreal_json_track_structure(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_unreal.json"
    export(full_cuesheet, out)
    data = json.loads(out.read_text())
    track = data["tracks"][0]
    assert "name" in track
    assert "type" in track
    assert "events" in track or "channel" in track or "keys" in track


def test_unreal_json_schema_version(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_unreal.json"
    export(full_cuesheet, out)
    data = json.loads(out.read_text())
    assert "schema_version" in data
    assert data["schema_version"] == "1.0"


def test_unreal_json_float_tracks_for_continuous(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_unreal.json"
    export(full_cuesheet, out)
    data = json.loads(out.read_text())
    float_tracks = [t for t in data["tracks"] if t["type"] == "float_curve"]
    assert len(float_tracks) >= 1  # energy track
