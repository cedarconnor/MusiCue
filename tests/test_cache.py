import json

from musicue.cache import Cache, build_audio_cache_key


def test_cache_miss_returns_none(tmp_path):
    cache = Cache(tmp_path)
    assert cache.get("abc123", "analysis.json") is None


def test_cache_put_and_get(tmp_path):
    cache = Cache(tmp_path)
    src = tmp_path / "source.json"
    src.write_text('{"hello": "world"}')
    cache.put("abc123", "analysis.json", src)
    result = cache.get("abc123", "analysis.json")
    assert result is not None
    assert json.loads(result.read_text()) == {"hello": "world"}


def test_cache_key_changes_with_audio_content(tmp_path):
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    wav_a.write_bytes(b"\x00\x01\x02")
    wav_b.write_bytes(b"\x00\x01\x03")
    config = {"demucs_model": "htdemucs_ft", "demucs_version": "4.0.1"}
    assert build_audio_cache_key(wav_a, config) != build_audio_cache_key(wav_b, config)


def test_cache_key_changes_with_model_version(tmp_path):
    wav = tmp_path / "song.wav"
    wav.write_bytes(b"\x00\x01\x02")
    config_a = {"demucs_version": "4.0.1"}
    config_b = {"demucs_version": "4.0.2"}
    assert build_audio_cache_key(wav, config_a) != build_audio_cache_key(wav, config_b)


def test_cache_key_is_stable(tmp_path):
    wav = tmp_path / "song.wav"
    wav.write_bytes(b"\x00\x01\x02")
    config = {"demucs_version": "4.0.1"}
    assert build_audio_cache_key(wav, config) == build_audio_cache_key(wav, config)


def test_cache_put_bytes_and_get(tmp_path):
    cache = Cache(tmp_path)
    cache.put_bytes("key1", "stems/drums.wav", b"\x01\x02\x03")
    result = cache.get("key1", "stems/drums.wav")
    assert result is not None
    assert result.read_bytes() == b"\x01\x02\x03"
