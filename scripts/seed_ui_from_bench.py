"""One-shot: import the existing benchmark analysis into the Web UI's
artifact tree so the MVP acceptance test doesn't require a fresh ~10 min
analyze run.

Reads from MusicTests/bench_cache/runs/<hash>/ and writes to
~/.musicue/songs/<source_sha>/analyses/<hash>/...
"""
from __future__ import annotations

import shutil
from pathlib import Path

from musicue.analysis.peaks import write_peaks
from musicue.ui.storage import UIStorage, sha256_of_file

SOURCE = Path("D:/MusiCue/MusicTests/Ambrosia_2191891511 - Siaynoq.m4a")
EXISTING_RUN = Path("D:/MusiCue/MusicTests/bench_cache/runs/b2d65b86a271")
TITLE = "Ambrosia - Siaynoq"


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"source missing: {SOURCE}")
    if not EXISTING_RUN.exists():
        raise SystemExit(f"existing run missing: {EXISTING_RUN}")
    analysis_json = EXISTING_RUN / "analysis.json"
    if not analysis_json.exists():
        raise SystemExit(f"analysis.json missing in {EXISTING_RUN}")

    storage = UIStorage(Path.home() / ".musicue")
    print(f"sha-ing source ({SOURCE.stat().st_size} bytes)...")
    rec = storage.register_source(SOURCE, title=TITLE)
    print(f"  song_id = {rec.id[:16]}...")

    analysis_id = EXISTING_RUN.name
    target = storage.analysis_dir(rec.id, analysis_id)
    target.mkdir(parents=True, exist_ok=True)

    print("copying analysis.json...")
    shutil.copy2(analysis_json, target / "analysis.json")

    print("generating mix peaks...")
    write_peaks(rec.source_path, target / "peaks.mix.json")

    stems_dir_src = EXISTING_RUN / "stems" / "htdemucs_ft" / SOURCE.stem
    if stems_dir_src.exists():
        for stem_name in ("drums", "bass", "vocals", "other"):
            stem_path = stems_dir_src / f"{stem_name}.wav"
            if stem_path.exists():
                print(f"generating {stem_name} peaks...")
                write_peaks(stem_path, target / f"peaks.{stem_name}.json")
            else:
                print(f"  skip {stem_name} (not found)")
    else:
        print(f"stems dir not found at {stems_dir_src}; skipping per-stem peaks")

    print()
    print(f"DONE. UI artifact tree at: {storage.song_dir(rec.id)}")
    print(f"  song_id   = {rec.id}")
    print(f"  analysis  = {analysis_id}")
    print(f"  open in browser: http://127.0.0.1:8765/editor/{rec.id}/{analysis_id}")


if __name__ == "__main__":
    main()
