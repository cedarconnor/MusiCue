import os
import sys

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from musicue.cli import app

if __name__ == "__main__":
    app()
