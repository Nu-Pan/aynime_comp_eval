import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib

    matplotlib.use("Agg", force=True)
except Exception:
    pass

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
