import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get("DTAF_DATA_DIR", ROOT/"data")).resolve()
