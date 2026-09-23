import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python scripts/seed_db.py`
from autopilot.db import connect
connect().close(); print("Database initialized")
