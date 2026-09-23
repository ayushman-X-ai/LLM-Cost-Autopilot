import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python scripts/inspect_logs.py`
from autopilot.settings import get_settings
p=Path(get_settings().log_path)
print(p.read_text(encoding="utf-8") if p.exists() else "No logs yet")
