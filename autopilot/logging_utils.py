import json, logging, time
from pathlib import Path
from .settings import get_settings

class JsonlHandler(logging.Handler):
    def emit(self, record):
        try:
            p = Path(get_settings().log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = {"timestamp": time.time(), "level": record.levelname, "message": record.getMessage()}
            if hasattr(record, "event_data"):
                payload.update(record.event_data)
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

def get_logger(name="autopilot"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler())
        logger.addHandler(JsonlHandler())
    return logger

def event(logger, name: str, **data):
    logger.info(name, extra={"event_data": {"event": name, **data}})
