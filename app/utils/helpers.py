import datetime as _dt
import random as _rand
from typing import List, Dict, Any, Optional


def generate_report_id(prefix: str = "eaiser") -> str:
    now = _dt.datetime.utcnow()
    report_number = str(int(now.strftime("%Y%m%d%H%M%S")) % 1000000).zfill(6)
    return f"{prefix}-{now.year}-{report_number}"


def get_authorities_by_zip_code(zip_code: Optional[str]) -> List[Dict[str, Any]]:
    # Minimal fallback: return a generic city department email.
    if not zip_code:
        zip_code = "00000"
    return [
        {
            "name": "City Department",
            "email": "chrishabh1000@gmail.com",
            "type": "general",
            "zip_code": zip_code,
        }
    ]