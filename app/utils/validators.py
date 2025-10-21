import re
from typing import Optional

EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
ZIP_REGEX = re.compile(r"^\d{5}(-\d{4})?$")

def validate_email(email: Optional[str]) -> bool:
    if not email:
        return False
    return bool(EMAIL_REGEX.match(email))

def validate_zip_code(zip_code: Optional[str]) -> bool:
    if not zip_code:
        return False
    return bool(ZIP_REGEX.match(zip_code))