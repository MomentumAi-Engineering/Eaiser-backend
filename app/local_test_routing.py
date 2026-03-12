
import asyncio
import os
import sys

sys.path.append(os.getcwd())

from routes.email_webhook import handle_inbound_email
from unittest.mock import MagicMock

async def test_routing():
    # Mocking the FastAPI Request object
    request = MagicMock()
    request.json = MagicMock(return_value=asyncio.Future())
    request.json.return_value.set_result({
        "From": "chrishabh1000@gmail.com",
        "Subject": "RE: EAiSER Alert – (ID: DDT62ZZ)",
        "TextBody": "LOCAL TEST: This is a test reply from the authority.",
        "HtmlBody": "<div>LOCAL TEST: This is a test reply from the authority.</div>"
    })
    
    print("Starting local routing test...")
    response = await handle_inbound_email(request)
    print(f"RESPONSE: {response}")

if __name__ == "__main__":
    asyncio.run(test_routing())
