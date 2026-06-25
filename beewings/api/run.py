"""Entry point for the ``beewings-api`` console script.

Starts a uvicorn server. Host/port come from BEEWINGS_API_HOST / BEEWINGS_API_PORT.
"""
from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    host = os.environ.get("BEEWINGS_API_HOST", "0.0.0.0")
    port = int(os.environ.get("BEEWINGS_API_PORT", "8000"))
    uvicorn.run("beewings.api.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
