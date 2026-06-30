#!/usr/bin/env python3
"""
Entry point — start the Call & Chat Summarizer HTTP server.

    python run.py                 # serve on the configured HOST:PORT (default :8077)
    PORT=9000 python run.py       # override via env
    python run.py --port 9000     # or via flag
"""
import argparse

import uvicorn

from summarizer.config import HOST, PORT


def main() -> None:
    ap = argparse.ArgumentParser(description="Call & Chat summarizer (Whisper + Gemma-4)")
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    uvicorn.run("summarizer.api:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
