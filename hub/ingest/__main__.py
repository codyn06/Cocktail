import argparse
import logging

import uvicorn

import protocol as p

from .server import create_app

ap = argparse.ArgumentParser(description="Cocktail hub ingest server")
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--port", type=int, default=p.DEFAULT_PORT)
a = ap.parse_args()
logging.basicConfig(level=logging.INFO)
uvicorn.run(create_app(), host=a.host, port=a.port)
