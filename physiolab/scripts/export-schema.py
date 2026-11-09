"""Run from physiolab: uv run --directory backend python ../scripts/export-schema.py"""
import json
from pathlib import Path
from physiolab.api import app
root = Path(__file__).resolve().parents[1]
(root / 'docs/openapi.json').write_text(json.dumps(app.openapi(), indent=2) + '\n')
