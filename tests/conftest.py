import sys
from pathlib import Path

# stitch.py lives under src/tests/proto and isn't part of an importable
# package, so make it importable by module name for the test suite.
PROTO_DIR = Path(__file__).resolve().parents[1] / "src" / "tests" / "proto"
sys.path.insert(0, str(PROTO_DIR))
