import sys
from pathlib import Path

# stitch.py and build_database.py aren't part of an importable package, so
# make them importable by module name for the test suite.
ROOT_DIR = Path(__file__).resolve().parents[1]
PROTO_DIR = ROOT_DIR / "src" / "tests" / "proto"
DATABASE_DIR = ROOT_DIR / "src" / "database"

sys.path.insert(0, str(PROTO_DIR))
sys.path.insert(0, str(DATABASE_DIR))
