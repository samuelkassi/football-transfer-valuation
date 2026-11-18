import sys
from pathlib import Path

# src/ modules import each other by bare name (e.g. `from models_v2 import ...`)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
