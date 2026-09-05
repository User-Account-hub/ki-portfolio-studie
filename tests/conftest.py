import sys
from pathlib import Path

# Erlaubt `from src...` unabhängig davon, von wo pytest aufgerufen wird.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
