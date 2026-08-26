"""Run the version selected by current_presidential_model.json."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_active_presidential_model_v32 import main


if __name__ == "__main__":
    main()
