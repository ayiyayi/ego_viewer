#!/usr/bin/env python3
"""Entry point executed by the dedicated SAM3 interpreter."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam3_hand_tracking.sam3_point import main

if __name__ == "__main__":
    main()
