"""Compare chosen recordings; shorter audio alone does not establish quality."""
import argparse
import json
from analyze_baseline import analyze_wav

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    args = parser.parse_args()
    try:
        print(json.dumps({"baseline": analyze_wav(args.baseline), "candidate": analyze_wav(args.candidate)}, indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Comparison failed: {exc}")
