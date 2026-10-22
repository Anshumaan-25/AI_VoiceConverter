"""Reproduce upstream fixes without committing models or recordings."""
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parent
BASE = "730930d337d171479eadf305f96cbed4bb393e77"

def run(*args, cwd=ROOT):
    return subprocess.run(list(args), cwd=cwd, check=True)

def main():
    svc = ROOT / "so-vits-svc"
    if not svc.exists():
        run("git", "clone", "https://github.com/svc-develop-team/so-vits-svc.git", str(svc))
        run("git", "checkout", BASE, cwd=svc)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=svc, text=True).strip()
    if head != BASE:
        raise RuntimeError("Upstream revision differs; refusing to overwrite checkout")
    patch = ROOT / "patches" / "so-vits-svc.patch"
    already = subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=svc, capture_output=True)
    if already.returncode == 0:
        print("Upstream patch is already installed")
        return
    run("git", "apply", "--check", str(patch), cwd=svc)
    run("git", "apply", str(patch), cwd=svc)

if __name__ == "__main__":
    main()
