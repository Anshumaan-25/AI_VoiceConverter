"""Restore the exact official RVC revision and the reviewed inference patch."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def restore():
    manifest = json.loads((ROOT/'rvc_manifest.json').read_text(encoding='utf-8'))
    target = ROOT/'vendor'/'rvc'
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout',
                        manifest['repository'], str(target)], check=True)
        subprocess.run(['git', '-C', str(target), 'checkout', '--detach', manifest['commit']], check=True)
    head = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
    if head != manifest['commit']:
        raise RuntimeError('Existing RVC checkout has a different revision; preserve it and select a new directory')
    patch = (ROOT/manifest['patch']).read_bytes()
    changed = subprocess.check_output(['git', '-C', str(target), 'diff', '--binary', 'HEAD'])
    if changed:
        if changed != patch:
            raise RuntimeError('Existing RVC changes differ from the reviewed patch; refusing to overwrite')
    else:
        subprocess.run(['git', '-C', str(target), 'apply', '--check', '-'], input=patch, check=True)
        subprocess.run(['git', '-C', str(target), 'apply', '-'], input=patch, check=True)
    print('RVC source and patch verified:', head)


if __name__ == '__main__':
    restore()
