"""Download only pinned content encoder/pitch assets with authoritative hashes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download():
    manifest = json.loads((ROOT/'rvc_manifest.json').read_text(encoding='utf-8'))
    for asset in manifest['assets'].values():
        target = ROOT/'vendor'/'rvc'/asset['file']
        if target.exists():
            if target.stat().st_size != asset['bytes'] or sha256(target) != asset['sha256']:
                raise RuntimeError(f'Existing asset differs from pinned hash: {target}')
            print('Verified:', target.name)
            continue
        if shutil.disk_usage(ROOT).free < asset['bytes']+10*1024**3:
            raise RuntimeError('Download would violate the 10 GiB free-space reserve')
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=target.parent, suffix='.download')
        os.close(fd)
        staged = Path(name)
        try:
            # Reuse the existing ContentVec asset only if it is byte-identical to HuBERT.
            candidates = [ROOT/'so-vits-svc'/'pretrain'/p for p in ('checkpoint_best_legacy_500.pt','rmvpe.pt')]
            cached = next((p for p in candidates if p.is_file() and p.stat().st_size == asset['bytes']
                           and sha256(p) == asset['sha256']), None)
            if cached:
                shutil.copyfile(cached, staged)
            else:
                url = ('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/'
                       +manifest['asset_revision']+'/'+asset['remote'])
                with urllib.request.urlopen(url, timeout=60) as response, staged.open('wb') as stream:
                    size = 0
                    while block := response.read(1024*1024):
                        size += len(block)
                        if size > asset['bytes']:
                            raise RuntimeError('Asset exceeds declared size')
                        stream.write(block)
            if staged.stat().st_size != asset['bytes'] or sha256(staged) != asset['sha256']:
                raise RuntimeError('Downloaded asset failed size/SHA256 verification')
            os.replace(staged, target)
            print('Verified:', target.name)
        finally:
            staged.unlink(missing_ok=True)


if __name__ == '__main__':
    download()
