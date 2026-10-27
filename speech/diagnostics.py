import json
import os
from pathlib import Path
import time


def fingerprint(path):
    """Optional checkpoint provenance; streaming avoids loading weights twice."""
    import hashlib
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return dict(file=path.name, bytes=path.stat().st_size, sha256=digest.hexdigest())


def atomic_json(path: Path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    # Windows readers/antivirus can briefly deny replacing an open destination.
    for attempt in range(50):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(.02)


class Trace:
    def __init__(self, directory: Path):
        self.directory = directory
        self.phase = 'load'
        self.chunk = -1

    def event(self, stage, **details):
        import sys
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            details['cuda'] = dict(allocated=torch.cuda.memory_allocated(),
                                  reserved=torch.cuda.memory_reserved(),
                                  peak_allocated=torch.cuda.max_memory_allocated(),
                                  peak_reserved=torch.cuda.max_memory_reserved())
        record = dict(stage=stage, phase=self.phase, chunk=self.chunk,
                      monotonic=time.monotonic(), pid=os.getpid(), **details)
        with (self.directory/'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        atomic_json(self.directory/'state.json', record)

    def begin_chunk(self, index):
        self.phase = 'chunk'
        self.chunk = index
        self.event('chunk_start')
