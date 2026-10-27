"""Subprocess fixture exercising containment, crashes and GPU cleanup."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

directory, mode = Path(sys.argv[1]), sys.argv[2]
while not (directory/'go').exists():
    time.sleep(.01)
(directory/'worker.pid').write_text(str(os.getpid()))
if mode in ('tree', 'crash'):
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
    (directory/'child.pid').write_text(str(child.pid))
if mode == 'gpu':
    import torch
    allocation = torch.ones(32*1024*1024, device='cuda')
    torch.cuda.synchronize()
    (directory/'allocated.json').write_text(json.dumps(dict(bytes=torch.cuda.memory_allocated())))
if mode == 'crash':
    raise SystemExit(7)
if mode != 'load':
    (directory/'state.json').write_text(json.dumps(dict(phase='chunk', chunk=0)))
while True:
    time.sleep(.05)
