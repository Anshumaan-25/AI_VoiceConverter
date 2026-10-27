"""A disposable controller used to verify kernel cleanup on controller death."""
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speech.contracts import SpeechOptions
from speech.supervisor import run_worker

directory = Path(sys.argv[1])
(directory/'controller.pid').write_text(str(os.getpid()))
run_worker([sys.executable, '-B', str(Path(__file__).with_name('worker_fixture.py')), str(directory), 'tree'],
           directory, SpeechOptions(), time.monotonic()+60)
