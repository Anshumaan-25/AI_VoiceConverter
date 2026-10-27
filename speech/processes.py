"""Kill-on-close Windows job: descendants die even if the controller crashes."""
import ctypes
from ctypes import wintypes
import os
import subprocess


class ProcessTree:
    def __init__(self, process):
        import psutil
        self.process = process
        self.handle = None
        self.members = {process.pid: psutil.Process(process.pid)}
        if os.name != 'nt':
            return
        class Basic(ctypes.Structure):
            _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                        ('flags', wintypes.DWORD), ('min_working', ctypes.c_size_t),
                        ('max_working', ctypes.c_size_t), ('active', wintypes.DWORD),
                        ('affinity', ctypes.c_size_t), ('priority', wintypes.DWORD),
                        ('scheduling', wintypes.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]
        class Extended(ctypes.Structure):
            _fields_ = [('basic', Basic), ('io', IO), ('process_memory', ctypes.c_size_t),
                        ('job_memory', ctypes.c_size_t), ('peak_process', ctypes.c_size_t),
                        ('peak_job', ctypes.c_size_t)]
        k = ctypes.WinDLL('kernel32', use_last_error=True)
        k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k.CreateJobObjectW.restype = wintypes.HANDLE
        k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel = k
        handle = k.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            k.CloseHandle(handle)
            raise ctypes.WinError(ctypes.get_last_error())
        if not k.AssignProcessToJobObject(handle, wintypes.HANDLE(int(process._handle))):
            k.CloseHandle(handle)
            raise ctypes.WinError(ctypes.get_last_error())
        self.handle = handle

    def resume(self):
        """Resume the primary thread only after the suspended process joins its job.

        Python's Windows venv launcher can spawn another interpreter immediately;
        suspension prevents that child from escaping before job assignment.
        """
        if os.name != 'nt':
            return
        import psutil
        k = self.kernel
        k.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.OpenThread.restype = wintypes.HANDLE
        k.ResumeThread.argtypes = [wintypes.HANDLE]
        k.ResumeThread.restype = wintypes.DWORD
        thread_id = psutil.Process(self.process.pid).threads()[0].id
        thread = k.OpenThread(0x0002, False, thread_id)
        if not thread:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if k.ResumeThread(thread) == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            k.CloseHandle(thread)

    def close(self):
        import psutil
        try:
            for child in psutil.Process(self.process.pid).children(recursive=True):
                self.members[child.pid] = child
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        if self.handle:
            if not self.kernel.CloseHandle(self.handle):
                raise RuntimeError('Could not close worker job; termination is unconfirmed')
            self.handle = None
        elif os.name == 'nt':
            if self.process.poll() is None:
                subprocess.run(['taskkill', '/PID', str(self.process.pid), '/T', '/F'],
                               capture_output=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            import signal
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('Worker termination could not be confirmed; no fallback may start') from exc
        _, alive = psutil.wait_procs(list(self.members.values()), timeout=10)
        if alive:
            raise RuntimeError('Worker descendants are still alive; no fallback may start')
