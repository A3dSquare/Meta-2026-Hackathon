"""Short per-thread sleeps without changing Windows' global timer resolution."""
import ctypes
import os
import time


class CaptureClock:
    def __init__(self):
        self._handle = None
        self._kernel = None
        if os.name == 'nt':
            try:
                from ctypes import wintypes as w
                kernel=ctypes.WinDLL('kernel32',use_last_error=True)
                kernel.CreateWaitableTimerExW.argtypes=[ctypes.c_void_p,w.LPCWSTR,w.DWORD,w.DWORD]
                kernel.CreateWaitableTimerExW.restype=w.HANDLE
                kernel.SetWaitableTimer.argtypes=[w.HANDLE,ctypes.POINTER(ctypes.c_longlong),w.LONG,
                                                 ctypes.c_void_p,ctypes.c_void_p,w.BOOL]
                kernel.SetWaitableTimer.restype=w.BOOL
                kernel.WaitForSingleObject.argtypes=[w.HANDLE,w.DWORD]
                kernel.WaitForSingleObject.restype=w.DWORD
                kernel.CloseHandle.argtypes=[w.HANDLE];kernel.CloseHandle.restype=w.BOOL
                # CREATE_WAITABLE_TIMER_HIGH_RESOLUTION; TIMER_MODIFY_STATE | SYNCHRONIZE.
                # https://learn.microsoft.com/windows/win32/api/synchapi/nf-synchapi-createwaitabletimerexw
                self._handle=kernel.CreateWaitableTimerExW(None,None,0x2,0x100002)
                self._kernel=kernel
            except (AttributeError,OSError):
                pass

    @property
    def high_resolution(self):
        return bool(self._handle)

    def sleep(self,seconds):
        if seconds <= 0:return
        if self._handle:
            due=ctypes.c_longlong(-max(1,round(seconds*10_000_000)))
            if self._kernel.SetWaitableTimer(self._handle,ctypes.byref(due),0,None,None,False):
                # Keep failure bounded; the capture caller checks its real deadline.
                self._kernel.WaitForSingleObject(self._handle,max(100,int(seconds*1000)+100))
                return
        time.sleep(seconds)

    def close(self):
        if self._handle:
            self._kernel.CloseHandle(self._handle)
            self._handle=None

    def __del__(self):
        self.close()
