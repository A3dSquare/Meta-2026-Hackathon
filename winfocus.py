"""Keep Geometry Dash the active window (Windows only; does nothing elsewhere).

Key presses (pydirectinput) go to whichever window is in FRONT. If the AI's
own overlay window, the console, or anything else is active, SPACE never
reaches GD: the cube doesn't jump, and the AI learns that pressing is
useless. These helpers find the GD window by its title and bring it back.
"""
from __future__ import annotations

import sys
import time

_IS_WIN = sys.platform == "win32"
_hwnd_cache = {"hwnd": None, "t": 0.0}


def _user32():
    import ctypes
    return ctypes.windll.user32


def find_gd_window(title_prefix: str = "geometry dash"):
    """Handle of the visible window whose title starts with 'Geometry Dash' (or None)."""
    if not _IS_WIN:
        return None
    now = time.time()
    h = _hwnd_cache["hwnd"]
    if h and now - _hwnd_cache["t"] < 30:
        try:
            if _user32().IsWindow(h):
                return h
        except Exception:
            pass
    try:
        import ctypes
        user32 = _user32()
        found = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        def cb(hwnd, _lparam):
            try:
                if user32.IsWindowVisible(hwnd):
                    n = user32.GetWindowTextLengthW(hwnd)
                    if n:
                        buf = ctypes.create_unicode_buffer(n + 1)
                        user32.GetWindowTextW(hwnd, buf, n + 1)
                        if buf.value.strip().lower().startswith(title_prefix):
                            found.append(hwnd)
            except Exception:
                pass
            return True

        user32.EnumWindows(cb, 0)
        h = found[0] if found else None
    except Exception:
        h = None
    _hwnd_cache.update(hwnd=h, t=now)
    return h


def gd_is_foreground() -> bool | None:
    """True / False, or None when it can't tell (not Windows, GD window not found)."""
    if not _IS_WIN:
        return None
    h = find_gd_window()
    if not h:
        return None
    try:
        return int(_user32().GetForegroundWindow() or 0) == int(h)
    except Exception:
        return None


def focus_gd() -> bool:
    """Bring the GD window to the front. Returns True if GD is now the active window."""
    if not _IS_WIN:
        return False
    h = find_gd_window()
    if not h:
        return False
    try:
        import ctypes
        user32 = _user32()
        kernel32 = ctypes.windll.kernel32
        if int(user32.GetForegroundWindow() or 0) == int(h):
            return True
        if user32.IsIconic(h):
            user32.ShowWindow(h, 9)                       # SW_RESTORE (it was minimised)
        # Windows only lets the program that owns the front window hand over focus:
        # briefly attach to that window's input thread so the switch is allowed
        fg = user32.GetForegroundWindow()
        me = kernel32.GetCurrentThreadId()
        other = user32.GetWindowThreadProcessId(fg, None) if fg else 0
        if other and other != me:
            user32.AttachThreadInput(other, me, True)
        try:
            user32.BringWindowToTop(h)
            user32.SetForegroundWindow(h)
        finally:
            if other and other != me:
                user32.AttachThreadInput(other, me, False)
        time.sleep(0.05)
        return int(user32.GetForegroundWindow() or 0) == int(h)
    except Exception:
        return False
