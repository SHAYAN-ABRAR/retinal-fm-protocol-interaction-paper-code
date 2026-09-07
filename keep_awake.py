"""Hold Windows awake for exactly as long as one training process lives.

Twice now a multi-hour run has been lost to the same failure: Windows Modern
Standby suspends the machine mid-epoch, and the run does not come back. It
deadlocks instead -- the main process spins on one core, the dataloader workers
stop accumulating CPU entirely, VRAM stays allocated and the GPU sits at 0%.
The run never fails, never exits, and never writes another line, so nothing
downstream notices; the APTOS RETFound seed-1 run sat like that for four hours
and forty-six minutes.

This asserts ES_SYSTEM_REQUIRED for the lifetime of a nominated process. It
changes **no** system setting: the request is held by this process, is visible
in `powercfg /requests`, and disappears when this process exits, whether that
is cleanly, by being killed, or by crashing. Nothing needs to be undone
afterwards, which is the point -- a changed sleep timeout would outlive the
experiment and quietly alter the machine.

The display is deliberately allowed to sleep; only the system is held.

Usage:
    python keep_awake.py --pid 1234
    python keep_awake.py --pid 1234 --poll 30
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040


def alive(pid: int) -> bool:
    """Is the process still running?

    OpenProcess with PROCESS_QUERY_LIMITED_INFORMATION succeeds for a process
    that has exited but still has an open handle somewhere, so the exit code is
    checked too: STILL_ACTIVE (259) means running.
    """
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == 259
    finally:
        kernel32.CloseHandle(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True,
                        help="process to stay awake for")
    parser.add_argument("--poll", type=float, default=30.0,
                        help="seconds between liveness checks")
    arguments = parser.parse_args()

    if not sys.platform.startswith("win"):
        print("not Windows; nothing to do")
        return 0

    if not alive(arguments.pid):
        print(f"pid {arguments.pid} is not running; refusing to hold the "
              f"machine awake for nothing")
        return 1

    set_state = ctypes.windll.kernel32.SetThreadExecutionState
    # Away mode keeps the system running with the display off. It is not
    # granted in every power configuration, so fall back to the plain system
    # request rather than silently holding nothing.
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
    if not set_state(flags):
        flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        if not set_state(flags):
            print("SetThreadExecutionState refused; the machine may still sleep")
            return 1
        granted = "system required (away mode refused)"
    else:
        granted = "system required + away mode"

    started = time.time()
    print(f"holding {granted} while pid {arguments.pid} runs; "
          f"polling every {arguments.poll:g}s", flush=True)
    try:
        while alive(arguments.pid):
            time.sleep(arguments.poll)
    except KeyboardInterrupt:
        pass
    finally:
        # Release explicitly. Exiting would release it anyway, but being
        # explicit means `powercfg /requests` is clean the moment this returns.
        set_state(ES_CONTINUOUS)

    hours = (time.time() - started) / 3600.0
    print(f"pid {arguments.pid} exited after {hours:.2f} h; request released",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
