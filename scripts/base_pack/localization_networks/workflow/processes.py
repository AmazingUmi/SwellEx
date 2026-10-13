"""Bound a remote command and clean its entire process group."""
from __future__ import annotations

from contextlib import contextmanager
import math
import os
import signal
import subprocess
import threading
import time

CLEANUP_GRACE_ENV = "SWELLEX_PROCESS_CLEANUP_GRACE_S"


@contextmanager
def _scoped_signals(handler):
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGHUP)}
    try:
        for signum in previous:
            signal.signal(signum, handler)
        yield
    finally:
        for signum, original in previous.items():
            signal.signal(signum, original)


def _interrupted(signum, frame):
    # unittest propagates KeyboardInterrupt, but catches SystemExit and can
    # continue launching tests. A terminated nested runner must stop immediately.
    raise KeyboardInterrupt(f"Command interrupted by signal {signum}")


def _signal_group(pgid: int, signum: int) -> bool:
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        return False
    return True


def _stop_group(child: subprocess.Popen, grace_s: float) -> None:
    # Descendants can outlive the group leader. Never gate group cleanup on
    # child.poll(), and escalate even when the leader has already exited.
    if not _signal_group(child.pid, signal.SIGTERM):
        return
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        child.poll()  # Reap the leader while allowing descendants to stop.
        if not _signal_group(child.pid, 0):
            return
        time.sleep(min(0.05, max(0, deadline - time.monotonic())))
    _signal_group(child.pid, signal.SIGKILL)


def run_process(argv, *, timeout, cwd=None, env=None, capture_output=False,
                text=False, stdout=None, stderr=None, grace_s=5.0) -> subprocess.CompletedProcess:
    """Return a completed command, or raise after stopping all its descendants.

    Commands always run in a new session with closed input. Captured output and
    file redirection follow subprocess.run's conventions; nonzero exits remain
    available to the caller through returncode.
    """
    if capture_output:
        if stdout is not None or stderr is not None:
            raise ValueError("capture_output cannot be combined with stdout/stderr")
        stdout = stderr = subprocess.PIPE
    child_env = dict(os.environ if env is None else env)
    inherited_grace = child_env.get(CLEANUP_GRACE_ENV)
    cleanup_grace = float(grace_s)
    if inherited_grace is not None:
        # Nested runners must finish escalating before their parent escalates.
        cleanup_grace = min(cleanup_grace, float(inherited_grace) / 2)
    if not math.isfinite(cleanup_grace) or cleanup_grace < 0:
        raise ValueError("Process cleanup grace must be finite and nonnegative")
    child_env[CLEANUP_GRACE_ENV] = str(cleanup_grace)
    child = None
    with _scoped_signals(_interrupted):
        try:
            child = subprocess.Popen(argv, cwd=cwd, env=child_env, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr, text=text,
                                     start_new_session=True)
            output, errors = child.communicate(timeout=timeout)
            _stop_group(child, cleanup_grace)
            return subprocess.CompletedProcess(argv, child.returncode, output, errors)
        except BaseException:
            if child is not None:
                # Repeated termination requests must not interrupt the cleanup
                # already in progress; the parent's SIGKILL remains effective.
                with _scoped_signals(signal.SIG_IGN):
                    _stop_group(child, cleanup_grace)
                    child.communicate()
            raise
