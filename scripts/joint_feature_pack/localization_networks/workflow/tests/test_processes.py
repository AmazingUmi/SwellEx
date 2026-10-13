"""Real process-group lifecycle checks; execute on the Linux compute host."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from scripts.joint_feature_pack.localization_networks.workflow.processes import run_process


@unittest.skipUnless(sys.platform == "linux", "process identity checks require Linux /proc")
class ProcessTests(unittest.TestCase):
    def live(self, pid):
        try:
            return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
        except FileNotFoundError:
            return False

    def test_capture_output_and_nonzero_result(self):
        result = run_process([sys.executable, "-c", "import sys; print('output'); print('error',file=sys.stderr); sys.exit(7)"],
                             timeout=5, capture_output=True, text=True)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stdout, "output\n")
        self.assertEqual(result.stderr, "error\n")

    def test_signal_handlers_are_restored_after_success_and_timeout(self):
        signals = (signal.SIGTERM, signal.SIGHUP)
        original = {signum: signal.getsignal(signum) for signum in signals}
        custom = lambda signum, frame: None
        try:
            for signum in signals:
                signal.signal(signum, custom)
            run_process([sys.executable, "-c", "pass"], timeout=5)
            self.assertTrue(all(signal.getsignal(signum) is custom for signum in signals))
            with self.assertRaises(subprocess.TimeoutExpired):
                run_process([sys.executable, "-c", "import time; time.sleep(60)"], timeout=0.2, grace_s=0.2)
            self.assertTrue(all(signal.getsignal(signum) is custom for signum in signals))
        finally:
            for signum, handler in original.items():
                signal.signal(signum, handler)

    def test_outer_timeout_propagates_cleanup_across_nested_process_groups(self):
        with tempfile.TemporaryDirectory(prefix="swellex-nested-processes-") as directory:
            root = Path(directory)
            marker, ready = root / "identity.json", root / "ready"
            leaf = ("import os,signal,time; from pathlib import Path; "
                    "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                    f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)")
            inner = ("import json,os,subprocess,sys,time; from pathlib import Path\n"
                     f"child=subprocess.Popen([sys.executable,'-c',{leaf!r}])\n"
                     f"ready=Path({str(ready)!r})\n"
                     "deadline=time.monotonic()+5\n"
                     "while not ready.exists() and time.monotonic()<deadline: time.sleep(0.01)\n"
                     "if not ready.exists(): raise RuntimeError('descendant did not start')\n"
                     f"Path({str(marker)!r}).write_text(json.dumps({{'leader':os.getpid(),'child':child.pid}}))\n"
                     "time.sleep(60)")
            outer = ("import subprocess,sys\n"
                     "from scripts.joint_feature_pack.localization_networks.workflow.processes import run_process\n"
                     f"run_process([sys.executable,'-c',{inner!r}],timeout=60,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n")
            try:
                with self.assertRaises(subprocess.TimeoutExpired):
                    run_process([sys.executable, "-c", outer], cwd=ROOT, timeout=1,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, grace_s=2)
                identity = json.loads(marker.read_text())
                deadline = time.monotonic()+2
                while any(self.live(pid) for pid in identity.values()) and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(all(not self.live(pid) for pid in identity.values()),
                                "nested process group survived outer timeout")
            finally:
                if marker.exists():
                    identity = json.loads(marker.read_text())
                    try:
                        os.killpg(identity["leader"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_term_resistant_descendant_is_killed_even_after_leader_exits(self):
        with tempfile.TemporaryDirectory(prefix="swellex-process-groups-") as directory:
            root = Path(directory)
            for outcome in ("timeout", "failure", "success"):
                with self.subTest(outcome=outcome):
                    marker = root / f"{outcome}.json"
                    ready = root / f"{outcome}.ready"
                    descendant = ("import os,signal,time; from pathlib import Path; "
                                  "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                                  f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)")
                    end = {"timeout": "time.sleep(60)", "failure": "sys.exit(7)", "success": "sys.exit(0)"}[outcome]
                    leader = ("import json,os,subprocess,sys,time; from pathlib import Path\n"
                              f"child=subprocess.Popen([sys.executable,'-c',{descendant!r}])\n"
                              f"ready=Path({str(ready)!r})\n"
                              "deadline=time.monotonic()+5\n"
                              "while not ready.exists() and time.monotonic()<deadline: time.sleep(0.01)\n"
                              "if not ready.exists(): raise RuntimeError('descendant did not start')\n"
                              f"Path({str(marker)!r}).write_text(json.dumps({{'leader':os.getpid(),'child':child.pid}}))\n"
                              + end)
                    try:
                        arguments = [sys.executable, "-c", leader]
                        if outcome == "timeout":
                            with self.assertRaises(subprocess.TimeoutExpired):
                                run_process(arguments, timeout=1, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL, grace_s=0.2)
                        else:
                            result = run_process(arguments, timeout=10, stdout=subprocess.DEVNULL,
                                                 stderr=subprocess.DEVNULL, grace_s=0.2)
                            self.assertEqual(result.returncode, 7 if outcome == "failure" else 0)
                        identity = json.loads(marker.read_text())
                        deadline = time.monotonic()+2
                        while self.live(identity["child"]) and time.monotonic() < deadline:
                            time.sleep(0.01)
                        self.assertFalse(self.live(identity["leader"]))
                        self.assertFalse(self.live(identity["child"]), "TERM-resistant descendant survived group cleanup")
                    finally:
                        if marker.exists():
                            identity = json.loads(marker.read_text())
                            try:
                                os.killpg(identity["leader"], signal.SIGKILL)
                            except ProcessLookupError:
                                pass


if __name__ == "__main__":
    unittest.main()
