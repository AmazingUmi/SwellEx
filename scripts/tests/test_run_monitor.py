"""Progress recovery must never resubmit work or hide terminal failures."""
from contextlib import redirect_stdout
from importlib import import_module
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.run_monitor import follow, LOG_PROGRESS_CODE, RemoteConnectionError


def packet(status="RUNNING", *, offset=0, text="", pending=False):
    return {"report": {"status": status, "checks": []},
            "logs": [{"name": "worker.log", "text": text}] if text else [],
            "offsets": {"worker.log": offset}, "pending": pending}


class FollowTests(unittest.TestCase):
    def test_reconnect_preserves_offsets_and_drains_final_logs_before_fetch(self):
        progress = Mock(side_effect=[packet(offset=5, text="first"), RemoteConnectionError("offline"),
                                    packet("PASS", offset=9, text="last", pending=True),
                                    packet("PASS", offset=9)])
        fetch = Mock(return_value={"status": "PASS"})
        summarize = Mock()
        with patch("scripts.run_monitor.time.sleep"), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(follow("run", progress=progress, fetch=fetch, summarize=summarize), 0)
        self.assertEqual([call.args[0] for call in progress.call_args_list],
                         [{}, {"worker.log": 5}, {"worker.log": 5}, {"worker.log": 9}])
        self.assertEqual(output.getvalue().count("first"), 1)
        self.assertEqual(output.getvalue().count("last"), 1)
        fetch.assert_called_once()
        self.assertEqual(summarize.call_count, 2)

    def test_transfer_retries_only_transport_failure_and_returns_worker_exit(self):
        progress = Mock(return_value=packet("FAIL"))
        fetch = Mock(side_effect=[RemoteConnectionError("lost"), {"status": "FAIL", "exit_code": 7}])
        with patch("scripts.run_monitor.time.sleep"), redirect_stdout(io.StringIO()):
            self.assertEqual(follow("run", progress=progress, fetch=fetch, summarize=Mock()), 7)
        self.assertEqual(fetch.call_count, 2)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "hash mismatch"):
            follow("run", progress=progress, fetch=Mock(side_effect=ValueError("hash mismatch")), summarize=Mock())

    def test_ctrl_c_only_stops_local_following(self):
        fetch = Mock()
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(follow("my-run", progress=Mock(side_effect=KeyboardInterrupt), fetch=fetch, summarize=Mock()), 130)
        fetch.assert_not_called()
        self.assertIn("Remote task continues", output.getvalue())
        self.assertIn("my-run", output.getvalue())

    def test_remote_application_error_is_not_retried_as_disconnect(self):
        for pack in ("base_pack", "joint_feature_pack"):
            remote = import_module(f"scripts.{pack}.localization_networks.workflow.remote")
            cfg = remote.load_config(remote.DEFAULT_CONFIG)
            for code, expected in ((255, RemoteConnectionError), (1, RuntimeError)):
                with patch.object(remote.subprocess, "run", return_value=subprocess.CompletedProcess([], code, "", "failure")):
                    with self.assertRaises(expected) as caught:
                        remote.remote_python(cfg, "probe")
                self.assertIs(type(caught.exception), expected)

    def test_log_probe_is_bounded_and_resumes_without_loss_or_external_files(self):
        from scripts.base_pack.localization_networks.workflow.remote import STATUS_CODE
        code = STATUS_CODE.replace("print(json.dumps(result))", LOG_PROGRESS_CODE)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            logs = run / "logs"
            logs.mkdir()
            (run / "report.json").write_text(json.dumps({"status": "PASS"}))
            expected = "abc\n" * 20000
            (logs / "worker.log").write_text(expected)
            (run / "private").write_text("must not read")
            (logs / "external.log").symlink_to(run / "private")
            offsets, actual = {}, ""
            for _ in range(3):
                result = subprocess.run([sys.executable, "-B", "-c", code],
                                        input=json.dumps({"run_dir": str(run), "offsets": offsets}),
                                        capture_output=True, text=True, check=True)
                update = json.loads(result.stdout)
                self.assertLessEqual(sum(len(row["text"]) for row in update["logs"]), 65536)
                self.assertNotIn("external.log", update["offsets"])
                actual += "".join(row["text"] for row in update["logs"])
                offsets = update["offsets"]
            self.assertEqual(actual, expected)
            self.assertFalse(update["pending"])


if __name__ == "__main__":
    unittest.main()
