"""Read-only progress following; connection loss never sends worker signals."""
from __future__ import annotations

import time


class RemoteConnectionError(RuntimeError):
    """The SSH transport failed; retrying a read cannot duplicate a run."""


# Append to each pack's status probe before its final print. Read bounded log
# chunks by byte offset, so reconnecting does not repeat delivered output.
LOG_PROGRESS_CODE = '''
offsets=p.get('offsets', {}); updates=[]; next_offsets={}; remaining=65536; pending=False
for path in sorted((run/'logs').glob('*.log')):
    if path.is_symlink() or not path.is_file(): continue
    size=path.stat().st_size; offset=offsets.get(path.name,0)
    if type(offset) is not int or offset<0: raise ValueError('Invalid log offset')
    if offset>size: offset=0
    with path.open('rb') as stream:
        stream.seek(offset); chunk=stream.read(remaining)
    next_offsets[path.name]=offset+len(chunk); remaining-=len(chunk)
    if chunk: updates.append({'name':path.name,'text':chunk.decode('utf-8',errors='replace')})
    if next_offsets[path.name]<size: pending=True
print(json.dumps({'report':result,'logs':updates,'offsets':next_offsets,'pending':pending}))
'''


def follow(run_id: str, *, progress, fetch, summarize) -> int:
    offsets = {}
    previous = None
    try:
        while True:
            try:
                update = progress(offsets)
            except (RemoteConnectionError, OSError) as exc:
                print(f"Connection lost; retrying in 5 s. Remote run continues: {run_id}\n{exc}", flush=True)
                time.sleep(5)
                continue
            report = update["report"]
            state = (report["status"], tuple((row["name"], row["status"]) for row in report.get("checks", [])))
            if state != previous:
                summarize(report)
                previous = state
            elif not update["logs"]:
                print(f"{run_id}: {report['status']}", flush=True)
            for chunk in update["logs"]:
                print(f"[{chunk['name']}]\n{chunk['text']}", end="" if chunk['text'].endswith('\n') else "\n", flush=True)
            offsets = update["offsets"]
            if report["status"] in {"PASS", "FAIL"} and not update["pending"]:
                try:
                    final = fetch()
                except RemoteConnectionError as exc:
                    print(f"Result transfer connection lost; retrying in 5 s: {run_id}\n{exc}", flush=True)
                    time.sleep(5)
                    continue
                return 0 if final["status"] == "PASS" else int(final.get("exit_code", 1)) or 1
            time.sleep(5)
    except KeyboardInterrupt:
        print(f"Stopped local monitoring. Remote task continues. Run ID: {run_id}", flush=True)
        return 130
