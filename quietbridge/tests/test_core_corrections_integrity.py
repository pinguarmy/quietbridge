"""corrections.jsonl survives torn writes, foreign lines and concurrent appends."""
import json
import threading

from quietbridge.corrections import CorrectionStore, summarize_corrections
from quietbridge.profiles import Profile


def _store(tmp_path):
    profile = Profile("alice", "Alice", tmp_path)
    profile.ensure_dirs()
    return profile, CorrectionStore(profile)


def test_record_after_torn_line_stays_readable(tmp_path):
    profile, store = _store(tmp_path)
    store.record(original="a", corrected="first", learn=True)
    with store.path.open("a", encoding="utf-8") as stream:
        stream.write('{"id": "torn", "corr')  # process died mid-write
    entry = store.record(original="c", corrected="after crash", learn=True)
    assert [item["corrected"] for item in store.list()] == ["first", "after crash"]
    assert store.list()[-1] == entry
    summary = summarize_corrections(profile)
    assert summary["texts"] == ["first", "after crash"]
    assert any("malformed" in warning for warning in summary["warnings"])


def test_non_object_and_undecodable_lines_are_reported_not_fatal(tmp_path):
    profile, store = _store(tmp_path)
    store.record(original="a", corrected="kept", learn=True)
    with store.path.open("ab") as stream:
        stream.write(b'[1, 2]\n"text"\n42\n\xff\xfe broken\n')
    summary = summarize_corrections(profile)
    assert summary["texts"] == ["kept"]
    assert summary["skipped_count"] == 4
    assert [item["corrected"] for item in store.list()] == ["kept"]


def test_concurrent_records_are_all_whole_lines(tmp_path):
    _, store = _store(tmp_path)
    barrier = threading.Barrier(8)

    def write(index):
        barrier.wait()
        for n in range(10):
            # ~19 KB lines: larger than one stdio buffer.
            store.record(original="o", corrected=f"{index}-{n}",
                         candidates=["c" * 1900] * 10, learn=True)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    lines = store.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 80
    assert len({json.loads(line)["corrected"] for line in lines}) == 80
