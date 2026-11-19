from lipflow.corrections import find_correction, inserted_span

P = "Hello Miguel, I am sending you a message with my new school."


def fix(before, after, pasted=P):
    return find_correction(pasted, inserted_span(before, after))


def test_small_fix_is_learned():
    b = "Chat so far. "
    assert fix(b, b + P, P) is None  # untouched
    assert fix(b, b + P.replace("school", "tool"), P) == "Hello Miguel, I am sending you a message with my new tool."


def test_typing_more_after_is_not_part_of_the_label():
    b = ""
    after = P.replace("school", "tool") + " Also can you review it by Friday?"
    assert fix(b, after) == "Hello Miguel, I am sending you a message with my new tool."


def test_text_after_the_cursor_is_ignored():
    b = "Before. After text"
    after = "Before. " + P.replace("school", "tool") + " After text"
    assert fix(b, after) == "Hello Miguel, I am sending you a message with my new tool."


def test_rewrite_or_deletion_is_not_a_correction():
    assert fix("", "Totally different words that I wrote instead of it") is None
    assert fix("", "") is None


def test_saved_corrections_load_back_and_bad_files_are_skipped(tmp_path, monkeypatch, capsys):
    import numpy as np
    from lipflow import corrections
    monkeypatch.setattr(corrections, "DIR", str(tmp_path))
    rois = np.zeros((3, 96, 96), np.uint8)
    # raw candidates as dicts would be stored as a pickled object array; it is never read back
    corrections.save(rois, "HELLO WORD", "hello world", [{"text": "HELLO WORD", "score": -1.0}])
    (tmp_path / "0.npz").write_bytes(b"not an archive")  # an interrupted save sorts first
    (tmp_path / "notes.txt").write_text("stray")
    items = corrections.load_all()
    assert len(items) == 1 and items[0]["text"] == "hello world" and items[0]["rois"].shape == (3, 96, 96)
    assert corrections.count() == 2  # the .npz files; the stray text file is not a clip
    assert "skipping unreadable correction clip 0.npz" in capsys.readouterr().out
