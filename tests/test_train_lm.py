"""lipflow train-lm validates its inputs before it needs any downloaded asset, and a missing
SentencePiece tokenizer is an actionable error, not a bare 'file not found' from sentencepiece."""
import pytest

from lipflow import train_lm


@pytest.fixture()
def no_tokenizer(monkeypatch, tmp_path):
    monkeypatch.setattr(train_lm, "SPM", str(tmp_path / "models" / "lm" / "unigram5000.model"))


@pytest.mark.usefixtures("no_tokenizer")
@pytest.mark.parametrize("phrases, validation, message", [
    ([], ["HELLO HOW ARE YOU"], "No valid training sentences"),
    (["HELLO HOW ARE YOU"], [], "No valid validation sentences"),
    (["1 2 3", "x"], None, "No valid sentences found"),  # digits and one-word lines are dropped
])
def test_invalid_phrases_fail_before_the_tokenizer_is_needed(phrases, validation, message):
    with pytest.raises(ValueError, match=message):
        train_lm.train(phrases=phrases, validation_phrases=validation)


@pytest.mark.usefixtures("no_tokenizer")
def test_missing_phrases_file_is_reported_before_the_tokenizer(monkeypatch, tmp_path):
    monkeypatch.setattr(train_lm, "PHRASES", str(tmp_path / "phrases.txt"))
    with pytest.raises(FileNotFoundError, match="import-wispr"):
        train_lm.train()


@pytest.mark.usefixtures("no_tokenizer")
def test_missing_tokenizer_names_the_file_and_the_fix():
    with pytest.raises(FileNotFoundError) as e:
        train_lm.train(phrases=["HELLO HOW ARE YOU"], validation_phrases=["SEE YOU LATER TODAY"])
    msg = str(e.value)
    assert "unigram5000.model" in msg and "setup" in msg


def test_cli_train_lm_reports_errors_in_one_line(monkeypatch, capsys):
    from lipflow.__main__ import main

    def fail():
        raise ValueError("No valid sentences found")

    monkeypatch.setattr(train_lm, "train", fail)
    with pytest.raises(SystemExit) as e:
        main(["train-lm"])
    assert e.value.code == "lipflow train-lm: No valid sentences found"
