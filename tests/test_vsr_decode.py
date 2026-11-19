"""LipReader.beam_search when ESPnet finds no finished hypothesis (it then returns [])."""
import types

import torch

from lipflow.vsr import LipReader


def _reader(hyps):
    reader = LipReader.__new__(LipReader)  # no weights needed: only the result handling is tested
    reader.beam = lambda enc: hyps
    reader.token_list = ["<blank>", "▁HI", "<eos>"]
    return reader


def test_no_hypothesis_reads_as_empty_text_not_an_index_error():
    reader = _reader([])
    assert reader.beam_search(torch.zeros(4, 8)) == ""
    assert reader.beam_search(torch.zeros(4, 8), nbest=5) == []


def test_a_hypothesis_is_returned_as_text(monkeypatch):
    hyp = types.SimpleNamespace(asdict=lambda: {"yseq": [2, 1, 2], "score": -1.5})
    monkeypatch.setattr("lipflow.vsr.add_results_to_json", lambda data, tokens: "▁HI<eos>")
    reader = _reader([hyp])
    assert reader.beam_search(torch.zeros(4, 8)) == "HI"
    assert reader.beam_search_details(torch.zeros(4, 8)) == [{"text": "HI", "score": -1.5}]
