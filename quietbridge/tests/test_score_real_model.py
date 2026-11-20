"""Teacher-forced phrase scoring with the REAL stock lip model.

Skipped with a reason when the models (CI on Linux) or the sample clip
(usually Windows CI) are absent. One clip of one professional speaker talking
aloud: these check that the scorer computes what the beam search computes and
behaves sensibly on real input, not recognition accuracy for anyone else.
"""
import math
import os

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS = os.path.join(ROOT, "models")
SAMPLE = os.path.join(ROOT, "samples", "2016-03-12.mov")
REQUIRED = [os.path.join(MODELS, *p) for p in (("vsr", "model.pth"), ("vsr", "model.json"),
                                               ("lm", "model.pth"), ("lm", "unigram5000.model"))]
HAVE_MODELS = all(os.path.isfile(p) for p in REQUIRED)
needs_models = pytest.mark.skipif(not HAVE_MODELS, reason="lip models missing: run ./setup.sh (Windows: setup.ps1)")
needs_sample = pytest.mark.skipif(
    not os.path.exists(SAMPLE) or not os.path.isfile(os.path.join(MODELS, "face_landmarker.task")),
    reason="sample clip or face model missing: run ./setup.sh --samples (Windows: setup.ps1 -Samples)")
TRUTH = "BORN IN NEW YORK CITY AND RAISED MOSTLY IN CHICAGO NANCY DAVIS GRADUATED FROM SMITH COLLEGE"


@pytest.fixture(scope="module")
def reader():
    import torch
    from lipflow.vsr import LipReader
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return LipReader(beam_size=4, personal=False, device="cpu")


@needs_models
def test_top_text_scores_exactly_what_the_beam_search_scored(reader):
    rng = np.random.default_rng(0)
    rois = rng.integers(60, 200, size=(40, 96, 96), dtype=np.uint8)
    enc = reader.encode(rois)
    details = reader.beam_search_details(enc, nbest=3)
    texts = [d["text"] for d in details if d["text"]]
    if not texts:
        pytest.skip("the decoder ended no hypothesis on noise input")
    scored = reader.score_texts(enc, texts + ["", "ROOM 112"])
    for detail, result in zip(details, scored):
        assert result["score"] == pytest.approx(detail["score"], abs=0.05)
        assert result["normalized"] == pytest.approx(result["score"] / result["tokens"])
    assert scored[-2] is None and scored[-1] is None  # nothing to write / digits above 20


@needs_models
def test_same_length_utterances_never_share_cached_memory_projections(reader):
    import torch
    rng = np.random.default_rng(1)
    first = reader.encode(rng.integers(0, 255, size=(30, 96, 96), dtype=np.uint8))
    second = reader.encode(rng.integers(0, 255, size=(30, 96, 96), dtype=np.uint8))
    fresh = reader.beam_search_details(second.clone(), nbest=1)
    buffer = torch.empty_like(first)
    buffer.copy_(first)
    reader.beam_search_details(buffer, nbest=1)
    buffer.copy_(second)  # same address and length: the old cache would reuse `first`
    assert reader.beam_search_details(buffer, nbest=1) == fresh
    texts = ["HELLO THERE", "THANK YOU VERY MUCH"]
    expected = reader.score_texts(second.clone(), texts)
    buffer.copy_(first)
    reader.score_texts(buffer, texts)
    buffer.copy_(second)
    assert [r["score"] for r in reader.score_texts(buffer, texts)] == pytest.approx([r["score"] for r in expected])


@needs_models
@needs_sample
def test_real_clip_prefers_the_spoken_sentence_and_the_offer_rule_holds(reader):
    from lipflow.face import FaceTracker, mouth_rois
    from lipflow.offline import load_clip
    from lipflow.vsr import LipReader
    from quietbridge import phrase_match as pm

    tracker = FaceTracker()
    try:
        ts, grays, anchors = load_clip(SAMPLE, tracker, 20.4, 28.1)
    finally:
        tracker.close()
    idx = LipReader.resample(ts, len(ts))
    enc = reader.encode(mouth_rois([grays[i] for i in idx], [anchors[i] for i in idx]))
    top = reader.beam_search_details(enc, nbest=3)[0]["text"]
    variants = [TRUTH.replace("CHICAGO", "BOSTON"), TRUTH.replace("NEW YORK", "NEW JERSEY"),
                TRUTH.replace("GRADUATED", "NOT GRADUATED"), "PLEASE OPEN THE WINDOW"]
    scores = reader.score_texts(enc, [top, TRUTH, *variants])
    top_score, truth = scores[0]["normalized"], scores[1]["normalized"]
    assert all(truth > s["normalized"] for s in scores[2:])
    assert all(math.isfinite(s["normalized"]) for s in scores)
    phrases = variants  # the spoken sentence is NOT saved, only its neighbours
    offers = pm.model_offers([top], phrases, {i: s["normalized"] for i, s in enumerate(scores[2:])}, top_score)
    if top == TRUTH:  # a correct result must not gain a one-word variant to confirm
        assert offers == []
    phrases = [TRUTH, *variants]
    offers = pm.model_offers([top], phrases, {i: s["normalized"] for i, s in enumerate(scores[1:])}, top_score)
    if top != TRUTH and truth >= top_score - pm.MODEL_OFFER_MARGIN:
        assert offers[0].phrase == TRUTH
