from lipflow.profile import corpus
from lipflow.practice import HARVARD, practice_sentences


def test_selected_domains_produce_twelve_recordable_phrases(tmp_path):
    words, phrases = corpus(['ai-coding', 'finance', 'education'])
    assert len(phrases) == 12
    assert 'API' in words and 'private equity' in words and 'mathematics' in words
    assert all(5 <= len(p.split()) <= 12 for p in phrases)
    path = tmp_path / 'phrases.txt'
    path.write_text('\n'.join(phrases))
    practice = practice_sentences(phrases_path=str(path))
    assert len(practice) == 24
    assert sum(s in HARVARD for s in practice) == 12


def test_recording_round_keeps_twelve_custom_prompts_and_skips_previous(tmp_path):
    _, phrases = corpus(['ai-coding', 'finance', 'education'])
    path = tmp_path / 'phrases.txt'
    path.write_text('\n'.join(phrases))
    round_one = practice_sentences(24, str(path), exclude={HARVARD[0]})
    assert len(round_one) == 24 and HARVARD[0] not in round_one
    assert sum(s in HARVARD for s in round_one) == 12


def test_domain_corpus_is_deduplicated_and_does_not_invent_personal_assertions():
    words, phrases = corpus(['ai-coding', 'ai-coding'])
    assert len(words) == len(set(words))
    assert not any('I study at' in p or 'I work at' in p for p in phrases)
