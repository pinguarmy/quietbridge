from lipflow.cloud import output_is_plausible


def test_rejects_unsupported_factual_expansion():
    assert not output_is_plausible(
        "This is Eisenhower's Farewell Address, delivered in 1961 to the American public.",
        ["WALLET OFFICER A FA WELL ADDRESS"], "faithful")


def test_rejects_negation_flip_even_when_text_is_fluent():
    assert not output_is_plausible("I think we should apply.", ["I DO NOT THINK WE SHOULD APPLY"], "polish")
    assert output_is_plausible("I don't think we should apply.", ["I DO NOT THINK WE SHOULD APPLY"], "polish")


def test_polish_filler_removal_allowed():
    assert output_is_plausible("I think we should send the email tomorrow.",
                               ["I THINK UM WE SHOULD WE SHOULD SEND THE EMAIL TOMORROW"], "polish")


def test_normal_correction_allowed_without_self_confidence():
    assert output_is_plausible("While in office, a farewell address.",
                               ["WALLET OFFICER A FA WELL ADDRESS"], "faithful")


def test_no_answer_wrappers_or_json_are_pasted():
    assert not output_is_plausible("Here is the corrected text: Hello.", ["HELLO"], "polish")
    assert not output_is_plausible('{"text": "Hello"}', ["HELLO"], "polish")
