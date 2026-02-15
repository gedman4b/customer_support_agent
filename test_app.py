from app import answer_question, find_best_match


def test_best_match_returns_eva_answer():
    match, score = find_best_match("Can you tell me what EVA does?")
    assert match is not None
    assert "verifying a patient" in match.answer
    assert score > 0.40


def test_unknown_question_uses_fallback():
    response, used_kb, score = answer_question("What's the weather in Boston?")
    assert used_kb is False
    assert score < 0.40
    assert "don't have a predefined" in response.lower()


def test_acronym_only_overlap_does_not_force_wrong_kb_match():
    response, used_kb, score = answer_question("What is EVA pricing?")
    assert used_kb is False
    assert score < 0.40
    assert "don't have a predefined" in response.lower()
