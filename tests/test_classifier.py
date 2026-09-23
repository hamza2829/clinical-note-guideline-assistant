from app.classifier import classify_with_rules


def test_rule_fallback_handles_roman_stage() -> None:
    result = classify_with_rules("Synthetic note documents stage III disease.")
    assert result.label == "III"
    assert result.method == "explicit-stage-rule"


def test_rule_fallback_handles_numeric_stage() -> None:
    assert classify_with_rules("This training case is stage 4.").label == "IV"

