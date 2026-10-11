from app.classifier import classify_with_rules, extract_category


def test_rule_fallback_handles_roman_stage() -> None:
    result = classify_with_rules("Synthetic note documents stage III disease.")
    assert result.label == "III"
    assert result.method == "explicit-stage-rule"


def test_rule_fallback_handles_numeric_stage() -> None:
    assert classify_with_rules("This training case is stage 4.").label == "IV"



def test_explicit_stage_wins_over_model(monkeypatch) -> None:
    def model_must_not_run(note):
        raise AssertionError("model should not run when the stage is explicit")

    monkeypatch.setattr("app.classifier.classify_with_pytorch", model_must_not_run)
    result = extract_category("Synthetic record: stage IV metastatic breast cancer with distant spread.")
    assert result.label == "IV"
    assert result.method == "explicit-stage-rule"
