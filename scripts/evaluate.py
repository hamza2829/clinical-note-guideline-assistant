import json
from pathlib import Path

from app.pipeline import run_pipeline


def main() -> None:
    cases = json.loads(Path("data/evaluation.json").read_text(encoding="utf-8"))
    category_correct = 0
    retrieval_correct = 0
    results = []

    for case in cases:
        result = run_pipeline(case["note"])
        category_ok = result.classification.label == case["expected_stage"]
        retrieval_ok = result.passage.stage == case["expected_stage"]
        category_correct += category_ok
        retrieval_correct += retrieval_ok
        results.append(
            {
                "id": case["id"],
                "expected": case["expected_stage"],
                "predicted": result.classification.label,
                "retrieved_stage": result.passage.stage,
                "method": result.classification.method,
                "category_correct": category_ok,
                "retrieval_correct": retrieval_ok,
            }
        )

    report = {
        "synthetic_cases": len(cases),
        "category_accuracy": category_correct / len(cases),
        "retrieval_accuracy": retrieval_correct / len(cases),
        "results": results,
        "limitations": (
            "Small synthetic set with explicit stage wording. This is a smoke test, not evidence of "
            "clinical validity or real-world generalization."
        ),
    }
    output = Path("outputs/evaluation_report.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

