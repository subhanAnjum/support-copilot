"""
Ragas eval: faithfulness, answer relevancy and context precision.
Writes eval_report.csv and exits non-zero if any metric is below threshold.

Usage: python app/eval/run_eval.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from app.graph import run_ticket
from app.config import EVAL_DATASET_PATH

THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.75,
    "context_precision": 0.70,
}


def load_eval_examples():
    examples = []
    with open(EVAL_DATASET_PATH) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def build_ragas_dataset(examples):
    questions, answers, contexts, ground_truths = [], [], [], []

    for ex in examples:
        result = run_ticket(ex["question"])
        questions.append(ex["question"])
        answers.append(result["response"])
        contexts.append(result.get("retrieved_chunks", []))
        ground_truths.append(ex["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def main():
    examples = load_eval_examples()
    print(f"Running {len(examples)} eval examples through the graph...")

    dataset = build_ragas_dataset(examples)

    print("Scoring with Ragas (faithfulness, answer_relevancy, context_precision)...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    df = result.to_pandas()
    report_path = Path(__file__).parent / "eval_report.csv"
    df.to_csv(report_path, index=False)
    print(f"\nFull report written to {report_path}")

    print("\n--- Summary ---")
    all_passed = True
    for metric, threshold in THRESHOLDS.items():
        score = df[metric].mean()
        status = "PASS" if score >= threshold else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"{metric:20s} {score:.3f}  (threshold {threshold})  [{status}]")

    print(f"\nOverall: {'PASS' if all_passed else 'FAIL - see eval_report.csv'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
