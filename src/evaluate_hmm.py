
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_DIR / "results"

predictions = pd.read_csv(
    RESULTS / "student_test_xgboost_predictions.csv",
    dtype={"trip_id": str}
)

hmm = pd.read_csv(
    RESULTS / "hmm_viterbi_trip_00000000.csv",
    dtype={"trip_id": str}
)

# Keep only observations with exactly one reference candidate.
valid = (
    predictions.groupby(["trip_id", "observation"])["label"]
    .sum()
    .reset_index(name="reference_count")
)

valid = valid[valid["reference_count"] == 1]

# Obtain the reference arc ID.
reference = predictions[predictions["label"] == 1][
    ["trip_id", "observation", "arc_id"]
].rename(columns={"arc_id": "reference_arc_id"})

# Classifier-only predictions.
classifier = (
    predictions.sort_values("tuned_probability", ascending=False)
    .drop_duplicates(["trip_id", "observation"])
    [["trip_id", "observation", "arc_id"]]
    .rename(columns={"arc_id": "classifier_arc_id"})
)

comparison = (
    reference
    .merge(classifier, on=["trip_id", "observation"])
    .merge(
        hmm[["trip_id", "observation", "arc_id"]].rename(
            columns={"arc_id": "hmm_arc_id"}
        ),
        on=["trip_id", "observation"]
    )
    .merge(valid[["trip_id", "observation"]], on=["trip_id", "observation"])
)

y_true = comparison["reference_arc_id"]

for name, column in [
    ("Classifier-only", "classifier_arc_id"),
    ("HMM + Viterbi", "hmm_arc_id")
]:
    print("\n" + name)
    print("Top-1 Accuracy:", round(
        accuracy_score(y_true, comparison[column]), 4
    ))
    print("Macro F1:", round(
        f1_score(y_true, comparison[column], average="macro", zero_division=0), 4
    ))

comparison.to_csv(
    RESULTS / "hmm_comparison_trip_00000000.csv",
    index=False
)

print("\nEvaluated observations:", len(comparison))
print("Comparison saved successfully.")
