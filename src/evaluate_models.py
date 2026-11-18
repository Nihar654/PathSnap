
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix
)

# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

DATA_FILE = Path("results/student_test_candidates.csv")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

RESULT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "distance_m",
    "road_length_m",
    "direction_difference",
    "projection",
    "gps_step_distance_m"
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "KNN": "knn.joblib",
    "Random Forest": "random_forest.joblib",
    "LightGBM": "lightgbm.joblib",
    "XGBoost": "xgboost.joblib"
}

CHUNK_SIZE = 100000

print("=" * 60)
print("PATHSNAP - STUDENT TEST MODEL EVALUATION")
print("=" * 60)

if not DATA_FILE.exists():
    raise FileNotFoundError(DATA_FILE)

# --------------------------------------------------
# 2. LOAD STUDENT-TEST DATA
# --------------------------------------------------

print("\nLoading student-test candidates...")

df = pd.read_csv(
    DATA_FILE,
    dtype={"trip_id": str}
)

X = df[FEATURES].astype("float32")
y = df["label"].astype(int)

print("Candidate rows:", len(df))
print("GPS observations:", df.groupby(
    ["trip_id", "observation"]
).ngroups)
print("Positive candidates:", int(y.sum()))
print("Negative candidates:", int((y == 0).sum()))

# --------------------------------------------------
# 3. LOAD FIVE TRAINED MODELS
# --------------------------------------------------

models = {}

for name, filename in MODEL_FILES.items():

    model_path = MODEL_DIR / filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model: {model_path}"
        )

    models[name] = joblib.load(model_path)

    print("Loaded:", name)

# --------------------------------------------------
# 4. EVALUATE EACH MODEL
# --------------------------------------------------

results = []
all_predictions = {}

for name, model in models.items():

    print("\n" + "=" * 50)
    print("Evaluating:", name)
    print("=" * 50)

    start_time = time.time()

    # Predict in chunks to limit memory usage.
    probability_parts = []

    for start in range(0, len(X), CHUNK_SIZE):

        end = min(start + CHUNK_SIZE, len(X))

        probabilities = model.predict_proba(
            X.iloc[start:end]
        )[:, 1]

        probability_parts.append(probabilities)

    probabilities = np.concatenate(probability_parts)

    inference_time = time.time() - start_time

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # Candidate-level metrics.
    accuracy = accuracy_score(y, predictions)

    precision = precision_score(
        y, predictions, zero_division=0
    )

    recall = recall_score(
        y, predictions, zero_division=0
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    loss = log_loss(
        y,
        np.column_stack((
            1 - probabilities,
            probabilities
        )),
        labels=[0, 1]
    )

    # Observation-level Top-1 Accuracy.
    scored = df[
        ["trip_id", "observation", "label"]
    ].copy()

    scored["score"] = probabilities

    best_indices = scored.groupby(
        ["trip_id", "observation"]
    )["score"].idxmax()

    best_candidates = scored.loc[best_indices]

    top1_accuracy = best_candidates["label"].mean()

    # Save predictions for later analysis.
    all_predictions[name] = predictions

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Macro_F1": macro_f1,
        "Top1_Accuracy": top1_accuracy,
        "Log_Loss": loss,
        "Inference_Time_Seconds": inference_time
    })

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Top-1 Accuracy:", round(top1_accuracy, 4))
    print("Log Loss:", round(loss, 4))
    print(
        "Inference time:",
        round(inference_time, 2),
        "seconds"
    )

# --------------------------------------------------
# 5. SAVE FINAL COMPARISON TABLE
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULT_DIR / "student_test_model_comparison.csv",
    index=False
)

print("\n" + "=" * 60)
print("STUDENT TEST MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

# --------------------------------------------------
# 6. SAVE CONFUSION MATRICES
# --------------------------------------------------

for name, predictions in all_predictions.items():

    cm = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    )

    print("\nConfusion Matrix:", name)
    print(cm)

# --------------------------------------------------
# 7. GENERATE COMPARISON GRAPH
# --------------------------------------------------

plot_df = results_df.set_index("Model")[
    ["Macro_F1", "Top1_Accuracy"]
]

ax = plot_df.plot(
    kind="bar",
    figsize=(11, 6)
)

ax.set_title("PathSnap - Student Test Model Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Machine Learning Models")
ax.set_ylim(0, 1)

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "student_test_model_comparison.png",
    dpi=200
)

plt.close()

# --------------------------------------------------
# 8. GENERATE LOG LOSS GRAPH
# --------------------------------------------------

ax = results_df.plot(
    x="Model",
    y="Log_Loss",
    kind="bar",
    figsize=(10, 5),
    legend=False
)

ax.set_title("PathSnap - Student Test Log Loss")
ax.set_ylabel("Log Loss")
ax.set_xlabel("Machine Learning Models")

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "student_test_log_loss.png",
    dpi=200
)

plt.close()

# --------------------------------------------------
# 9. FINAL OUTPUT
# --------------------------------------------------

print("\n" + "=" * 60)
print("STUDENT TEST EVALUATION COMPLETED")
print("=" * 60)

print("\nGenerated files:")
print("results/student_test_model_comparison.csv")
print("results/student_test_model_comparison.png")
print("results/student_test_log_loss.png")

print("\nEvaluation completed successfully.")
