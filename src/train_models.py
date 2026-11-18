
from pathlib import Path
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

CSV = Path("results/training_candidates.csv")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "distance_m",
    "road_length_m",
    "direction_difference",
    "projection",
    "gps_step_distance_m"
]

CHUNK_SIZE = 200000
RANDOM_STATE = 42

print("=" * 60)
print("PATHSNAP - FIVE MODEL EXPERIMENT")
print("=" * 60)

if not CSV.exists():
    raise FileNotFoundError(f"Dataset not found: {CSV}")

# --------------------------------------------------
# 2. SPLIT TRIPS (GROUP-AWARE HOLDOUT)
# --------------------------------------------------

trip_ids = sorted(
    p.name
    for p in Path("data/Student folder/train").iterdir()
    if p.is_dir()
)

if len(trip_ids) < 5:
    raise ValueError("Not enough training trips found.")

rng = np.random.RandomState(RANDOM_STATE)
shuffled_trips = rng.permutation(trip_ids)

train_trips = set(shuffled_trips[:64])
valid_trips = set(shuffled_trips[64:])

print("\nTRIP SPLIT")
print("Total trips:", len(trip_ids))
print("Training trips:", len(train_trips))
print("Validation trips:", len(valid_trips))

# --------------------------------------------------
# 3. READ LARGE CSV IN CHUNKS
# --------------------------------------------------

positive_parts = []
negative_parts = []
validation_parts = []

print("\nReading candidate dataset in chunks...")

for chunk_no, chunk in enumerate(
    pd.read_csv(
        CSV,
        chunksize=CHUNK_SIZE,
        dtype={"trip_id": str}
    ),
    start=1
):
    training = chunk[
        chunk["trip_id"].isin(train_trips)
    ]

    validation = chunk[
        chunk["trip_id"].isin(valid_trips)
    ]

    # Retain all positive training candidates.
    positive = training[
        training["label"] == 1
    ]

    # Sample up to three negative candidates
    # per positive candidate in each chunk.
    negative = training[
        training["label"] == 0
    ]

    if len(positive) > 0:
        positive_parts.append(positive)

        if len(negative) > 0:
            negative_parts.append(
                negative.sample(
                    n=min(
                        len(negative),
                        3 * len(positive)
                    ),
                    random_state=RANDOM_STATE + chunk_no
                )
            )

    if len(validation) > 0:
        validation_parts.append(validation)

    print(f"Processed chunk {chunk_no}")

if not positive_parts or not validation_parts:
    raise ValueError(
        "Training or validation data is empty. "
        "Check trip IDs in the CSV and folder names."
    )

# --------------------------------------------------
# 4. PREPARE TRAINING AND VALIDATION DATA
# --------------------------------------------------

print("\nCombining training data...")

positive_df = pd.concat(
    positive_parts,
    ignore_index=True
)

negative_df = pd.concat(
    negative_parts,
    ignore_index=True
)

# Keep the training set manageable.
positive_df = positive_df.sample(
    n=min(100000, len(positive_df)),
    random_state=RANDOM_STATE
)

negative_df = negative_df.sample(
    n=min(300000, len(negative_df)),
    random_state=RANDOM_STATE
)

train_df = pd.concat(
    [positive_df, negative_df],
    ignore_index=True
)

train_df = train_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

valid_df = pd.concat(
    validation_parts,
    ignore_index=True
)

# --------------------------------------------------
# 5. LIMIT VALIDATION OBSERVATIONS
# --------------------------------------------------

# Select observations that have at least one positive
# candidate. This allows meaningful Top-1 evaluation.

positive_observations = valid_df[
    valid_df["label"] == 1
][["trip_id", "observation"]].drop_duplicates()

if positive_observations.empty:
    raise ValueError(
        "No positive candidates found in validation trips."
    )

selected_observations = positive_observations.sample(
    n=min(5000, len(positive_observations)),
    random_state=RANDOM_STATE
)

valid_df = valid_df.merge(
    selected_observations,
    on=["trip_id", "observation"],
    how="inner"
)

# --------------------------------------------------
# 6. CREATE FEATURES AND LABELS
# --------------------------------------------------

X_train = train_df[FEATURES].astype("float32")
y_train = train_df["label"].astype("int8")

X_valid = valid_df[FEATURES].astype("float32")
y_valid = valid_df["label"].astype("int8")

print("\nDATASET SUMMARY")
print("-" * 40)
print("Training rows:", len(train_df))
print("Training positives:", int(y_train.sum()))
print("Training negatives:", int((y_train == 0).sum()))
print("Validation rows:", len(valid_df))
print(
    "Validation observations:",
    len(selected_observations)
)

# --------------------------------------------------
# 7. DEFINE FIVE CLASSIFIERS
# --------------------------------------------------

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=300,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    ),

    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=2
        )
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=2
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=10,
        verbosity=-1,
        random_state=RANDOM_STATE,
        n_jobs=2
    ),

    "XGBoost": XGBClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.08,
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=2
    )
}

# --------------------------------------------------
# 8. TRAIN AND EVALUATE
# --------------------------------------------------

results = []

for name, model in models.items():

    print("\n" + "=" * 50)
    print("TRAINING:", name)
    print("=" * 50)

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    probabilities = model.predict_proba(X_valid)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    macro_f1 = f1_score(
        y_valid,
        predictions,
        average="macro",
        zero_division=0
    )

    accuracy = accuracy_score(
        y_valid,
        predictions
    )

    # Observation-level Top-1 Accuracy:
    # Select the candidate with the highest probability
    # for each GPS observation.

    scored_df = valid_df[
        ["trip_id", "observation", "label"]
    ].copy()

    scored_df["score"] = probabilities

    best_indices = scored_df.groupby(
        ["trip_id", "observation"]
    )["score"].idxmax()

    best_candidates = scored_df.loc[
        best_indices
    ]

    top1_accuracy = (
        best_candidates["label"].mean()
    )

    # Save trained model.
    model_filename = (
        name.lower().replace(" ", "_") + ".joblib"
    )

    joblib.dump(
        model,
        MODEL_DIR / model_filename
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro_F1": macro_f1,
        "Top1_Accuracy": top1_accuracy,
        "Training_Time_Seconds": training_time
    })

    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Top-1 Accuracy:", round(top1_accuracy, 4))
    print(
        "Training time:",
        round(training_time, 2),
        "seconds"
    )
    print("Saved:", model_filename)

# --------------------------------------------------
# 9. SAVE COMPARISON TABLE
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULT_DIR / "model_comparison.csv",
    index=False
)

# --------------------------------------------------
# 10. GENERATE COMPARISON GRAPH
# --------------------------------------------------

ax = results_df.plot(
    x="Model",
    y=["Macro_F1", "Top1_Accuracy"],
    kind="bar",
    figsize=(11, 6)
)

ax.set_title("PathSnap - Model Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_xlabel("Machine Learning Models")

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "model_comparison.png",
    dpi=200
)

plt.close()

# --------------------------------------------------
# 11. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("ALL FIVE MODELS COMPLETED")
print("=" * 60)

print(
    results_df.to_string(index=False)
)

print("\nGENERATED FILES")
print("models/")
print("  logistic_regression.joblib")
print("  knn.joblib")
print("  random_forest.joblib")
print("  lightgbm.joblib")
print("  xgboost.joblib")

print("\nresults/")
print("  model_comparison.csv")
print("  model_comparison.png")

print("\nEXPERIMENT FINISHED")
