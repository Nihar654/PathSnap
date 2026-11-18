
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
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
DATA_DIR = Path("data/Student folder/train")
RESULT_DIR = Path("results")

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
FOLDS = 5

MAX_POSITIVES = 100000
MAX_NEGATIVES = 300000
MAX_VALIDATION_OBSERVATIONS = 5000


# --------------------------------------------------
# 2. DEFINE THE SAME FIVE CLASSIFIERS
# --------------------------------------------------

def create_models():

    return {
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
# 3. LOAD ONE FOLD USING CHUNKS
# --------------------------------------------------

def load_fold_data(train_trips, valid_trips):

    positive_parts = []
    negative_parts = []
    validation_parts = []

    print("\nReading CSV in chunks...")

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

        positive = training[
            training["label"] == 1
        ]

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
        raise RuntimeError(
            "Training or validation data is empty."
        )

    # Combine training samples.
    positive_df = pd.concat(
        positive_parts,
        ignore_index=True
    )

    negative_df = pd.concat(
        negative_parts,
        ignore_index=True
    )

    # Same limits as the original experiment.
    positive_df = positive_df.sample(
        n=min(MAX_POSITIVES, len(positive_df)),
        random_state=RANDOM_STATE
    )

    negative_df = negative_df.sample(
        n=min(MAX_NEGATIVES, len(negative_df)),
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

    # Prepare validation observations.
    valid_df = pd.concat(
        validation_parts,
        ignore_index=True
    )

    positive_observations = valid_df[
        valid_df["label"] == 1
    ][
        ["trip_id", "observation"]
    ].drop_duplicates()

    if positive_observations.empty:
        raise RuntimeError(
            "No positive validation observations in this fold."
        )

    selected_observations = positive_observations.sample(
        n=min(
            MAX_VALIDATION_OBSERVATIONS,
            len(positive_observations)
        ),
        random_state=RANDOM_STATE
    )

    valid_df = valid_df.merge(
        selected_observations,
        on=["trip_id", "observation"],
        how="inner"
    )

    return train_df, valid_df


# --------------------------------------------------
# 4. START CROSS VALIDATION
# --------------------------------------------------

print("=" * 65)
print("PATHSNAP - 5-FOLD GROUP-AWARE CROSS VALIDATION")
print("=" * 65)

if not CSV.exists():
    raise FileNotFoundError(CSV)

trip_ids = sorted(
    p.name
    for p in DATA_DIR.iterdir()
    if p.is_dir()
)

print("\nTotal training trips:", len(trip_ids))

if len(trip_ids) != 80:
    raise ValueError(
        f"Expected 80 training trips, found {len(trip_ids)}."
    )

group_kfold = GroupKFold(n_splits=FOLDS)

fold_results = []

# Split the trip IDs, not individual candidate rows.
for fold_number, (train_idx, valid_idx) in enumerate(
    group_kfold.split(
        np.zeros(len(trip_ids)),
        groups=trip_ids
    ),
    start=1
):

    train_trips = set(
        np.array(trip_ids)[train_idx]
    )

    valid_trips = set(
        np.array(trip_ids)[valid_idx]
    )

    assert train_trips.isdisjoint(valid_trips)

    print("\n" + "=" * 65)
    print(f"FOLD {fold_number}/{FOLDS}")
    print("=" * 65)

    print("Training trips:", len(train_trips))
    print("Validation trips:", len(valid_trips))

    train_df, valid_df = load_fold_data(
        train_trips,
        valid_trips
    )

    X_train = train_df[FEATURES].astype("float32")
    y_train = train_df["label"].astype("int8")

    X_valid = valid_df[FEATURES].astype("float32")
    y_valid = valid_df["label"].astype("int8")

    print("\nTraining rows:", len(train_df))
    print("Validation rows:", len(valid_df))

    print(
        "Validation observations:",
        valid_df.groupby(
            ["trip_id", "observation"]
        ).ngroups
    )

    models = create_models()

    # --------------------------------------------------
    # 5. TRAIN AND EVALUATE EACH MODEL
    # --------------------------------------------------

    for model_name, model in models.items():

        print("\n" + "-" * 50)
        print("FOLD", fold_number, "-", model_name)
        print("-" * 50)

        start_time = time.time()

        model.fit(X_train, y_train)

        training_time = time.time() - start_time

        probabilities = model.predict_proba(
            X_valid
        )[:, 1]

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

        # Observation-level Top-1 accuracy.
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

        top1_accuracy = float(
            best_candidates["label"].mean()
        )

        fold_results.append({
            "Fold": fold_number,
            "Model": model_name,
            "Macro_F1": macro_f1,
            "Candidate_Accuracy": accuracy,
            "Top1_Accuracy": top1_accuracy,
            "Training_Time_Seconds": training_time,
            "Training_Trips": len(train_trips),
            "Validation_Trips": len(valid_trips),
            "Training_Rows": len(train_df),
            "Validation_Rows": len(valid_df)
        })

        print("Macro F1:", round(macro_f1, 4))
        print("Candidate Accuracy:", round(accuracy, 4))
        print("Top-1 Accuracy:", round(top1_accuracy, 4))
        print("Training time:", round(training_time, 2), "seconds")

    del train_df, valid_df
    del X_train, X_valid, y_train, y_valid


# --------------------------------------------------
# 6. SAVE FOLD RESULTS
# --------------------------------------------------

fold_df = pd.DataFrame(fold_results)

fold_df.to_csv(
    RESULT_DIR / "cross_validation_fold_results.csv",
    index=False
)


# --------------------------------------------------
# 7. CALCULATE MEAN AND STANDARD DEVIATION
# --------------------------------------------------

summary = fold_df.groupby("Model").agg(
    Mean_Macro_F1=("Macro_F1", "mean"),
    Std_Macro_F1=("Macro_F1", "std"),
    Mean_Candidate_Accuracy=("Candidate_Accuracy", "mean"),
    Mean_Top1_Accuracy=("Top1_Accuracy", "mean"),
    Std_Top1_Accuracy=("Top1_Accuracy", "std"),
    Mean_Training_Time=("Training_Time_Seconds", "mean")
).reset_index()

summary.to_csv(
    RESULT_DIR / "cross_validation_summary.csv",
    index=False
)


# --------------------------------------------------
# 8. DISPLAY FINAL RESULTS
# --------------------------------------------------

print("\n" + "=" * 65)
print("5-FOLD CROSS VALIDATION SUMMARY")
print("=" * 65)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nGenerated files:")
print("results/cross_validation_fold_results.csv")
print("results/cross_validation_summary.csv")

print("\nAll 25 model-fold evaluations completed.")
