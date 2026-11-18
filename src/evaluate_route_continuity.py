
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

DATA_FILE = Path("results/student_test_candidates.csv")
TRIP_FOLDER = Path("data/Student folder/student_test")
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

print("=" * 60)
print("PATHSNAP - ROUTE CONTINUITY EVALUATION")
print("=" * 60)

# --------------------------------------------------
# 2. LOAD CANDIDATE DATA
# --------------------------------------------------

df = pd.read_csv(
    DATA_FILE,
    dtype={"trip_id": str}
)

print("Candidate rows:", len(df))

# --------------------------------------------------
# 3. LOAD ROAD NETWORKS
# --------------------------------------------------

road_networks = {}

for trip_id in df["trip_id"].unique():

    folder = TRIP_FOLDER / trip_id

    arcs = np.loadtxt(
        folder / f"{trip_id}.arcs",
        dtype=int
    )

    arcs = np.atleast_2d(arcs)

    road_networks[trip_id] = arcs

print("Road networks loaded:", len(road_networks))

# --------------------------------------------------
# 4. EVALUATE EACH MODEL
# --------------------------------------------------

overall_results = []
trip_results = []

for model_name, filename in MODEL_FILES.items():

    print("\n" + "=" * 50)
    print("Evaluating:", model_name)
    print("=" * 50)

    model_path = MODEL_DIR / filename

    model = joblib.load(model_path)

    X = df[FEATURES].astype("float32")

    probabilities = model.predict_proba(X)[:, 1]

    scored = df[
        ["trip_id", "observation", "arc_id", "label"]
    ].copy()

    scored["score"] = probabilities

    # Select highest-scoring road candidate per GPS observation.
    best_indices = scored.groupby(
        ["trip_id", "observation"]
    )["score"].idxmax()

    best = scored.loc[best_indices].copy()

    best = best.sort_values(
        ["trip_id", "observation"]
    )

    total_pairs = 0
    connected_pairs = 0

    total_observations = len(best)
    correct_observations = int(best["label"].sum())

    observations_with_candidates = (
        best.groupby("trip_id").size().sum()
    )

    # Count GPS observations with no positive candidate.
    positive_per_observation = df.groupby(
        ["trip_id", "observation"]
    )["label"].sum()

    missed_observations = int(
        (positive_per_observation == 0).sum()
    )

    # --------------------------------------------------
    # 5. CHECK CONTINUITY FOR EACH TRIP
    # --------------------------------------------------

    for trip_id, group in best.groupby("trip_id"):

        group = group.sort_values("observation")

        arcs = road_networks[trip_id]

        selected_arc_ids = group["arc_id"].to_numpy()

        selected_labels = group["label"].to_numpy()

        valid_ids = (
            (selected_arc_ids >= 0)
            & (selected_arc_ids < len(arcs))
        )

        if not np.all(valid_ids):
            raise ValueError(
                f"Invalid candidate arc ID in trip {trip_id}"
            )

        selected_arcs = arcs[selected_arc_ids]

        # Consecutive predicted road segments should connect:
        # previous segment's end node == next segment's start node.

        if len(selected_arcs) > 1:

            connected = (
                selected_arcs[:-1, 1]
                == selected_arcs[1:, 0]
            )

            pair_count = len(connected)
            connected_count = int(connected.sum())

        else:

            pair_count = 0
            connected_count = 0

        total_pairs += pair_count
        connected_pairs += connected_count

        trip_continuity = (
            connected_count / pair_count
            if pair_count > 0 else 0.0
        )

        trip_accuracy = float(
            selected_labels.mean()
        )

        exact_sequence = int(
            np.all(selected_labels == 1)
        )

        trip_results.append({
            "Model": model_name,
            "Trip_ID": trip_id,
            "Observations": len(group),
            "Correct_Selections": int(selected_labels.sum()),
            "Top1_Accuracy": trip_accuracy,
            "Connected_Pairs": connected_count,
            "Total_Pairs": pair_count,
            "Route_Continuity": trip_continuity,
            "Exact_Sequence": exact_sequence
        })

    # --------------------------------------------------
    # 6. OVERALL METRICS
    # --------------------------------------------------

    route_continuity = (
        connected_pairs / total_pairs
        if total_pairs > 0 else 0.0
    )

    top1_accuracy = (
        correct_observations / total_observations
        if total_observations > 0 else 0.0
    )

    candidate_miss_rate = (
        missed_observations / total_observations
        if total_observations > 0 else 0.0
    )

    exact_trip_rate = np.mean([
        item["Exact_Sequence"]
        for item in trip_results
        if item["Model"] == model_name
    ])

    overall_results.append({
        "Model": model_name,
        "Top1_Accuracy": top1_accuracy,
        "Route_Continuity": route_continuity,
        "Candidate_Miss_Rate": candidate_miss_rate,
        "Exact_Trip_Sequence_Rate": exact_trip_rate,
        "Connected_Pairs": connected_pairs,
        "Total_Pairs": total_pairs
    })

    print("Top-1 Accuracy:", round(top1_accuracy, 4))
    print("Route Continuity:", round(route_continuity, 4))
    print("Candidate Miss Rate:", round(candidate_miss_rate, 4))
    print("Exact Trip Sequence Rate:", round(exact_trip_rate, 4))

# --------------------------------------------------
# 7. SAVE RESULTS
# --------------------------------------------------

overall_df = pd.DataFrame(overall_results)

trip_df = pd.DataFrame(trip_results)

overall_df.to_csv(
    RESULT_DIR / "route_continuity_comparison.csv",
    index=False
)

trip_df.to_csv(
    RESULT_DIR / "route_continuity_per_trip.csv",
    index=False
)

print("\n" + "=" * 60)
print("FINAL ROUTE CONTINUITY COMPARISON")
print("=" * 60)

print(
    overall_df.to_string(index=False)
)

print("\nGenerated files:")
print("results/route_continuity_comparison.csv")
print("results/route_continuity_per_trip.csv")

print("\nRoute continuity evaluation completed.")
