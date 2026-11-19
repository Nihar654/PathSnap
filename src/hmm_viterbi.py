
from pathlib import Path
import time

import numpy as np
import pandas as pd
import networkx as nx


# ==========================================
# 1. PROJECT PATHS AND CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "Student folder" / "student_test"

PREDICTIONS_FILE = (
    PROJECT_DIR / "results" / "student_test_xgboost_predictions.csv"
)

OUTPUT_DIR = PROJECT_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. HMM CONFIGURATION
# ==========================================

N_CANDIDATES = 32

# Transition distance tolerance in metres.
# We will validate and tune this value later.
BETA = 50.0

# Small value to prevent log(0).
EPSILON = 1e-12

# Select which XGBoost scores to evaluate:
# "baseline_probability" or "tuned_probability"
SCORE_COLUMN = "tuned_probability"


# ==========================================
# 3. LOAD PREDICTIONS
# ==========================================


def load_predictions():

    predictions = pd.read_csv(
        PREDICTIONS_FILE,
        dtype={"trip_id": str}
    )

    candidates_file = (
        PROJECT_DIR / "results" / "student_test_candidates.csv"
    )

    candidate_features = pd.read_csv(
        candidates_file,
        dtype={"trip_id": str},
        usecols=[
            "trip_id",
            "observation",
            "arc_id",
            "projection",
            "gps_step_distance_m"
        ]
    )

    predictions = predictions.merge(
        candidate_features,
        on=["trip_id", "observation", "arc_id"],
        how="left",
        validate="one_to_one"
    )

    if predictions["projection"].isna().any():
        raise ValueError("Some candidate projections are missing.")

    if predictions["gps_step_distance_m"].isna().any():
        raise ValueError("Some GPS step distances are missing.")

    predictions = predictions.sort_values(
        ["trip_id", "observation", "arc_id"]
    ).reset_index(drop=True)

    return predictions


# ==========================================
# 4. BUILD DIRECTED ROAD GRAPH
# ==========================================

def build_road_graph(trip_id):

    trip_dir = DATA_DIR / trip_id

    nodes_file = trip_dir / f"{trip_id}.nodes"
    arcs_file = trip_dir / f"{trip_id}.arcs"
    track_file = trip_dir / f"{trip_id}.track"

    nodes = np.atleast_2d(np.loadtxt(nodes_file))
    arcs = np.atleast_2d(
        np.loadtxt(arcs_file, dtype=int)
    )
    track = np.atleast_2d(np.loadtxt(track_file))

    # Convert longitude and latitude into approximate metres.
    reference_latitude = np.mean(track[:, 1])

    scale_x = (
        111320 * np.cos(np.radians(reference_latitude))
    )
    scale_y = 110540

    node_x = nodes[:, 0] * scale_x
    node_y = nodes[:, 1] * scale_y

    graph = nx.MultiDiGraph()

    # Add road-network nodes.
    for node_id in range(len(nodes)):

        graph.add_node(
            node_id,
            x=float(node_x[node_id]),
            y=float(node_y[node_id])
        )

    # Add directed road segments with physical lengths.
    for arc_id, (start_node, end_node) in enumerate(arcs):

        dx = node_x[end_node] - node_x[start_node]
        dy = node_y[end_node] - node_y[start_node]

        distance = float(np.hypot(dx, dy))

        graph.add_edge(
            int(start_node),
            int(end_node),
            key=int(arc_id),
            weight=distance,
            arc_id=int(arc_id)
        )

    return graph, arcs, nodes, track


# ==========================================
# 5. PREPARE HMM EMISSION SCORES
# ==========================================

def prepare_emissions(trip_predictions):

    trip_predictions = trip_predictions.copy()

    # Read XGBoost candidate scores.
    scores = trip_predictions[SCORE_COLUMN].to_numpy(
        dtype=float
    )

    # Prevent negative or invalid scores.
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    scores = np.clip(scores, EPSILON, 1.0)

    trip_predictions["emission"] = scores

    # Normalize candidate scores within each observation.
    totals = trip_predictions.groupby(
        "observation"
    )["emission"].transform("sum")

    trip_predictions["emission"] = (
        trip_predictions["emission"] / totals
    )

    return trip_predictions



# ==========================================
# 6. SHORTEST-PATH DISTANCE CACHE
# ==========================================

def create_shortest_path_cache(graph):

    cache = {}

    def get_distances(source_node):

        if source_node not in cache:

            cache[source_node] = nx.single_source_dijkstra_path_length(
                graph,
                source_node,
                weight="weight"
            )

        return cache[source_node]

    return get_distances


# ==========================================
# 7. DISTANCE BETWEEN TWO CANDIDATE ARCS
# ==========================================

def candidate_transition_distance(
    current_arc_id,
    next_arc_id,
    current_projection,
    next_projection,
    arcs,
    graph,
    get_distances,
    arc_lengths
):

    current_start, current_end = arcs[current_arc_id]
    next_start, next_end = arcs[next_arc_id]

    # Remaining distance on the current arc.
    remaining_current = (
        (1.0 - current_projection)
        * arc_lengths[current_arc_id]
    )

    # Distance travelled along the next arc.
    travelled_next = (
        next_projection
        * arc_lengths[next_arc_id]
    )

    # If both candidates are the same directed arc,
    # handle movement along that arc directly.
    if current_arc_id == next_arc_id:

        if next_projection >= current_projection:
            return (
                (next_projection - current_projection)
                * arc_lengths[current_arc_id]
            )

        return np.inf

    # Find the shortest directed path between
    # the end of the current arc and start of next arc.
    distances = get_distances(int(current_end))

    middle_distance = distances.get(
        int(next_start),
        np.inf
    )

    if not np.isfinite(middle_distance):
        return np.inf

    return (
        remaining_current
        + middle_distance
        + travelled_next
    )



# ==========================================
# 8. CALCULATE ROAD ARC LENGTHS
# ==========================================

def calculate_arc_lengths(arcs, nodes, track):

    reference_latitude = np.mean(track[:, 1])

    scale_x = (
        111320 * np.cos(np.radians(reference_latitude))
    )
    scale_y = 110540

    node_x = nodes[:, 0] * scale_x
    node_y = nodes[:, 1] * scale_y

    start_nodes = arcs[:, 0]
    end_nodes = arcs[:, 1]

    dx = node_x[end_nodes] - node_x[start_nodes]
    dy = node_y[end_nodes] - node_y[start_nodes]

    lengths = np.hypot(dx, dy)

    return lengths


# ==========================================
# 9. VITERBI DECODING
# ==========================================

def viterbi_decode(
    trip_emissions,
    arcs,
    graph,
    get_distances,
    arc_lengths
):
    trip_emissions = trip_emissions.sort_values(
        ["observation", "arc_id"]
    )

    observation_groups = [
        group.copy().reset_index(drop=True)
        for _, group in trip_emissions.groupby("observation")
    ]

    if not observation_groups:
        return pd.DataFrame(), 0

    decoded_rows = []
    fallback_count = 0

    # Split into continuous observation sequences.
    sequences = []
    current_sequence = [observation_groups[0]]

    for group in observation_groups[1:]:
        previous_obs = int(
            current_sequence[-1]["observation"].iloc[0]
        )
        current_obs = int(group["observation"].iloc[0])

        if current_obs == previous_obs + 1:
            current_sequence.append(group)
        else:
            sequences.append(current_sequence)
            current_sequence = [group]

    sequences.append(current_sequence)

    for sequence in sequences:

        # Initial Viterbi scores.
        first = sequence[0]

        delta = np.log(
            np.clip(
                first["emission"].to_numpy(dtype=float),
                EPSILON,
                1.0
            )
        )

        backpointers = []

        # Process consecutive observations.
        for t in range(1, len(sequence)):

            previous = sequence[t - 1]
            current = sequence[t]

            previous_count = len(previous)
            current_count = len(current)

            gps_distance = float(
                current["gps_step_distance_m"].iloc[0]
            )

            transition_log = np.full(
                (previous_count, current_count),
                -np.inf
            )

            # Calculate transition scores.
            for i, previous_candidate in previous.iterrows():

                previous_index = previous.index.get_loc(i)

                for j, current_candidate in current.iterrows():

                    current_index = current.index.get_loc(j)

                    distance = candidate_transition_distance(
                        int(previous_candidate["arc_id"]),
                        int(current_candidate["arc_id"]),
                        float(previous_candidate["projection"]),
                        float(current_candidate["projection"]),
                        arcs,
                        graph,
                        get_distances,
                        arc_lengths
                    )

                    if np.isfinite(distance):

                        transition_log[
                            previous_index, current_index
                        ] = -abs(
                            distance - gps_distance
                        ) / BETA

            # Normalize each previous-state transition row.
            for i in range(previous_count):

                row = transition_log[i]
                finite = np.isfinite(row)

                if not finite.any():
                    # Fallback when no next candidate is reachable.
                    transition_log[i, :] = -np.log(current_count)
                    fallback_count += 1

                else:
                    maximum = np.max(row[finite])
                    row[finite] -= maximum

                    denominator = np.log(
                        np.exp(row[finite]).sum()
                    )

                    row[finite] -= denominator

            # Viterbi recurrence.
            scores = delta[:, None] + transition_log

            best_previous = np.argmax(scores, axis=0)
            best_scores = np.max(scores, axis=0)

            current_emissions = np.log(
                np.clip(
                    current["emission"].to_numpy(dtype=float),
                    EPSILON,
                    1.0
                )
            )

            delta = best_scores + current_emissions
            backpointers.append(best_previous)

        # Trace back the best path.
        last_state = int(np.argmax(delta))

        best_path = [last_state]

        for pointer in reversed(backpointers):
            last_state = int(pointer[last_state])
            best_path.append(last_state)

        best_path.reverse()

        for group, state_index in zip(sequence, best_path):
            decoded_rows.append(group.iloc[state_index])

    decoded = pd.DataFrame(decoded_rows).reset_index(drop=True)

    return decoded, fallback_count
    

if __name__ == "__main__":

    predictions = load_predictions()

    # Normalize trip IDs so folder names like 00000000 match IDs like 0.
    predictions["_trip_key"] = (
        predictions["trip_id"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )

    trip_folders = sorted(
        folder for folder in DATA_DIR.iterdir()
        if folder.is_dir()
    )

    print("Predictions loaded successfully!")
    print("Total candidate rows:", len(predictions))
    print("Student-test trips found:", len(trip_folders))

    summary = []
    failed_trips = []

    for index, folder in enumerate(trip_folders, start=1):

        trip_id = folder.name.zfill(8)

        print("\n" + "=" * 55)
        print(f"PROCESSING TRIP {index}/{len(trip_folders)}: {trip_id}")
        print("=" * 55)

        try:
            trip_predictions = predictions[
                predictions["_trip_key"] == trip_id
            ].copy()

            if trip_predictions.empty:
                print("No predictions found. Skipping this trip.")
                failed_trips.append(trip_id)
                continue

            graph, arcs, nodes, track = build_road_graph(trip_id)

            emissions = prepare_emissions(trip_predictions)

            arc_lengths = calculate_arc_lengths(arcs, nodes, track)

            get_distances = create_shortest_path_cache(graph)

            start_time = time.time()

            decoded, fallback_count = viterbi_decode(
                trip_emissions=emissions,
                arcs=arcs,
                graph=graph,
                get_distances=get_distances,
                arc_lengths=arc_lengths
            )

            elapsed_time = time.time() - start_time

            output_file = OUTPUT_DIR / f"hmm_viterbi_trip_{trip_id}.csv"
            decoded.to_csv(output_file, index=False)

            print("GPS observations:", len(track))
            print("Decoded observations:", len(decoded))
            print("Fallback transitions:", fallback_count)
            print("Execution time:", round(elapsed_time, 2), "seconds")
            print("Saved to:", output_file)

            summary.append({
                "trip_id": trip_id,
                "observations": len(decoded),
                "fallback_transitions": fallback_count,
                "execution_time_seconds": round(elapsed_time, 2),
                "status": "Success"
            })

        except Exception as error:
            print("ERROR processing trip:", trip_id)
            print(error)
            failed_trips.append(trip_id)

    # Save an overall processing summary.
    summary_file = OUTPUT_DIR / "hmm_batch_summary.csv"

    pd.DataFrame(summary).to_csv(summary_file, index=False)

    print("\n" + "=" * 55)
    print("BATCH PROCESSING COMPLETED")
    print("=" * 55)
    print("Successfully processed:", len(summary))
    print("Failed trips:", len(failed_trips))
    print("Summary saved to:", summary_file)

    if failed_trips:
        print("Trips requiring attention:", failed_trips)
