
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

DATA = Path("data/Student folder/student_test")
OUTPUT = Path("results")

OUTPUT.mkdir(exist_ok=True)

K = 32
all_rows = []

# --------------------------------------------------
# 2. POINT-TO-SEGMENT DISTANCE
# --------------------------------------------------

def point_segment_distances(px, py, ax, ay, bx, by):

    dx = bx - ax
    dy = by - ay

    denom = dx * dx + dy * dy
    denom = np.maximum(denom, 1e-12)

    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = np.clip(t, 0, 1)

    qx = ax + t * dx
    qy = ay + t * dy

    distances = np.sqrt(
        (px - qx) ** 2 + (py - qy) ** 2
    )

    return distances, t


# --------------------------------------------------
# 3. FIND STUDENT-TEST TRIPS
# --------------------------------------------------

if not DATA.exists():
    raise FileNotFoundError(
        f"Student-test folder not found: {DATA}"
    )

folders = sorted(
    p for p in DATA.iterdir()
    if p.is_dir()
)

print("=" * 55)
print("PATHSNAP - STUDENT TEST CANDIDATE GENERATION")
print("=" * 55)

print("Student-test trips found:", len(folders))

if not folders:
    raise RuntimeError("No student-test trip folders found.")


# --------------------------------------------------
# 4. PROCESS EACH TRIP
# --------------------------------------------------

for trip_number, folder in enumerate(folders, 1):

    trip_id = folder.name

    try:

        track = np.loadtxt(
            folder / f"{trip_id}.track"
        )

        nodes = np.loadtxt(
            folder / f"{trip_id}.nodes"
        )

        arcs = np.loadtxt(
            folder / f"{trip_id}.arcs",
            dtype=int
        )

        route = np.loadtxt(
            folder / f"{trip_id}.route",
            dtype=int
        )

        track = np.atleast_2d(track)
        nodes = np.atleast_2d(nodes)
        arcs = np.atleast_2d(arcs)
        route = np.atleast_1d(route).astype(int)

        if np.any(route < 0) or np.any(route >= len(arcs)):
            print(f"Skipping {trip_id}: invalid route IDs")
            continue

        # Convert longitude and latitude into approximate metres.
        lat0 = np.mean(track[:, 1])

        scale_x = 111320 * np.cos(
            np.radians(lat0)
        )

        scale_y = 110540

        node_x = nodes[:, 0] * scale_x
        node_y = nodes[:, 1] * scale_y

        starts = arcs[:, 0]
        ends = arcs[:, 1]

        if (
            np.any(starts < 0)
            or np.any(ends < 0)
            or np.any(starts >= len(nodes))
            or np.any(ends >= len(nodes))
        ):
            print(f"Skipping {trip_id}: invalid arc node references")
            continue

        ax, ay = node_x[starts], node_y[starts]
        bx, by = node_x[ends], node_y[ends]

        # Build nearest-road-segment midpoint index.
        midpoints = np.column_stack((
            (ax + bx) / 2,
            (ay + by) / 2
        ))

        tree = cKDTree(midpoints)

        # Reference route segments.
        route_ax = ax[route]
        route_ay = ay[route]
        route_bx = bx[route]
        route_by = by[route]

        # GPS coordinates in approximate metres.
        gps_x = track[:, 0] * scale_x
        gps_y = track[:, 1] * scale_y

        # --------------------------------------------------
        # 5. GENERATE CANDIDATES FOR EACH GPS OBSERVATION
        # --------------------------------------------------

        for i in range(len(track)):

            px = gps_x[i]
            py = gps_y[i]

            # Find nearby road candidates.
            _, candidate_ids = tree.query(
                [px, py],
                k=min(K, len(arcs))
            )

            candidate_ids = np.atleast_1d(
                candidate_ids
            )

            # Find closest reference-route segment.
            route_distances, _ = point_segment_distances(
                px,
                py,
                route_ax,
                route_ay,
                route_bx,
                route_by
            )

            nearest_route_position = np.argmin(
                route_distances
            )

            correct_arc_id = route[
                nearest_route_position
            ]

            # GPS movement direction.
            if i > 0:

                dx_gps = gps_x[i] - gps_x[i - 1]
                dy_gps = gps_y[i] - gps_y[i - 1]

                gps_heading = np.degrees(
                    np.arctan2(dx_gps, dy_gps)
                )

                step_distance = np.hypot(
                    dx_gps,
                    dy_gps
                )

            else:

                gps_heading = 0.0
                step_distance = 0.0

            # Candidate distances and projections.
            distances, projections = point_segment_distances(
                px,
                py,
                ax[candidate_ids],
                ay[candidate_ids],
                bx[candidate_ids],
                by[candidate_ids]
            )

            for j, arc_id in enumerate(candidate_ids):

                road_heading = np.degrees(
                    np.arctan2(
                        bx[arc_id] - ax[arc_id],
                        by[arc_id] - ay[arc_id]
                    )
                )

                angle = abs(
                    (gps_heading - road_heading + 180) % 360 - 180
                )

                all_rows.append({

                    "trip_id": trip_id,

                    "observation": i,

                    "arc_id": int(arc_id),

                    "distance_m": float(
                        distances[j]
                    ),

                    "road_length_m": float(
                        np.hypot(
                            bx[arc_id] - ax[arc_id],
                            by[arc_id] - ay[arc_id]
                        )
                    ),

                    "direction_difference": float(angle),

                    "projection": float(
                        projections[j]
                    ),

                    "gps_step_distance_m": float(
                        step_distance
                    ),

                    "label": int(
                        arc_id == correct_arc_id
                    )
                })

        print(
            f"[{trip_number}/{len(folders)}] "
            f"Processed trip {trip_id}"
        )

    except Exception as error:

        print(
            f"Skipped trip {trip_id}: {error}"
        )


# --------------------------------------------------
# 6. SAVE STUDENT-TEST DATASET
# --------------------------------------------------

if not all_rows:
    raise RuntimeError(
        "No student-test candidates were generated."
    )

df = pd.DataFrame(all_rows)

output_file = OUTPUT / "student_test_candidates.csv"

df.to_csv(
    output_file,
    index=False
)

# --------------------------------------------------
# 7. DATASET SUMMARY
# --------------------------------------------------

observation_counts = df.groupby(
    ["trip_id", "observation"]
)["label"].sum()

positive_observations = int(
    (observation_counts > 0).sum()
)

total_observations = len(observation_counts)

coverage = (
    positive_observations / total_observations * 100
)

print("\n" + "=" * 55)
print("STUDENT-TEST DATA SUMMARY")
print("=" * 55)

print("Total candidate rows:", len(df))

print("Positive labels:", int(df["label"].sum()))

print(
    "Negative labels:",
    int((df["label"] == 0).sum())
)

print("Trips processed:", df["trip_id"].nunique())

print("Total GPS observations:", total_observations)

print(
    "Observations with positive candidate:",
    positive_observations
)

print(
    "Observations without positive candidate:",
    total_observations - positive_observations
)

print("Candidate coverage:", round(coverage, 2), "%")

print("Output:", output_file)

print("\nStudent-test data preparation completed.")
