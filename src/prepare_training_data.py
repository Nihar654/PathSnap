
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DATA = Path("data/Student folder/train")
OUTPUT = Path("results")
OUTPUT.mkdir(exist_ok=True)

K = 32
all_rows = []

def point_segment_distances(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    denom = np.maximum(denom, 1e-12)

    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = np.clip(t, 0, 1)

    qx = ax + t * dx
    qy = ay + t * dy

    return np.sqrt((px - qx)**2 + (py - qy)**2), t

folders = sorted(p for p in DATA.iterdir() if p.is_dir())

print("PREPARING TRAINING DATA")
print("Training trips found:", len(folders))

for trip_number, folder in enumerate(folders, 1):
    trip_id = folder.name

    try:
        track = np.loadtxt(folder / f"{trip_id}.track")
        nodes = np.loadtxt(folder / f"{trip_id}.nodes")
        arcs = np.loadtxt(folder / f"{trip_id}.arcs", dtype=int)
        route = np.loadtxt(folder / f"{trip_id}.route", dtype=int)

        track = np.atleast_2d(track)
        nodes = np.atleast_2d(nodes)
        arcs = np.atleast_2d(arcs)
        route = np.atleast_1d(route).astype(int)

        if np.any(route < 0) or np.any(route >= len(arcs)):
            print(f"Skipping {trip_id}: invalid route IDs")
            continue

        # Convert longitude/latitude into approximate local metres.
        lat0 = np.mean(track[:, 1])
        scale_x = 111320 * np.cos(np.radians(lat0))
        scale_y = 110540

        node_x = nodes[:, 0] * scale_x
        node_y = nodes[:, 1] * scale_y

        starts = arcs[:, 0]
        ends = arcs[:, 1]

        ax, ay = node_x[starts], node_y[starts]
        bx, by = node_x[ends], node_y[ends]

        midpoints = np.column_stack(((ax + bx) / 2, (ay + by) / 2))
        tree = cKDTree(midpoints)

        route_ax = ax[route]
        route_ay = ay[route]
        route_bx = bx[route]
        route_by = by[route]

        gps_x = track[:, 0] * scale_x
        gps_y = track[:, 1] * scale_y

        for i in range(len(track)):
            px, py = gps_x[i], gps_y[i]

            # Find nearby road-segment candidates.
            _, candidate_ids = tree.query(
                [px, py], k=min(K, len(arcs))
            )
            candidate_ids = np.atleast_1d(candidate_ids)

            # Find the closest reference-route segment.
            route_distances, _ = point_segment_distances(
                px, py,
                route_ax, route_ay,
                route_bx, route_by
            )
            nearest_route_position = np.argmin(route_distances)
            correct_arc_id = route[nearest_route_position]

            # GPS movement direction.
            if i > 0:
                dx_gps = gps_x[i] - gps_x[i - 1]
                dy_gps = gps_y[i] - gps_y[i - 1]
                gps_heading = np.degrees(np.arctan2(dx_gps, dy_gps))
                step_distance = np.hypot(dx_gps, dy_gps)
            else:
                gps_heading = 0.0
                step_distance = 0.0

            distances, projection = point_segment_distances(
                px, py,
                ax[candidate_ids], ay[candidate_ids],
                bx[candidate_ids], by[candidate_ids]
            )

            for j, arc_id in enumerate(candidate_ids):
                road_heading = np.degrees(np.arctan2(
                    bx[arc_id] - ax[arc_id],
                    by[arc_id] - ay[arc_id]
                ))

                angle = abs((gps_heading - road_heading + 180) % 360 - 180)

                all_rows.append({
                    "trip_id": trip_id,
                    "observation": i,
                    "arc_id": int(arc_id),
                    "distance_m": float(distances[j]),
                    "road_length_m": float(np.hypot(
                        bx[arc_id] - ax[arc_id],
                        by[arc_id] - ay[arc_id]
                    )),
                    "direction_difference": float(angle),
                    "projection": float(projection[j]),
                    "gps_step_distance_m": float(step_distance),
                    "label": int(arc_id == correct_arc_id)
                })

        print(f"[{trip_number}/{len(folders)}] Processed trip {trip_id}")

    except Exception as error:
        print(f"Skipped trip {trip_id}: {error}")

if not all_rows:
    raise RuntimeError("No training candidates were generated.")

df = pd.DataFrame(all_rows)

df.to_csv(OUTPUT / "training_candidates.csv", index=False)

print("\nTRAINING DATA SUMMARY")
print("=" * 40)
print("Total candidate rows:", len(df))
print("Positive labels:", int(df["label"].sum()))
print("Negative labels:", int((df["label"] == 0).sum()))
print("Trips processed:", df["trip_id"].nunique())
print("Output:", OUTPUT / "training_candidates.csv")
print("\nTraining data preparation completed.")
