
import csv
from collections import Counter

file = "results/training_candidates.csv"

positive_counts = Counter()
candidate_counts = Counter()

with open(file, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        key = (row["trip_id"], row["observation"])
        candidate_counts[key] += 1

        if int(row["label"]) == 1:
            positive_counts[key] += 1

total = len(candidate_counts)
with_positive = sum(1 for key in candidate_counts if positive_counts[key] > 0)
without_positive = total - with_positive
multiple_positive = sum(1 for key in candidate_counts if positive_counts[key] > 1)

print("CANDIDATE DATA QUALITY CHECK")
print("=" * 40)
print("Total GPS observations:", total)
print("Observations with a positive candidate:", with_positive)
print("Observations without a positive candidate:", without_positive)
print("Observations with multiple positive candidates:", multiple_positive)
print("Candidate coverage:", round(with_positive / total * 100, 2), "%")
