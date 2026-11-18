
from pathlib import Path

trip = Path("data/Student folder/train/00000001")

nodes = (trip / "00000001.nodes").read_text().splitlines()

arcs = [
    list(map(int, line.split()))
    for line in (trip / "00000001.arcs").read_text().splitlines()
]

valid = 0
invalid = 0

for start, end in arcs:
    if 0 <= start < len(nodes) and 0 <= end < len(nodes):
        valid += 1
    else:
        invalid += 1

print("ROAD NODE VALIDATION")
print("=" * 35)
print("Total road connections:", len(arcs))
print("Valid connections:", valid)
print("Invalid connections:", invalid)

if invalid == 0:
    print("\nSUCCESS: All road connections refer to valid nodes!")
