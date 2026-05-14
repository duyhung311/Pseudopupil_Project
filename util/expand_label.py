import json
import os

# Your existing labels presumably look like:
# {"mat_file": "0Azimuth/file.mat", "angle": "imagesS0", "cx": 1024.0, "cy": 750.0}

with open("labels2.json") as f:
    original_labels = json.load(f)

expanded = []
for entry in original_labels:
    # Original entry — uses both I1+I2 as 3-channel input (existing behaviour)
    # Reframe as: I1 only and I2 only as separate single-channel inputs
    expanded.append({
        "mat_file": entry["mat_file"],
        "angle":    entry["angle"],
        "cx":       float(entry["cx"]),
        "cy":       float(entry["cy"]),
        "exposure": "i1"    # ← new field
    })
    expanded.append({
        "mat_file": entry["mat_file"],
        "angle":    entry["angle"],
        "cx":       float(entry["cx"]),
        "cy":       float(entry["cy"]),
        "exposure": "i2"    # ← new field
    })
    expanded.append({
        "mat_file": entry["mat_file"],
        "angle":    entry["angle"],
        "cx":       float(entry["cx"]),
        "cy":       float(entry["cy"]),
        "exposure": "both"    # ← new field
    })

with open("labels_expanded2.json", "w") as f:
    json.dump(expanded, f, indent=2)

print(f"Original: {len(original_labels)} → Expanded: {len(expanded)} samples")
# Original: 415 → Expanded: 830 samples