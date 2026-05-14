import json
import os

with open("labels_expanded2.json") as f:
    labels = json.load(f)

fixed = []
for entry in labels:
    mat_file = entry["mat_file"]
    
    # Fix truncated filename: site.mat → site_1.mat
    if mat_file.endswith("site.mat"):
        mat_file = mat_file.replace("site.mat", "site_1.mat")
    
    # Fix path separators to be consistent (forward slash)
    mat_file = mat_file.replace("\\", "/")
    
    fixed.append({**entry, "mat_file": mat_file})

# Verify fixes before saving
print("Sample fixed paths:")
for e in fixed[:3]:
    print(f"  {e['mat_file']}")

# Check all files actually exist
MAT_DIR = "./Aeschna_isoceles"
missing = []
for e in fixed:
    path = os.path.join(MAT_DIR, e["mat_file"])
    if not os.path.exists(path):
        missing.append(path)

if missing:
    print(f"\nStill missing {len(missing)} files:")
    for p in missing[:10]:
        print(f"  {p}")
else:
    print(f"\nAll {len(fixed)} files found!")
    with open("labels_fixed.json", "w") as f:
        json.dump(fixed, f, indent=2)
    print("Saved → labels_fixed.json")