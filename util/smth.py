import json
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_float
from skimage.exposure import rescale_intensity
import os

with open("labels.json") as f:
    labels = json.load(f)

# Plot a random sample of 20 labels on their images
sample = np.random.choice(len(labels), min(20, len(labels)), replace=False)
fig, axes = plt.subplots(4, 5, figsize=(25, 20))

for ax, idx in zip(axes.flat, sample):
    entry    = labels[idx]
    mat_path = os.path.join("./Aeschna_isoceles", entry["mat_file"])
    mat      = sio.loadmat(mat_path)
    imdat    = mat["imdat"]
    I1 = imdat[entry["angle"]][0][0]["presetcapture"][0][0]["image"][0][0]
    I1 = rescale_intensity(img_as_float(np.squeeze(I1)), out_range=(0.0,1.0))

    cx, cy = float(entry["cx"]), float(entry["cy"])

    ax.imshow(I1, cmap="gray")
    ax.plot(cx, cy, "r+", markersize=15, markeredgewidth=2)
    # Draw a 30px circle around the label to see where it sits
    circle = plt.Circle((cx, cy), 30, color="yellow", fill=False, linewidth=1)
    ax.add_patch(circle)
    ax.set_title(f"{entry['angle']}\n({cx:.0f},{cy:.0f})", fontsize=7)
    ax.axis("off")

plt.suptitle("Label audit — red cross = your label, yellow circle = 30px radius", 
             fontsize=12)
plt.tight_layout()
plt.savefig("label_audit1.png", dpi=100)
plt.close()
print("Saved label_audit.png")