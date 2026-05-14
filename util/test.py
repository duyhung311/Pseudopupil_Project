# Run this on one failing file and one training file side by side
import scipy.io as sio
import matplotlib.pyplot as plt

# One failing file
mat_new = sio.loadmat(".\\Apocordulia_macrops\\12Azimuth\\20260127_232709_azimuth_000012_elevation_000-42_site_1.mat")
I1_new = mat_new["imdat"]["imagesS0"][0][0]["presetcapture"][0][0]["image"][0][1]

# One training file
mat_train = sio.loadmat(".\Aeschna_isoceles\\12Azimuth\\20250624_130846_azimuth_000012_elevation_000-42_site_1.mat")
I1_train = mat_train["imdat"]["imagesS0"][0][0]["presetcapture"][0][0]["image"][0][1]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(I1_train, cmap="gray"); axes[0].set_title("Training file")
axes[1].imshow(I1_new,   cmap="gray"); axes[1].set_title("New session file")
plt.savefig("comparison.png", dpi=100)

# Quick check — print dimensions of new vs training files
# import scipy.io as sio
# import glob

# # Check one new file
# mat = sio.loadmat(".\\Apocordulia_macrops\\12Azimuth\\20260127_232709_azimuth_000012_elevation_000-42_site_1.mat")
# I1 = mat["imdat"]["imagesS0"][0][0]["presetcapture"][0][0]["image"][0][1]
# print(f"New session shape:      {I1.shape}")   # should be (H, W)

# # Check one training file  
# mat = sio.loadmat(".\Aeschna_isoceles\\12Azimuth\\20250624_130846_azimuth_000012_elevation_000-42_site_1.mat")
# I1 = mat["imdat"]["imagesS0"][0][0]["presetcapture"][0][0]["image"][0][1]
# print(f"Training session shape: {I1.shape}")