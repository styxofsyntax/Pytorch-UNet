import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2

mask_path = 'output.jpg'
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

colors = [
    (0, 0, 0),         # 0: Background - black
    (0, 0.6, 1),       # 1: Card - blue
    (1, 0, 0),         # 2: Damages - red
]

cmap = mcolors.ListedColormap(colors)
bounds = [0, 1, 2, 3]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(8, 8))
plt.imshow(mask, cmap=cmap, norm=norm)
cbar = plt.colorbar(ticks=[0.5, 1.5, 2.5])
cbar.ax.set_yticklabels(['Background', 'Card', 'Damages'])
plt.title("Predicted Mask")
plt.axis('off')
plt.tight_layout()
plt.show()
