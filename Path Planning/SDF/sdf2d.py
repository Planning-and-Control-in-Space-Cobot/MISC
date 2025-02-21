import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import skfmm

# Define environment limits
x_limits = [0, 200]
y_limits = [0, 200]

# Define obstacles using shapely
obstacles = [
    Point(50, 50).buffer(40),
    Point(150, 125).buffer(35),
    Polygon([(0, 100), (90, 100), (90, 110), (0, 110)]),
    Polygon([(100, 20), (100, 100), (120, 100), (120, 20)])
]

# Resolution of the grid
resolution = 1
x = np.arange(x_limits[0], x_limits[1], resolution)
y = np.arange(y_limits[0], y_limits[1], resolution)
xv, yv = np.meshgrid(x, y)

# Create a binary grid for obstacles
phi = np.full(xv.shape, 1, dtype=np.float64)  # Initialize with -1 (outside obstacles)

for obstacle in obstacles:
    for i in range(xv.shape[0]):
        for j in range(xv.shape[1]):
            point = Point(xv[i, j], yv[i, j])
            if obstacle.contains(point):
                phi[i, j] = -0.5  # Inside obstacles (arbitrary positive value)

# Compute signed distance field (SDF)
sd = skfmm.distance(phi, dx=resolution)

# Plot binary obstacle grid
plt.figure(figsize=(8, 6))
plt.title("Binary Obstacle Grid")
plt.imshow(phi, extent=(x_limits[0], x_limits[1], y_limits[0], y_limits[1]), origin="lower", cmap="gray")
plt.colorbar(label="Obstacle Value")
plt.xticks([])
plt.yticks([])
plt.show()

# Plot signed distance field
plt.figure(figsize=(8, 6))
plt.title("Signed Distance Field (SDF)")
plt.imshow(sd, extent=(x_limits[0], x_limits[1], y_limits[0], y_limits[1]), origin="lower", cmap="jet")
plt.colorbar(label="Distance")
plt.xticks([])
plt.yticks([])
plt.show()