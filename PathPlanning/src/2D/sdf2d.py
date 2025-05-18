import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from scipy.ndimage import gaussian_filter

import skfmm
import time

from Map import Map


class SDF:
    def __init__(self, map):
        self.map = map
        self.sdf = self.compute_sdf()
        self.sdf_gaussian = self.compute_gaussian_filtered_sdf()

    def set_max_sdf_value(self, max_value):
        if max_value <= 0:
            raise ValueError("Max value must be greater than 0")

        # Clip the values of sdf between 0 and max value
        if np.any(np.isnan(self.sdf)):
            raise ValueError("SDF contains NaN values")

        if np.any(np.isinf(self.sdf)):
            raise ValueError("SDF contains infinite values")

        self.sdf = np.clip(self.sdf, 0, max_value)
        self.sdf = np.round(self.sdf, 2)

    def compute_sdf(self):
        map_image = self.map.get_map_image()
        return skfmm.distance(map_image)

    def compute_gaussian_filtered_sdf(self, sigma=1.0):
        return gaussian_filter(self.sdf, sigma=sigma)

    def query(self, x, y):
        if self.map.contains(x, y):
            return self.sdf[y, x]
        else:
            return None

    def get_sdf(self):
        return self.sdf

    def get_gaussian_filtered_sdf(self):
        return self.sdf_gaussian

    def plot_sdf(self):
        plt.figure(figsize=(8, 6))
        plt.title("Signed Distance Field (SDF)")
        plt.imshow(self.sdf, origin="lower", cmap="jet")
        plt.colorbar(label="Distance")
        plt.xticks([])
        plt.yticks([])
        plt.show()
