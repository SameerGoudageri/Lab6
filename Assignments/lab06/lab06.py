# -*- coding: utf-8 -*-
"""Lab 06 — Scale Invariant Blob Detector.

Detect blobs (flower heads) at their natural scale, using a Laplacian-of-
Gaussian (LoG) scale space. The work is split into small steps, each one a
method below that takes an array in and returns the result out:

  1. gauss_kernel(sigma)                   1D Gaussian kernel
  2. gauss_2nd_deriv_kernel(sigma)         1D second-derivative-of-Gaussian kernel
  3. second_derivatives(image, sigma)      Lxx, Lyy by separable convolution
  4. laplacian(image, sigma)               scale-normalised LoG at one scale
  5. build_scale_space(image)              LoG stacked over every scale
  6. candidate_points(scale_space)         points above the threshold
  7. lap_max_det(scale_space, row, col, k) is this point a local maximum?
  8. find_maxima(scale_space)              every local maximum
  9. blob_circles(maxima)                  centre and radius of each blob
 10. detect_blobs(image)                   put the steps together
"""
import numpy as np
import os
import sys
from scipy import signal
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from helpers.dataloader import DATA_DIR, download_image, norm_img  # noqa: F401

FLOWERS_IMAGE_URL = (
    "https://github.com/tonyscan6003/etivities/blob/main/sunflowers_temp.jpg?raw=true"
)


class ScaleInvDet:
    """Scale Invariant Blob Detector using a Laplacian-of-Gaussian (LoG) scale space.

    All settings have defaults, so no arguments are needed::

        sid = ScaleInvDet()
        blobs = sid.detect_blobs(image)

    Any setting can be changed with a keyword argument, e.g.::

        sid = ScaleInvDet(sigma_max=10, thresh=60)
    """

    def __init__(self, **kwargs):
        self.sigma_min = kwargs.get('sigma_min', 1)
        self.sigma_max = kwargs.get('sigma_max', 36)
        self.scale_fac = kwargs.get('scale_fac', 1.05)
        self.thresh = kwargs.get('thresh', 40)

        # The scales of the scale space, worked out from the settings above.
        self.num_scales = int((self.sigma_max - self.sigma_min) / self.scale_fac)
        self.sigma_vals = np.linspace(self.sigma_min, self.sigma_max, self.num_scales)

        # Intermediate results, filled in by detect_blobs().
        self.laplace_scale_space = None
        self.maxima = None
        self.blobs = None

    def gauss_kernel(self, sigma):
        """Return a 1D Gaussian kernel as a row vector of shape ``(1, k)``."""
        k = int(6 * sigma + 1)
        half = (k - 1) // 2
        x = np.arange(-half, half + 1, dtype=float)
        kernel = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -(x ** 2) / (2 * sigma ** 2)
        )
        return np.expand_dims(kernel, axis=0)

    def gauss_2nd_deriv_kernel(self, sigma):
        """Return the 1D second derivative of a Gaussian as a row vector."""
        k = int(6 * sigma + 1)
        half = (k - 1) // 2
        x = np.arange(-half, half + 1, dtype=float)
        gaussian = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -(x ** 2) / (2 * sigma ** 2)
        )
        kernel = gaussian * (x ** 2 - sigma ** 2) / sigma ** 4
        return np.expand_dims(kernel, axis=0)

    def second_derivatives(self, image, sigma):
        """Return the Gaussian-smoothed second derivatives ``(Lxx, Lyy)``."""
        gaussian = self.gauss_kernel(sigma)
        second = self.gauss_2nd_deriv_kernel(sigma)

        # x is the column direction: filter across columns, then down rows.
        Lxx = signal.convolve2d(image, second, mode='same', boundary='symm')
        Lxx = signal.convolve2d(Lxx, gaussian.T, mode='same', boundary='symm')

        # y is the row direction: filter down rows, then across columns.
        Lyy = signal.convolve2d(image, second.T, mode='same', boundary='symm')
        Lyy = signal.convolve2d(Lyy, gaussian, mode='same', boundary='symm')
        return Lxx, Lyy

    def laplacian(self, image, sigma):
        """Return the scale-normalised Laplacian of Gaussian at one scale."""
        Lxx, Lyy = self.second_derivatives(image, sigma)
        return sigma ** 2 * (Lxx + Lyy)

    def build_scale_space(self, image):
        """Stack the LoG response at every configured scale into a 3D array."""
        layers = []
        for sigma in tqdm(self.sigma_vals, desc="Building scale space"):
            layers.append(self.laplacian(image, sigma))
        return np.dstack(layers)

    def candidate_points(self, scale_space):
        """Return interior ``(row, col, scale)`` points above the threshold."""
        interior = scale_space[1:-1, 1:-1, 1:-1]
        points = np.argwhere(interior > self.thresh)
        return points + 1

    def lap_max_det(self, scale_space, row, col, k):
        """Return a point's ``[magnitude, row, col, sigma]`` if it is a maximum."""
        rows, cols, scales = scale_space.shape
        if not (1 <= row < rows - 1 and 1 <= col < cols - 1 and 1 <= k < scales - 1):
            return []

        value = scale_space[row, col, k]
        neighbourhood = scale_space[row - 1:row + 2, col - 1:col + 2, k - 1:k + 2]
        if value + 0.001 > np.max(neighbourhood):
            return [value, row, col, self.sigma_vals[k]]
        return []

    def find_maxima(self, scale_space):
        """Return every candidate that is a local maximum as an ``(N, 4)`` array."""
        maxima = []
        for row, col, k in tqdm(
            self.candidate_points(scale_space), desc="Finding maxima"
        ):
            result = self.lap_max_det(scale_space, row, col, k)
            if result:
                maxima.append(result)

        if not maxima:
            return np.empty((0, 4))
        return np.array(maxima)

    def blob_circles(self, maxima):
        """Convert maxima to integer ``[row, col, radius]`` blob circles."""
        radii = np.ceil(np.sqrt(2) * maxima[:, 3])
        return np.column_stack((maxima[:, 1], maxima[:, 2], radii)).astype(int)

    def detect_blobs(self, image):
        """Run the full detector, retaining its scale space, maxima, and blobs."""
        self.laplace_scale_space = self.build_scale_space(image)
        self.maxima = self.find_maxima(self.laplace_scale_space)
        self.blobs = self.blob_circles(self.maxima)
        return self.blobs
