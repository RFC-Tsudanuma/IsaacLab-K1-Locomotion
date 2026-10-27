"""Shared rough tile geometry and matching piecewise-linear height queries."""
import numpy as np
import torch


def make_tile(size, resolution, amplitude):
    coordinates = np.linspace(-size / 2., size / 2., resolution, dtype=np.float32)
    heights = np.random.uniform(0., amplitude, (resolution, resolution)).astype(np.float32)
    heights[[0, -1], :] = 0.
    heights[:, [0, -1]] = 0.
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    vertices = np.stack((x, y, heights), axis=-1).reshape(-1, 3)
    faces = []
    for row in range(resolution - 1):
        for col in range(resolution - 1):
            a = row * resolution + col
            faces.extend(((a, a + resolution, a + 1),
                          (a + 1, a + resolution, a + resolution + 1)))
    return vertices, np.asarray(faces, dtype=np.int32), heights


def tile_height(points, heights, size):
    """Match the two triangles used by make_tile, including flat exterior."""
    count = heights.shape[0]
    uv = (points[..., :2] + size / 2.) * ((count - 1) / size)
    inside = ((uv >= 0.) & (uv < count - 1)).all(-1)
    uv = uv.clamp(0., count - 1.00001)
    cell = uv.floor().long()
    u, v = (uv - cell).unbind(-1)
    i, j = cell.unbind(-1)
    h00, h10 = heights[i, j], heights[i + 1, j]
    h01, h11 = heights[i, j + 1], heights[i + 1, j + 1]
    lower = h00 + u * (h10 - h00) + v * (h01 - h00)
    upper = h11 + (1. - u) * (h01 - h11) + (1. - v) * (h10 - h11)
    return torch.where(inside, torch.where(u + v <= 1., lower, upper), 0.)
