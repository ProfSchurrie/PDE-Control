import sys; sys.path.append('../PhiFlow')
import numpy as np
from phi.geom import Geometry
import matplotlib.pyplot as plt

class Triangle(Geometry):
    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)

    def value_at(self, points):
        # points: (..., 2)
        p = np.reshape(points, (-1, 2))
        v0 = self.c - self.a
        v1 = self.b - self.a
        v2 = p - self.a

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot11 = np.dot(v1, v1)
        dot02 = np.einsum('ij,j->i', v2, v0)
        dot12 = np.einsum('ij,j->i', v2, v1)

        denom = dot00 * dot11 - dot01 * dot01
        u = (dot11 * dot02 - dot01 * dot12) / denom
        v = (dot00 * dot12 - dot01 * dot02) / denom

        inside = (u >= 0) & (v >= 0) & (u + v <= 1)
        return np.reshape(inside.astype(np.float32), points.shape[:-1] + (1,))

if __name__ == '__main__':

    from phi.geom import AABox
    from phi.physics.field import CenteredGrid

    resolution = (64, 64)
    # Create dummy domain
    domain_box = AABox(0, resolution)
    points = CenteredGrid.getpoints(domain_box, resolution).data

    # Define triangle points (centered roughly)
    a = (32, 16)
    b = (48, 48)
    c = (16, 48)

    triangle = Triangle(a, b, c)
    values = triangle.value_at(points)[0, ..., 0]  # Remove batch & channel dims

    # Plot result
    plt.imshow(values, origin='lower', cmap='Greys')
    plt.title("Triangle shape test")
    plt.colorbar()
    plt.show()