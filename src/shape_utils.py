import os
import random

import numpy as np
from phi.geom import Sphere, box

from skimage.draw import polygon

def rasterize_triangle(coords, resolution):
    """Rasterize triangle from 3 xy coords into a 2D boolean mask."""
    rr, cc = polygon([p[1] for p in coords], [p[0] for p in coords], resolution)
    mask = np.zeros(resolution, dtype=bool)
    mask[rr, cc] = True
    return mask

def rasterize_polygon(xy_coords, resolution):
    rr, cc = polygon([p[1] for p in xy_coords], [p[0] for p in xy_coords], resolution)
    mask = np.zeros(resolution, dtype=bool)
    mask[rr, cc] = True
    return mask

def load_shapes(shape_dir):
    files = [os.path.join(shape_dir, file) for file in os.listdir(shape_dir) if file.endswith('.npz')]
    npzs = [np.load(file) for file in files]
    arrays = [npz[npz.files[0]][..., 0] for npz in npzs]
    return arrays



# def make_crescent(domain, batch_size):
#     size = np.random.randint(10, 14, (batch_size, 2))
#     radius = size[..., 0] / 2.0
#     pos0 = np.random.randint(10, 56, (batch_size, 2))
#     center_offset = np.array([-size[:, 0] // 4, -size[:, 1] // 4]).T
#     center = pos0 + center_offset
#
#     crescents = []
#     for b in range(batch_size):
#         cx, cy = center[b]
#         r = radius[b]
#
#         outer = Sphere(center=(cx, cy), radius=r)
#         inner = Sphere(center=(cx + r * 0.4, cy), radius=r * 0.6)
#
#         outer_mask = outer.approximate_signed_distance(domain.center_points()) < 0
#         inner_mask = inner.approximate_signed_distance(domain.center_points()) < 0
#
#         crescent_mask = outer_mask & ~inner_mask
#         value = crescent_mask.astype(np.float32)
#         crescents.append(value)
#
#     return np.stack(crescents, axis=0)

def make_crescent(domain, batch_size):
    H, W = domain.resolution
    points = domain.center_points()

    size = np.random.randint(10, 14, (batch_size, 2))
    radius = size[..., 0] / 2.0
    pos0 = np.random.randint(10, 56, (batch_size, 2))
    center_offset = np.array([-size[:, 0] // 4, -size[:, 1] // 4]).T
    center = pos0 + center_offset

    crescents = []
    for b in range(batch_size):
        cx, cy = center[b]
        r = radius[b]

        outer = Sphere(center=(cx, cy), radius=r)
        inner = Sphere(center=(cx + r * 0.4, cy), radius=r * 0.6)

        outer_mask = outer.approximate_signed_distance(points) < 0
        inner_mask = inner.approximate_signed_distance(points) < 0

        crescent_mask = outer_mask & ~inner_mask
        mask = crescent_mask.astype(np.float32).reshape(H, W)
        crescents.append(mask)

    batch = np.stack(crescents, axis=0)        # (B, H, W)
    return batch[..., None]                    # (B, H, W, 1)

def make_vortex(domain, batch_size):
    H, W = domain.resolution
    coords = domain.center_points()  # shape: (H*W, 2)
    coords = coords.reshape(H, W, 2)

    # Convert grid to polar coordinates (centered)
    cx, cy = W // 2, H // 2
    x = coords[..., 0] - cx
    y = coords[..., 1] - cy
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Normalize
    r_norm = r / np.max(r)
    theta_norm = (theta + np.pi) / (2 * np.pi)  # [0,1]

    vortices = []
    for b in range(batch_size):
        # Parameters per sample
        n_arms = np.random.randint(2, 6)  # spiral arms
        tightness = np.random.uniform(2.0, 6.0)  # spiral twist
        thickness = np.random.uniform(0.05, 0.15)

        # Spiral function: sin(n * theta + tightness * r)
        spiral = 0.5 * (1 + np.sin(n_arms * theta + tightness * r))
        mask = (spiral > (1 - thickness)) & (r_norm < 0.8)  # trim at max radius

        vortices.append(mask.astype(np.float32))

    return add_channel_dim(np.stack(vortices, axis=0))



def make_cross(domain, batch_size):
    size = np.random.randint(8, 14, (batch_size,))
    thickness = np.random.randint(2, 5, (batch_size,))

    positions = np.random.randint(20, 44, (batch_size, 2))
    points = domain.center_points()
    H, W = domain.resolution

    crosses = []
    for b in range(batch_size):
        cx, cy = positions[b]
        s = size[b]
        t = thickness[b]

        # Horizontal bar
        hbox = box[cx - s/2 : cx + s/2, cy - t/2 : cy + t/2]
        # Vertical bar
        vbox = box[cx - t/2 : cx + t/2, cy - s/2 : cy + s/2]

        mask = (hbox.approximate_signed_distance(points) < 0) | \
               (vbox.approximate_signed_distance(points) < 0)

        crosses.append(mask.reshape(H, W).astype(np.float32))

    return add_channel_dim(np.stack(crosses, axis=0))



def make_T(domain, batch_size):
    width = np.random.randint(8, 14, (batch_size,))
    height = np.random.randint(6, 10, (batch_size,))
    stem_width = np.random.randint(2, 4, (batch_size,))

    positions = np.random.randint(20, 44, (batch_size, 2))
    points = domain.center_points()
    H, W = domain.resolution

    Ts = []
    for b in range(batch_size):
        cx, cy = positions[b]
        w = width[b]
        h = height[b]
        sw = stem_width[b]

        # Top bar
        top = box[cx - w/2 : cx + w/2, cy - sw/2 : cy + sw/2]
        # Stem (extends downward from center)
        stem = box[cx - sw/2 : cx + sw/2, cy - sw/2 : cy + h/2]

        mask = (top.approximate_signed_distance(points) < 0) | \
               (stem.approximate_signed_distance(points) < 0)

        Ts.append(mask.reshape(H, W).astype(np.float32))

    return add_channel_dim(np.stack(Ts, axis=0))

def make_hexagon(domain, batch_size):
    H, W = domain.resolution
    hexagons = []

    for b in range(batch_size):
        r = np.random.randint(6, 10)
        cx = np.random.randint(20, W - 20)
        cy = np.random.randint(20, H - 20)

        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        coords = np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles)], axis=1)

        mask = rasterize_polygon(coords, (H, W))
        hexagons.append(mask.astype(np.float32))

    return add_channel_dim(np.stack(hexagons, axis=0))

def make_U(domain, batch_size):
    H, W = domain.resolution
    points = domain.center_points()

    Us = []
    for b in range(batch_size):
        cx, cy = np.random.randint(20, 44, size=2)
        w = np.random.randint(10, 14)
        h = np.random.randint(10, 14)
        t = np.random.randint(2, 4)

        # Left bar
        left = box[cx - w/2 : cx - w/2 + t, cy - h/2 : cy + h/2]
        # Right bar
        right = box[cx + w/2 - t : cx + w/2, cy - h/2 : cy + h/2]
        # Bottom bar
        bottom = box[cx - w/2 : cx + w/2, cy + h/2 - t : cy + h/2]

        mask = (left.approximate_signed_distance(points) < 0) | \
               (right.approximate_signed_distance(points) < 0) | \
               (bottom.approximate_signed_distance(points) < 0)

        Us.append(mask.reshape(H, W).astype(np.float32))

    return add_channel_dim(np.stack(Us, axis=0))

def make_L(domain, batch_size):
    H, W = domain.resolution
    points = domain.center_points()

    Ls = []
    for b in range(batch_size):
        cx, cy = np.random.randint(20, 44, size=2)
        w = np.random.randint(8, 12)
        h = np.random.randint(10, 14)
        t = np.random.randint(2, 4)

        # Vertical bar
        vert = box[cx - w/2 : cx - w/2 + t, cy - h/2 : cy + h/2]
        # Horizontal foot
        horiz = box[cx - w/2 : cx + w/2, cy + h/2 - t : cy + h/2]

        mask = (vert.approximate_signed_distance(points) < 0) | \
               (horiz.approximate_signed_distance(points) < 0)

        Ls.append(mask.reshape(H, W).astype(np.float32))

    return add_channel_dim(np.stack(Ls, axis=0))

# def make_arrow(domain, batch_size):
#     shaft_length = np.random.randint(10, 14, (batch_size,))
#     shaft_width = np.random.randint(2, 4, (batch_size,))
#     head_size = np.random.randint(6, 9, (batch_size,))
#
#     positions = np.random.randint(20, 44, (batch_size, 2))
#     H, W = domain.resolution
#     points = domain.center_points()
#
#     arrows = []
#     for b in range(batch_size):
#         cx, cy = positions[b]
#         l = shaft_length[b]
#         w = shaft_width[b]
#         hs = head_size[b]
#
#         # Shaft using phi.box
#         shaft = box[cx - w/2 : cx + w/2,
#                     cy - l/2 : cy + l/2]
#
#         shaft_mask = shaft.approximate_signed_distance(points) < 0
#         shaft_mask = shaft_mask.reshape(H, W)
#
#         # Triangle head rasterized manually
#         tip_coords = np.array([
#             [cx - hs, cy - l // 2],
#             [cx + hs, cy - l // 2],
#             [cx, cy - l // 2 - hs]
#         ])
#         tip_mask = rasterize_triangle(tip_coords, (H, W))
#
#         combined = shaft_mask | tip_mask
#         arrows.append(combined.astype(np.float32))
#
#     return add_channel_dim(np.stack(arrows, axis=0))


# def make_arrow(domain, batch_size):
#     shaft_length = np.random.randint(10, 14, (batch_size,))
#     shaft_width = np.random.randint(2, 4, (batch_size,))
#     head_size = np.random.randint(6, 9, (batch_size,))
#
#     positions = np.random.randint(20, 44, (batch_size, 2))
#     H, W = domain.resolution
#     points = domain.center_points()
#
#     arrows = []
#     for b in range(batch_size):
#         cx, cy = positions[b]
#         l = shaft_length[b]
#         w = shaft_width[b]
#         hs = head_size[b]
#
#         # Shaft pointing right (horizontal rectangle)
#         dx = l // 2 - w / 2
#         dy = l
#
#         shaft = box[
#                 cx + w / 2 - w + dx: cx + w / 2 + dx,
#                 cy - l / 2 - l + dy: cy - l / 2 + dy
#                 ]
#
#         shaft_mask = shaft.approximate_signed_distance(points) < 0
#         shaft_mask = shaft_mask.reshape(H, W)
#
#         # Triangle head pointing right
#         tip_coords = np.array([
#             [cx + l // 2 + hs, cy],              # tip
#             [cx + l // 2,       cy - hs],        # bottom corner
#             [cx + l // 2,       cy + hs]         # top corner
#         ])
#         tip_mask = rasterize_triangle(tip_coords, (H, W))
#
#         combined = shaft_mask | tip_mask
#         arrows.append(combined.astype(np.float32))
#
#     return add_channel_dim(np.stack(arrows, axis=0))


def distribute_random_shape(resolution, batch_size, shape_library, margin=1):
    array = np.zeros((batch_size,) + tuple(resolution) + (1,), np.float32)
    for batch in range(batch_size):
        shape = random.choice(shape_library)
        y = random.randint(margin, resolution[0] - margin - shape.shape[0] - 2)
        x = random.randint(margin, resolution[1] - margin - shape.shape[1] - 2)
        array[batch, y:(y + shape.shape[0]), x:(x + shape.shape[1]), 0] = shape
    assert array.dtype == np.float32
    return array

from scipy.ndimage import rotate, zoom                # needed for the make_* functions
# (import make_arrow, make_L, … here or place this code in the same file)

# ---------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------
def _transform_mask(mask, scale, angle, resolution):
    """
    Scale → pad or crop to target resolution → rotate.
    """
    from scipy.ndimage import zoom, rotate

    H, W = resolution
    h, w = mask.shape

    # Scale the mask
    scaled = zoom(mask, scale, order=0)
    h_s, w_s = scaled.shape

    # Center crop or pad to (H, W)
    pad_crop = []

    for target, current in zip((H, W), (h_s, w_s)):
        delta = target - current
        if delta >= 0:
            # Need padding
            pad_before = delta // 2
            pad_after = delta - pad_before
            pad_crop.append((pad_before, pad_after))
        else:
            # Need cropping
            crop_start = (-delta) // 2
            crop_end = crop_start + target
            pad_crop.append((crop_start, crop_end))

    if h_s < H or w_s < W:
        # Padding case
        scaled = np.pad(scaled, pad_crop, mode='constant')
    else:
        # Cropping case
        scaled = scaled[pad_crop[0][0]:pad_crop[0][1],
                        pad_crop[1][0]:pad_crop[1][1]]

    # Rotate
    rotated = rotate(scaled, angle, reshape=False, order=0)
    return rotated



def add_channel_dim(batch):
    """
    Ensures the shape is (B, H, W, 1).
    Input:  (B, H, W)
    Output: (B, H, W, 1)
    """
    if batch.ndim == 3:
        return batch[..., np.newaxis]
    raise ValueError(f"Expected shape (B, H, W), got {batch.shape}")

def remove_channel_dim(batch):
    """
    Reduces shape (B, H, W, 1) → (B, H, W)
    or (H, W, 1) → (H, W)
    """
    if batch.ndim == 4 and batch.shape[-1] == 1:
        return batch[..., 0]
    elif batch.ndim == 3 and batch.shape[-1] == 1:
        return batch[..., 0]
    raise ValueError(f"Expected shape (..., 1), got {batch.shape}")


# ---------------------------------------------------------------------
# main sampler
# ---------------------------------------------------------------------
def sample_shape_batch(domain,
                       batch_size,
                       shape_library,
                       margin=1,
                       generators=None):
    """
    Returns a (batch, H, W, 1) float32 array.
      – 3/11 chance: pick a pre-ras­terised shape from `shape_library`
      – 1/11 chance each: make_arrow, make_L, make_U, make_hexagon,
                         make_T, make_cross, make_vortex, make_crescent
    Each “live” shape is randomly rotated (0, 90, 180, 270 deg)
    and uniformly scaled s ∈ [0.5, 1.5].
    """
    if generators is None:     # default 8 specialised generators
        generators = [
            make_L, make_U, make_hexagon,
            make_T,    make_cross, make_vortex, make_crescent
        ]

    H, W   = domain.resolution
    output = np.zeros((batch_size, H, W, 1), dtype=np.float32)

    for b in range(batch_size):

        draw = random.randint(0, 9)     # 0–2  → “library path”, 3–10 → generators[draw-3]

        # ------------------------------------------------------------------
        # (A) pre-made shape from .npz library: re-use your old logic
        # ------------------------------------------------------------------
        if draw < 3:
            # print("square triangle circle")
            shape = random.choice(shape_library)          # numpy array (h, w)
            shape_h, shape_w = shape.shape

            # random top-left corner ensuring a margin
            y = random.randint(margin, H - margin - shape_h)
            x = random.randint(margin, W - margin - shape_w)

            output[b, y:y+shape_h, x:x+shape_w, 0] = shape.astype(np.float32)

        # ------------------------------------------------------------------
        # (B) on-the-fly shape with rotation + scale
        # ------------------------------------------------------------------
        else:
            gen_fn = generators[draw - 3]  # pick the 1/11-chance generator
            # print(gen_fn.__name__)

            for attempt in range(5):  # Retry up to 5 times
                try:
                    mask = remove_channel_dim(gen_fn(domain, batch_size=1))[0]  # (H, W)
                    angle = random.choice([0, 90, 180, 270])
                    scale = random.uniform(1.0, 2.0)

                    mask_tr = _transform_mask(mask, scale, angle, (H, W))

                    # optional random translation so the object is not always centred
                    shift_y = random.randint(- (H // 4), H // 4)
                    shift_x = random.randint(- (W // 4), W // 4)
                    mask_tr = np.roll(np.roll(mask_tr, shift_y, axis=0), shift_x, axis=1)

                    output[b, ..., 0] = mask_tr.astype(np.float32)
                    break  # success, no need to retry

                except ValueError as e:
                    print(f"[sample_shape_batch] Attempt {attempt + 1} failed with: {e}")
                    if attempt == 4:
                        raise RuntimeError(f"Failed to generate valid shape after 5 attempts.") from e

    return output

if __name__ == '__main__':
    from phi.flow import Domain
    from pprint import pprint
    domain = Domain([64, 64])
    batch_size = 1
    shape_library = load_shapes('../notebooks/shapes')
    state = sample_shape_batch(domain,batch_size, shape_library)
    import pylab
    batch_count = state.shape[0]
    batches = list(range(min(3, batch_count)))  # show up to 3 batches safely

    pylab.subplots(len(batches), 1, figsize=(6, 6))
    for i, batch in enumerate(batches):
        pylab.subplot(len(batches), 1, i + 1)
        pylab.imshow(state[batch, ..., 0], origin='lower', cmap='Greys')
        pylab.title(f"Target shape, batch {batch}")
    pylab.tight_layout()
    pylab.show()
