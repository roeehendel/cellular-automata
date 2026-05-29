import math

import jax
import jax.numpy as jnp


def _get_circular_offsets(radius: float = 2.0):
    """
    Return a list of (dx, dy) offsets for all cells whose center
    is within 'radius' in Euclidean distance, excluding (0,0).
    """
    offsets = []
    r_sq = radius * radius
    max_d = int(math.ceil(radius))
    for dx in range(-max_d, max_d + 1):
        for dy in range(-max_d, max_d + 1):
            if dx == 0 and dy == 0:
                continue
            dist_sq = dx * dx + dy * dy
            if dist_sq <= r_sq:
                offsets.append((dx, dy))
    return offsets


# Precompute the offsets just once
CIRCULAR_OFFSETS = _get_circular_offsets(radius=3.0)


def greenberg_hastings_step_fn(state: jnp.ndarray, dt: float) -> jnp.ndarray:
    """
    Greenberg–Hastings update with a circular neighborhood:
      - -1 -> refractory
      -  0 -> resting
      -  1 -> excited
    """
    x = state[:, :, 0]  # shape: (n, n)

    # Pad the array so we don't go out of bounds when checking neighbors
    # Padding with 2 is enough for radius=2
    x_padded = jnp.pad(x, pad_width=2, mode="constant", constant_values=0)

    # We'll define a helper to check if cell (i,j) has any excited neighbor
    def has_excited_neighbor(i, j):
        ip = i + 2  # shift due to padding
        jp = j + 2
        # Gather neighbor values within the circular radius
        neighbor_vals = []
        for dx, dy in CIRCULAR_OFFSETS:
            neighbor_vals.append(x_padded[ip + dx, jp + dy])
        return jnp.any(jnp.array(neighbor_vals) == 1)

    # Vectorize over all i, j
    i_coords = jnp.arange(x.shape[0])
    j_coords = jnp.arange(x.shape[1])

    def check_row(i):
        return jax.vmap(lambda j: has_excited_neighbor(i, j))(j_coords)

    excited_neighbor = jax.vmap(check_row)(i_coords)

    # Standard GH transitions:
    #  (0) resting -> (1) excited if any neighbor is excited
    #  (1) excited -> (-1) refractory
    #  (-1) refractory -> (0) resting
    new_x = jnp.where(
        x == 0, jnp.where(excited_neighbor, 1, 0), jnp.where(x == 1, -1, 0)
    )

    return new_x[..., None]
