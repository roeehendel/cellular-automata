import logging
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

logger = logging.getLogger(__name__)


@partial(jax.jit, static_argnums=(1,))
def apply_rule(state: jnp.ndarray, step_fn):
    """
    Generalized rule application function that can handle both local and point-wise rules

    state.shape = (rows, cols, channels)
    step_fn signature depends on needs_neighbors:
        if True: step_fn(center, north, south, east, west) -> updated_center
        if False: step_fn(value) -> updated_value
    """
    rows, cols, channels = state.shape

    def update_cell(i, j):
        center = state[i, j]
        north = state[(i - 1) % rows, j]
        south = state[(i + 1) % rows, j]
        west = state[i, (j - 1) % cols]
        east = state[i, (j + 1) % cols]
        return step_fn(center, north, south, east, west)

    # Create a grid of all coordinates
    i, j = jnp.meshgrid(jnp.arange(rows), jnp.arange(cols), indexing="ij")
    coords = jnp.stack([i, j], axis=-1)
    vectorized_update = jax.vmap(lambda idx: update_cell(idx[0], idx[1]))
    return vectorized_update(coords.reshape(-1, 2)).reshape(rows, cols, -1)


def simulate(
    init_fn,
    step_fn,
    boundary_init_fn,
    boundary_step_fn,
    grid_shape: tuple[int, int],
    dt: float,
    total_time: float,
):
    """
    Simulate with separate evolution rules for boundary and non-boundary regions.

    boundary_step_fn should be a point-wise function that only depends on the previous state
    step_fn remains a local rule that depends on neighbors
    """
    initial_state = init_fn(grid_shape)
    boundary_initial_state = boundary_init_fn(grid_shape)

    boundary_mask = ~jnp.isinf(boundary_initial_state).any(axis=-1)

    step_fn_dt = partial(step_fn, dt=dt)
    boundary_step_fn_dt = partial(boundary_step_fn, dt=dt)

    def scan_fn(state: jnp.ndarray, _):
        # Update boundary points (point-wise rule)
        boundary_update = boundary_step_fn_dt(state)

        # Update non-boundary points (local rule)
        interior_update = apply_rule(state, step_fn_dt)

        # Combine updates using the mask
        new_state = jnp.where(
            boundary_mask[..., None], boundary_update, interior_update
        )
        return new_state, new_state

    steps = int(total_time / dt)

    logger.info("Simulating")
    _, states = lax.scan(scan_fn, initial_state, None, length=steps)
    logger.info("Simulation complete")

    return states
