import logging
from functools import partial

import jax.numpy as jnp
from jax import jit

from cellular_automata.engine import simulate
from cellular_automata.video import default_to_color_fn, export_video


def wave_init_fn(grid_size: tuple[int, int]) -> jnp.ndarray:
    """
    Initialize the grid with zeros
    """
    return jnp.zeros((*grid_size, 2))


@jit
def wave_step_fn(middle, north, south, east, west, dt: float, c: float = 0.5):
    x = middle[0]
    v = middle[1]
    x_n = north[0]
    x_s = south[0]
    x_e = east[0]
    x_w = west[0]

    neighbors_avg_x = (x_n + x_s + x_e + x_w) / 4.0
    laplacian = 4.0 * (neighbors_avg_x - x)

    v_next = v + c**2 * laplacian * dt
    x_next = x + v_next * dt

    return jnp.array([x_next, v_next])


def wave_boundary_init_fn(
    grid_shape: tuple[int, int], source_x: int, source_y: int, source_r: int
) -> jnp.ndarray:
    """
    Creates a barrier with two slits and marks a circular source region
    """
    rows, cols = grid_shape

    # Create coordinate grids
    y, x = jnp.meshgrid(jnp.arange(cols), jnp.arange(rows))

    # Initialize state array
    state = jnp.full((rows, cols, 2), -jnp.inf)

    # Define key positions and dimensions
    barrier_x = rows // 2
    slit_width = 5
    gap_between_slits = 20
    slit1_center = cols // 2 - gap_between_slits // 2
    slit2_center = cols // 2 + gap_between_slits // 2

    # Create masks for different regions
    edges_mask = (x == 0) | (x == rows - 1) | (y == 0) | (y == cols - 1)
    barrier_mask = x == barrier_x
    source_mask = ((x - source_x) ** 2 + (y - source_y) ** 2) <= source_r**2
    slit1_mask = (x == barrier_x) & (jnp.abs(y - slit1_center) <= slit_width // 2)
    slit2_mask = (x == barrier_x) & (jnp.abs(y - slit2_center) <= slit_width // 2)

    # Combine all boundary conditions
    boundary_mask = edges_mask | barrier_mask | source_mask
    # Remove slits from boundary
    boundary_mask = boundary_mask & ~(slit1_mask | slit2_mask)

    # Apply all masks at once
    state = state.at[boundary_mask].set(0.0)

    return state


@jit
def wave_boundary_step_fn(
    state: jnp.ndarray,
    dt: float,
    source_x: int,
    source_y: int,
    source_r: int,
    frequency: float = 0.05,
    amplitude: float = 1.0,
):
    """
    Boundary cells: fixed at 0 except for circular source which oscillates
    """
    rows, cols = state.shape[:2]

    # Get current time from a reference point in the source
    t = state[source_x, source_y, 1]  # Use second channel as time
    new_t = t + dt

    # Calculate new oscillation value
    new_value = amplitude * jnp.sin(2 * jnp.pi * frequency * t)

    # Create coordinate grids
    y, x = jnp.meshgrid(jnp.arange(cols), jnp.arange(rows))

    # Calculate distances from source center
    distances = (x - source_x) ** 2 + (y - source_y) ** 2

    # Create the update using where instead of boolean indexing
    new_displacement = jnp.where(distances <= source_r**2, new_value, state[..., 0])
    new_time = jnp.where(distances <= source_r**2, new_t, state[..., 1])

    state = state.at[..., 0].set(new_displacement)
    state = state.at[..., 1].set(new_time)

    return state


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    rows, cols = 600, 600
    grid_shape = (rows, cols)

    source_x = rows // 3
    source_y = cols // 2
    source_r = 5

    boundary_init_fn = partial(
        wave_boundary_init_fn,
        source_x=source_x,
        source_y=source_y,
        source_r=source_r,
    )
    boundary_step_fn = partial(
        wave_boundary_step_fn,
        source_x=source_x,
        source_y=source_y,
        source_r=source_r,
    )

    states = simulate(
        init_fn=wave_init_fn,
        step_fn=wave_step_fn,
        boundary_init_fn=boundary_init_fn,
        boundary_step_fn=boundary_step_fn,
        grid_shape=grid_shape,
        dt=0.1,
        total_time=1200,
    )

    export_video(
        states=states,
        simulation_dt=0.1,
        video_dt=200.0,
        video_fps=30,
        to_color_fn=default_to_color_fn,
        out_filename="wave.mp4",
        # play=True,
    )
