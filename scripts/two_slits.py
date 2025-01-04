import logging
from functools import partial

import jax.numpy as jnp

from cellular_automata.dynamics.wave import wave_step_fn
from cellular_automata.simulation import simulate
from cellular_automata.video import default_to_color_fn, export_video


def init_fn(state_shape: tuple) -> jnp.ndarray:
    """Initialize the grid with zeros"""
    return jnp.zeros(state_shape)


def boundary_init_fn(
    state_shape: tuple,
) -> jnp.ndarray:
    """Creates a barrier with two slits and marks a line source region at the top"""
    rows, cols = state_shape[:2]

    # Create coordinate grids
    y, x = jnp.meshgrid(jnp.arange(cols), jnp.arange(rows))

    # Initialize state array with zeros
    state = jnp.full(state_shape, jnp.inf)

    # Define barrier and slits
    barrier_x = rows // 2
    slit_width = 5
    gap_between_slits = 20
    slit1_center = cols // 2 - gap_between_slits // 2
    slit2_center = cols // 2 + gap_between_slits // 2

    # Create masks for different regions
    source_mask = x == 0  # Line source near the top
    barrier_mask = x == barrier_x
    slit1_mask = (x == barrier_x) & (jnp.abs(y - slit1_center) <= slit_width // 2)
    slit2_mask = (x == barrier_x) & (jnp.abs(y - slit2_center) <= slit_width // 2)
    barrier_mask = barrier_mask & ~(slit1_mask | slit2_mask)

    # Combine masks
    boundary_mask = barrier_mask | source_mask

    # Set fixed boundaries to zero, except source which starts at [1.0, 0.0]
    state = state.at[boundary_mask].set(jnp.array([0.0, 0.0]))
    state = state.at[source_mask].set(jnp.array([1.0, 0.0]))

    return state


def boundary_step_fn(
    state: jnp.ndarray,
    dt: float,
    frequency: float,
    amplitude: float = 1.0,
) -> jnp.ndarray:
    """Update line source with oscillating wave and handle boundaries"""
    rows, cols = state.shape[:2]
    y, x = jnp.meshgrid(jnp.arange(cols), jnp.arange(rows))

    # Get current position and velocity at source (using first point of line)
    current_pos = state[0, 0, 0]
    current_vel = state[0, 0, 1]

    # Calculate new position and velocity for harmonic oscillator
    omega = 2 * jnp.pi * frequency
    new_pos = current_pos + current_vel * dt
    new_vel = current_vel - (omega**2) * current_pos * dt

    # Update line source region
    source_mask = x == 0
    new_x = jnp.where(source_mask, new_pos, state[..., 0])
    new_v = jnp.where(source_mask, new_vel, state[..., 1])

    # Create barrier with slits
    barrier_x = rows // 2
    slit_width = 5
    gap_between_slits = 20
    slit1_center = cols // 2 - gap_between_slits // 2
    slit2_center = cols // 2 + gap_between_slits // 2

    barrier_mask = (x == barrier_x) & ~(
        (jnp.abs(y - slit1_center) <= slit_width // 2)
        | (jnp.abs(y - slit2_center) <= slit_width // 2)
    )

    # Set barrier points to zero
    new_x = jnp.where(barrier_mask, 0.0, new_x)
    new_v = jnp.where(barrier_mask, 0.0, new_v)

    return jnp.stack([new_x, new_v], axis=-1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    rows, cols, channels = 600, 600, 2
    state_shape = (rows, cols, channels)
    dt = 0.1
    c = 0.5

    frequency = 0.05
    amplitude = 2.0

    simulation_duration = 2000

    video_duration = 10  # seconds
    video_fps = 30

    output_rate = simulation_duration / video_duration

    states = simulate(
        init_fn=init_fn,
        step_fn=lambda state, dt=dt: wave_step_fn(state, dt, c=c),
        boundary_init_fn=boundary_init_fn,
        boundary_step_fn=partial(
            boundary_step_fn,
            frequency=frequency,
            amplitude=amplitude,
        ),
        state_shape=state_shape,
        simulation_duration=simulation_duration,
        simulation_dt=dt,
        output_rate=output_rate,
        output_fps=video_fps,
    )

    size_in_bytes = states.nbytes
    size_in_mb = size_in_bytes / (1024 * 1024)
    print(f"States array size: {size_in_mb:.2f} MB")

    export_video(
        states=states,
        video_fps=video_fps,
        to_color_fn=default_to_color_fn,
        # out_filename="wave.mp4",
        play=True,
    )
