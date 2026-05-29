import logging

import jax.numpy as jnp

from cellular_automata.dynamics.greenberg_hastings import greenberg_hastings_step_fn
from cellular_automata.simulation import simulate
from cellular_automata.video import default_to_color_fn, export_video


def init_fn(state_shape: tuple) -> jnp.ndarray:
    rows, cols, _ = state_shape

    x, y = jnp.meshgrid(jnp.arange(cols), jnp.arange(rows))

    x_center = cols // 2
    y_center = rows // 2

    circular_bump_mask = (x - x_center) ** 2 + (y - y_center) ** 2 < 10**2

    state = jnp.zeros(state_shape)
    state = state.at[circular_bump_mask, 0].set(1)

    return state


def boundary_init_fn(state_shape: tuple) -> jnp.ndarray:
    state = jnp.full(state_shape, jnp.inf)

    state = state.at[0, :, :].set(0)
    state = state.at[-1, :, :].set(0)
    state = state.at[:, 0, :].set(0)
    state = state.at[:, -1, :].set(0)

    return state


def boundary_step_fn(state: jnp.ndarray, dt: float) -> jnp.ndarray:
    return state


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cols, rows, channels = 501, 501, 1
    state_shape = (cols, rows, channels)
    dt = 1.0

    simulation_duration = 100
    video_fps = 30
    video_duration = simulation_duration // video_fps

    output_rate = simulation_duration / video_duration

    states = simulate(
        init_fn=init_fn,
        step_fn=lambda state, dt=dt: greenberg_hastings_step_fn(state, dt),
        boundary_init_fn=boundary_init_fn,
        boundary_step_fn=boundary_step_fn,
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
        play=True,
        out_filename="greenberg_hastings.mp4",
        scale=1,
    )
