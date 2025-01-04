import logging

import jax.numpy as jnp

from cellular_automata.dynamics.wave import wave_step_fn
from cellular_automata.simulation import simulate
from cellular_automata.video import default_to_color_fn, export_video


def init_fn(state_shape: tuple) -> jnp.ndarray:
    rows, cols, channels = state_shape
    state = jnp.zeros((rows, cols, channels))

    x, y = jnp.meshgrid(jnp.arange(cols), jnp.arange(rows))

    # init a sharp gaussian in the center of the grid
    x_center = cols // 4
    y_center = rows // 2

    sigma = 10
    amplitude = 2.0

    gaussian = (
        jnp.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * sigma**2))
        * amplitude
    )

    state = state.at[..., 0].set(gaussian)

    return state


def boundary_init_fn(state_shape: tuple) -> jnp.ndarray:
    state = jnp.full(state_shape, jnp.inf)

    state.at[0, :, :].set(0)
    state.at[-1, :, :].set(0)
    state.at[:, 0, :].set(0)
    state.at[:, -1, :].set(0)

    return state


def boundary_step_fn(state: jnp.ndarray, dt: float) -> jnp.ndarray:
    return state


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    rows, cols, channels = 400, 400, 2
    state_shape = (rows, cols, channels)
    dt = 0.1
    c = 1.0

    simulation_duration = (rows // 2) * 4
    video_duration = 4

    output_rate = simulation_duration / video_duration

    video_fps = 60

    states = simulate(
        init_fn=init_fn,
        step_fn=lambda state, dt=dt: wave_step_fn(state, dt, c),
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
    )
