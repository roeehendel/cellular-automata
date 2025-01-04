import logging
import time
from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import config

# config.update("jax_platforms", "METAL")
config.update("jax_platforms", "cpu")

logger = logging.getLogger(__name__)

DEFAULT_DTYPE = jnp.float16


def simulate(
    init_fn: Callable[[tuple], jnp.ndarray],
    step_fn: Callable[[jnp.ndarray, float], jnp.ndarray],
    boundary_init_fn: Callable[[tuple], jnp.ndarray],
    boundary_step_fn: Callable[[jnp.ndarray, float], jnp.ndarray],
    state_shape: tuple,
    simulation_duration: float,
    simulation_dt: float,
    output_rate: float = 1.0,
    output_fps: int = 30,
    state_dtype: jnp.dtype = jnp.float16,
) -> jnp.ndarray:
    """
    Simulate the system for the given time period.
    Returns an array of states over time.

    Args:
        init_fn (Callable[[tuple], jnp.ndarray]): takes a shape tuple and returns initial state array.
        step_fn (Callable[[jnp.ndarray, float], jnp.ndarray]): takes state and dt, returns next state.
        boundary_init_fn (Callable[[tuple], jnp.ndarray]): takes a shape tuple and returns initial boundary state array.
        boundary_step_fn (Callable[[jnp.ndarray, float], jnp.ndarray]): takes state and dt, returns next boundary state.
        state_shape (tuple): Shape of the state array.
        simulation_duration (float): Total simulation time.
        simulation_dt (float): Time step size.
        output_rate (float, optional): Output speed over simulation speed. Defaults to 1.0.
        output_fps (int, optional): Frames per second to output. Defaults to 30.
        state_dtype (jnp.dtype, optional): Data type for state arrays. Defaults to jnp.float16.

    Returns:
        jnp.ndarray: Array of states over time with shape [steps, *state_shape].
    """

    state = init_fn(state_shape).astype(state_dtype)
    boundary_state = boundary_init_fn(state_shape).astype(state_dtype)

    boundary_mask = (~jnp.isinf(boundary_state).any(axis=-1))[..., None]

    init_state = jnp.where(boundary_mask, boundary_state, state)

    simulation_fps = 1 / simulation_dt
    simulation_steps_per_output_step = int(output_rate * (simulation_fps / output_fps))
    if simulation_steps_per_output_step < 1:
        raise ValueError(
            f"{simulation_steps_per_output_step=} must be greater than 1.\n"
            f"output_rate: {output_rate}, simulation_fps (1/dt): {simulation_fps}, output_fps: {output_fps}\n"
            f"simulation_steps_per_output_step = int(output_rate * (simulation_fps / output_fps))"
        )

    @jax.jit
    def simulation_step(state: jnp.ndarray) -> jnp.ndarray:
        interior_result = step_fn(state.copy(), simulation_dt).astype(state_dtype)
        boundary_result = boundary_step_fn(state.copy(), simulation_dt).astype(
            state_dtype
        )
        return jnp.where(boundary_mask, boundary_result, interior_result)

    @jax.jit
    def output_step(state: jnp.ndarray, _) -> tuple[jnp.ndarray, jnp.ndarray]:
        state = lax.fori_loop(
            lower=0,
            upper=simulation_steps_per_output_step,
            body_fun=lambda _, s: simulation_step(s),
            init_val=state,
        )
        return state, state

    simulation_steps = int(simulation_duration / simulation_dt)
    output_steps = simulation_steps // simulation_steps_per_output_step

    _, states = lax.scan(
        output_step,
        init=init_state,
        length=output_steps,
    )

    # Include initial state in output
    states = jnp.concatenate([init_state[None, ...], states], axis=0)

    logging.info("Simulating")
    tic = time.perf_counter()
    states = jax.block_until_ready(states)
    toc = time.perf_counter()
    logging.info(f"Simulation complete in {toc - tic:.2f} seconds")

    return states
