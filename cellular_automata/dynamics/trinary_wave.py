import jax.numpy as jnp


def trinary_wave_step_fn(state: jnp.ndarray, dt: float) -> jnp.ndarray:
    """
    Compute wave equation update where state values are -1,0,1
    state shape: [n, n, 2] where [..., 0] is position and [..., 1] is velocity
    """
    x = state[..., 0]
    v = state[..., 1]

    # Create padded versions of x that include the boundary values
    x_padded = jnp.pad(x, pad_width=1, mode="edge")

    # Get all neighbors using the padded array
    x_n = x_padded[:-2, 1:-1]  # North
    x_s = x_padded[2:, 1:-1]  # South
    x_e = x_padded[1:-1, 2:]  # East
    x_w = x_padded[1:-1, :-2]  # West

    laplacian = jnp.sign(x_n + x_s + x_e + x_w - 4 * x)
    new_v = jnp.sign(v + laplacian)
    new_x = jnp.sign(x + new_v)

    return jnp.stack([new_x, new_v], axis=-1)
