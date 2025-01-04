import jax.numpy as jnp


def trinary_wave_step_fn(
    state: jnp.ndarray,
    dt: float,
    c: float,
    damping: float = 0.0,
) -> jnp.ndarray:
    """
    Compute wave equation update where state values are -1,0,1
    state shape: [n, n, 2] where [..., 0] is position and [..., 1] is velocity
    """
    x = state[..., 0]
    v = state[..., 1]

    # Create padded versions of x that include the boundary values
    x_padded_n = jnp.vstack([x, x[-1:, :]])  # Pad bottom
    x_padded_s = jnp.vstack([x[:1, :], x])  # Pad top
    x_padded_e = jnp.hstack([x, x[:, -1:]])  # Pad right
    x_padded_w = jnp.hstack([x[:, :1], x])  # Pad left

    # Now we can safely slice to get neighbors
    x_n = x_padded_n[1:, :]  # Values to the north (below)
    x_s = x_padded_s[:-1, :]  # Values to the south (above)
    x_e = x_padded_e[:, 1:]  # Values to the east (right)
    x_w = x_padded_w[:, :-1]  # Values to the west (left)

    laplacian = jnp.sign(x_n + x_s + x_e + x_w - 4 * x)
    new_v = jnp.sign(v + laplacian)
    new_x = jnp.sign(x + new_v)

    return jnp.stack([new_x, new_v], axis=-1)
