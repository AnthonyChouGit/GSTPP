from jaxtyping import Array, Float
import jax
import jax.numpy as jnp
import equinox as eqx

@eqx.filter_jit
def get_latent_graph(E1: Float[Array, "nb_nodes feature_dim"], E2: Float[Array, "nb_nodes feature_dim"]):
    A = jnp.matmul(E1, jnp.matrix_transpose(E2)) - jnp.matmul(E2, jnp.matrix_transpose(E1))
    A = jax.nn.softplus(A)
    return A

@eqx.filter_jit
def get_dist_graph(node_loc: Float[Array, "nb_nodes loc_dim"]):
    temp = node_loc[:, None, :] - node_loc[None, :, :]
    temp = jnp.where(jnp.eye(node_loc.shape[0])[:, :, None], 0., temp)
    dist = jnp.linalg.norm(temp, axis=-1)
    A = jnp.exp(-dist)
    A = A - jnp.eye(A.shape[0])
    return A