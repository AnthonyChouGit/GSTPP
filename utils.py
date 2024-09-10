from jaxtyping import Array, Float, PyTree
import jax
import jax.numpy as jnp
import equinox as eqx
import math

@eqx.filter_jit
def forward_pass(module_list: list, x: Float[Array, "d_in"]):
    for layer in module_list:
        x = layer(x)
    return x

batched_forward = jax.vmap(forward_pass, (None, 0), 0)

@eqx.filter_jit
def normal_ll(mu: Float[Array, "hdim"], sigma: Float[Array, "hdim"], x: Float[Array, "hdim"]):
    ll = jax.scipy.stats.norm.logpdf(x, mu, sigma).sum(-1)
    return ll

@eqx.filter_jit
def normalize(x: Float[Array, "... dim"]):
    norm = jnp.linalg.norm(x, axis=-1) # (...)
    norm = jnp.where(norm>1e-12, norm, 1e-12)
    unit = x / norm[..., None]
    return unit, norm

def cosine_decay(lr, cur_step, decay_steps):
    cosine_decay = 0.5 * (1 + math.cos(math.pi * cur_step / decay_steps)) * lr
    return cosine_decay

def lr_schedule(step, warmups, base_rate, training_steps):
    if step <= warmups:
        lr = step / warmups * base_rate
    else:
        lr = cosine_decay(base_rate, step - warmups, training_steps - warmups)
    return lr

@eqx.filter_jit
def sample_mix(mu: Float[Array, "nb_nodes hdim"], sigma: Float[Array, "nb_nodes hdim"], log_weights: Float[Array, 'nb_nodes'], num_samples: int, key: Array):
    key1, key2 = jax.random.split(key)
    log_weights = jnp.broadcast_to(log_weights, (num_samples, log_weights.shape[-1]))
    clusters = jax.random.categorical(key1, log_weights) # (num_samples, )
    # clusters = log_weights.argmax(-1)
    mu = mu[clusters]
    sigma = sigma[clusters]
    eps = jax.random.normal(key2, (num_samples, mu.shape[-1]))
    loc = eps * sigma + mu
    return loc
