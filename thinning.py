import jax.numpy as jnp
from jaxtyping import PyTree, Array, Float
import jax
import equinox as eqx

@eqx.filter_jit
def __extrapolate(func: callable, x0: PyTree, t0: Float[Array, '[]'], dts: Float[Array, 'T']):
    intensities, x1 = jax.vmap(func, (0, None, None), 0)(dts, x0, t0)
    return intensities, x1 # (T, )  (T, hdim)

@eqx.filter_jit
def __get_sample_rate(func: callable, x0: PyTree, t0: Float[Array, '[]'], key: Array, boundary: float, bound_samples: int, oversample_rate: float):
    times_for_bound = jax.random.uniform(key, (bound_samples, ), maxval=boundary)
    intensities_for_bound, _ = __extrapolate(func, x0, t0, times_for_bound)
    sample_rate = intensities_for_bound.max() * oversample_rate
    return sample_rate

def __forward_sample(func: callable, x0: PyTree, t0: Float[Array, '[]'], key: Array, boundary: float, num_exp: int, num_samples: int, sample_rate: float):
    key1, key2 = jax.random.split(key)
    exp_numbers = jax.random.exponential(key1, (num_exp, ))
    sampled_times = (exp_numbers / sample_rate).cumsum(-1).clip(max=boundary) # (num_exp, )

    sampled_intensities, x1 = __extrapolate(func, x0, t0, sampled_times)
    U = jax.random.uniform(key2, (num_samples, num_exp))

    criterion = U * sample_rate / sampled_intensities # (num_samples, num_exp)
    min_cri = criterion.min(-1) # (num_samples, )
    has_accepted = min_cri<1. # (num_samples, )
    temp = jnp.where(criterion>=1., boundary+1, sampled_times) # (num_samples, num_exp)
    ind = jnp.argmin(temp, -1) # (num_samples, )
    accepted_time = sampled_times[ind]
    accepted_x = jax.tree.map(lambda h: h[ind], x1)
    return accepted_time, accepted_x, has_accepted, sampled_times[-1], jax.tree.map(lambda h: h[-1], x1) # TODO

def thinning(func: callable, x0: PyTree, t0: Float[Array, '[]'], key: Array, boundary: float, num_samples: int, num_exp: int, bound_samples: int, oversample_rate: float):
    cur_key, key = jax.random.split(key)
    sample_rate = __get_sample_rate(func, x0, t0, cur_key, boundary, bound_samples, oversample_rate)
    last_dt = 0.
    have_samples = 0
    sample_dts = list()
    sample_xs = list()
    while True:
        need_sample_num = num_samples - have_samples
        cur_key, key = jax.random.split(key)
        times, xs, has_accepted, max_sample, next_x0 = __forward_sample(func, x0, t0, cur_key, boundary-last_dt, num_exp, need_sample_num, sample_rate)
        accepted_time = times[has_accepted]
        # accepted_x = xs[has_accepted] # TODO
        accepted_x = jax.tree.map(lambda h: h[has_accepted], xs)
        sample_dts.append(accepted_time+last_dt)
        sample_xs.append(accepted_x)
        valid_samples = accepted_time.shape[0]
        have_samples += valid_samples
        assert have_samples <= num_samples
        if have_samples == num_samples:
            break
        t0 = t0 + max_sample
        last_dt = last_dt + max_sample

        if last_dt >= boundary:
            x_boundary = jax.tree.map(lambda x: jnp.expand_dims(x, 0), next_x0)
            sample_dts.append(jnp.full((num_samples-have_samples), boundary))
            sample_xs.extend([x_boundary for _ in range(num_samples-have_samples)])
            break

        x0 = next_x0
    
    sample_dts = jnp.concatenate(sample_dts, 0) # TODO
    out_sample_xs = list()
    for i in range(len(x0)):
        x_cat = jnp.concatenate([x[i] for x in sample_xs], 0)
        out_sample_xs.append(x_cat)
    sample_xs = out_sample_xs
    return sample_dts, sample_xs
