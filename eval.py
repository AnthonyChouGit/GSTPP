import jax
import torch.utils
import torch.utils.data
from gstpp import GSTPP
import equinox as eqx
import torch
from data import get_array
from jaxtyping import Array, Float
import jax.numpy as jnp

def train_loss(model: GSTPP, ts: Float[Array, "N T"], ss: Float[Array, "N T loc_dim"], mask: Float[Array, "N T"], t0: float, t1: float):
    out = jax.vmap(model.loss, (0, 0, 0, None, None), 0)(ts, ss, mask, t0, t1)[1].mean()
    # jax.debug.breakpoint()
    return out

def lax_train_loss(model: GSTPP, ts: Float[Array, "N T"], ss: Float[Array, "N T loc_dim"], mask: Float[Array, "N T"], t0: float, t1: float):
    out = jax.vmap(model.loss_lax, (0, 0, 0, None, None), 0)(ts, ss, mask, t0, t1)[1].mean()
    # jax.debug.breakpoint()
    return out

def lax_validate_ll(model: GSTPP, test_loader: torch.utils.data.DataLoader, t0: float, t1: float):
    model = jax.tree.map(disable_gradient, model)
    model = eqx.nn.inference_mode(model)
    total_events = 0
    total_ll = 0
    total_time_ll = 0
    total_space_ll = 0
    func = jax.vmap(model.get_ll_lax, (0, 0, 0, None, None), 0)
    for batch in test_loader:
        ts, ss, mask = get_array(batch)
        # cur_key, key = jnp.split(jax.vmap(jax.random.split)(key), 2, -2)
        # cur_key = cur_key.squeeze(-2)
        # key = key.squeeze(-2)
        time_ll, space_ll = func(ts, ss, mask, t0, t1)
        time_ll = time_ll.sum()
        space_ll = space_ll.sum()
        space_time_ll = time_ll + space_ll
        num_events = mask.sum()
        total_events = num_events + total_events
        total_ll = total_ll + space_time_ll
        total_time_ll = time_ll + total_time_ll
        total_space_ll = space_ll + total_space_ll
    total_ll /= total_events
    total_time_ll /= total_events
    total_space_ll /= total_events
    return total_ll, total_time_ll, total_space_ll

def validate_ll(model: GSTPP, test_loader: torch.utils.data.DataLoader, t0: float, t1: float):
    model = jax.tree.map(disable_gradient, model)
    model = eqx.nn.inference_mode(model)
    total_events = 0
    total_ll = 0
    total_time_ll = 0
    total_space_ll = 0
    func = jax.vmap(model.get_ll, (0, 0, 0, None, None), 0)
    for batch in test_loader:
        ts, ss, mask = get_array(batch)
        # cur_key, key = jnp.split(jax.vmap(jax.random.split)(key), 2, -2)
        # cur_key = cur_key.squeeze(-2)
        # key = key.squeeze(-2)
        time_ll, space_ll = func(ts, ss, mask, t0, t1)
        time_ll = time_ll.sum()
        space_ll = space_ll.sum()
        space_time_ll = time_ll + space_ll
        num_events = mask.sum()
        total_events = num_events + total_events
        total_ll = total_ll + space_time_ll
        total_time_ll = time_ll + total_time_ll
        total_space_ll = space_ll + total_space_ll
    total_ll /= total_events
    total_time_ll /= total_events
    total_space_ll /= total_events
    return total_ll, total_time_ll, total_space_ll

def disable_gradient(elem):
    if eqx.is_array(elem):
        elem = jax.lax.stop_gradient(elem)
    else:
        elem = elem
    return elem