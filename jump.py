from typing import Any
import equinox as eqx
from jaxtyping import Array, Float, PyTree
from net import GRUNet, GCN
import jax
import jax.numpy as jnp
from utils import normalize

class GRUJump(eqx.Module):
    gru_net: GRUNet

    def __init__(self, hdim: int, input_dim: int, key: Array):
        self.gru_net = GRUNet(hdim, input_dim, key)

    def __call__(self, t: Float[Array, "[]"], h_in: Float[Array, "hdim"], x_in: Float[Array, "in_dim"]) -> Float[Array, "hdim"]:
        z, g = self.gru_net(t, h_in, x_in)
        out = z*h_in + (1-z)*g
        return out

class GRUNodeJump(eqx.Module):
    gru_jump: GRUJump

    def __init__(self, hdim: int, input_dim: int, key: Array):
        self.gru_jump = GRUJump(hdim, input_dim, key)

    def __call__(self, t: Float[Array, "[]"], H_in: Float[Array, "nb_nodes hdim"], X_in: Float[Array, "nb_nodes in_dim"], args=None) -> Float[Array, "nb_nodes hdim"]:
        out = jax.vmap(self.gru_jump, (None, 0, 0), 0)(t, H_in, X_in)
        return out

class GRUGCNNodeJump(eqx.Module):
    gcn: GCN
    gru_net: GRUNet

    def __init__(self, hdim: int, input_dim: int, gdep: int, support_len: int, beta: float, key: Array):
        key1, key2 = jax.random.split(key)
        self.gcn = GCN(gdep, hdim, support_len, beta, key1)
        self.gru_net = GRUNet(hdim, input_dim, key2)

    def __call__(self, t: Float[Array, "[]"], H_in: Float[Array, "nb_nodes hdim"], X_in: Float[Array, "nb_nodes in_dim"], args: PyTree) -> Float[Array, "nb_nodes hdim"]:
        A, C = args
        H_G = self.gcn(H_in, A, C) # (nb_nodes, hdim)
        z, g = jax.vmap(self.gru_net, (None, 0, 0), 0)(t, H_G, X_in)
        out = z*H_in + (1-z)*g
        return out

class StateJump(eqx.Module):
    node_jump: eqx.Module
    global_jump: GRUJump
    loc_proj: eqx.nn.Linear
    log_decay_rate: jax.Array

    def __init__(self, hdim: int, loc_dim: int, node_jump: str, key: Array, **kwargs):
        key1, key2, key3 = jax.random.split(key, 3)
        if node_jump == 'gru':
            self.node_jump = GRUNodeJump(hdim, hdim, key1)
        elif node_jump == 'gru-gcn':
            self.node_jump = GRUGCNNodeJump(hdim, hdim, kwargs['gdep'], kwargs['support_len'], kwargs['beta'], key1)
        self.global_jump = GRUJump(hdim, loc_dim, key2)
        self.loc_proj = eqx.nn.Linear(loc_dim, hdim, False, key=key3)
        self.log_decay_rate = jnp.zeros(hdim)

    eqx.filter_jit
    def __call__(self, t: Float[Array, "[]"], state: PyTree, loc: Float[Array, "loc_dim"], node_loc: Float[Array, "nb_nodes loc_dim"], args: PyTree) -> Any:
        H_in, h_in = state
        rel_loc = loc[None, :] - node_loc # (nb_nodes, loc_dim)
        direction, dists = normalize(rel_loc)
        loc_feature = jax.vmap(self.loc_proj)(direction) # (nb_nodes, hdim)
        decay_rate = jnp.exp(self.log_decay_rate) # hdim
        loc_feature = loc_feature * jnp.exp(-decay_rate*dists[:, None])
        H = self.node_jump(t, H_in, loc_feature, args)
        h = self.global_jump(t, h_in, loc)
        return H, h
