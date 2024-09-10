import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PyTree
from net import GCN, GRUNet_noinput
from utils import forward_pass

class GRUFunc(eqx.Module):
    gru_net: GRUNet_noinput

    def __init__(self, hdim: int, key: Array):
        self.gru_net = GRUNet_noinput(hdim, key)

    @eqx.filter_jit
    def __call__(self, t: Float[Array, "[]"], h_in: Float[Array, "hdim"]) -> Float[Array, "hdim"]:
        z, g = self.gru_net(t, h_in)
        out = (1-z)*(g-h_in)
        return out

class GRUNodeFunc(eqx.Module):
    gru_func: GRUFunc

    def __init__(self, hdim: int, key: Array):
        self.gru_func = GRUFunc(hdim, key)

    @eqx.filter_jit
    def __call__(self, t: Float[Array, "[]"], H_in: Float[Array, "nb_nodes hdim"], args=None) -> Float[Array, "nb_nodes hdim"]:
        out = jax.vmap(self.gru_func, (None, 0), 0)(t, H_in)
        return out

class GRUGCNNodeFunc(eqx.Module):
    gcn: GCN
    gru_net: GRUNet_noinput

    def __init__(self, hdim: int, gdep: int, support_len: int, beta: float, key: Array):
        key1, key2 = jax.random.split(key)
        self.gru_net = GRUNet_noinput(hdim, key1)
        self.gcn = GCN(gdep, hdim, support_len, beta, key2)

    @eqx.filter_jit
    def __call__(self, t: Float[Array, "[]"], H_in: Float[Array, "nb_nodes hdim"], args) -> Float[Array, "nb_nodes hdim"]:
        A, C = args
        H_G = self.gcn(H_in, A, C) # (nb_nodes, hdim)
        z, g = jax.vmap(self.gru_net, (None, 0), 0)(t, H_G)
        out = (1-z)*(g-H_in)
        return out
    
class IntensityODEFunc(eqx.Module):
    node_func: eqx.Module
    global_func: GRUFunc
    intensity_fn: list

    def __init__(self, hdim: int, node_func: str, key, **kwargs):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        if node_func == 'gru':
            self.node_func = GRUNodeFunc(hdim, key1)
        elif node_func == 'gru-gcn':
            self.node_func = GRUGCNNodeFunc(hdim, kwargs['gdep'], kwargs['support_len'], kwargs['beta'], key1)
        else:
            raise NotImplementedError()
        self.global_func = GRUFunc(hdim, key2)
        self.intensity_fn = [
            eqx.nn.Linear(hdim, 2*hdim, key=key3),
            jax.nn.silu,
            eqx.nn.Linear(2*hdim, 1, key=key4)
        ]

    @eqx.filter_jit
    def get_intensity(self, h: Float[Array, "hdim"]):
        temp = forward_pass(self.intensity_fn, h)
        intensity = jax.nn.sigmoid(temp - 2.) * 50 # 1
        return intensity.squeeze(-1)

    @eqx.filter_jit
    def __call__(self, t: Float[Array, "[]"], state: PyTree, args: PyTree) -> PyTree:
        _, _, H, h = state # H: (nb_nodes, hdim) h: (hdim,)
        dH = self.node_func(t, H, args)
        dh = self.global_func(t, h)
        intensity = self.get_intensity(h)
        d_energy = (intensity**2 + (dH**2).sum() + (dh**2).sum()) / (1+jnp.size(dH)+jnp.size(dh))
        d_state = d_energy, intensity, dH, dh
        return d_state

