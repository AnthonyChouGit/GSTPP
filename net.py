import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float
from utils import forward_pass

class TimeDependentLinear(eqx.Module):
    layer: Array
    hyper_bias: eqx.nn.Linear

    def __init__(self, d_in: int, d_out: int, key: Array):
        key1, key2 = jax.random.split(key)
        self.layer = eqx.nn.Linear(d_in, d_out, key=key1)
        self.hyper_bias = eqx.nn.Linear(1, d_out, False, key=key2)
        self.hyper_bias = eqx.tree_at(lambda linear: linear.weight, self.hyper_bias, jnp.zeros_like(self.hyper_bias.weight))

    def __call__(self, t: Float[Array, "[]"], x: Float[Array, "d_in"]) -> Float[Array, "d_out"]:
        lhs = self.layer(x)
        rhs = self.hyper_bias(jnp.broadcast_to(t, (1,)))
        rst = lhs + rhs
        return rst

class TimeDependentSwish(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, dim: int, key: Array):
        key1, key2 = jax.random.split(key)
        self.linear1 = eqx.nn.Linear(1, 2*dim, key=key1)
        self.linear2 = eqx.nn.Linear(2*dim, dim, key=key2)

    def __call__(self, t: Float[Array, "[]"], x: Float[Array, "dim"]) -> Float[Array, "dim"]:

        temp = self.linear1(jnp.broadcast_to(t, (1,)))
        temp = jax.nn.silu(temp)

        temp = self.linear2(temp)
        temp = jax.nn.silu(temp)

        rst = x * jax.nn.sigmoid(x * temp)
        return rst

class TimeDependentMLP(eqx.Module):
    in_proj: TimeDependentLinear
    out_proj: TimeDependentLinear
    activation: TimeDependentSwish

    def __init__(self, d_in: int, d_out: int, hdim: int, key: Array):
        key1, key2, key3 = jax.random.split(key, 3)
        self.in_proj = TimeDependentLinear(d_in, hdim, key1)
        self.out_proj = TimeDependentLinear(hdim, d_out, key2)
        self.activation = TimeDependentSwish(hdim, key3)

    @eqx.filter_jit
    def __call__(self, t: Float[Array, "[]"], x: Float[Array, "d_in"]) -> Float[Array, "dim"]:
        temp = self.in_proj(t, x)
        temp = self.activation(t, temp)
        rst = self.out_proj(t, temp)
        return rst

class GCN(eqx.Module):
    gdep: int
    beta: int
    out_proj: eqx.nn.Linear
    support_len: int

    def __init__(self, gdep: int, hdim: int, support_len:int, beta: float, key: Array):
        self.out_proj = eqx.nn.Linear((gdep*support_len+1)*hdim, hdim, key=key)
        self.gdep = gdep
        self.beta = beta
        self.support_len = support_len

    @eqx.filter_jit
    def __call__(self, H_in: Float[Array, "nb_nodes hdim"], A: Float[Array, "support_len nb_nodes nb_nodes"], C: Float[Array, "nb_nodes nb_nodes hdim"]) -> Float[Array, "nb_nodes hdim"]:
        assert A.shape[0] == self.support_len
        nb_nodes, hdim = H_in.shape
        out = [H_in, ]
        A = A + jnp.eye(A.shape[-1])
        d = A.sum(-1) # (support_len, nb_nodes)
        A = A / d[:, :, None] # (support_len, nb_nodes, nb_nodes)
        H = H_in[None, :, :] # (1, nb_nodes, hdim)
        for i in range(self.gdep):
            H_C = H[:, None, :, :] * C # (support_len, nb_nodes, nb_nodes, hdim)
            H = self.beta * H + (1-self.beta) * (A[:, :, :, None] * H_C).sum(-2) # (support_len, nb_nodes, hdim)
            temp = jnp.transpose(H, (1, 0, 2)).reshape((nb_nodes, self.support_len*hdim)) # (nb_nodes, support_len*hdim)
            out.append(temp)
        out = jnp.concatenate(out, -1) # (nb_nodes, (gdep*support_len+1)*hdim)
        out = jax.vmap(self.out_proj)(out)
        return out

class GRUNet_noinput(eqx.Module):
    rz_net: list
    g_net: list

    def __init__(self, hdim: int, key: Array):
        key1, key2 = jax.random.split(key)
        self.rz_net = [
            eqx.nn.Linear(hdim, 2*hdim, key=key1),
            jax.nn.sigmoid
        ]
        self.g_net = [
            eqx.nn.Linear(hdim, hdim, key=key2),
            jax.nn.tanh
        ]

    @eqx.filter_jit
    def __call__(self, t: Float[Array, "[]"], h_in: Float[Array, "hdim"]) -> tuple:
        rz = forward_pass(self.rz_net, h_in)
        r, z = jnp.split(rz, 2, -1)
        g = forward_pass(self.g_net, r*h_in)
        return z, g

class GRUNet(eqx.Module):
    rz_net: list
    g_net: list

    def __init__(self, hdim: int, in_dim: int, key: Array):
        key1, key2 = jax.random.split(key)
        self.rz_net = [
            eqx.nn.Linear(hdim+in_dim, 2*hdim, key=key1),
            jax.nn.sigmoid
        ]
        self.g_net = [
            eqx.nn.Linear(hdim+in_dim, hdim, key=key2),
            jax.nn.tanh
        ]

    @eqx.filter_jit
    def __call__(self, t: Float[Array, "[]"], h_in: Float[Array, "hdim"], x_in: Float[Array, "in_dim"]) -> tuple:
        hx = jnp.concatenate((h_in, x_in), -1)
        rz = forward_pass(self.rz_net, hx)
        r, z = jnp.split(rz, 2, -1)
        hx = jnp.concatenate((r*h_in, x_in), -1)
        g = forward_pass(self.g_net, hx)
        return z, g