import equinox as eqx
from jaxtyping import Array, Float, Bool, PyTree
import jax
from jump import StateJump
from func import IntensityODEFunc
from graph import get_dist_graph, get_latent_graph
from net import TimeDependentMLP
import ode
import jax.numpy as jnp
from utils import batched_forward, normal_ll, normalize, sample_mix
from thinning import thinning

class GSTPP(eqx.Module):
    node_loc: jax.Array
    jump: StateJump
    func: IntensityODEFunc
    init_node: jax.Array
    init_global: jax.Array
    E1: jax.Array
    E2: jax.Array
    node_prob_net: TimeDependentMLP
    loc_pred_net: TimeDependentMLP
    p_net: list
    energy_reg: float
    max_len: int
    dist_graph: bool
    latent_graph: bool
    node_func: str
    node_jump: str

    def __init__(self, hdim: int, node_loc: Float[Array, "nb_nodes loc_dim"], node_func: str, node_jump: str, energy_reg: float, key: Array, max_len: int, dist_graph=True, latent_graph=True, **kwargs):
        nb_nodes, loc_dim = node_loc.shape
        self.node_loc = node_loc
        self.max_len = max_len
        key1, key2, key3, key4, key5, key6, key7, key8, key9, key10 = jax.random.split(key, 10)
        gdep = None if 'gdep' not in kwargs.keys() else kwargs['gdep']
        beta = None if 'beta' not in kwargs.keys() else kwargs['beta']
        if node_func == 'gru-gcn' or node_jump == 'gru-gcn':
            assert gdep is not None
            assert beta is not None
        self.dist_graph = dist_graph
        self.latent_graph = latent_graph
        self.jump = StateJump(hdim, loc_dim, node_jump, key1, gdep=gdep, beta=beta, support_len=dist_graph+latent_graph)
        self.func = IntensityODEFunc(hdim, node_func, key2, gdep=gdep, beta=beta, support_len=dist_graph+latent_graph)
        self.init_node = jax.nn.tanh(jax.random.normal(key3, (nb_nodes, hdim)))
        self.init_global = jax.nn.tanh(jax.random.normal(key4, (hdim,)))

        self.E1 = None
        self.E2 = None
        self.p_net = None

        if node_func == 'gru-gcn' or node_jump == 'gru-gcn':
            self.p_net = [
                eqx.nn.Linear(loc_dim, hdim, key=key9),
                jax.nn.silu,
                eqx.nn.Linear(hdim, hdim, key=key10),
                jax.nn.tanh
            ]
            if latent_graph:
                self.E1 = jax.random.normal(key5, (nb_nodes, hdim))
                self.E2 = jax.random.normal(key6, (nb_nodes, hdim))

        self.node_prob_net = TimeDependentMLP(hdim, 1, 2*hdim, key7)
        self.loc_pred_net = TimeDependentMLP(hdim, 2*loc_dim+1, 2*hdim, key8)
        self.energy_reg = energy_reg
        self.node_func = node_func
        self.node_jump = node_jump

    # Internal method used to get the graph (A, C) based on current configuration
    def __get_graph_param(self):
        if self.node_func == 'gru-gcn' or self.node_jump == 'gru-gcn':
            A1 = get_latent_graph(self.E1, self.E2) if self.latent_graph else None
            A2 = get_dist_graph(self.node_loc) if self.dist_graph else None
            if A1==None and A2!=None:
                A = A2[None, :, :]
            elif A1!=None and A2==None:
                A = A1[None, :, :]
            elif A1!=None and A2!=None:
                A = jnp.stack((A1, A2)) # (2, nb_nodes, nb_nodes)
            P = self.node_loc[:, None, :] - self.node_loc[None, :, :] # (nb_nodes, nb_nodes, loc_dim)
            C = jax.vmap(batched_forward, (None, 0), 0)(self.p_net, P) # (nb_nodes, nb_nodes, hdim)
        else:
            A = None
            C = None
        return A, C

    # Generate the Gaussian mixture parameters from the local states H
    @eqx.filter_jit
    def _get_spatial_dist(self, t: Float[Array, "[]"], H: Float[Array, "nb_nodes hdim"]):
        temp = jax.vmap(self.node_prob_net, (None, 0), 0)(t, H).squeeze(-1) # (nb_nodes, )
        log_node_prob = jax.nn.log_softmax(temp)
        temp = jax.vmap(self.loc_pred_net, (None, 0), 0)(t, H) # (nb_nodes, 2*loc_dim+1)
        loc_mean_param, loc_std_param = jnp.split(temp[:, :-1], 2, -1) # (nb_nodes, loc_dim)
        loc_std = jax.nn.softplus(loc_std_param)
        dist = jax.nn.softplus(temp[:, -1]) # (nb_nodes,)
        loc_direction, _ = normalize(loc_mean_param)
        loc_mean = self.node_loc + loc_direction * dist[:, None]
        return log_node_prob, loc_mean, loc_std

    # Compute T-LL from the sequence of global prejump states hs
    @eqx.filter_jit
    def __timell_get(self, hs: Float[Array, "max_len hdim"], mask: Bool[Array, "max_len"], Lambda: Float[Array, "[]"]):
        intensity = jax.vmap(self.func.get_intensity)(hs)
        intensity = jnp.where(mask, intensity, 1.)
        intensity = jnp.where(intensity==0, intensity+1e-6, intensity)
        time_ll = jnp.log(intensity).sum() - Lambda
        return time_ll
    
    # Compute S-LL from the sequence of local prejump states Hs, given time ts and location locs
    @eqx.filter_jit
    def __spacell_get(self, H: Float[Array, "max_len nb_nodes hdim"], ts: Float[Array, "max_len"], locs: Float[Array, "max_len loc_dim"], mask: Bool[Array, "max_len"]):
        log_node_prob, loc_mean, loc_std = jax.vmap(self._get_spatial_dist, (0, 0), 0)(ts, H)
        ll = jax.vmap(jax.vmap(normal_ll))(loc_mean, loc_std, jnp.broadcast_to(locs[:, None, :], loc_mean.shape))
        temp = log_node_prob + ll
        space_ll = jax.nn.logsumexp(temp, -1) # (T,)
        space_ll = jnp.where(mask, space_ll, 0.)
        space_ll = space_ll.sum()
        return space_ll

    def _get__loglikelihood(self, prejump_H: Float[Array, "T nb_nodes hdim"], prejump_h: Float[Array, "T hdim"], 
                    t: Float[Array, "T"], locs: Float[Array, "T loc_dim"], Lambda: Float[Array, "[]"], mask: Bool[Array, "T"]):
        assert jnp.ndim(prejump_H) == 3
        assert jnp.ndim(prejump_h) == 2
        assert jnp.ndim(t) == 1
        assert jnp.ndim(locs) == 2
        assert jnp.ndim(Lambda) == 0
        assert jnp.ndim(mask) == 1

        # Pad hs and mask to the maximum length to avoid repeated compilation of __timell_get
        assert self.max_len>=prejump_h.shape[0]
        padded_h = jnp.concatenate([prejump_h, jnp.zeros((self.max_len-prejump_h.shape[0], prejump_h.shape[1]))], 0)
        padded_mask = jnp.concatenate([mask, jnp.zeros((self.max_len-mask.shape[0],)).astype(mask.dtype)], 0)

        time_ll = self.__timell_get(padded_h, padded_mask, Lambda)


        # Pad Hs, ts, locs to the maximum length to avoid repeated compilation of __spacell_get
        padded_H = jnp.concatenate([prejump_H, jnp.zeros((self.max_len-prejump_H.shape[0], prejump_H.shape[1], prejump_H.shape[2]))], 0)
        padded_t = jnp.concatenate([t, jnp.zeros((self.max_len-t.shape[0],))], 0)
        padded_locs = jnp.concatenate([locs, jnp.zeros((self.max_len-locs.shape[0], locs.shape[1]))], 0)

        space_ll = self.__spacell_get(padded_H, padded_t, padded_locs, padded_mask)

        return time_ll, space_ll

    def run(self, ts: Float[Array, "seq_len"], locs: Float[Array, "seq_len loc_dim"], mask: Bool[Array, "seq_len"], t0: float, t1: float): 
        mask = mask.astype(bool)
        assert jnp.ndim(ts) == 1
        assert jnp.ndim(locs) == 2
        assert jnp.ndim(mask) == 1
        assert jnp.ndim(t0) == 0
        assert jnp.ndim(t1) == 0
        T = ts.shape[0]
        H = self.init_node
        h = self.init_global
        prejump_node_list = list()
        prejump_global_list = list()
        afterjump_node_list = list()
        afterjump_global_list = list()
        A, C = self.__get_graph_param()
        t0_cur = jnp.asarray(t0)
        energy = jnp.asarray(0.)
        Lambda = jnp.asarray(0.)
        for i in range(T):
            t1_cur = ts[i]
            loc_cur = locs[i, :]
            mask_cur = mask[i]

            # State extrapolation
            energy, Lambda, H, h = extrapolate(self.func, t0_cur, t1_cur, (energy, Lambda, H, h), (A, C), mask_cur)

            # Store the current prejump states
            prejump_node_list.append(H)
            prejump_global_list.append(h)
            if i<T-1:
                
                # State jump
                H, h = update(self.jump, t1_cur, (H, h), loc_cur, self.node_loc, (A, C), mask_cur)

                # Store the current afterjump state
                afterjump_node_list.append(H)
                afterjump_global_list.append(h)
            t0_cur = jnp.where(mask_cur, t1_cur, t0_cur)
            
        # Attend to the surval probability
        energy, Lambda, _, _ = extrapolate(self.func, t0_cur, jnp.asarray(t1), (energy, Lambda, H, h), (A, C), jnp.ones_like(t0_cur, dtype=bool))

        # Stack up all the states in to matrices
        all_prejump_H = jnp.stack(prejump_node_list)
        all_prejump_h = jnp.stack(prejump_global_list)
        all_afterjump_H = jnp.stack(afterjump_node_list)
        all_afterjump_h = jnp.stack(afterjump_global_list)

        return (all_prejump_H, all_prejump_h), (all_afterjump_H, all_afterjump_h), Lambda, energy
    
    def get_ll(self, ts: Float[Array, "seq_len"], locs: Float[Array, "seq_len loc_dim"], mask: Bool[Array, "seq_len"], t0: float, t1: float):
        prejump, _, Lambda, _ = self.run(ts, locs, mask, t0, t1)
        H, h = prejump
        time_ll, space_ll = self._get__loglikelihood(H, h, ts, locs, Lambda, mask)
        return time_ll, space_ll
    
    def loss(self, ts: Float[Array, "seq_len"], locs: Float[Array, "seq_len loc_dim"], mask: Bool[Array, "seq_len"], t0: float, t1: float):
        prejump, _, Lambda, energy = self.run(ts, locs, mask, t0, t1)
        H, h = prejump
        time_ll, space_ll = self._get__loglikelihood(H, h, ts, locs, Lambda, mask)
        loss = - (time_ll + space_ll)
        loss_with_reg = loss + self.energy_reg * energy
        return loss, loss_with_reg

    # Internal method used from intensity and state extrapolation (only used in temporal sampling)
    def _get_intensity(self, dt: Float[Array, '()'], state: PyTree, t0: Float[Array, '()']):
        A, C = self.__get_graph_param()
        energy, Lambda, H, h = extrapolate(self.func, t0, t0+dt, state, (A, C), jnp.ones_like(t0))
        intensity = self.func.get_intensity(h)
        return intensity, (energy, Lambda, H, h)
    
    # Generate multiple samples given afterjump states using the thinning algorithm
    def _sample_dt(self, t: Float[Array, "N"], afterjump: PyTree, num_samples: int, boundary: float, key: Array, oversample_rate: float=5.):
        dts = list()
        Hs = list()
        for i in range(t.shape[0]):
            t0 = t[i]
            x0 = tuple(x[i] for x in afterjump)
            cur_key, key = jax.random.split(key)
            sample_dts, sample_xs = thinning(self._get_intensity, x0, t0, cur_key, boundary, num_samples, 2000, 2000, oversample_rate) # (num_samples, )
            _, _, H, _ = sample_xs
            dts.append(sample_dts)
            Hs.append(H)
        dts = jnp.stack(dts, 0) # (N, num_samples, )
        Hs = jnp.stack(Hs, 0) # (N, num_samples, nb_nodes, hdim)
        return dts, Hs
    
    @eqx.filter_jit
    def _sample_loc(self, t: Float[Array, "N"], prejump_H: Float[Array, "N nb_nodes hdim"], num_samples: int, key:Array):
        log_node_prob, loc_mean, loc_std = jax.vmap(self._get_spatial_dist, (0, 0), 0)(t, prejump_H) # TODO
        key = jax.random.split(key, loc_mean.shape[0])
        loc_samples = jax.vmap(sample_mix, (0, 0, 0, None, 0), 0)(loc_mean, loc_std, log_node_prob, num_samples, key) # (N, num_samples, loc_dim) TODO
        return loc_samples
    
    def sample(self, ts: Float[Array, "seq_len"], locs: Float[Array, "seq_len loc_dim"], mask: Bool[Array, "seq_len"], t0: float, key: Array, num_samples: int, boundary: float, oversample_rate: float=5.):
        t1 = ts.max() + 1e-5
        key1, key2 = jax.random.split(key)
        ts = ts[mask][:-1]
        locs = locs[mask][:-1]
        mask = mask[mask][:-1]
        _, afterjumps, _, _ = self.run(ts, locs, mask, t0, t1)
        H, h = afterjumps
        states = (jnp.zeros(H.shape[0]), jnp.zeros(H.shape[0]), H, h)
        dts, Hs = self._sample_dt(ts, states, num_samples, boundary, key1, oversample_rate) # Hs: (seq_len-1, num_samples, nb_nodes, hdim) dts: (N, num_smaples)
        
        with jax.disable_jit():
            loc_samples = self._sample_loc((ts[:, None]+dts).flatten(), Hs.reshape((-1, Hs.shape[-2], Hs.shape[-1])), 1, key2).squeeze(1) # (N*num_samples, loc_dim)
        loc_samples = loc_samples.reshape(ts.shape[0], num_samples, -1)
        return dts, loc_samples

# Extrapolate energy, Lambda, H and h simultaneously
@eqx.filter_jit
def extrapolate(func: IntensityODEFunc, t0: Float[Array, "[]"], t1: Float[Array, "[]"], state: PyTree, args: PyTree, mask: Float[Array, "[]"]):
    t1 = jnp.where(mask, t1, t0+1e-6)
    t1 = jnp.where(t1>t0, t1, t0+1e-6)
    energy, Lambda, H, h = state
    prejump_energy, prejump_Lambda, prejump_H, prejump_h = ode.integrate(func, t0, t1, state, args)
    energy = jnp.where(mask, prejump_energy, energy)
    Lambda = jnp.where(mask, prejump_Lambda, Lambda)
    H = jnp.where(mask, prejump_H, H)
    h = jnp.where(mask, prejump_h, h)
    return energy, Lambda, H, h

# Update H and h simultaneously using the new event location loc
@eqx.filter_jit
def update(jump: StateJump, t: Float[Array, "[]"], state: PyTree, loc: Float[Array, "loc_dim"], node_loc: Float[Array, "nb_nodes loc_dim"], args: PyTree, mask: Float[Array, "[]"]):
    H, h = state
    after_H, after_h = jump(t, state, loc, node_loc, args)
    H = jnp.where(mask, after_H, H)
    h = jnp.where(mask, after_h, h)
    return H, h

