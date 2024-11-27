import arg_parse
import os

args = arg_parse.parse()

# args.cuda = '3'
# args.load_path = 'save/earthquakes_100_save/model-100.eqx'

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
from gstpp import GSTPP
import setproctitle
from data import load_data
import jax
import equinox as eqx
from data import get_array
import jax.numpy as jnp
from sklearn.cluster import KMeans
import numpy as np
from eval import disable_gradient
from jaxtyping import PyTree, Array, Float, Bool
from sklearn.metrics import root_mean_squared_error
import time

# jax.config.update("jax_debug_nans", True)

def sample_get(model: GSTPP, ts: Float[Array, "T"], ss: Float[Array, "T loc_dim"], mask: Bool[Array, "T"], t0: float, key: Array, num_samples: int):

    loc_sample = model.sample_loc_cond(ts, ss, mask, t0, key, num_samples) # 0 - -2
    ss = ss[mask][1:] # 
    return loc_sample, ss


setproctitle.setproctitle(args.title)
eval_batch_size = args.eval_batch_size
dataname = args.dataname
base_path = f'datasets/{dataname}'
train_data = load_data(f'{base_path}/train.pkl')
val_data = load_data(f'{base_path}/validate.pkl')
test_data = load_data(f'{base_path}/test.pkl')
s_mean, s_std = train_data.get_spatial_stats()
train_data.normalize_(s_mean, s_std, 1.)
val_data.normalize_(s_mean, s_std, 1.)
test_data.normalize_(s_mean, s_std, 1.)
t_max = max([train_data.get_tmax(), val_data.get_tmax(), test_data.get_tmax()]) + 1e-5
test_loader = test_data.get_dataloader(eval_batch_size)
dt_max = max([train_data.get_dtmax(), val_data.get_dtmax(), test_data.get_dtmax()])
hdim = args.hdim
num_clusters = args.clusters
scale = jnp.log(jnp.prod(get_array(s_std)))
max_len = max([train_data.get_max_len(), val_data.get_max_len(), test_data.get_max_len()])

node_loc = jnp.empty((num_clusters, test_data.get_spatial_dim()))
init_key = jax.random.PRNGKey(123321)
model = eqx.filter_eval_shape(GSTPP, hdim, node_loc, args.node_func, args.node_jump, args.energy_reg, init_key, max_len, not args.no_dist_graph, not args.no_latent_graph, gdep=args.gdep, beta=args.beta)
model = eqx.tree_deserialise_leaves(args.load_path, model)
sample_key = jax.random.PRNGKey(69)
model = jax.tree.map(disable_gradient, model)
model = eqx.nn.inference_mode(model)
real_locs = list()
pred_locs = list()
t0 = 0.

for batch in test_loader:
    start_time = time.time()
    batch_ts, batch_ss, batch_mask = get_array(batch)
    for i in range(batch_ts.shape[0]):
        ts = batch_ts[i]
        ss = batch_ss[i]
        mask = batch_mask[i]
        mask = mask.astype(bool)
        cur_key, sample_key = jax.random.split(sample_key)
        # dts_samples: (T, num_samples) loc_sample: (T, num_samples, loc_dim)   select_dts: (T, )   select_ss: (T, loc_dim)
        loc_sample, select_ss = sample_get(model, ts, ss, mask, t0, cur_key, 50)
        real_locs.append(select_ss)
        pred_locs.append(loc_sample)
    print(f'Batch finished in {time.time()-start_time} seconds.')

real_locs = jnp.concatenate(real_locs, 0) # (N, loc_dim)
pred_locs = jnp.concatenate(pred_locs, 0) # (N, num_samples, loc_dim)
s_std = get_array(s_std)
s_mean = get_array(s_mean)
real_locs = real_locs * s_std + s_mean
pred_locs = pred_locs * s_std + s_mean
dist = jnp.sqrt(((real_locs[:, None, :]-pred_locs)**2).sum(-1))
mean_dist = dist.mean((0, 1))
print(mean_dist)
