import arg_parse
import os
args = arg_parse.parse()
args.cuda = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
from gstpp import GSTPP
import setproctitle
from data import load_data
# from kmeans import KMeansJax
import jax
import equinox as eqx
from data import get_array
import jax.numpy as jnp
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from eval import disable_gradient

# jax.config.update("jax_debug_nans", True)
args.load_path = 'save/covid19_100_save/model-100.eqx'
args.eval_batch_size = 1
args.dataname = 'covid19'
args.clusters = 100

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
dt_max = max([train_data.get_dtmax(), val_data.get_dtmax(), test_data.get_dtmax()])
hdim = args.hdim
num_clusters = args.clusters
scale = jnp.log(jnp.prod(get_array(s_std)))
max_len = max([train_data.get_max_len(), val_data.get_max_len(), test_data.get_max_len()])

node_loc = jnp.zeros((num_clusters, test_data.get_spatial_dim()))
init_key = jax.random.PRNGKey(123321)
model = eqx.filter_eval_shape(GSTPP, hdim, node_loc, args.node_func, args.node_jump, args.energy_reg, init_key, max_len, not args.no_dist_graph, not args.no_latent_graph, gdep=args.gdep, beta=args.beta)
model = eqx.tree_deserialise_leaves(args.load_path, model)
model = jax.tree.map(disable_gradient, model)
model = eqx.nn.inference_mode(model)

t0 = 0.
t1 = (max([train_data.get_tmax(), val_data.get_tmax(), test_data.get_tmax()]) + 1e-5).item()
ts, ss = test_data[77]
ts = jnp.array(ts)
ss = jnp.array(ss)
mask = jnp.ones_like(ts)
prejump, afterjump, _, _ = model.run(ts, ss, mask, t0, t1)
H, h = prejump
H_last = H[-1]
t_last = ts[-1]
sample_key = jax.random.PRNGKey(69)
loc_samples = model._sample_loc(t_last[None, ...], H_last[None, ...], 1000, sample_key)[0]
loc_samples = loc_samples* s_std.numpy() + s_mean.numpy()

sns.set(font_scale=2)
sns.set_style("white")
df = pd.DataFrame.from_records(loc_samples.tolist(), columns=['x', 'y'])
# temp = df['y']
sns.kdeplot(data=df, x='x', y='y', fill=True)
df = pd.DataFrame.from_records([(ss[-1]* s_std.numpy() + s_mean.numpy()).tolist()], columns=['x', 'y'])
sns.scatterplot(data=df, x='x', y='y')
plt.savefig('temp.jpg', bbox_inches='tight')
# print()