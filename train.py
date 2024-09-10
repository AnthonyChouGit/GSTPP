import arg_parse
import os
args = arg_parse.parse()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
from gstpp import GSTPP
import setproctitle
from data import load_data
import jax
import equinox as eqx
from data import get_array
import jax.numpy as jnp
import optax
from torch.utils.tensorboard import SummaryWriter
from csv_utils import read_keyval, write_keyval
import time
from eval import validate_ll, train_loss
from jaxtyping import PyTree
from sklearn.cluster import KMeans
import numpy as np

# jax.config.update("jax_debug_nans", True)

setproctitle.setproctitle(args.title)
batch_size = args.batch_size
eval_batch_size = args.eval_batch_size
val_steps = args.val_steps
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
train_loader = train_data.get_dataloader(batch_size, True)
val_loader = val_data.get_dataloader(eval_batch_size)
test_loader = test_data.get_dataloader(eval_batch_size)
dt_max = max([train_data.get_dtmax(), val_data.get_dtmax(), test_data.get_dtmax()])
hdim = args.hdim
num_clusters = args.clusters
scale = jnp.log(jnp.prod(get_array(s_std)))
max_len = max([train_data.get_max_len(), val_data.get_max_len(), test_data.get_max_len()])

epoch_num = args.max_epoch
kmeans = KMeans(num_clusters, random_state=999, max_iter=2000).fit(get_array(train_data.get_all_locs()))
node_loc = jnp.array(kmeans.cluster_centers_)
init_key = jax.random.PRNGKey(123321)
if args.load_path is not None:
    model = eqx.filter_eval_shape(GSTPP, hdim, node_loc, args.node_func, args.node_jump, args.energy_reg, init_key, max_len, not args.no_dist_graph, not args.no_latent_graph, gdep=args.gdep, beta=args.beta)
    model = eqx.tree_deserialise_leaves(args.load_path, model)
else:
    model = GSTPP(hdim, node_loc, args.node_func, args.node_jump, args.energy_reg, init_key, max_len, not args.no_dist_graph, not args.no_latent_graph, gdep=args.gdep, beta=args.beta)
base_lr = args.lr
batch_num = len(train_data) // batch_size + 1
scheduler = optax.warmup_cosine_decay_schedule(base_lr/args.warmups, base_lr, args.warmups, epoch_num*batch_num)
optim = optax.adamw(scheduler)
opt_state = optim.init(eqx.filter(model, eqx.is_array))
t0 = 0.
t1 = (max([train_data.get_tmax(), val_data.get_tmax(), test_data.get_tmax()]) + 1e-5).item()
tb_writer = SummaryWriter(f'results/{args.title}')
save_path = f'save/{args.title}_save'
if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(f'{save_path}/eval.csv'):
    loss_dict = dict()
else:
    loss_dict = read_keyval(f'{save_path}/eval.csv', int, float)

@eqx.filter_jit
def update_model(model: GSTPP, opt_state: PyTree, grads: PyTree):
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state

for epoch in range(args.start_epoch, epoch_num+1):
    start_time = time.time()
    loss_total = 0.
    num_events = 0
    print(f'epoch {epoch}')
    for batch in train_loader:
        # batch_start = time.time()
        ts, ss, mask = get_array(batch)
        loss, grads = eqx.filter_value_and_grad(train_loss)(model, ts, ss, mask, t0, t1)
        if epoch < args.node_freeze:
            grads = eqx.tree_at(lambda tree: tree.node_loc, grads, jnp.zeros_like(grads.node_loc))
        
        model, opt_state = update_model(model, opt_state, grads)

        num = mask.sum()
        loss_total += loss*ts.shape[0]
        num_events += num
        # print(time.time() - batch_start)
    loss_avg = loss_total / num_events
    tb_writer.add_scalar("train/LL", np.asarray(loss_avg+scale), epoch)
    train_time = time.time() - start_time
    print(f'Epoch {epoch} finished in {train_time} seconds.')
    if epoch % val_steps == 0:
        print('evaluating...')
        start_time = time.time()
        total_ll, total_time_ll, total_space_ll = validate_ll(model, test_loader, t0, t1)
        loss_dict[epoch] = -total_ll.item()
        write_keyval(f'{save_path}/eval.csv', loss_dict)
        tb_writer.add_scalar('val/LL', np.asarray(-total_ll+scale), epoch)
        tb_writer.add_scalar('val/time LL', np.asarray(-total_time_ll), epoch)
        tb_writer.add_scalar('val/space LL', np.asarray(-total_space_ll+scale), epoch)
        eqx.tree_serialise_leaves(f'{save_path}/model-{epoch}', model)
        end_time = time.time()
        runtime = end_time - start_time
        print(f'Finished in {runtime} seconds.')
tb_writer.close()
