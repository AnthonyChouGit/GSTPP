import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--val_steps', type=int, default=10)
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--dataname', type=str, default='covid19')
    parser.add_argument('--gdep', type=int, default=1)
    parser.add_argument('--hdim', type=int, default=32)
    parser.add_argument('--clusters', type=int, default=100)
    parser.add_argument('--title', type=str, default='test')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmups', type=int, default=10)
    parser.add_argument('--activation', type=str, default='sigmoid')
    parser.add_argument('--beta', type=float, default=.2)
    parser.add_argument('--node_func', type=str, default='gru-gcn')
    parser.add_argument('--node_jump', type=str, default='gru')
    parser.add_argument('--energy_reg', type=float, default=1e-4)
    parser.add_argument('--node_freeze', type=int, default=0)
    parser.add_argument('--no_latent_graph', action='store_true')
    parser.add_argument('--no_dist_graph', action='store_true')
    args = parser.parse_args()
    assert args.warmups < args.max_epoch
    
    return args
