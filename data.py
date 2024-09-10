import torch
import pickle
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp

class STDataset(torch.utils.data.Dataset):
    def __init__(self, data): 
        time_seqs = list()
        space_seqs = list()
        for seq in data:
            ts = [e[0] for e in seq]
            ss = [e[1:] for e in seq]
            time_seqs.append(ts)
            space_seqs.append(ss)
        self.ts = time_seqs
        self.ss = space_seqs

    def get_max_len(self):
        max_len = max([len(seq) for seq in self.ts])
        return max_len

    def __len__(self):
        return len(self.ts)
    
    def get_spatial_dim(self):
        return len(self.ss[0][0])
    
    def get_all_locs(self):
        all_ss = torch.tensor([e for seq in self.ss for e in seq])
        return all_ss
    
    def get_tmax(self):
        all_ts = [torch.tensor(t) for t in self.ts]
        all_ts = torch.cat(all_ts)
        t_max = all_ts.max()
        return t_max
    
    def get_dtmax(self):
        all_dts = list()
        for i in range(len(self.ts)):
            t = torch.tensor([0.]+self.ts[i])
            dt = t[1:] - t[:-1]
            all_dts.extend(dt.tolist())
        dt_max = max(all_dts)
        return dt_max
    
    def get_spatial_stats(self):
        all_ss = torch.cat([torch.tensor(s) for s in self.ss], dim=0) # (XXX, D)
        s_mean = all_ss.mean(0)
        s_std = all_ss.std(0) # (D)
        return s_mean, s_std

    def __getitem__(self, ind):
        return self.ts[ind], self.ss[ind]
    
    def normalize_(self, s_mean, s_scale, dt_scale):
        num_seq = len(self.ts)
        for i in range(num_seq):
            ss_i = torch.tensor(self.ss[i])
            ts_i = torch.tensor(self.ts[i])
            ss_i = (ss_i - s_mean) / s_scale
            temp = torch.cat([torch.zeros(1), ts_i])
            dts_i = temp[1:] - temp[:-1]
            dts_i /= dt_scale
            ts_i = torch.cumsum(dts_i, dim=0)
            self.ss[i] = ss_i.tolist()
            self.ts[i] = ts_i.tolist()

    def get_dataloader(self, batch_size, shuffle=False):
        dataloader = DataLoader(
            dataset=self,
            collate_fn=collate_fn,
            shuffle=shuffle,
            batch_size=batch_size
        )
        return dataloader


def collate_fn(data):
    '''
        input is a list of tuples:
        [(ts, ss), (ts, ss), ...]
    '''
    all_ts = [torch.tensor(seq_tuple[0]) for seq_tuple in data]
    all_ss = [torch.tensor(seq_tuple[1]) for seq_tuple in data]
    all_mask = [torch.ones(len(seq_tuple[0])) for seq_tuple in data]
    ts = torch.nn.utils.rnn.pad_sequence(all_ts, True) # (N, T)
    ss = torch.nn.utils.rnn.pad_sequence(all_ss, True) # (N, T, D)
    mask = torch.nn.utils.rnn.pad_sequence(all_mask, True) # (N, T)
    return ts, ss, mask

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    dataset = STDataset(data)
    return dataset

def get_array(batch):
    out = jax.tree.map(lambda tensor: jnp.array(tensor.numpy()), batch)
    return out