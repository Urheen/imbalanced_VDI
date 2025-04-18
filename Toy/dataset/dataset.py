import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pickle, random
from torch.utils.data import Sampler
import math
from collections import deque
import itertools

def read_pickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data


def write_pickle(data, name):
    with open(name, 'wb') as f:
        pickle.dump(data, f)


class ToyDataset(Dataset):

    def __init__(self, pkl, domain_id, opt=None):
        idx = pkl['domain'] == domain_id
        self.data = pkl['data'][idx].astype(np.float32)
        self.label = pkl['label'][idx].astype(np.int64)
        self.domain = domain_id

        if opt.normalize_domain:
            print('===> Normalize in every domain')
            self.data_m, self.data_s = self.data.mean(
                0, keepdims=True), self.data.std(0, keepdims=True)
            self.data = (self.data - self.data_m) / self.data_s

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.domain

    def __len__(self):
        return len(self.data)


class SeqToyDataset(Dataset):
    # the size may change because of the toy dataset!!
    def __init__(self, datasets, size=3 * 200):
        self.datasets = datasets
        self.size = size
        # print(self.size)
        # exit(0)
        # print('SeqDataset Size {} Sub Size {}'.format(
        #     size, [len(ds) for ds in datasets]))

    def __len__(self):
        return self.size
    
    def __getitem__(self, i):
        return [ds[i] for ds in self.datasets]
    
class ZiyanDataset(Dataset):
    def __init__(self, datasets):
        """
        datasets: List of sub-datasets, each a list of tensors.
        """
        self.datasets = datasets
        self.num_subdatasets = len(datasets)
        self.num_samples = len(datasets[0])  # Assuming all sub-datasets have the same length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Index format: (subdataset_id, sample_id)
        # print(index)
        if isinstance(index, (tuple, list)):
            subdataset_id, sample_id = index
            return self.datasets[subdataset_id][sample_id]

class ZiyanDatasetSampler(Sampler):
    def __init__(self, datasets, batch_size, K, shuffle=False, drop_last=False, allow_padding=False):
        """
        datasets: List of sub-datasets.
        batch_size: Number of samples per batch.
        K: Number of sub-datasets per batch.
        shuffle: Whether to shuffle data.
        drop_last: Whether to drop the last incomplete batch.
        allow_padding: Whether to pad when samples are missing.
        """
        self.datasets = datasets
        self.batch_size = batch_size
        self.K = K
        self.num_domains = len(datasets)  # Number of sub-datasets
        self.num_samples = len(datasets[0])  # Assuming all sub-datasets have the same length
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.allow_padding = allow_padding
        self.batches = []
        self.reset_order()

    def reset_order(self):
        """
        Reset the sampling order for a new epoch.
        """
        self.batches = []
        if self.shuffle:
            index_order = torch.randperm(self.num_samples).tolist()
        else:
            index_order = list(range(self.num_samples))
        index_iter = iter(index_order)
        batch = list(itertools.islice(index_iter, self.batch_size))
        # batch.sort()

        while batch:
            if self.shuffle:
                domain_order = torch.randperm(self.num_domains).tolist()
            else:
                domain_order = list(range(self.num_domains))

            domain_iter = iter(domain_order)
            domain_groups = list(itertools.islice(domain_iter, self.K))
            domain_groups.sort()

            while domain_groups:
                # Construct batch tuples: (subdataset_id, sample_id)
                this_batches = list(itertools.product(domain_groups, batch))

                self.batches.append(this_batches)
                domain_groups = list(itertools.islice(domain_iter, self.K))
                domain_groups.sort()

            batch = list(itertools.islice(index_iter, self.batch_size))

            if len(batch) == 0 or len(batch) == self.batch_size:
                pass
            else:
                if self.drop_last:
                    if len(batch) < self.batch_size:
                        batch = list(itertools.islice(index_iter, self.batch_size))
                else:
                    flag = self.batch_size - len(batch)
                    for elem in index_order:
                        if elem not in batch:
                            batch.append(elem)
                            flag -= 1
                            if flag == 0:
                                break
            # batch.sort()

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
    
def ziyan_collate(batch_list):
    data, label, domain = [], [], []
    # print(batch_list[0])
    for (x,y,d) in batch_list:
        # print(x,y,d)
        data.append(torch.from_numpy(x))
        label.append(torch.LongTensor([y]))
        domain.append(torch.LongTensor([d]))
        # print(data[-1], label[-1], domain[-1])
    data = torch.stack(data, dim=0)  # shape [batch_size*K, D]
    label = torch.stack(label, dim=0) 
    domain = torch.stack(domain, dim=0) 
    num_domain_per_batch = torch.unique(domain).flatten().shape[0]
    data_dim = data.shape[-1]

    data = data.view(num_domain_per_batch, -1, data_dim)  # shape [K, batch_size, D]
    label = label.view(num_domain_per_batch, -1)
    domain = domain.view(num_domain_per_batch, -1)
    # print(data.shape, label.shape, domain.shape)
    # print(domain)
    # batch_tensor = batch_tensor.permute(1, 0, 2).contiguous()  # shape [batch_size, K, D]
    outputs = []
    for k in range(num_domain_per_batch):
        outputs.append((data[k], label[k], domain[k]))
    return outputs
    

# class ziyanSeqToyDataset(Dataset):
#     def __init__(self, datasets, size=3*200, k=None):
#         self.datasets = datasets
#         self.size = size
#         self.k = k if k is not None else len(datasets)
#         self._current_subset = None
        
#         # 验证所有子数据集长度一致
#         assert all(len(ds) == len(datasets[0]) for ds in datasets)
#         print(f'ziyanSeqToyDataset Size {size}, Sub Size {[len(ds) for ds in datasets]}')

#     def set_current_subset(self, subset):
#         """设置当前batch使用的子数据集索引"""
#         self._current_subset = subset

#     def __len__(self):
#         return self.size

#     def __getitem__(self, i):
#         if self._current_subset is None:
#             # 默认模式（k=10时的原始行为）
#             return [ds[i] for ds in self.datasets]
        
#         # 使用当前设置的子数据集
#         return [self.datasets[idx][i] for idx in self._current_subset]

# class KSubsetDataLoader:
#     def __init__(self, dataset, batch_size, k, shuffle=False, drop_last=False):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.k = k
#         self.shuffle = shuffle
#         self.drop_last = drop_last

#         self.num_domains = len(dataset.datasets)
#         assert self.num_domains % k == 0, "Number of subsets must be divisible by k"
#         self.num_groups = self.num_domains // k
#         self.n = len(dataset.datasets[0])
        
#         # Calculate number of batches per group
#         if drop_last:
#             self.batches_per_domain = self.n // batch_size
#         else:
#             self.batches_per_domain = (self.n + batch_size - 1) // batch_size
        
#         self.num_batches = self.num_groups * self.batches_per_domain

#     def __iter__(self):
#         m = len(self.dataset.datasets)
#         g = self.num_groups
#         k = self.k
#         n = self.n
#         batch_size = self.batch_size

#         groups = []
#         for _ in range(self.batches_per_domain):
#             subset_indices = np.arange(m)
#             if self.shuffle:
#                 np.random.shuffle(subset_indices)
#             groups.extend([subset_indices[i*k : (i+1)*k] for i in range(g)])
#         if self.shuffle:
#             np.random.shuffle(groups)
        
#         all_batches = []
#         indices_order = np.arange(n)
#         if self.shuffle:
#             np.random.shuffle(indices_order)
#         indices = [indices_order for _ in range(m)]

#         # if self.shuffle:
#         #     for elem in indices:
#         #         np.random.shuffle(elem)
#         for group in groups:
#             # Split into batches
#             batches = []
#             for j in group:
#                 batch_indices = indices[j][:batch_size]
#                 indices[j] = indices[j][batch_size:]
                
#                 batches.append(batch_indices)
            
#             # Record group and batch indices
#             all_batches.append((group, batches))
#         # print('all batch', all_batches)
#         # for elem in all_batches:
#         #     print(elem)
#         # print('end all batch')
#         # print(f"before {len(all_batches)}, {all_batches}")
#         # print('ziyan', len(all_batches), len(all_batches[0][1]))
#         if self.shuffle:
#             np.random.shuffle(all_batches)

#         # print(f"after {len(all_batches)}, {all_batches}")
#         # exit(0)
#         for group, batch_indices in all_batches:
#             # Collect data for this batch
#             batch_data = []
#             group_argsorted = np.argsort(group)
#             for idx in range(k):
#                 this_i = group[group_argsorted[idx]]
#                 this_batch_indices = batch_indices[group_argsorted[idx]]
#                 sample = [self.dataset.datasets[this_i][i] for i in this_batch_indices]
#                 batch_data.append(sample)
#             yield self.collate_fn(batch_data)

#     @staticmethod
#     def collate_fn(batch):
#         """batch 结构： [ K个[], 每个[] 32个tuple, 每个tuple (x,y,domain)]"""
#         # print(batch[0])
#         # print(len(batch), len(batch[0]), batch[0][0][0].shape)
        
#         # 将原始batch结构转换为三维张量
#         data = torch.stack([
#             torch.stack([torch.from_numpy(s[0]) for s in sample])
#             for sample in batch
#         ])  # (batch_size, k, features)
        
#         labels = torch.stack([
#             torch.LongTensor([s[1] for s in sample])
#             for sample in batch
#         ])  # (batch_size, k)
        
#         domains = torch.stack([
#             torch.LongTensor([s[2] for s in sample])
#             for sample in batch
#         ])  # (batch_size, k)

#         # print(data.shape, labels.shape, domains.shape)
#         # print(domains)
#         # 按子数据集拆分并重组
#         k = data.shape[0]
#         output = []
#         for i in range(k):
#             output.append((
#                 data[i, :, :],    # (batch_size, features)
#                 labels[i, :],     # (batch_size,)
#                 domains[i, :]     # (batch_size,)
#             ))
#         # print(output)
#         # print(len(output), len(output[0]), output[0][0].shape, output[0][1].shape, output[0][2].shape)
#         return output

#     def __len__(self):
#         return self.num_batches


# class OnlineSeqToyDataset(Dataset):
#     # the size may change because of the toy dataset!!
#     def __init__(self, datasets, size=3 * 200):
#         self.datasets = datasets
#         self.size = size
#         print('SeqDataset Size {} Sub Size {}'.format(
#             size, [len(ds) for ds in datasets]))

#     def __len__(self):
#         return self.size

#     def __getitem__(self, batch_info):
#         selected_subsets, batch_indices = batch_info
#         # print(selected_subsets)
#         batch_data = []

#         for ds in selected_subsets:
#             samples = [self.datasets[ds][idx] for idx in batch_indices[ds]]
#             # for feature_list in zip(*samples):
#             #     print(torch.tensor(np.stack(feature_list)).shape)
#             # exit(0)
#             stacked_tensors = tuple(torch.tensor(feature_list) for feature_list in zip(*samples))
#             # print(stacked_tensors[0].shape)
#             batch_data.append(stacked_tensors)
#         # print(batch_data[0][0].shape)  # 32,2
#         return batch_data

# class OnlineBatchSampler(Sampler):
#     def __init__(self, num_subsets, k, batch_size, size=3 * 200):
#         """
#         dataset_size: 每个子dataset的大小（假设所有子dataset长度相同）
#         num_subsets: 总子数据集数（比如 10 个）
#         k: 每个 batch 选择的子数据集数（比如 3 个）
#         batch_size: 每个 batch 的数据量（比如 32，每个子数据集都取 32）
#         """
#         self.dataset_size = size
#         self.num_subsets = num_subsets
#         self.k = k
#         self.batch_size = batch_size

#         self._reset()

#     def _reset(self):
#         self.remaining_indices = {i: list(range(self.dataset_size)) for i in range(self.num_subsets)}

#         # **随机打乱每个子数据集的索引**
#         for i in self.remaining_indices:
#             random.shuffle(self.remaining_indices[i])

#     def __iter__(self):
#         while True:
#             # **2️⃣ 找到至少有 batch_size 个数据的子数据集**
#             available_subsets = [i for i in self.remaining_indices if len(self.remaining_indices[i]) >= self.batch_size]

#             # **3️⃣ 如果可用子数据集小于 `k`，则终止**
#             if len(available_subsets) < self.k:
#                 # print(self.remaining_indices)
#                 break  # 没有足够的子数据集来填充 batch 了

#             # **4️⃣ 随机选择 `k` 个可用子数据集**
#             selected_subsets = random.sample(available_subsets, self.k)
#             selected_subsets = sorted(selected_subsets, reverse=False)

#             # **5️⃣ 取 `batch_size` 个数据**
#             batch_indices = {}
#             for dataset_idx in selected_subsets:
#                 batch_indices[dataset_idx] = self.remaining_indices[dataset_idx][:self.batch_size]
#                 self.remaining_indices[dataset_idx] = self.remaining_indices[dataset_idx][self.batch_size:]  # **移除已使用索引**

#             yield selected_subsets, batch_indices

#     def __len__(self):
#         total_samples = sum(len(indices) for indices in self.remaining_indices.values())
#         return math.ceil(total_samples / self.batch_size)  # 计算 batch 数