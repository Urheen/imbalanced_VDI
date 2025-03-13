import random
import itertools
import torch
from torch.utils.data import Dataset, Sampler, DataLoader

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
    def __init__(self, datasets, batch_size, K, shuffle=True, drop_last=False, allow_padding=True):
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
        # index_order = list(range(self.num_samples))  # Sample indices
        # if self.shuffle:
        #     random.shuffle(index_order)

        # index_iter = iter(index_order)
        if self.shuffle:
            index_order = torch.randperm(self.num_samples).tolist()
        else:
            index_order = list(range(self.num_samples))
        index_iter = iter(index_order)
        batch = list(itertools.islice(index_iter, self.batch_size))

        while batch:
            # Shuffle sub-dataset order for each batch
            # domain_order = list(range(self.N))
            # if self.shuffle:
            #     random.shuffle(domain_order)
            # domain_iter = iter(domain_order)
            
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

                if len(domain_groups) < self.K and self.allow_padding:
                    # Fill missing sub-dataset slots with last seen dataset
                    while len(domain_groups) < self.K:
                        domain_groups.append(domain_groups[-1])

                self.batches.append(this_batches)
                domain_groups = list(itertools.islice(domain_iter, self.K))
                domain_groups.sort()

            batch = list(itertools.islice(index_iter, self.batch_size))

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
    

def ziyan_collate(batch_list):
    x, y = [], []
    print(batch_list[0])
    for elem in batch_list:
        x.append(elem[0])
        y.append(elem[1])
    batch_tensor = torch.stack(x, dim=0)  # shape [batch_size*K, D]
    batch_label = torch.stack(y, dim=0) 
    # print(batch_tensor.shape, batch_label)
    this_doamin_num = torch.unique(batch_label).flatten().shape[0]
    data_dim = batch_tensor.shape[-1]
    # assert this_doamin_num == K
    batch_tensor = batch_tensor.view(this_doamin_num, -1, data_dim)  # shape [K, batch_size, D]
    batch_label = batch_label.view(this_doamin_num, -1, 1)
    # print(batch_label)
    # batch_tensor = batch_tensor.permute(1, 0, 2).contiguous()  # shape [batch_size, K, D]
    return batch_tensor, batch_label

# Create synthetic dataset for testing
N = 9            # Number of sub-datasets
M = 30           # Samples per sub-dataset
batch_size = 12  # Number of samples per batch
K = 3            # Number of sub-datasets per batch
D = 2            # Feature dimension

# Generate sub-datasets
datasets = []
for i in range(N):
    dataset = [(torch.FloatTensor([i,j]), torch.LongTensor([i])) for j in range(M)]
    datasets.append(dataset)

# Create dataset and sampler
multi_dataset = ZiyanDataset(datasets)
sampler = ZiyanDatasetSampler(datasets, batch_size=batch_size, K=K, shuffle=True, allow_padding=True)

# Create DataLoader
dataloader = DataLoader(multi_dataset, batch_sampler=sampler, collate_fn=ziyan_collate)

# Verification
print("### Verification: Batch shapes and sub-dataset IDs ###")
seen_subdatasets = [set() for _ in range(N)]

for epoch in range(2):  # Run for two epochs
    print(f"Epoch {epoch}")
    for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
        sub_ids_in_batch = set()
        for sample in batch_data.view(-1, D):
            sub_id = int(sample[0].item())
            samp_id = int(sample[1].item())
            sub_ids_in_batch.add(sub_id)
            seen_subdatasets[sub_id].add(samp_id)

        print(f" Batch {batch_idx}: batch_data.shape = {tuple(batch_data.shape)}, sub-dataset IDs = {sorted(list(sub_ids_in_batch))}")
    dataloader.batch_sampler.reset_order()

print("\n### Coverage Check ###")
for sub_id in range(N):
    seen_count = len(seen_subdatasets[sub_id])
    print(f"Sub-dataset {sub_id}: {seen_count}/{M} samples seen.")
