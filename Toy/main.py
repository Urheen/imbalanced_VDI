import os
import torch
import numpy as np
import random
import pickle
import argparse
import importlib.util
import warnings
warnings.filterwarnings("ignore")

# load the config files
parser = argparse.ArgumentParser(description='Choose the configs to run.')
parser.add_argument('-c', '--config', type=str, required=True)
args = parser.parse_args()

use_config_spec = importlib.util.spec_from_file_location(
    args.config, "configs/{}.py".format(args.config))
config_module = importlib.util.module_from_spec(use_config_spec)
use_config_spec.loader.exec_module(config_module)
opt = config_module.opt

# set which gpu to use
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_device

# random seed specification
np.random.seed(opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)

# init model
from model.model import VDI as Model

model = Model(opt).to(opt.device)

data_source = opt.dataset

# load the data
from dataset.dataset import *

data_source = opt.dataset

with open(data_source, "rb") as data_file:
    data_pkl = pickle.load(data_file)
print(f"Data: {data_pkl['data'].shape}\nLabel: {data_pkl['label'].shape}")

try:
    opt.angle = data_pkl['angle']
except:
    print("The dataset has no angle data.")

data = data_pkl['data']
data_mean = data.mean(0, keepdims=True)
data_std = data.std(0, keepdims=True)
data_pkl['data'] = (data - data_mean) / data_std  # normalize the raw data
datasets = [ToyDataset(data_pkl, i, opt)
            for i in range(opt.num_domain)]  # sub dataset for each domain

# multi_dataset = ZiyanDataset(datasets)
# sampler = ZiyanDatasetSampler(datasets, batch_size=opt.batch_size, K=opt.k, shuffle=True, drop_last=True)
# dataloader = DataLoader(multi_dataset, batch_sampler=sampler, collate_fn=ziyan_collate)

# dataset_off = SeqToyDataset(datasets, size=len(datasets[0]))  # mix sub dataset to a large one
# dataloader_off = DataLoader(dataset=dataset_off, batch_size=32, shuffle=False, drop_last=False)
# print('hey')
# for epoch in range(5):  # Run for two epochs
#     print(f"Epoch {epoch}")
#     for batch_idx, elem in enumerate(dataloader):
#         x_seq, y_seq, domain_seq = [d[0][None, :, :] for d in elem
#                                         ], [d[1][None, :] for d in elem
#                                             ], [d[2][None, :] for d in elem]
#         x_seq_tmp = torch.cat(x_seq, 0)
#         y_seq_tmp = torch.cat(y_seq, 0)
#         domain_seq_tmp = torch.cat(domain_seq, 0)
#         print('on', x_seq_tmp.shape, y_seq_tmp.shape, domain_seq_tmp.shape, domain_seq_tmp[:,0].flatten())
#     dataloader.batch_sampler.reset_order()
# exit(0)
#     # for batch_idx, elem in enumerate(dataloader_off):
#     #     x_seq, y_seq, domain_seq = [d[0][None, :, :] for d in elem
#     #                                     ], [d[1][None, :] for d in elem
#     #                                         ], [d[2][None, :] for d in elem]
#     #     x_seq_tmp = torch.cat(x_seq, 0)
#     #     y_seq_tmp = torch.cat(y_seq, 0)
#     #     domain_seq_tmp = torch.cat(domain_seq, 0)
#     #     print('off', x_seq_tmp.shape, y_seq_tmp.shape, domain_seq_tmp.shape, domain_seq_tmp)
#     #     break
# exit(0)

if opt.online:
    # dataset = OnlineSeqToyDataset(datasets, size=len(datasets[0]))  # mix sub dataset to a large one
    # test_sampler = OnlineBatchSampler(len(datasets), opt.num_domain, opt.batch_size, size=len(datasets[0]))
    # test_dataloader = DataLoader(dataset=dataset, sampler=test_sampler)
    # sampler = OnlineBatchSampler(len(datasets), opt.k, opt.batch_size, size=len(datasets[0]))
    # dataloader = DataLoader(dataset=dataset, sampler=sampler)
    # dataset = ziyanSeqToyDataset(datasets, size=len(datasets[0]))  # mix sub dataset to a large one
    # dataloader = KSubsetDataLoader(
    #     dataset=dataset,
    #     batch_size=opt.batch_size,
    #     k=opt.k,
    #     shuffle=True,
    #     drop_last=True
    # )
    multi_dataset = ZiyanDataset(datasets)
    sampler = ZiyanDatasetSampler(datasets, batch_size=opt.batch_size, K=opt.k, shuffle=True)
    dataloader = DataLoader(multi_dataset, batch_sampler=sampler, collate_fn=ziyan_collate)

    # dataset = SeqToyDataset(datasets, size=len(datasets[0]))  # mix sub dataset to a large one
    # dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True)
else:
    dataset = SeqToyDataset(datasets, size=len(datasets[0]))  # mix sub dataset to a large one
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True)

test_dataset = SeqToyDataset(datasets, size=len(datasets[0]))  # mix sub dataset to a large one
test_loader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size)

# train
for epoch in range(opt.num_epoch):
    if epoch == 0:
        model.test(epoch, test_loader)
        if opt.online:
            print(f"The domain for each batch is {opt.k}.")
            if opt.use_selector:
                print(f"The number of the selected samples is {opt.num_filtersamples}.")

    model.learn(epoch, dataloader)
    if opt.online:
        dataloader.batch_sampler.reset_order()

    if (epoch + 1) % opt.save_interval == 0 or (epoch + 1) == opt.num_epoch:
        model.save()
    if (epoch + 1) % opt.test_interval == 0 or (epoch + 1) == opt.num_epoch:
        model.test(epoch, test_loader)
print(f"{opt.online}")