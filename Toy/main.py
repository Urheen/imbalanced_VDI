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

# dataset = SeqToyDataset(datasets, size=len(datasets[0]))  # mix sub dataset to a large one
# dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True)

# random.seed(13)
# dataset = ziyanSeqToyDataset(datasets, size=len(datasets[0]))  # mix sub dataset to a large one
# dataloader_te = KSubsetDataLoader(
#     dataset=dataset,
#     batch_size=32,
#     k=3,
#     shuffle=True
# )

# for epoch in range(1):

#     for elem in dataloader:
#         for d in elem:
#             print('off', epoch, len(elem), len(d), d[0].shape, d[1].shape, d[2].shape)
#             break

#     for elem in dataloader_te:
#         for d in elem:
#             print('on', epoch, len(elem), len(d), d[0].shape, d[1].shape, d[2].shape)
#             break
# exit(0)

    

if opt.online:
    # dataset = OnlineSeqToyDataset(datasets, size=len(datasets[0]))  # mix sub dataset to a large one
    # test_sampler = OnlineBatchSampler(len(datasets), opt.num_domain, opt.batch_size, size=len(datasets[0]))
    # test_dataloader = DataLoader(dataset=dataset, sampler=test_sampler)
    # sampler = OnlineBatchSampler(len(datasets), opt.k, opt.batch_size, size=len(datasets[0]))
    # dataloader = DataLoader(dataset=dataset, sampler=sampler)
    dataset = ziyanSeqToyDataset(datasets, size=len(datasets[0]))  # mix sub dataset to a large one
    dataloader = KSubsetDataLoader(
        dataset=dataset,
        batch_size=opt.batch_size,
        k=opt.k,
        shuffle=True,
        drop_last=True
    )
    test_loader = KSubsetDataLoader(
        dataset=dataset,
        batch_size=opt.batch_size,
        k=opt.num_domain,
    )


# TODO: this is the test to check whether the online environment is correctly simulated.
# for epoch in range(3):
#     print()
#     if opt.online:
#         random.seed(13+epoch)
#         sampler = OnlineBatchSampler(len(datasets), opt.k, opt.batch_size, size=len(datasets[0]))
#         dataloader = DataLoader(dataset=dataset, sampler=sampler)
#     for data in dataloader:
#         print([torch.unique(d[2]) for d in data])



# exit(0)
# train
for epoch in range(opt.num_epoch):
    if epoch == 0:
        # if opt.online:
            # random.seed(13+epoch)
            # test_dataloader.sampler._reset()
        model.test(epoch, test_loader)

    verbose = epoch >= 79

    # if opt.online:
    #     dataloader.sampler._reset()
    model.learn(epoch, dataloader, verbose=verbose)

    if (epoch + 1) % opt.save_interval == 0 or (epoch + 1) == opt.num_epoch:
        model.save()
    if (epoch + 1) % opt.test_interval == 0 or (epoch + 1) == opt.num_epoch:
        # if opt.online:
        #     random.seed(13+epoch)
        #     sampler = OnlineBatchSampler(len(datasets), opt.num_domain, opt.batch_size, size=len(datasets[0]))
        #     dataloader = DataLoader(dataset=dataset, sampler=sampler)
        # if opt.online:
        #     test_dataloader.sampler._reset()
        model.test(epoch, test_loader, (epoch >= 49))
