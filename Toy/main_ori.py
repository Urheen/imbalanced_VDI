import os
import torch
import numpy as np
import random
import pickle
import argparse
import importlib.util
import scipy.stats as stats


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

dataset = SeqToyDataset(datasets, size=len(
    datasets[0]))  # mix sub dataset to a large one
dataloader = DataLoader(dataset=dataset,
                        shuffle=True,
                        batch_size=opt.batch_size)
test_loader = DataLoader(dataset=dataset,
                        batch_size=opt.batch_size)

def get_loader(opt, t, return_test_loader=False):
    data_source = f"{opt.dataset}_{t}.pkl"
    data_source = "data/toy_d15_quarter_circle.pkl"
    print(data_source)

    with open(data_source, "rb") as data_file:
        data_pkl = pickle.load(data_file)
    if t == 0:
        print(f"Data: {data_pkl['data'].shape}\nLabel: {data_pkl['label'].shape}")

    try:
        opt.angle = data_pkl['angle']
    except:
        if t == 0:
            print("The dataset has no angle data.")

    data = data_pkl['data']

    # TODO: Plot_dataset

    data_mean = data.mean(0, keepdims=True)
    data_std = data.std(0, keepdims=True)
    data_pkl['data'] = (data - data_mean) / data_std  # normalize the raw data
    datasets = [ToyDataset(data_pkl, i, opt)
                for i in range(opt.num_domain)]  # sub dataset for each domain
    
    dataset = SeqToyDataset(datasets, size=len(datasets[0]))  # mix sub dataset to a large one
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True)

    if return_test_loader:
        test_loader = DataLoader(dataset=dataset, batch_size=opt.batch_size)
    else:
        test_loader = None

    return dataloader, test_loader

# alpha = torch.ones(opt.num_domain)
# dirichlet_weights = torch.distributions.dirichlet.Dirichlet(alpha).sample()
# selected_domains = torch.topk(dirichlet_weights, opt.k).indices.tolist()
alpha = np.ones(opt.num_domain)
rng_numpy = np.random.default_rng(seed=42)
dirichlet_weights = rng_numpy.dirichlet(alpha)

rng_scipy = np.random.default_rng(seed=42)
Poss = stats.poisson
Poss.random_state = rng_scipy 
poisson_probs = Poss.pmf(np.arange(opt.num_domain), opt.imbal_lambda) + 1e-6
poisson_probs_normalized = poisson_probs / np.sum(poisson_probs)
domain_weights = np.array(poisson_probs_normalized)
poisson_probs = stats.poisson.pmf(np.arange(opt.num_domain), opt.imbal_lambda) + 1e-6
poisson_probs_normalized = poisson_probs / np.sum(poisson_probs)
domain_weights = np.array(poisson_probs_normalized)


dataloader, test_loader = get_loader(opt, 0, return_test_loader=True)

for epoch in range(opt.num_epoch):
    if epoch == 0:
        model.test(epoch, test_loader)
        # for warm_epoch in range(opt.warm_epoch):
        #     model.learn(warm_epoch, dataloader, domain_weights=np.ones_like(domain_weights))
        #     test_flag = (warm_epoch + 1) % opt.test_interval == 0 or (warm_epoch + 1) == opt.warm_epoch
        #     if test_flag:
        #         model.test(warm_epoch, test_loader)
        # assert warm_epoch == -1, f"Warm-up training is not finished with warm_epoch {warm_epoch}!!!"
        # print(f"warm up training DONE!")
        
        # model.test(epoch, test_loader)
        # # exit(0)
        model.learn(epoch, dataloader)
        continue

    # dataloader, test_loader = get_loader(opt, epoch, return_test_loader=test_flag)

    model.learn(epoch, dataloader)
    # np.random.shuffle(domain_weights)

    save_flag = (epoch + 1) % opt.save_interval == 0 or (epoch + 1) == opt.num_epoch
    test_flag = (epoch + 1) % opt.test_interval == 0 or (epoch + 1) == opt.num_epoch

    if save_flag:
        model.save()
    if test_flag:
        model.test(epoch, test_loader)
# # train
# for epoch in range(opt.num_epoch):
#     if epoch == 0:
#         model.test(epoch, test_loader)
#         model.test(epoch, test_loader)
#         model.learn(epoch, dataloader, domain_weights)
#         continue
#     model.learn(epoch, dataloader, domain_weights)
#     if (epoch + 1) % opt.save_interval == 0 or (epoch + 1) == opt.num_epoch:
#         model.save()
#     if (epoch + 1) % opt.test_interval == 0 or (epoch + 1) == opt.num_epoch:
#         model.test(epoch, test_loader)
print('hrewl')