import os
import torch
import numpy as np
import random
import pickle
import argparse
import importlib.util
import warnings
import scipy.stats as stats
import matplotlib.pyplot as plt
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

def get_loader(opt, t, data_mean, data_std):
    if data_mean is None:
        with open("data/toy_d15_quarter_circle.pkl", "rb") as data_file:
            ref_data_pkl = pickle.load(data_file)
            data = ref_data_pkl['data']
            data_mean = data.mean(0, keepdims=True)
            data_std = data.std(0, keepdims=True)
            
    warm_epoch = 0 if opt.use_pretrain_model_warmup else opt.warm_epoch
    if t < warm_epoch:
        data_source = "data/toy_d15_quarter_circle.pkl"
    else:
        data_source = f"{opt.dataset}_{t-warm_epoch}.pkl"
    # data_source = "data/growing_circle/data/toy_d30_pi_-999.pkl"
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

    data_pkl['data'] = (data - data_mean) / data_std  # normalize the raw data
    datasets = [ToyDataset(data_pkl, i, opt)
                for i in range(opt.num_domain)]  # sub dataset for each domain
    
    dataset = SeqToyDataset(datasets, size=len(datasets[0]))  # mix sub dataset to a large one
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=dataset, batch_size=opt.batch_size)

    if t < warm_epoch:
        return dataloader, test_loader, data_mean, data_std
    else:
        return dataloader, test_loader, None, None

with open("data/toy_d15_quarter_circle.pkl", "rb") as data_file:
    ref_data_pkl = pickle.load(data_file)
    data = ref_data_pkl['data']
    data_mean = data.mean(0, keepdims=True)
    data_std = data.std(0, keepdims=True)
    ref_data = (ref_data_pkl['data'] - data_mean) / data_std

with open("data/growing_circle/data/toy_d30_pi_0.pkl", "rb") as data_file:
    curr_data_pkl = pickle.load(data_file)
    curr_data_0 = (curr_data_pkl['data'] - data_mean) / data_std

with open("data/growing_circle/data/toy_d30_pi_1.pkl", "rb") as data_file:
    next_data_pkl = pickle.load(data_file)
    next_data = (next_data_pkl['data'] - data_mean) / data_std

print(curr_data_pkl['data'].mean(), curr_data_pkl['label'].sum())

print(ref_data_pkl['data'].mean(), ref_data_pkl['label'].sum())

l_style_self = ['k*', 'r*', 'b*', 'y*', 'k.', 'r.']
c_style_self = ['red', 'blue']
l_style_ref = ['k*', 'r*', 'b*', 'y*', 'k.', 'r.']
c_style_ref = ['green', 'yellow']
for i in range(2):
    data_sub = curr_data_0[curr_data_pkl['label'] == i, :]
    print(data_sub.shape[0], 'our')
    plt.plot(data_sub[:, 0], data_sub[:, 1], l_style_self[i], color=c_style_self[i], alpha=0.5, label=f"curr_{i}")

    data_sub = next_data[next_data_pkl['label'] == i, :]
    print(data_sub.shape[0], 'ref')
    plt.plot(data_sub[:, 0], data_sub[:, 1], l_style_ref[i], color=c_style_ref[i], alpha=0.5, label=f"next_{i}")
plt.title(f"Circle dataset with time frame.")
plt.legend()
plt.show()
plt.savefig(f"./test_self_0.png")

plt.clf()
for i in range(2):
    data_sub = curr_data_0[curr_data_pkl['label'] == i, :]
    print(data_sub.shape[0], 'our')
    plt.plot(data_sub[:, 0], data_sub[:, 1], l_style_self[i], color=c_style_self[i], alpha=0.5, label=f"curr_{i}")

    data_sub = ref_data[ref_data_pkl['label'] == i, :]
    print(data_sub.shape[0], 'ref')
    plt.plot(data_sub[:, 0], data_sub[:, 1], l_style_ref[i], color=c_style_ref[i], alpha=0.5, label=f"ref_{i}")
plt.title(f"Circle dataset with time frame.")
plt.legend()
plt.show()
plt.savefig(f"./test_self_1.png")
plt.clf()
# exit(0)

# 千万不能un-comment，会降低performance
# alpha = np.ones(opt.num_domain)
# dirichlet_weights = torch.distributions.dirichlet.Dirichlet(alpha, generator=local_generator).sample()
# from numpy.random import Generator, PCG64

# rng_scipy = np.random.default_rng(seed=42)
# Poss = stats.poisson
# Poss.random_state = rng_scipy 
# poisson_probs = Poss.pmf(np.arange(opt.num_domain), opt.imbal_lambda) + 1e-6
# poisson_probs_normalized = poisson_probs / np.sum(poisson_probs)
# domain_weights = np.array(poisson_probs_normalized)

# print(dirichlet_weights)
# print(domain_weights)

# plt.bar(np.arange(len(dirichlet_weights)), sorted(dirichlet_weights))
# plt.savefig('./np.png')
# plt.clf()
# plt.bar(np.arange(len(dirichlet_weights)), domain_weights)
# plt.savefig('./torch.png')
# plt.clf()
# exit(0)
# domain_weights = None

alpha = np.ones(opt.num_domain)
rng_numpy = np.random.default_rng(seed=42)
from copy import deepcopy
ref_opt = deepcopy(opt)
ref_opt.use_pretrain_model_all = True
# dirichlet_weights = rng_numpy.dirichlet(alpha)
# domain_weights = dirichlet_weights


def test_model(this_epoch, this_opt, this_dataloader):
    ref_model = Model(this_opt).to(opt.device)
    ref_model.test(this_epoch, this_dataloader)
    del ref_model

dataloader, test_loader, mu, std = get_loader(opt, -1, None, None)
# test_model(0, ref_opt, test_loader)
# dataloader, test_loader, _, _ = get_loader(opt, opt.warm_epoch, mu, std)
# test_model(opt.warm_epoch, ref_opt, test_loader)
# exit(0)

# train without warmup
if opt.use_pretrain_model_warmup:
    for epoch in range(opt.num_epoch):
        model.save()  # save warm up model
        print(f"Warm up training results.")
        test_model(epoch, ref_opt, test_loader)

        dataloader, test_loader, mu, std = get_loader(opt, epoch, None, None)
        warmup = False
        dirichlet_weights = rng_numpy.dirichlet(alpha)
        domain_weights = dirichlet_weights
        domain_weights = None

        print(f"Warm up training results at first time round.")
        test_model(epoch, ref_opt, test_loader)
        print(f"WARMUP training is COMPLETE.")
        exit(0)


# train with warmup
for epoch in range(opt.warm_epoch + opt.num_epoch):
    if epoch < opt.warm_epoch:
        dataloader, test_loader, mu, std = get_loader(opt, epoch, None, None)
        warmup = True
        dirichlet_weights = rng_numpy.dirichlet(alpha)
        domain_weights = np.ones_like(dirichlet_weights)
        domain_weights = None
    elif epoch == opt.warm_epoch:
        model.save()  # save warm up model
        print(f"Warm up training results.")
        test_model(epoch, ref_opt, test_loader)

        dataloader, test_loader, _, _ = get_loader(opt, epoch, mu, std)
        warmup = False
        dirichlet_weights = rng_numpy.dirichlet(alpha)
        domain_weights = dirichlet_weights
        domain_weights = None

        print(f"Warm up training results at first time round.")
        test_model(epoch, ref_opt, test_loader)
        print(f"WARMUP training is COMPLETE.")
        exit(0)

    # if epoch == 0:
    #     ref_model = Model(opt).to(opt.device)
    #     ref_model.test(epoch, test_loader)
    #     del ref_model

    if warmup:
        model.learn(epoch, dataloader)
        if epoch == opt.warm_epoch - 1:
            model.save(True)

    else:
        test_flag = (epoch + 1) % opt.test_interval == 0 or (epoch + 1) == opt.warm_epoch + opt.num_epoch
        if test_flag:
            test_model(epoch, ref_opt, test_loader)

        for _ in range(100):
            model.learn(epoch, dataloader)

        if test_flag:
            model.save()
            test_model(epoch, ref_opt, test_loader)

# last
dataloader, test_loader, _, _ = get_loader(opt, opt.warm_epoch + opt.num_epoch, mu, std)
print(f"Final Test on the last dataset.")
test_model(ref_opt, test_loader)
print(f"Training is COMPLETE.")
print('h4', opt.test_interval)