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
import time
start_time = time.time()
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
    try:
        radius = data_pkl['radius']
    except:
        radius = 5

    data_pkl['data'] = (data - data_mean) / data_std  # normalize the raw data
    datasets = [ToyDataset(data_pkl, i, opt)
                for i in range(opt.num_domain)]  # sub dataset for each domain
    
    dataset = SeqToyDataset(datasets, size=len(datasets[0]))  # mix sub dataset to a large one
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=dataset, batch_size=opt.batch_size)

    return dataloader, test_loader, radius, data_mean, data_std


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
rng_dirichlet = np.random.default_rng(seed=42)
rng_choice = np.random.default_rng(seed=42)
from copy import deepcopy
ref_opt = deepcopy(opt)
ref_opt.use_pretrain_model_all = True
# dirichlet_weights = rng_numpy.dirichlet(alpha)
# domain_weights = dirichlet_weights


def test_model(this_epoch, this_opt, this_dataloader, use_warmup=False):
    ref_model = Model(this_opt).to(opt.device)
    test_acc, _ = ref_model.test(this_epoch, this_dataloader)
    del ref_model
    return test_acc

# model.save()
_, test_loader, r, mu, std = get_loader(opt, -1, None, None)
# test_model(0, ref_opt, test_loader)
# dataloader, test_loader, _, _ = get_loader(opt, 0, mu, std)
# test_model(opt.warm_epoch, ref_opt, test_loader)
# exit(0)

test_results = []
from tqdm import tqdm
print(opt.epoch_per_T)
# train without warmup
if opt.use_pretrain_model_warmup:
    for epoch in tqdm(range(opt.num_epoch)):
        if epoch == 0:
            model.save()  # save warm up model
            print(f"Warm up training results for radius {r}.")
            test_model(epoch, ref_opt, test_loader)

        dataloader, test_loader, r, mu, std = get_loader(opt, epoch, mu, std)
        model.__reset_schedulers__()
        dirichlet_weights = rng_dirichlet.dirichlet(alpha)
        domain_weights = dirichlet_weights

        if not opt.upperbound:
            domain_sampled = rng_choice.choice(opt.num_domain, size=opt.k, replace=False, p=domain_weights)
            domain_sel = np.sort(domain_sampled).tolist()
            re_norm_log = domain_weights[domain_sel]
            domain_weight_renorm = re_norm_log / np.sum(re_norm_log)
            # domain_weight_renorm = np.ones_like(domain_sel)
        else:
            domain_sel = np.arange(opt.num_domain).tolist()
            domain_weight_renorm = np.ones_like(domain_sel)

        # domain_weights = None

        # print(f"Warm up training results at first time round.")
        # test_model(epoch, ref_opt, test_loader)
        # print(f"WARMUP training is COMPLETE.")

        # test_flag = epoch == 0 or (epoch + 1) % opt.test_interval == 0 or (epoch + 1) == opt.warm_epoch + opt.num_epoch
        test_flag = True
        if test_flag:
            print(f"Prior training for radius {r}.")
            test_model(epoch, ref_opt, test_loader)
        # assert False

        for tt in range(opt.epoch_per_T):
            model.learn(epoch, dataloader, domain_weight_renorm=domain_weight_renorm, domain_sel=domain_sel, save_buffer_data=opt.use_buffer, verbose=(tt==opt.epoch_per_T-1 or tt==0))

        model.save()

        if test_flag:
            print(f"Post training for radius {r}.")
            testacc = test_model(epoch, ref_opt, test_loader)
            test_results.append(testacc)
    _, test_loader, r, _, _ = get_loader(opt, opt.num_epoch, mu, std)
    print(f"Final Test on the last dataset for radius {r}.")
    testacc = test_model(opt.num_epoch, ref_opt, test_loader)
    test_results.append(testacc)
else:
    # train with warmup
    for epoch in range(opt.warm_epoch + opt.num_epoch):
        if epoch < opt.warm_epoch:
            dataloader, test_loader, _, mu, std = get_loader(opt, epoch, None, None)
            warmup = True
            dirichlet_weights = rng_numpy.dirichlet(alpha)
            domain_weights = np.ones_like(dirichlet_weights)
            domain_weights = None
        elif epoch == opt.warm_epoch:
            model.save()  # save warm up model
            print(f"Warm up training results.")
            test_model(epoch, ref_opt, test_loader)

            dataloader, test_loader, _, _, _ = get_loader(opt, epoch, mu, std)
            warmup = False
            dirichlet_weights = rng_numpy.dirichlet(alpha)
            domain_weights = dirichlet_weights
            domain_weights = None

            print(f"Warm up training results at first time round.")
            test_model(epoch, ref_opt, test_loader)
            print(f"WARMUP training is COMPLETE.")
            print(model.x_buffer[0].shape)
            assert False

        if warmup:
            model.learn(epoch, dataloader, save_buffer_data=(epoch == opt.warm_epoch - 1))
            if epoch == opt.warm_epoch - 1:
                # continue
                model.save(True)
        else:
            test_flag = (epoch + 1) % opt.test_interval == 0 or (epoch + 1) == opt.warm_epoch + opt.num_epoch
            if test_flag:
                test_model(epoch, ref_opt, test_loader)

            for _ in range(100):
                model.learn(epoch, dataloader)

            if test_flag:
                # model.save()
                test_model(epoch, ref_opt, test_loader)
    # last
    dataloader, test_loader, _, _, _ = get_loader(opt, opt.warm_epoch + opt.num_epoch, mu, std)
    print(f"Final Test on the last dataset.")
    test_model(ref_opt, test_loader)



print(f"Training is COMPLETE.")
plt.plot(np.arange(len(test_results)), test_results)
plt.xlabel('Time round')
plt.ylabel('Accuracy')
plt.show()
plt.savefig(f"./growing_new_{opt.epoch_per_T}.png")
plt.clf()
avg_test = sum(test_results) / len(test_results)
print(f"average test accuracy is {avg_test}")
for i in range(0, opt.num_epoch, 20):
    tmp_test_res = test_results[i:i+20]
    tmp_avg_test = sum(tmp_test_res) / len(tmp_test_res)
    print(f"average test accuracy is {tmp_avg_test} for region [{i}: {i+20})")
print('h424', opt.epoch_per_T, opt.k, f"Total: {time.time()-start_time}s")