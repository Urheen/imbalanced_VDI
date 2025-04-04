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

import scipy.stats as stats
import matplotlib.pyplot as plt

# init model
from model.model import VDI as Model

model = Model(opt).to(opt.device)

# load the data
from dataset.dataset import *


def get_loader(opt, t, return_test_loader=False):
    data_source = f"{opt.dataset}_{t}.pkl"
    # data_source = "data/toy_d15_quarter_circle.pkl"
    print(data_source)

    with open(data_source, "rb") as data_file:
        data_pkl = pickle.load(data_file)
    print(f"Data: {data_pkl['data'].shape}\nLabel: {data_pkl['label'].shape}")

    try:
        opt.angle = data_pkl['angle']
    except:
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
        test_dataset = SeqToyDataset(datasets, size=len(datasets[0]))  # mix sub dataset to a large one
        test_loader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size)
    else:
        test_loader = None

    return dataloader, test_loader

# with open("data/growing_circle/data/toy_d30_pi_500.pkl", "rb") as data_file:
#     curr_data_pkl = pickle.load(data_file)

# with open("data/toy_d15_quarter_circle.pkl", "rb") as data_file:
#     ref_data_pkl = pickle.load(data_file)

# print(curr_data_pkl['data'].mean(), curr_data_pkl['label'].sum())

# print(ref_data_pkl['data'].mean(), ref_data_pkl['label'].sum())

# l_style_self = ['k*', 'r*', 'b*', 'y*', 'k.', 'r.']
# c_style_self = ['red', 'blue']
# l_style_ref = ['k*', 'r*', 'b*', 'y*', 'k.', 'r.']
# c_style_ref = ['green', 'yellow']
# for i in range(2):
#     data_sub = curr_data_pkl['data'][curr_data_pkl['label'] == i, :]
#     plt.plot(data_sub[:, 0], data_sub[:, 1], l_style_self[i], color=c_style_self[i], alpha=0.5)

#     data_sub = ref_data_pkl['data'][ref_data_pkl['label'] == i, :]
#     plt.plot(data_sub[:, 0], data_sub[:, 1], l_style_ref[i], color=c_style_ref[i], alpha=0.5)
# plt.title(f"Circle dataset with time frame.")
# plt.show()
# plt.savefig(f"./test_self.png")
# plt.clf()

alpha = torch.ones(opt.num_domain)
dirichlet_weights = torch.distributions.dirichlet.Dirichlet(alpha).sample()
selected_domains = torch.topk(dirichlet_weights, opt.k).indices.tolist()
import numpy as np
import matplotlib.pyplot as plt


poisson_probs = stats.poisson.pmf(np.arange(opt.num_domain), opt.imbal_lambda) + 1e-6
poisson_probs_normalized = poisson_probs / np.sum(poisson_probs)
domain_weights = torch.from_numpy(poisson_probs_normalized)
# sampled_classes = np.random.choice(opt.num_domain, size=3, p=poisson_probs_normalized, replace=False)
# selected = torch.multinomial(domain_weights, 3)
dataloader, test_loader = get_loader(opt, 0, return_test_loader=True)
# train
for epoch in range(opt.num_epoch):
    if epoch == 0:
        # dataloader, test_loader = get_loader(opt, epoch, return_test_loader=True)
        model.test(epoch, test_loader)
        # if opt.online:
        #     print(f"The domain for each batch is {opt.k}.")
        #     if opt.use_selector:
        #         print(f"The number of the selected samples is {opt.num_filtersamples}.")
        # warmup training
        model.learn(epoch, dataloader, domain_weights=torch.ones_like(domain_weights))
        continue

    save_flag = (epoch + 1) % opt.save_interval == 0 or (epoch + 1) == opt.num_epoch
    test_flag = (epoch + 1) % opt.test_interval == 0 or (epoch + 1) == opt.num_epoch
    # dataloader, test_loader = get_loader(opt, epoch, return_test_loader=flag)

    model.learn(epoch, dataloader, domain_weights=domain_weights)

    if save_flag:
        model.save()
    if test_flag:
        model.test(epoch, test_loader)
