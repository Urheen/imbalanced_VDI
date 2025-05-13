import os
import torch
import numpy as np
import random
import argparse
import importlib.util

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

from dataset.feature_dataset import FeatureDataloader
import matplotlib.pyplot as plt

dataloader = FeatureDataloader(opt)


# tmp = set()
# for t in range(opt.T):
#     flag = 0
#     for data in dataloader.get_data(t):
#         flag += 1
#         # print(len(data), len(data[0]), len(data[1]))
#         for elem in data:
#             # print(len(elem))
#             for e in elem[2]:
#                 tmp.add(e.item())
#             # print(tmp)
#         # print(tmp)
#     print(flag, tmp)
# exit(0)

alpha = np.ones(opt.num_domain)
rng_dirichlet = np.random.default_rng(seed=49)
rng_choice = np.random.default_rng(seed=49)
from copy import deepcopy
ref_opt = deepcopy(opt)
ref_opt.use_pretrain_model_all = True
# dirichlet_weights = rng_numpy.dirichlet(alpha)
# domain_weights = dirichlet_weights


def test_model(this_epoch, this_opt, this_dataloader, t):
    ref_model = Model(this_opt).to(opt.device)
    test_acc, _ = ref_model.test(this_epoch, this_dataloader, t)
    del ref_model
    return test_acc

import time
sta = time.time()
prev_test_res, post_test_res = [], []
from tqdm import tqdm
print(opt.epoch_per_T)
# train without warmup
if opt.use_pretrain_model_warmup:
    for epoch in tqdm(range(opt.num_epoch)):
        if epoch == 0:
            model.save()  # save warm up model
            print(f"Warm up training results for year 0.")
            test_model(epoch, ref_opt, dataloader, 0)
            continue
        # print(epoch)
        # assert False
        model.__reset_schedulers__()
        
        if not opt.upperbound:
            dirichlet_weights = rng_dirichlet.dirichlet(alpha)
            domain_weights = dirichlet_weights
            if opt.k > 1:
                domain_sampled = rng_choice.choice(np.arange(1, opt.num_domain), size=opt.k-1, replace=False, p=np.ones((opt.num_domain-1))/(opt.num_domain-1))
                domain_sel = [0] + np.sort(domain_sampled).tolist()
                # domain_sel = [elem % opt.num_domain for elem in domain_sel]
            else:
                domain_sel = [0]
            re_norm_log = domain_weights[domain_sel]
            domain_weight_renorm = re_norm_log / np.sum(re_norm_log)
            domain_weight_renorm *= opt.batch_size / domain_weight_renorm.max()
            domain_weight_renorm = np.clip(domain_weight_renorm.astype(int), a_min=opt.num_buffersamples, a_max=opt.batch_size)
            assert 0 in domain_sel, "Source domain not in selection"
        else:
            domain_sel = np.arange(opt.num_domain).tolist()
            domain_weight_renorm = np.ones_like(domain_sel) * opt.batch_size

        print(domain_sel, domain_weight_renorm)

        test_flag = True
        if test_flag:
            print(f"Prior training for year {epoch}.")
            testacc = test_model(epoch, ref_opt, dataloader, epoch)
            prev_test_res.append(testacc)
        # assert False

        for tt in range(opt.epoch_per_T):
            model.learn(epoch, dataloader, t=epoch, domain_weight_renorm=domain_weight_renorm, domain_sel=domain_sel, save_buffer_data=opt.use_buffer, verbose=(tt==opt.epoch_per_T-1 or tt==0))

        model.save()

        if test_flag:
            print(f"Post training for year {epoch}.")
            testacc = test_model(epoch, ref_opt, dataloader, epoch)
            post_test_res.append(testacc)

else:
    # train with warmup
    t = 0
    for epoch in tqdm(range(opt.warm_epoch+1)):
        if epoch < opt.warm_epoch:
            warmup = True
        elif epoch == opt.warm_epoch:
            model.save()  # save warm up model
            print(f"Warm up training results.")
            test_model(epoch, ref_opt, dataloader, t)
            warmup = False

            print(f"Warm up training results at first time round.")
            test_model(epoch, ref_opt, dataloader, t)
            test_model(epoch, ref_opt, dataloader, t)
            test_model(epoch, ref_opt, dataloader, t)
            print(f"WARMUP training is COMPLETE.")
            print(model.x_buffer[0].shape, model.x_buffer.keys())

        if warmup:
            model.learn(epoch, dataloader, t, save_buffer_data=(epoch == opt.warm_epoch - 1))
            if epoch == opt.warm_epoch - 1:
                # continue
                model.save(True)
        else:
            break

delta = time.time() - sta

print(f"prev: {prev_test_res}")
print(f"post: {post_test_res}")

print(f"Training is COMPLETE.")
assert len(prev_test_res) == len(post_test_res)
plt.plot(np.arange(len(prev_test_res)), prev_test_res)
plt.xlabel('Results for Next Round.')
plt.ylabel('Accuracy')
plt.show()
plt.savefig(f"./res_figure/growing_NEXT_{opt.epoch_per_T}.png")
plt.clf()

avg_prev = sum(prev_test_res) / len(prev_test_res)
print(f"NEXT round average test accuracy is {avg_prev}")
for i in range(0, opt.num_epoch-1):
    tmp_test_res = [prev_test_res[i]]
    tmp_avg_test = sum(tmp_test_res) / len(tmp_test_res)
    print(f"NEXT round average test accuracy is {tmp_avg_test} for year {i}")

plt.plot(np.arange(len(post_test_res)), post_test_res)
plt.xlabel('Results for This Round.')
plt.ylabel('Accuracy')
plt.show()
plt.savefig(f"./res_figure/growing_POST_{opt.epoch_per_T}.png")
plt.clf()

avg_post = sum(post_test_res) / len(post_test_res)
print(f"THIS round average test accuracy is {avg_post}")
for i in range(0, opt.num_epoch-1):
    tmp_test_res = [post_test_res[i]]
    tmp_avg_test = sum(tmp_test_res) / len(tmp_test_res)
    print(f"THIS round average test accuracy is {tmp_avg_test} for year {i}")


print('h424', opt.seed, opt.epoch_per_T, opt.k, f"Total: {delta}s")

# import time
# sta = time.time()
# # train
# for epoch in range(opt.num_epoch):
#     print(epoch)
#     model.learn(epoch, dataloader)
#     if (epoch + 1) % opt.save_interval == 0 or (epoch + 1) == opt.num_epoch:
#         model.save()
#     if (epoch + 1) % opt.test_interval == 0 or (epoch + 1) == opt.num_epoch:
#         model.test(epoch, dataloader)

