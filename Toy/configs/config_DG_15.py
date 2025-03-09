from easydict import EasyDict
import numpy as np

from time import localtime, strftime
# set experiment configs
opt = EasyDict()

opt.src_domain_idx = [0, 12, 3, 4, 14, 8]
opt.tgt_domain_idx = [1, 2, 5, 6, 7, 9, 10, 11, 13]
# opt.src_domain_idx = [0, 3, 4, 8]
# opt.tgt_domain_idx = [1, 2, 5, 6, 7, 9]
opt.all_domain_idx = opt.src_domain_idx + opt.tgt_domain_idx
opt.num_source = len(opt.src_domain_idx)
opt.num_target = len(opt.tgt_domain_idx)
opt.num_domain = len(opt.all_domain_idx)

opt.dataset = "data/toy_d15_spiral_tight_boundary.pkl"
opt.d_loss_type = "GRDA_loss"  # "DANN_loss" # "CIDA_loss" # "DANN_loss_mean" "GRDA_loss"

opt.use_pretrain_R = True
opt.pretrain_R_path = "data/netR_4_dann.pth"
opt.pretrain_U_path = "data/netU_4_dann.pth"
opt.fix_u_r = False

opt.use_pretrain_model_all = False

opt.lambda_gan = 0.6
opt.lambda_reconstruct = 10
opt.lambda_u_concentrate = 1
opt.lambda_beta = 0.8
opt.lambda_beta_alpha = 0.1

# for warm up
opt.init_lr = 1e-6
opt.peak_lr_e = 3.2 * 1e-4
opt.peak_lr_d = 3.2 * 1e-4
opt.final_lr = 1e-8
opt.warmup_steps = 70

opt.seed = 2333
opt.num_epoch = 2000
opt.batch_size = 32

opt.use_visdom = False  # True
opt.visdom_port = 2000
opt.test_on_all_dmn = False
tmp_time = localtime()
opt.outf = "result_save/{}".format(strftime("%Y-%m-%d %H:%M:%S", tmp_time))

opt.save_interval = 100
opt.test_interval = 2  # 20

opt.device = "cuda"
opt.gpu_device = "0"
opt.gamma = 100
opt.beta1 = 0.9
opt.weight_decay = 5e-4
opt.no_bn = True  # do not use batch normalization
opt.normalize_domain = False

# network parameter
opt.num_hidden = 512
opt.num_class = 2  # how many classes for classification input data x
opt.input_dim = 2  # the dimension of input data x

opt.u_dim = 4  # the dimension of local domain index u
opt.beta_dim = 2  # the dimension of global domain index beta

# for grda discriminator
opt.sample_v = 10

# online settings, # of domain in each batch
opt.online = True
# opt.online = False

if opt.online:
    opt.k = opt.num_domain
    opt.use_selector = False
    opt.n_neighbors = opt.batch_size - 1
    opt.num_filtersamples = opt.batch_size
else:
    opt.k = opt.num_domain
    opt.use_selector = False
    opt.n_neighbors = 0
    opt.num_filtersamples = 0