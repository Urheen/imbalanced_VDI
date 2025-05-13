import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from model.modules import *
import os
from visdom import Visdom
import pickle
import json

from model.lr_scheduler import TransformerLRScheduler
from sklearn.manifold import MDS
from geomloss import SamplesLoss
# const
LARGE_NUM = 1e9


# ========================
def to_np(x):
    return x.detach().cpu().numpy()


def to_tensor(x, device="cuda"):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
    else:
        x = x.to(device)
    return x


def flat(x):
    n, m = x.shape[:2]
    return x.reshape(n * m, *x.shape[2:])


def write_pickle(data, name):
    with open(name, 'wb') as f:
        pickle.dump(data, f)

def repeat_data(data, expand_size):
    repeat_times = (expand_size + data.shape[0] - 1) // data.shape[0]
    return data.repeat(repeat_times, 1)[:expand_size]

# =========================
# the base model
class BaseModel(nn.Module):

    def __init__(self, opt):
        super(BaseModel, self).__init__()
        # set output format
        np.set_printoptions(suppress=True, precision=6)

        self.opt = opt
        self.device = opt.device
        self.use_visdom = opt.use_visdom
        if opt.use_visdom:
            self.env = Visdom(port=opt.visdom_port)
            self.test_pane = dict()
            self.test_pane_init = False

        self.num_domain = opt.num_domain

        self.outf = self.opt.outf
        self.train_log = self.outf + "/loss.log"
        self.model_path = self.outf + '/model.pth'
        if not os.path.exists(self.opt.outf):
            os.mkdir(self.opt.outf)

        if not os.path.exists(self.opt.outf_warm):
            os.mkdir(self.opt.outf_warm)

        with open(self.train_log, 'w') as f:
            f.write("log start!\n")

        # save all the config to json file
        with open("{}/config.json".format(self.outf), "w") as outfile:
            json.dump(self.opt, outfile, indent=2)

        mask_list = np.zeros(opt.num_domain)
        mask_list[opt.src_domain_idx] = 1
        self.domain_mask = torch.IntTensor(mask_list).to(opt.device)

        # nan flag
        self.nan_flag = False

        # beta:
        self.use_beta_seq = None

        self.init_test = False

        self.x_buffer = {}
        self.y_buffer = {}
        self.domain_buffer = {}
        self.buffer_sel = torch.ones((self.num_domain, self.opt.batch_size)).to(self.device) * -1.
        self.use_buffer = self.opt.use_buffer

    def reset_buffer(self):
        del self.x_buffer, self.y_buffer, self.domain_buffer, self.buffer_sel
        self.x_buffer = {}
        self.y_buffer = {}
        self.domain_buffer = {}
        self.buffer_sel = torch.ones((self.num_domain, self.opt.batch_size)).to(self.device) * -1.
        self.use_buffer = self.opt.use_buffer
        return

    def learn(self, epoch, dataloader, t, domain_weight_renorm=None, domain_sel=None, save_buffer_data=False, verbose=False):
        self.train()

        self.epoch = epoch
        loss_values = {loss: 0 for loss in self.loss_names}
        self.new_u = []
        self.new_u_buffer = [[] for _ in range(self.num_domain)]

        if self.opt.use_pretrain_model_warmup:
            self.domain_sel = domain_sel
        else:
            self.domain_sel = np.arange(self.opt.num_domain).tolist()
            domain_weight_renorm = np.ones_like(self.domain_sel) * self.opt.batch_size

        count = 0
        for data in dataloader.get_data(t):
            # print(count)
            count += 1
            self.__set_input__(data, domain_weight_renorm)
            self.__train_forward__()
            new_loss_values = self.__optimize__()

            # for the loss visualization
            for key, loss in new_loss_values.items():
                loss_values[key] += loss

            for elem in range(self.u_seq.shape[0]):
                valid = self.u_seq[elem][self.domain_sel_mask[elem] == 1]
                # print(valid.shape)
                self.new_u_buffer[elem].append(valid.unsqueeze(0))
            # self.new_u.append(self.u_seq)

            if save_buffer_data:
                self.__buffer_input__()

        for elem in range(len(self.new_u_buffer)):
            self.new_u_buffer[elem] = self.my_cat(self.new_u_buffer[elem])
        self.use_beta_seq = self.generate_beta(self.new_u_buffer, is_global=True)

        for key, _ in new_loss_values.items():
            loss_values[key] /= count

        if self.use_visdom:
            self.__vis_loss__(loss_values)

        if verbose:
            if (self.epoch + 1) % 10 == 0 or self.epoch == 0:
                print("epoch {}, loss: {}, lambda gan: {}".format(
                    self.epoch, loss_values, self.opt.lambda_gan))

        # learning rate decay
        for lr_scheduler in self.lr_schedulers:
            if verbose:
                print(lr_scheduler.update_steps)
            lr_scheduler.step()

        # check nan
        if any(np.isnan(val) for val in loss_values.values()):
            self.nan_flag = True
        else:
            self.nan_flag = False

    def test(self, epoch, dataloader, t):
        # Assuming test on all domains!
        self.eval()
        self.epoch = epoch

        # init the sample number:
        if not self.init_test:
            # drop last
            batch_num = np.floor(dataloader.sudo_len / self.opt.batch_size)
            factor = np.ceil(self.opt.save_sample / batch_num)
            self.save_sample = int(factor * batch_num)
            self.factor = int(factor)
            self.save_sample_idx = np.arange(self.factor)
            self.init_test = True

        acc_num = torch.zeros(self.num_domain).to(self.device)
        l_x = torch.zeros(self.num_domain, self.save_sample,
                          self.opt.input_dim).to(self.device)
        l_y = torch.zeros(self.num_domain, self.save_sample).to(self.device)
        # l_r_x = []
        l_domain = torch.zeros(self.num_domain,
                               self.save_sample).to(self.device)
        l_label = torch.zeros(self.num_domain,
                              self.save_sample).to(self.device)
        l_encode = torch.zeros(self.num_domain, self.save_sample,
                               self.opt.num_hidden).to(self.device)
        l_u = torch.zeros(self.num_domain, self.opt.u_dim).to(self.device)
        l_u_all = torch.zeros(self.num_domain, self.save_sample,
                              self.opt.u_dim).to(self.device)

        sample_count = 0
        # sample a few datapoints for visualization
        count = 0
        self.domain_sel = np.arange(self.opt.num_domain).tolist()

        for data in dataloader.get_data(t):
            self.__set_input__(data, train=False)

            # forward
            with torch.no_grad():
                self.__test_forward__()
                # drop last batch
                if self.tmp_batch_size < self.opt.batch_size:
                    continue
                # y dim: domain x batch x dim
                sample_count += self.y_seq.shape[1]
                count += 1

                acc_num += self.g_seq.eq(self.y_seq).to(torch.float).sum(-1)
                l_x[:, (count - 1) * self.factor:count *
                    self.factor, :] = self.x_seq[:, self.save_sample_idx, :]
                l_y[:, (count - 1) * self.factor:count *
                    self.factor] = self.y_seq[:, self.save_sample_idx]
                l_domain[:, (count - 1) * self.factor:count *
                         self.factor] = self.domain_seq[:,
                                                        self.save_sample_idx]
                l_encode[:, (count - 1) * self.factor:count *
                         self.factor, :] = self.q_z_seq[:, self.
                                                        save_sample_idx, :]
                l_label[:, (count - 1) * self.factor:count *
                        self.factor] = self.g_seq[:, self.save_sample_idx]
                l_u += self.u_seq.sum(1)
                l_u_all[:, (count - 1) * self.factor:count *
                        self.factor, :] = self.u_seq[:,
                                                     self.save_sample_idx, :]

        acc = to_np(acc_num / sample_count)
        test_acc = acc[self.opt.tgt_domain_idx].sum() / (
            self.opt.num_target) * 100
        acc_msg = '[Test][{}] Accuracy: total average {:.1f}, test average {:.1f}, in each domain {}'.format(
            epoch,
            acc.mean() * 100, test_acc, np.around(acc * 100, decimals=1))
        self.__log_write__(acc_msg)
        if self.use_visdom:
            self.__vis_test_error__(test_acc, 'test acc')

        d_all = dict()
        d_all['acc_msg'] = acc_msg
        d_all['data'] = flat(to_np(l_x))
        d_all['gt_label'] = flat(to_np(l_y))
        d_all['domain'] = flat(to_np(l_domain))
        d_all['label'] = flat(to_np(l_label))
        d_all['encodeing'] = flat(to_np(l_encode))
        d_all['u'] = to_np(l_u / self.save_sample)
        # d_all['r_x'] = flat(r_x_all)
        d_all['u_all'] = flat(to_np(l_u_all))
        d_all['beta'] = to_np(self.beta_seq)

        if (
                self.epoch + 1
        ) % self.opt.save_interval == 0 or self.epoch + 1 == self.opt.num_epoch:
            write_pickle(d_all, self.opt.outf + '/' + str(epoch) + '_pred.pkl')

        return test_acc, self.nan_flag
    
    def inference(self, dataloader):
        self.test(epoch=self.opt.num_epoch-1, dataloader=dataloader)

    def my_cat(self, new_u_seq):
        # concatenation of local domain index u
        st = new_u_seq[0]
        idx_end = len(new_u_seq)
        for i in range(1, idx_end):
            st = torch.cat((st, new_u_seq[i]), dim=1)
        return st

    def __vis_test_error__(self, loss, title):
        if not self.test_pane_init:
            self.test_pane[title] = self.env.line(X=np.array([self.epoch]),
                                                  Y=np.array([loss]),
                                                  opts=dict(title=title))
            self.test_pane_init = True
        else:
            self.env.line(X=np.array([self.epoch]),
                          Y=np.array([loss]),
                          win=self.test_pane[title],
                          update='append')

    def save(self, warm=False):
        fpath = self.opt.outf_warm if warm else self.outf 
        torch.save(self.netU.state_dict(), fpath + '/netU.pth')
        torch.save(self.netUCon.state_dict(), fpath + '/netUCon.pth')
        torch.save(self.netZ.state_dict(), fpath + '/netZ.pth')
        torch.save(self.netF.state_dict(), fpath + '/netF.pth')
        torch.save(self.netR.state_dict(), fpath + '/netR.pth')
        torch.save(self.netD.state_dict(), fpath + '/netD.pth')
        torch.save(self.netBeta.state_dict(), fpath + '/netBeta.pth')
        torch.save(self.netBeta2U.state_dict(), fpath + '/netBeta2U.pth')
        torch.save([self.x_buffer, self.y_buffer, self.domain_buffer], fpath + '/buffer.pth')

    def __set_input__(self, data, domain_weight_renorm=None, train=True):
        """
        :param
            x_seq: Number of domain x Batch size x  Data dim
            y_seq: Number of domain x Batch size x Predict Data dim
            (testing: Number of domain x Batch size x test len x Predict Data dim)
            one_hot_seq: Number of domain x Batch size x Number of vertices (domains)
            domain_seq: Number of domain x Batch size x domain dim (1)
            idx_seq: Number of domain x Batch size x 1 (the order in the whole dataset)
            y_value_seq: Number of domain x Batch size x Predict Data dim
        """
        x_seq, y_seq, domain_seq = [d[0][None, :, :] for d in data
                                        ], [d[1][None, :] for d in data
                                            ], [d[2][None, :] for d in data]

        self.x_seq_tmp = torch.cat(x_seq, 0).to(self.device)
        self.y_seq_tmp = torch.cat(y_seq, 0).to(self.device)
        self.domain_seq_tmp = torch.cat(domain_seq, 0).to(self.device)
        # print(torch.unique(self.domain_seq_tmp))
        self.domain_seq_tmp = (self.domain_seq_tmp % self.num_domain).detach()
        # print(torch.unique(self.domain_seq_tmp))
        # if train:
        #     assert False
        # self.domain_sel = torch.unique(self.domain_seq_tmp).tolist()

        self.x_seq = torch.zeros((self.num_domain, *self.x_seq_tmp.shape[1:])).to(self.device)
        self.y_seq = torch.zeros((self.num_domain, *self.y_seq_tmp.shape[1:])).to(device=self.device, dtype=torch.long)
        self.domain_seq = torch.zeros((self.num_domain, *self.domain_seq_tmp.shape[1:])).to(device=self.device, dtype=torch.long)
        self.domain_sel_mask = torch.zeros((self.num_domain, self.x_seq_tmp.shape[1])).to(self.device)
        self.domain_sel_mask[self.domain_sel, :] = 1.0
        self.domain_sel_mask.requires_grad = False
        # print(self.domain_sel_mask.shape)
        self.tmp_batch_size = self.x_seq.shape[1]

        for idx, elem in enumerate(self.domain_sel):
            # print(idx, elem)
            # print(self.x_seq[elem].shape, self.x_seq_tmp[idx].shape)
            self.x_seq[elem, :, :] = self.x_seq_tmp[elem]
            self.y_seq[elem] = self.y_seq_tmp[elem]
            self.domain_seq[elem] = self.domain_seq_tmp[elem]

            if train:
                curr_sample = domain_weight_renorm[idx]
                self.domain_sel_mask[elem, curr_sample:] = 0.0
                
        # if train:
        #     print(self.domain_sel)
        #     print(self.x_buffer[0].shape, self.x_buffer.keys())
        #     assert False
        if self.opt.use_buffer:
            for idx, elem in enumerate(range(self.opt.num_domain)):
                # TODO: union to the batch data
                if elem in self.domain_sel: 
                    continue  # if current batch already contains this domain data
                self.x_seq[elem, :, :] = self.x_buffer[elem].clone()
                self.y_seq[elem] = self.y_buffer[elem].clone()
                self.domain_seq[elem] = self.domain_buffer[elem].clone()
                self.domain_sel_mask[elem, :self.opt.num_buffersamples] = 1.0  # TODO: here we need to mask out the fake samples
            self.x_seq.requires_grad = True

        # here need to be fixed......
        if self.opt.use_buffer:
            self.tmp_total_domain_sel = range(self.opt.num_domain)
        else:
            self.tmp_total_domain_sel = self.domain_sel
        self.tmp_num_domain = len(self.tmp_total_domain_sel)
        self.tmp_total_domain_sel = torch.LongTensor(self.tmp_total_domain_sel).to(self.device)

    def __buffer_input__(self):
        for this_domain in self.domain_sel:
            this_domain_data = self.x_seq[this_domain].detach()
            this_domain_label = self.y_seq[this_domain]
            this_domain_domain = self.domain_seq[this_domain]

            rng = np.random.default_rng(seed=2766249141)  
            random_values = rng.random(self.tmp_batch_size)        # 不影响全局 np.random

            # random_values = np.random.rand(self.tmp_batch_size)
            density = np.ones_like(random_values)
            keys = random_values ** (1 / density)  # Weighted Random Sampling
        
            selected_indices = np.argsort(-keys)[:self.opt.num_buffersamples]
            # selected_indices = []
            # for elem in np.argsort(-keys):
            #     if self.domain_sel_mask[this_domain, elem]:
            #         selected_indices.append(elem)
            #         if len(selected_indices) == self.opt.num_buffersamples:
            #             break
            # assert len(selected_indices) == self.opt.num_buffersamples

            valid = int(self.domain_sel_mask[this_domain, :].sum().item())

            if not self.opt.use_pretrain_model_warmup or valid >= self.opt.num_buffersamples:
                self.x_buffer[this_domain] = repeat_data(this_domain_data[selected_indices], self.tmp_batch_size)
                self.y_buffer[this_domain] = repeat_data(this_domain_label[selected_indices].unsqueeze(-1), self.tmp_batch_size).squeeze(-1)
                self.domain_buffer[this_domain] = repeat_data(this_domain_domain[selected_indices].unsqueeze(-1), self.tmp_batch_size).squeeze(-1)
                self.buffer_sel[this_domain, :] = -1
            else:
                # todo 
                # valid = self.domain_sel_mask[this_domain, :].sum().item()
                
                valid_idx = selected_indices[:valid]

                # print(this_domain_label[valid_idx].shape, self.y_buffer[this_domain].shape)
                # assert False
                self.x_buffer[this_domain] = torch.cat([this_domain_data[valid_idx], self.x_buffer[this_domain][:-valid, :]], dim=0)
                self.y_buffer[this_domain] = torch.cat([this_domain_label[valid_idx], self.y_buffer[this_domain][:-valid]], dim=0)
                self.domain_buffer[this_domain] = torch.cat([this_domain_domain[valid_idx], self.domain_buffer[this_domain][:-valid]], dim=0)
                # print(self.x_buffer[this_domain].shape, self.y_buffer[this_domain].shape, self.domain_buffer[this_domain].shape)
                # exit(0)
    
    def __reweight_domain_index__(self):
        domain_idx = self.beta_seq.detach()
        val = self.domain_sel_mask.sum(dim=1)
        self.reweights = torch.zeros((domain_idx.shape[0], 1)).to(self.x_seq.device, self.x_seq.dtype)
        self.reweights.requires_grad = False
        for idx in range(domain_idx.shape[0]):
            dists = torch.cdist(domain_idx[idx].unsqueeze(0), domain_idx, p=2)
            # print(dists.shape, dists)
            w = torch.exp(- dists** 2 / (2. * self.opt.kernel_sigmaSQ)) / np.sqrt(2 * torch.pi * self.opt.kernel_sigmaSQ)
            # w = torch.exp(- dists ) 
            # print(w.shape, w)
            # assert False
            self.reweights[idx] = 1. /  torch.sqrt((w * val).sum() / w.sum())
        # print(reweights, 1./reweights)
        # assert False
        # print(freq_weight, self.domain_sel)
        # for idx, elem in enumerate(range(self.opt.num_domain)):
        #     if elem in self.domain_sel:
        #         continue
        #     reweights[elem] *= (self.opt.num_buffersamples / self.opt.batch_size)
        # print(freq_weight)
        # assert False

    def __train_forward__(self):
        self.u_seq, self.u_mu_seq, self.u_log_var_seq = self.netU(self.x_seq)

        self.u_con_seq = self.netUCon(self.u_seq)

        # TODO: when datas selector enter, need to re-assign the value for u-graph
        if self.use_beta_seq != None:
            self.beta_seq, self.beta_log_var_seq = self.netBeta(
                self.use_beta_seq, self.use_beta_seq)
        else:
            self.tmp_beta_seq = self.generate_beta(self.u_seq)
            self.beta_seq, self.beta_log_var_seq = self.netBeta(
                self.tmp_beta_seq, self.tmp_beta_seq)
        if self.train:
            if self.opt.imbal:
                self.__reweight_domain_index__()
            else:
                self.reweights = torch.ones((self.num_domain, 1)).to(self.device)
                self.reweights.requires_grad = False
        
        self.beta_U_seq = self.netBeta2U(self.beta_seq)

        self.q_z_seq, self.q_z_mu_seq, self.q_z_log_var_seq, self.p_z_seq, self.p_z_mu_seq, self.p_z_log_var_seq, = self.netZ(
            self.x_seq, self.u_seq, self.beta_seq)

        self.r_x_seq = self.netR(self.u_seq)
        self.f_seq = self.netF(self.q_z_seq)

        self.d_seq = self.netD(self.q_z_seq)
        self.loss_D = self.__loss_D__(self.d_seq)

    def __test_forward__(self):
        self.u_seq, self.u_mu_seq, self.u_log_var_seq = self.netU(self.x_seq)

        # if self.use_beta_seq != None:
        #     self.beta_seq, _ = self.netBeta(self.use_beta_seq,
        #                                     self.use_beta_seq)
        # else:
        #     self.tmp_beta_seq = self.generate_beta(self.u_seq)
        #     self.beta_seq, _ = self.netBeta(self.tmp_beta_seq,
        #                                     self.tmp_beta_seq)
        self.tmp_beta_seq = self.generate_beta(self.u_seq)
        self.beta_seq, _ = self.netBeta(self.tmp_beta_seq,
                                        self.tmp_beta_seq)

        self.q_z_seq, self.q_z_mu_seq, self.q_z_log_var_seq, self.p_z_seq, self.p_z_mu_seq, self.p_z_log_var_seq, = self.netZ(
            self.x_seq, self.u_seq, self.beta_seq)
        self.f_seq = self.netF(self.q_z_seq)
        self.g_seq = torch.argmax(self.f_seq.detach(), dim=2)

        # test use only
        # self.r_x_seq = self.netR(self.u_seq)

    def __optimize__(self):
        loss_value = dict()
        loss_value['D'], loss_value['E_pred'], loss_value['Q_u_x'], loss_value['Q_z_x_u'], loss_value['P_z_x_u'], \
            loss_value['U_concentrate'], loss_value['R'], loss_value['U_beta_R'], loss_value['P_beta_alpha'] \
                = self.__optimize_DUZF__()

        return loss_value

    def contrastive_loss(self, u_con_seq, temperature=1):
        # print(f"this is contrastive loss input u_con_seq {u_con_seq.shape}")
        # exit(0)
        u_con_seq = u_con_seq.reshape(self.tmp_batch_size * self.num_domain,
                                      -1)
        u_con_seq = nn.functional.normalize(u_con_seq, p=2, dim=1)

        # calculate the cosine similarity between each pair
        logits = torch.matmul(u_con_seq, torch.t(u_con_seq)) / temperature

        # we only choose the one that is:
        # 1, belongs to one domain
        # 2, next to each other
        # as the pair that we want to concentrate them, and all the others will be cancel out

        # the first 2 steps will generate matrix in this format:
        # [0, 1, 0, 0]
        # [0, 0, 1, 0]
        # [0, 0, 0, 1]
        # [1, 0, 0, 0]
        base_m = torch.diag(torch.ones(self.tmp_batch_size - 1),
                            diagonal=1).to(self.device)
        base_m[self.tmp_batch_size - 1, 0] = 1

        # Then we generate the "complementary" matrix in this format:
        # [1, 0, 1, 1]
        # [1, 1, 0, 1]
        # [1, 1, 1, 0]
        # [0, 1, 1, 1]
        # which will be used in the mask
        base_m = torch.ones(self.tmp_batch_size, self.tmp_batch_size).to(
            self.device) - base_m
        # generate the true mask with the base matrix as block.
        # [1, 0, 1, 1, 0, 0, 0, 0 ...]
        # [1, 1, 0, 1, 0, 0, 0, 0 ...]
        # [1, 1, 1, 0, 0, 0, 0, 0 ...]
        # [0, 1, 1, 1, 0, 0, 0, 0 ...]
        # [0, 0, 0, 0, 1, 0, 1, 1 ...]
        # [0, 0, 0, 0, 1, 1, 0, 1 ...]
        # [0, 0, 0, 0, 1, 1, 1, 0 ...]
        # [0, 0, 0, 0, 0, 1, 1, 1 ...]
        # 这里也有问题，需要改
        masks = torch.block_diag(*([base_m] * self.num_domain))
        logits = logits - masks * LARGE_NUM

        # label: which similarity should maximize. We only maximize the similarity of datapoints that:
        # belongs to one domain
        # next to each other
        label = torch.arange(self.tmp_batch_size * self.num_domain).to(
            self.device)
        label = torch.remainder(label + 1, self.tmp_batch_size) + label.div(
            self.tmp_batch_size, rounding_mode='floor') * self.tmp_batch_size
        label[~flat(self.domain_sel_mask).bool()] = -1

        # loss_u_concentrate = F.cross_entropy(logits, label, reduction='none')
        # loss_u_concentrate *= flat(self.domain_sel_mask)
        # print(logits.shape, label.shape, flat(self.domain_sel_mask).shape)
        # assert False
        loss = F.cross_entropy(logits, label, reduction='none', ignore_index=-1)

        mask = (label != -1)
        # print(loss.shape, self.reweights.shape, self.reweights.repeat(1, self.opt.batch_size).shape)
        loss = (loss * mask.float() * flat(self.reweights.repeat(1, self.opt.batch_size))).sum() 
        loss /= mask.float().sum()
        return loss

    def __optimize_DUZF__(self):
        self.train()

        self.optimizer_UZF.zero_grad()

        # - E_q[log q(u|x)]
        # u is multi-dimensional
        # loss_q_u_x = torch.mean((0.5 * flat(self.u_log_var_seq)).sum(1), dim=0)
        loss_q_u_x = (0.5 * flat(self.u_log_var_seq)).sum(1) * flat(self.domain_sel_mask)
        loss_q_u_x = loss_q_u_x.reshape(self.num_domain, -1) * self.reweights
        loss_q_u_x = torch.mean(loss_q_u_x)

        # - E_q[log q(z|x,u)]
        # remove all the losses about log var and use 1 as var
        # loss_q_z_x_u = torch.mean((0.5 * flat(self.q_z_log_var_seq)).sum(1),dim=0)
        loss_q_z_x_u = (0.5 * flat(self.q_z_log_var_seq)).sum(1) * flat(self.domain_sel_mask)
        loss_q_z_x_u = loss_q_z_x_u.reshape(self.num_domain, -1) * self.reweights
        loss_q_z_x_u = torch.mean(loss_q_z_x_u)

        # E_q[log p(z|x,u)]
        # first one is for normal
        loss_p_z_x_u = -0.5 * flat(self.p_z_log_var_seq) - 0.5 * (
            torch.exp(flat(self.q_z_log_var_seq)) +
            (flat(self.q_z_mu_seq) - flat(self.p_z_mu_seq))**2) / flat(
                torch.exp(self.p_z_log_var_seq))
        loss_p_z_x_u = loss_p_z_x_u.sum(1) * flat(self.domain_sel_mask)
        loss_p_z_x_u = loss_p_z_x_u.reshape(self.num_domain, -1) * self.reweights
        loss_p_z_x_u = torch.mean(loss_p_z_x_u)


        # domain_valid = self.domain_sel_mask.sum(dim=1).bool()
        # mask_comb = (self.domain_mask == 1) & domain_valid
        # # E_q[log p(y|z)]
        # y_seq_source = self.y_seq[mask_comb]
        # f_seq_source = self.f_seq[mask_comb]

        source_valid = torch.zeros_like(self.y_seq) # 30,16
        source_valid[self.domain_mask==1, :] = 1.0
        mask_comb = source_valid.bool() & self.domain_sel_mask.bool()
        # mask_comb = source_valid.bool()
        y_seq_source = flat(self.y_seq)
        f_seq_source = flat(self.f_seq)
        y_seq_source[~flat(mask_comb)] = -1

        loss_p_y_z = -F.nll_loss(f_seq_source, y_seq_source, reduction='none', ignore_index=-1) 
        valid_mask = (y_seq_source != -1)
        loss_p_y_z = (loss_p_y_z * valid_mask.float() * flat(self.reweights.repeat(1, self.opt.batch_size))).sum() 
        loss_p_y_z /= (valid_mask.float()).sum()

        # y_seq_source = flat(self.y_seq)[flat(mask_comb)]
        # f_seq_source = flat(self.f_seq)[flat(mask_comb)]
        # loss_p_y_z = -F.nll_loss(f_seq_source, y_seq_source)
        # todo
        # y_seq_source = self.y_seq[self.domain_mask == 1]
        # f_seq_source = self.f_seq[self.domain_mask == 1]
        # loss_p_y_z = -F.nll_loss(
        #     flat(f_seq_source).squeeze(), flat(y_seq_source))
        
        # E_q[log p(\beta|\alpha)]
        # assuming alpha mean = 0
        # print(var_beta_mask.sum(), var_beta_mask.shape[0])
        # if var_beta_mask.sum() != self.opt.k: 
        #     print(var_beta_mask)
        #     exit(0)
        var_beta = torch.exp(self.beta_log_var_seq) 
        # To reproduce the exact result of our experiment, use the following line to replace the loss_beta_alpha:
        loss_beta_alpha = (var_beta**2).sum(dim=1)  # [30]
        loss_beta_alpha = loss_beta_alpha.unsqueeze(1) * self.reweights
        loss_beta_alpha = - torch.mean(loss_beta_alpha)
        # Actually the previous line is wrong because based on our formula, it should be var_beta, not var_beta**2.
        # However, all our parameter tunning is based on the previous one. Thus, to ensure that you can reproduce our results, please use the previous line.
        # The correct line should be: 
        # loss_beta_alpha = -torch.mean(var_beta.sum(dim=1))
        # Uncomment the previous line to use the correct formula.

        # loss for u and beta
        # log p(u|beta)
        
        beta_t = self.beta_U_seq.unsqueeze(dim=1).expand(
            -1, self.tmp_batch_size, -1) 
        # now beta_t is domain x batch x domain idx dim
        # print(beta_t.shape) 
        loss_p_u_beta = ((self.u_seq - beta_t)**2).sum(2) * self.domain_sel_mask  # [30, 16]
        loss_p_u_beta *= self.reweights
        loss_p_u_beta = -torch.mean(loss_p_u_beta) 

        # concentrate loss
        loss_u_concentrate = self.contrastive_loss(self.u_con_seq)

        # reconstruction loss (p(x|u))
        loss_p_x_u = ((flat(self.x_seq) - flat(self.r_x_seq))**2).sum(1) * flat(self.domain_sel_mask)
        loss_p_x_u = loss_p_x_u.reshape(self.num_domain, -1) * self.reweights
        loss_p_x_u = -torch.mean(loss_p_x_u) 

        # gan loss (adversarial loss)
        if self.opt.lambda_gan != 0:
            if self.opt.d_loss_type == "ADDA_loss":
                d_seq_target = self.d_seq[self.domain_mask == 0]
                adda_reweight = self.reweights[self.domain_mask == 0].unsqueeze(-1)
                # print(d_seq_target.shape, adda_reweight.shape)
                # assert False
                loss_E_gan = (-torch.log(d_seq_target + 1e-10) * adda_reweight).mean()
            else:
                loss_E_gan = -self.loss_D
        else:
            loss_E_gan = torch.tensor(0,
                                      dtype=torch.double,
                                      device=self.opt.device)
            
        loss_E = loss_E_gan * self.opt.lambda_gan + self.opt.lambda_u_concentrate * loss_u_concentrate - (
            self.opt.lambda_reconstruct * loss_p_x_u + self.opt.lambda_beta *
            loss_p_u_beta + self.opt.lambda_beta_alpha * loss_beta_alpha +
            loss_p_y_z + loss_q_u_x + loss_q_z_x_u + loss_p_z_x_u)

        self.optimizer_D.zero_grad()
        self.loss_D.backward(retain_graph=True)
        self.optimizer_UZF.zero_grad()
        loss_E.backward()

        self.optimizer_D.step()
        self.optimizer_UZF.step()

        return self.loss_D.item(), -loss_p_y_z.item(), loss_q_u_x.item(
        ), loss_q_z_x_u.item(), loss_p_z_x_u.item(), loss_u_concentrate.item(
        ), -loss_p_x_u.item(), -loss_p_u_beta.item(), -loss_beta_alpha.item()

    def __log_write__(self, loss_msg):
        print(loss_msg)
        with open(self.train_log, 'a') as f:
            f.write(loss_msg + "\n")

    def __vis_loss__(self, loss_values):
        if self.epoch == 0:
            self.panes = {
                loss_name: self.env.line(
                    X=np.array([self.epoch]),
                    Y=np.array([loss_values[loss_name]]),
                    opts=dict(title='loss for {} on epochs'.format(loss_name)))
                for loss_name in self.loss_names
            }
        else:
            for loss_name in self.loss_names:
                self.env.line(X=np.array([self.epoch]),
                              Y=np.array([loss_values[loss_name]]),
                              win=self.panes[loss_name],
                              update='append')

    def __init_weight__(self, net=None):
        if net is None:
            net = self
        for m in net.modules():
            if isinstance(m, nn.Linear):
                # print("init linear weight!")
                nn.init.normal_(m.weight, mean=0, std=0.01)
                #                 nn.init.normal_(m.weight, mean=0, std=0.1)
                #                 nn.init.xavier_normal_(m.weight, gain=10)
                nn.init.constant_(m.bias, val=0)


class VDI(BaseModel):
    #########
    # VDI (Variational Domain Index) Model
    #########

    def __init__(self, opt, search_space=None):
        super(VDI, self).__init__(opt)

        self.bayesian_opt = False
        if search_space != None:
            self.bayesian_opt = True

        self.netU = UNet(opt).to(opt.device)
        self.netUCon = UConcenNet(opt).to(opt.device)
        self.netZ = Q_ZNet_beta(opt).to(opt.device)
        self.netF = PredNet(opt).to(opt.device)
        self.netR = ReconstructNet(opt).to(opt.device)

        self.netBeta = BetaNet(opt).to(opt.device).float()
        self.netBeta2U = Beta2UNet(opt).to(opt.device).float()

        # for DANN-style discriminator loss & MDS aggregation
        if self.opt.d_loss_type == "DANN_loss":
            self.netD = ClassDiscNet(opt).to(opt.device)
            self.__loss_D__ = self.__loss_D_dann__
            self.generate_beta = self.__reconstruct_u_graph__
        # for DANN-style discriminator loss & mean aggregation
        elif self.opt.d_loss_type == "DANN_loss_mean":
            assert self.opt.u_dim == self.opt.beta_dim, "When you use \"mean\" as aggregation, you should make sure local domain index and global domain index have the same dimension."
            self.netD = ClassDiscNet(opt).to(opt.device)
            self.__loss_D__ = self.__loss_D_dann__
            self.generate_beta = self.__u_mean__
            self.netBeta2U = nn.Identity().to(opt.device)
        # for ADDA-style discriminator loss & MDS aggregation
        elif self.opt.d_loss_type == "ADDA_loss":
            self.netD = DiscNet(opt).to(opt.device)
            self.__loss_D__ = self.__loss_D_adda__
            # self.generate_beta = self.__u_mean__
            self.generate_beta = self.__reconstruct_u_graph__
        # for CIDA-style discriminator loss & MDS aggregation
        elif self.opt.d_loss_type == "CIDA_loss":
            self.netD = DiscNet(opt).to(opt.device)
            self.__loss_D__ = self.__loss_D_cida__
            self.generate_beta = self.__u_mean__
        # for GRDA-style discriminator & MDS aggregation
        elif self.opt.d_loss_type == "GRDA_loss":
            self.netD = GraphDNet(opt).to(opt.device)
            self.__loss_D__ = self.__loss_D_grda__
            self.generate_beta = self.__reconstruct_u_graph__
        self.__init_weight__()

        if self.opt.use_pretrain_R:
            pretrain_model_U = torch.load(self.opt.pretrain_U_path)
            pretrain_model_R = torch.load(self.opt.pretrain_R_path)

            self.netU.load_state_dict(pretrain_model_U)
            self.netR.load_state_dict(pretrain_model_R)

        if self.opt.use_pretrain_model_all:
            print(f"use_pretrained_all")
            self.netU.load_state_dict(torch.load(self.opt.pretrain_model_all_path + '/netU.pth'))
            self.netUCon.load_state_dict(torch.load(self.opt.pretrain_model_all_path + '/netUCon.pth'))
            self.netZ.load_state_dict(torch.load(self.opt.pretrain_model_all_path + '/netZ.pth'))
            self.netF.load_state_dict(torch.load(self.opt.pretrain_model_all_path + '/netF.pth'))
            self.netR.load_state_dict(torch.load(self.opt.pretrain_model_all_path + '/netR.pth'))
            self.netD.load_state_dict(torch.load(self.opt.pretrain_model_all_path + '/netD.pth'))
            self.netBeta.load_state_dict(torch.load(self.opt.pretrain_model_all_path + '/netBeta.pth'))
            self.netBeta2U.load_state_dict(torch.load(self.opt.pretrain_model_all_path + '/netBeta2U.pth'))
            # self.x_buffer, self.y_buffer, self.domain_buffer = torch.load(self.opt.pretrain_model_all_path + '/buffer.pth')
            
        elif self.opt.use_pretrain_model_warmup:
            print(f"use_pretrained_warmup")
            self.netU.load_state_dict(torch.load(self.opt.pretrain_model_warmup_path + '/netU.pth'))
            self.netUCon.load_state_dict(torch.load(self.opt.pretrain_model_warmup_path + '/netUCon.pth'))
            self.netZ.load_state_dict(torch.load(self.opt.pretrain_model_warmup_path + '/netZ.pth'))
            self.netF.load_state_dict(torch.load(self.opt.pretrain_model_warmup_path + '/netF.pth'))
            self.netR.load_state_dict(torch.load(self.opt.pretrain_model_warmup_path + '/netR.pth'))
            self.netD.load_state_dict(torch.load(self.opt.pretrain_model_warmup_path + '/netD.pth'))
            self.netBeta.load_state_dict(torch.load(self.opt.pretrain_model_warmup_path + '/netBeta.pth'))
            self.netBeta2U.load_state_dict(torch.load(self.opt.pretrain_model_warmup_path + '/netBeta2U.pth'))
            self.x_buffer, self.y_buffer, self.domain_buffer = torch.load(self.opt.pretrain_model_warmup_path + '/buffer.pth')

        if self.opt.fix_u_r:
            UZF_parameters = list(self.netZ.parameters()) + list(
                self.netF.parameters())
        else:
            UZF_parameters = list(self.netU.parameters()) + list(
                self.netZ.parameters()) + list(self.netF.parameters()) + list(
                    self.netR.parameters()) + list(self.netUCon.parameters())

        UZF_parameters += list(self.netBeta.parameters()) + list(
            self.netBeta2U.parameters())

        self.optimizer_UZF = optim.Adam(UZF_parameters,
                                        lr=opt.init_lr,
                                        betas=(opt.beta1, 0.999))
        self.optimizer_D = optim.Adam(self.netD.parameters(),
                                      lr=opt.init_lr,
                                      betas=(opt.beta1, 0.999))
        self.lr_scheduler_UZF = TransformerLRScheduler(
            optimizer=self.optimizer_UZF,
            init_lr=opt.init_lr,
            peak_lr=opt.peak_lr_e,
            warmup_steps=opt.warmup_steps,
            decay_steps=opt.total_epoch - opt.warmup_steps,
            gamma=0.5**(1 / 100),
            final_lr=opt.final_lr)
        self.lr_scheduler_D = TransformerLRScheduler(
            optimizer=self.optimizer_D,
            init_lr=opt.init_lr,
            peak_lr=opt.peak_lr_d,
            warmup_steps=opt.warmup_steps,
            decay_steps=opt.total_epoch - opt.warmup_steps,
            gamma=0.5**(1 / 100),
            final_lr=opt.final_lr)

        self.lr_schedulers = [self.lr_scheduler_UZF, self.lr_scheduler_D]
        self.loss_names = [
            'D', 'E_pred', 'Q_u_x', 'Q_z_x_u', 'P_z_x_u', 'U_beta_R',
            'U_concentrate', 'R', 'P_beta_alpha'
        ]

        # for mds(u)
        self.embedding = MDS(n_components=self.opt.beta_dim,
                             dissimilarity='precomputed')
        
    def __reset_schedulers__(self):
        self.lr_scheduler_UZF.set_step(self.opt.total_epoch - 100)
        self.lr_scheduler_D.set_step(self.opt.total_epoch - 100)

        self.lr_schedulers = [self.lr_scheduler_UZF, self.lr_scheduler_D]

    def __u_mean__(self, u_seq):
        mu_beta = u_seq.mean(1).detach()
        mu_beta_mean = mu_beta.mean(0, keepdim=True)
        mu_beta_std = mu_beta.std(0, keepdim=True)
        mu_beta_std = torch.maximum(mu_beta_std,
                                    torch.ones_like(mu_beta_std) * 1e-12)
        mu_beta = (mu_beta - mu_beta_mean) / mu_beta_std
        return mu_beta

    def __reconstruct_u_graph__(self, u_seq, is_global=False):
        with torch.no_grad():
            A = torch.zeros(self.num_domain, self.num_domain)
            if isinstance(u_seq, list):
                new_u = [elem.detach() for elem in u_seq]
            else:
                new_u = u_seq.detach()  #５, 16, 8
            # ~ Wasserstein Loss
            loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
            for i in range(self.num_domain):
                for j in range(i + 1, self.num_domain):
                    if i not in self.tmp_total_domain_sel or j not in self.tmp_total_domain_sel:
                        A[i][j] = LARGE_NUM
                        A[j][i] = LARGE_NUM
                    else:
                        try:
                            if is_global:
                                A[i][j] = loss(new_u[i], new_u[j])
                                A[j][i] = A[i][j]
                            else:
                                i_mask = self.domain_sel_mask[i]
                                j_mask = self.domain_sel_mask[j]
                                A[i][j] = loss(new_u[i][i_mask==1], new_u[j][j_mask==1])
                                A[j][i] = A[i][j]
                        except:
                            assert False


            A_np = to_np(A)
            bound = np.sort(A.flatten())[int(self.tmp_total_domain_sel.shape[0]**2 * 1 / 4)]
            # generate self.A
            self.A = (A_np < bound)

            # calculate the beta seq
            mu_beta = self.embedding.fit_transform(A_np)
            mu_beta = torch.from_numpy(mu_beta).to(self.device)
            # new normalization:
            mu_beta_mean = mu_beta.mean(0, keepdim=True)
            mu_beta_std = mu_beta.std(0, keepdim=True)
            mu_beta_std = torch.maximum(mu_beta_std,
                                        torch.ones_like(mu_beta_std) * 1e-12)
            mu_beta = (mu_beta - mu_beta_mean) / mu_beta_std
            # if self.epoch:
            #     print(self.tmp_total_domain_sel)
            #     print(A_np)
            #     print(mu_beta)
            #     assert False
            return mu_beta

    def __loss_D_dann__(self, d_seq):
        # this is for DANN
        return F.nll_loss(flat(d_seq),
                          flat(self.domain_seq))  # , self.u_seq.mean(1)

    def __loss_D_adda__(self, d_seq):
        # print(d_seq[self.domain_mask == 1].shape, self.reweights[self.domain_mask == 1].shape)
        # assert False
        d_seq_source = d_seq[self.domain_mask == 1] * self.reweights[self.domain_mask == 1].unsqueeze(-1)
        d_seq_target = d_seq[self.domain_mask == 0] * self.reweights[self.domain_mask == 0].unsqueeze(-1)
        # D: discriminator loss from classifying source v.s. target
        loss_D = (-torch.log(d_seq_source + 1e-10).mean() -
                  torch.log(1 - d_seq_target + 1e-10).mean())
        return loss_D

    def __loss_D_cida__(self, d_seq):
        # this is for CIDA
        # use L1 instead of L2
        return F.l1_loss(flat(d_seq),
                         flat(self.u_seq.detach()))  # , self.u_seq.mean(1)

    def __loss_D_grda__(self, d_seq):
        # this is for GRDA
        A = self.A

        criterion = nn.BCEWithLogitsLoss()
        d = d_seq
        # random pick subchain and optimize the D
        # balance coefficient is calculate by pos/neg ratio
        # A is the adjancency matrix
        sub_graph = self.__sub_graph__(my_sample_v=self.opt.sample_v, A=A)

        errorD_connected = torch.zeros((1, )).to(self.device)  # .double()
        errorD_disconnected = torch.zeros((1, )).to(self.device)  # .double()

        count_connected = 0
        count_disconnected = 0

        for i in range(self.opt.sample_v):
            v_i = sub_graph[i]
            # no self loop version!!
            for j in range(i + 1, self.opt.sample_v):
                v_j = sub_graph[j]
                label = torch.full(
                    (self.tmp_batch_size, ),
                    A[v_i][v_j],
                    device=self.device,
                )
                # dot product
                if v_i == v_j:
                    idx = torch.randperm(self.tmp_batch_size)
                    output = (d[v_i][idx] * d[v_j]).sum(1)
                else:
                    output = (d[v_i] * d[v_j]).sum(1)

                if A[v_i][v_j]:  # connected
                    errorD_connected += criterion(output, label)
                    count_connected += 1
                else:
                    errorD_disconnected += criterion(output, label)
                    count_disconnected += 1

        # prevent nan
        if count_connected == 0:
            count_connected = 1
        if count_disconnected == 0:
            count_disconnected = 1

        errorD = 0.5 * (errorD_connected / count_connected +
                        errorD_disconnected / count_disconnected)
        # this is a loss balance
        return errorD * self.num_domain

    def __sub_graph__(self, my_sample_v, A):
        # sub graph tool for grda loss
        if np.random.randint(0, 2) == 0:
            return np.random.choice(self.num_domain,
                                    size=my_sample_v,
                                    replace=False)

        # subsample a chain (or multiple chains in graph)
        left_nodes = my_sample_v
        choosen_node = []
        vis = np.zeros(self.num_domain)
        while left_nodes > 0:
            chain_node, node_num = self.__rand_walk__(vis, left_nodes, A)
            choosen_node.extend(chain_node)
            left_nodes -= node_num

        return choosen_node

    def __rand_walk__(self, vis, left_nodes, A):
        # graph random sampling tool for grda loss
        chain_node = []
        node_num = 0
        # choose node
        node_index = np.where(vis == 0)[0]
        st = np.random.choice(node_index)
        vis[st] = 1
        chain_node.append(st)
        left_nodes -= 1
        node_num += 1

        cur_node = st
        while left_nodes > 0:
            nx_node = -1
            node_to_choose = np.where(vis == 0)[0]
            num = node_to_choose.shape[0]
            node_to_choose = np.random.choice(node_to_choose,
                                              num,
                                              replace=False)

            for i in node_to_choose:
                if cur_node != i:
                    # have an edge and doesn't visit
                    if A[cur_node][i] and not vis[i]:
                        nx_node = i
                        vis[nx_node] = 1
                        chain_node.append(nx_node)
                        left_nodes -= 1
                        node_num += 1
                        break
            if nx_node >= 0:
                cur_node = nx_node
            else:
                break
        return chain_node, node_num
