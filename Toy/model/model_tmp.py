import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from model.modules import *
import os
from visdom import Visdom
import pickle
from model.lr_scheduler import TransformerLRScheduler
import ot
from sklearn.manifold import MDS
import json
import random
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

        self.x_filtered = {}
        self.y_filtered = {}
        self.domain_filtered = {}
        self.filter_sel = set()
        self.use_selector = self.opt.use_selector

    def reset_buffer(self):
        del self.x_filtered, self.y_filtered, self.domain_filtered, self.filter_sel
        self.x_filtered = {}
        self.y_filtered = {}
        self.domain_filtered = {}
        self.filter_sel = set()
        self.use_selector = self.opt.use_selector
        return

    def learn(self, epoch, dataloader, verbose=False):
        self.train()

        self.epoch = epoch
        loss_values = {loss: 0 for loss in self.loss_names}
        self.new_u = []

        # self.reset_buffer()
        flag = 0
        if self.opt.use_selector:
            while len(self.filter_sel) != self.num_domain:
                for data in dataloader:
                    self.__set_input__(data)
                    if self.opt.use_selector:
                        self.__filter_input__()
                flag += 1

        count = 0
        for data in dataloader:
            count += 1
            self.__set_input__(data)
            self.__train_forward__()
            new_loss_values = self.__optimize__()
        
            # for the loss visualization
            for key, loss in new_loss_values.items():
                loss_values[key] += loss

            # self.new_u.append(self.u_seq)

            if self.opt.use_selector:
                self.__filter_input__()

        # self.new_u = self.my_cat(self.new_u)
        # self.use_beta_seq = self.generate_beta(self.new_u)
        self.use_beta_seq = None

        for key, _ in new_loss_values.items():
            loss_values[key] /= count

        if self.use_visdom:
            self.__vis_loss__(loss_values)

        if (self.epoch + 1) % 10 == 0 or self.epoch == 0:
            print("epoch {}, loss: {}, lambda gan: {}".format(
                self.epoch, loss_values, self.opt.lambda_gan))

        # learning rate decay
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()

        # check nan
        if any(np.isnan(val) for val in loss_values.values()):
            self.nan_flag = True
        else:
            self.nan_flag = False

    def test(self, epoch, dataloader, verbose=False):
        self.eval()
        self.epoch = epoch

        acc_curve = []
        l_x = []
        # l_r_x = []
        l_domain = []
        l_label = []
        l_encode = []
        l_y = []
        l_u = np.zeros((self.num_domain, self.opt.u_dim))
        l_u_all = []

        for data in dataloader:
            self.__set_input__(data)

            # forward
            with torch.no_grad():
                self.__test_forward__()
            
            acc_curve.append(self.g_seq.eq(self.y_seq).to(torch.float).sum(-1,keepdim=True).detach())

            if self.opt.normalize_domain:
                x_np = to_np(self.x_seq)
                for i in range(len(x_np)):
                    x_np[i] = x_np[i] * self.data_s[i] + self.data_m[i]
                l_x.append(x_np)
            else:
                l_x.append(to_np(self.x_seq))

            l_y.append(to_np(self.y_seq))
            # l_r_x.append(to_np(self.r_x_seq)) # test use only
            l_domain.append(to_np(self.domain_seq))
            l_encode.append(to_np(self.q_z_seq))
            l_label.append(to_np(self.g_seq))
            l_u += to_np(self.u_seq.sum(1))
            l_u_all.append(to_np(self.u_seq))

        x_all = np.concatenate(l_x, axis=1)
        y_all = np.concatenate(l_y, axis=1)
        # r_x_all = np.concatenate(l_r_x, axis=1) # test use only
        e_all = np.concatenate(l_encode, axis=1)
        domain_all = np.concatenate(l_domain, axis=1)
        label_all = np.concatenate(l_label, axis=1)
        u = l_u / x_all.shape[1]
        u_all = np.concatenate(l_u_all, axis=1)
        beta_all = to_np(self.beta_seq)

        d_all = dict()

        d_all['data'] = flat(x_all)
        d_all['gt_label'] = flat(y_all)
        d_all['domain'] = flat(domain_all)
        d_all['label'] = flat(label_all)
        d_all['encodeing'] = flat(e_all)
        d_all['u'] = u
        # d_all['r_x'] = flat(r_x_all) # test use only
        d_all['u_all'] = flat(u_all)
        d_all['beta'] = beta_all

        acc = to_np(torch.cat(acc_curve, 1).sum(-1))
        # if verbose:
        #     print(acc.shape, y_all.shape, acc)
        #     exit(0)
        acc = acc / y_all.shape[
            1]  # this is the total number of the domain. Assume that we could get them
        test_acc = acc[self.opt.tgt_domain_idx].sum() / (
            self.opt.num_target) * 100
        acc_msg = '[Test][{}] Accuracy: total average {:.1f}, test average {:.1f}, in each domain {}'.format(
            epoch,
            acc.mean() * 100, test_acc, np.around(acc * 100, decimals=1))
        self.__log_write__(acc_msg)
        if self.use_visdom:
            self.__vis_test_error__(test_acc, 'test acc')

        d_all['acc_msg'] = acc_msg

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

    def save(self):
        torch.save(self.netU.state_dict(), self.outf + '/netU.pth')
        torch.save(self.netUCon.state_dict(), self.outf + '/netUCon.pth')
        torch.save(self.netZ.state_dict(), self.outf + '/netZ.pth')
        torch.save(self.netF.state_dict(), self.outf + '/netF.pth')
        torch.save(self.netR.state_dict(), self.outf + '/netR.pth')
        torch.save(self.netD.state_dict(), self.outf + '/netD.pth')
        torch.save(self.netBeta.state_dict(), self.outf + '/netBeta.pth')
        torch.save(self.netBeta2U.state_dict(), self.outf + '/netBeta2U.pth')

    def __set_input__(self, data, train=True):
        # :param
        #   x_seq: Number of domain x Batch size x  Data dim
        #   y_seq: Number of domain x Batch size x Predict Data dim
        #   (testing: Number of domain x Batch size x test len x Predict Data dim)
        #   one_hot_seq: Number of domain x Batch size x Number of vertices (domains)
        #   domain_seq: Number of domain x Batch size x domain dim (1)
        #   idx_seq: Number of domain x Batch size x 1 (the order in the whole dataset)
        #   y_value_seq: Number of domain x Batch size x Predict Data dim
        x_seq, y_seq, domain_seq = [d[0][None, :, :] for d in data
                                        ], [d[1][None, :] for d in data
                                            ], [d[2][None, :] for d in data]

        self.x_seq_tmp = torch.cat(x_seq, 0).to(self.device)
        self.y_seq_tmp = torch.cat(y_seq, 0).to(self.device)
        self.domain_seq_tmp = torch.cat(domain_seq, 0).to(self.device)
        self.domain_sel = torch.unique(self.domain_seq_tmp).tolist()

        self.x_seq = torch.zeros((self.num_domain, *self.x_seq_tmp.shape[1:])).to(self.device)
        self.y_seq = torch.zeros((self.num_domain, *self.y_seq_tmp.shape[1:])).to(device=self.device, dtype=torch.long)
        self.domain_seq = torch.zeros((self.num_domain, *self.domain_seq_tmp.shape[1:])).to(device=self.device, dtype=torch.long)
        self.domain_sel_mask = torch.zeros((self.num_domain, self.x_seq_tmp.shape[1])).to(self.device)
        self.domain_sel_mask[self.domain_sel, :] = 1.0
        # self.domain_sel_mask.requires_grad = False
        # print(self.domain_sel_mask.shape)

        for idx, elem in enumerate(self.domain_sel):
            # print(idx, elem)
            # print(self.x_seq[elem].shape, self.x_seq_tmp[idx].shape)
            self.x_seq[elem, :, :] = self.x_seq_tmp[idx]
            self.y_seq[elem] = self.y_seq_tmp[idx]
            self.domain_seq[elem] = self.domain_seq_tmp[idx]

        # print(self.domain_sel, self.filter_sel)
        if len(self.filter_sel):
            # print(self.domain_sel, self.filter_sel)
            for idx, elem in enumerate(self.filter_sel):
                # TODO: union to the batch data
                if elem in self.domain_sel: continue  # if current batch already contains this domain data
                # print(self.y_seq[elem].shape, self.y_filtered[elem].shape)
                # print(self.x_seq[elem, :, :].shape, self.x_filtered[elem].shape)
                self.x_seq[elem, :, :] = self.x_filtered[elem].clone()
                self.y_seq[elem] = self.y_filtered[elem].clone()
                self.domain_seq[elem] = self.domain_filtered[elem].clone()
                self.domain_sel_mask[elem, :self.opt.num_filtersamples] = 1.0  # TODO: here we need to mask out the fake samples

        self.x_seq.requires_grad = True

        # here need to be fixed......
        self.tmp_batch_size = self.x_seq.shape[1]
        self.tmp_total_domain_sel = list(set(list(self.filter_sel) + self.domain_sel))
        self.tmp_num_domain = len(self.tmp_total_domain_sel)
        self.tmp_total_domain_sel = torch.LongTensor(self.tmp_total_domain_sel).to(self.device)

        # if self.epoch >= 1:
        #     print(self.tmp_num_domain)
        #     print(self.domain_sel_mask)
        # print('&&&&&&&&&&&&&&&&&&&&', self.tmp_num_domain, self.filter_sel)
        # print(self.domain_sel)  # current selected domain
        # print(self.x_seq.shape, self.y_seq.shape, self.domain_seq.shape, self.tmp_batch_size, self.domain_sel, self.domain_sel_mask)

    def __filter_input__(self):
        from sklearn.neighbors import NearestNeighbors

        for this_domain in self.domain_sel:
            # if this_domain in self.filter_sel:
            #     continue  # if already select the sample
            nn = NearestNeighbors(n_neighbors=self.opt.n_neighbors)
            this_domain_data = self.x_seq[this_domain].detach()
            this_domain_label = self.y_seq[this_domain]
            this_domain_domain = self.domain_seq[this_domain]
            # print(this_domain, self.domain_sel, self.filter_sel, torch.unique(self.domain_seq), self.domain_seq.shape)
            # print(self.x_seq[this_domain].shape, self.y_seq[this_domain].shape, self.domain_seq[this_domain].shape)
            nn.fit(this_domain_data.cpu().numpy())
            distances, _ = nn.kneighbors(this_domain_data.cpu().numpy())

            avg_distances = distances[:, 1:].mean(axis=1)  # 排除自身距离
            density = 1 / (avg_distances + 1e-8)  # 防止除零

            n = this_domain_data.shape[0]
            random_values = np.random.rand(n)
            keys = random_values ** (1 / density)  # Efraimidis-Spirakis算法
        
            # 选取关键值最大的前num_samples个索引
            selected_indices = np.argsort(-keys)[:self.opt.num_filtersamples]

            self.x_filtered[this_domain] = repeat_data(this_domain_data[selected_indices], self.tmp_batch_size)
            self.y_filtered[this_domain] = repeat_data(this_domain_label[selected_indices].unsqueeze(-1), self.tmp_batch_size).squeeze(-1)
            self.domain_filtered[this_domain] = repeat_data(this_domain_domain[selected_indices].unsqueeze(-1), self.tmp_batch_size).squeeze(-1)
            self.filter_sel.add(this_domain)

        # if len(self.filter_sel) == self.num_domain:
        #     self.use_selector = False  # already touch all the domains

    def __train_forward__(self):
        self.u_seq, self.u_mu_seq, self.u_log_var_seq = self.netU(self.x_seq)

        self.u_con_seq = self.netUCon(self.u_seq)

        # TODO: when datas selector enter, need to re-assign the value for u-graph
        if self.opt.online:
            self.tmp_beta_seq = self.generate_beta(self.u_seq)
            self.beta_seq, self.beta_log_var_seq = self.netBeta(
                self.tmp_beta_seq, self.tmp_beta_seq)
        else:
            if self.use_beta_seq != None:
                self.beta_seq, self.beta_log_var_seq = self.netBeta(
                    self.use_beta_seq, self.use_beta_seq)
            else:
                self.tmp_beta_seq = self.generate_beta(self.u_seq)
                self.beta_seq, self.beta_log_var_seq = self.netBeta(
                    self.tmp_beta_seq, self.tmp_beta_seq)

        self.beta_U_seq = self.netBeta2U(self.beta_seq)

        self.q_z_seq, self.q_z_mu_seq, self.q_z_log_var_seq, self.p_z_seq, self.p_z_mu_seq, self.p_z_log_var_seq, = self.netZ(
            self.x_seq, self.u_seq, self.beta_seq)

        self.r_x_seq = self.netR(self.u_seq)
        self.f_seq = self.netF(self.q_z_seq)

        self.d_seq = self.netD(self.q_z_seq)
        self.loss_D = self.__loss_D__(self.d_seq)

    def __test_forward__(self):
        self.u_seq, self.u_mu_seq, self.u_log_var_seq = self.netU(self.x_seq)

        if self.use_beta_seq != None:
            self.beta_seq, _ = self.netBeta(self.use_beta_seq,
                                            self.use_beta_seq)
        else:
            self.tmp_beta_seq = self.generate_beta(self.u_seq)
            self.beta_seq, _ = self.netBeta(self.tmp_beta_seq,
                                            self.tmp_beta_seq)

        self.q_z_seq, self.q_z_mu_seq, self.q_z_log_var_seq, self.p_z_seq, self.p_z_mu_seq, self.p_z_log_var_seq, = self.netZ(
            self.x_seq, self.u_seq, self.beta_seq)
        self.f_seq = self.netF(self.q_z_seq)
        self.g_seq = torch.argmax(self.f_seq.detach(), dim=2)
        # print(self.f_seq.shape, self.g_seq.shape)
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
        # 这里需要换成选择的label
        label = torch.arange(self.tmp_batch_size * self.num_domain).to(
            self.device)
        label = torch.remainder(label + 1, self.tmp_batch_size) + label.div(
            self.tmp_batch_size, rounding_mode='floor') * self.tmp_batch_size

        loss_u_concentrate = F.cross_entropy(logits, label, reduction='none')
        loss_u_concentrate *= flat(self.domain_sel_mask)
        return loss_u_concentrate.mean()

    def __optimize_DUZF__(self):
        self.train()

        self.optimizer_UZF.zero_grad()

        gradient_mask = self.domain_sel_mask

        # - E_q[log q(u|x)]
        # u is multi-dimensional
        # loss_q_u_x = torch.mean((0.5 * flat(self.u_log_var_seq)).sum(1), dim=0)
        loss_q_u_x = ((0.5 * flat(self.u_log_var_seq)).sum(1) * flat(self.domain_sel_mask)).mean()

        # - E_q[log q(z|x,u)]
        # remove all the losses about log var and use 1 as var
        # loss_q_z_x_u = torch.mean((0.5 * flat(self.q_z_log_var_seq)).sum(1),dim=0)
        loss_q_z_x_u = ((0.5 * flat(self.q_z_log_var_seq)).sum(1) * flat(self.domain_sel_mask)).mean()

        # E_q[log p(z|x,u)]
        # first one is for normal
        loss_p_z_x_u = -0.5 * flat(self.p_z_log_var_seq) - 0.5 * (
            torch.exp(flat(self.q_z_log_var_seq)) +
            (flat(self.q_z_mu_seq) - flat(self.p_z_mu_seq))**2) / flat(
                torch.exp(self.p_z_log_var_seq))
        loss_p_z_x_u = (loss_p_z_x_u.sum(1) * flat(self.domain_sel_mask)).mean()

        # print(f"the shape for y_seq:{self.y_seq.shape}, domain mask {self.domain_mask.shape}")
        # print((self.domain_mask == 1).to(self.y_seq.device))
        # print(torch.isin(torch.arange(self.domain_mask.shape[0]).to(self.y_seq.device), self.domain_sel))
        mask_comb = (self.domain_mask == 1).unsqueeze(-1) & (self.domain_sel_mask == 1)
        # print(f"{self.y_seq[mask1 & mask2].shape}")
        # exit(0)
        # E_q[log p(y|z)]
        # y_seq_source = self.y_seq[self.domain_mask == 1]
        # f_seq_source = self.f_seq[self.domain_mask == 1]
        y_seq_source = self.y_seq * mask_comb
        f_seq_source = self.f_seq * mask_comb.unsqueeze(-1)

        if y_seq_source.shape[0] == 0:
            loss_p_y_z = torch.tensor([0]).to(y_seq_source.device)
        else:
            loss_p_y_z = -F.nll_loss(
                flat(f_seq_source).squeeze(), flat(y_seq_source))
        
        # E_q[log p(\beta|\alpha)]
        # assuming alpha mean = 0
        # print(self.beta_log_var_seq.shape)  # num_domain, 2
        var_beta_mask = (self.domain_sel_mask.sum(dim=1) != 0)
        # if var_beta_mask.sum() != self.opt.k: 
        #     print(var_beta_mask)
        #     exit(0)
        var_beta = torch.exp(self.beta_log_var_seq) 
        # To reproduce the exact result of our experiment, use the following line to replace the loss_beta_alpha:
        loss_beta_alpha = -((var_beta**2).sum(dim=1) * var_beta_mask).mean()
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
        # print(beta_t.shape)  # [15,32,4]
        loss_p_u_beta = ((self.u_seq - beta_t)**2).sum(2) * self.domain_sel_mask
        loss_p_u_beta = -torch.mean(loss_p_u_beta) 

        # concentrate loss
        loss_u_concentrate = self.contrastive_loss(self.u_con_seq)

        # reconstruction loss (p(x|u))
        loss_p_x_u = ((flat(self.x_seq) - flat(self.r_x_seq))**2).sum(1) * flat(self.domain_sel_mask)
        loss_p_x_u = -torch.mean(loss_p_x_u) 

        # gan loss (adversarial loss)
        loss_E_gan = -self.loss_D
            
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
                if m.bias is not None:
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
            self.netU.load_state_dict(torch.load(self.opt.pretrain_model_all_path + '/netU.pth'))
            self.netUCon.load_state_dict(torch.load(self.opt.pretrain_model_all_path + '/netUCon.pth'))
            self.netZ.load_state_dict(torch.load(self.opt.pretrain_model_all_path + '/netZ.pth'))
            self.netF.load_state_dict(torch.load(self.opt.pretrain_model_all_path + '/netF.pth'))
            self.netR.load_state_dict(torch.load(self.opt.pretrain_model_all_path + '/netR.pth'))
            self.netD.load_state_dict(torch.load(self.opt.pretrain_model_all_path + '/netD.pth'))
            self.netBeta.load_state_dict(torch.load(self.opt.pretrain_model_all_path + '/netBeta.pth'))
            self.netBeta2U.load_state_dict(torch.load(self.opt.pretrain_model_all_path + '/netBeta2U.pth'))

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
            decay_steps=opt.num_epoch - opt.warmup_steps,
            gamma=0.5**(1 / 100),
            final_lr=opt.final_lr)
        self.lr_scheduler_D = TransformerLRScheduler(
            optimizer=self.optimizer_D,
            init_lr=opt.init_lr,
            peak_lr=opt.peak_lr_d,
            warmup_steps=opt.warmup_steps,
            decay_steps=opt.num_epoch - opt.warmup_steps,
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

    def __u_mean__(self, u_seq):
        mu_beta = u_seq.mean(1).detach()
        mu_beta_mean = mu_beta.mean(0, keepdim=True)
        mu_beta_std = mu_beta.std(0, keepdim=True)
        mu_beta_std = torch.maximum(mu_beta_std,
                                    torch.ones_like(mu_beta_std) * 1e-12)
        mu_beta = (mu_beta - mu_beta_mean) / mu_beta_std
        mu_beta *= (self.domain_sel_mask.sum(1) != 0).unsqueeze(-1)  # TODO: 如果没有该domain 数据，则此处的值应为0
        # print(mu_beta.shape)  # [num_domain, u_dim]
        # exit(0)
        return mu_beta

    def __reconstruct_u_graph__(self, u_seq):
        A = np.zeros((self.opt.num_domain, self.opt.num_domain))

        new_u = to_np(u_seq)
        A = np.zeros((self.num_domain, self.num_domain))
        tmp_size = new_u.shape[1]
        for i in range(self.num_domain):
            for j in range(i + 1, self.num_domain):
                try:
                    # ZIYANTODO: fix it
                    # if self.epoch >= 1:
                    #     print(i, j)
                    #     print(self.tmp_total_domain_sel)
                    #     print(self.tmp_total_domain_sel == i)
                    #     print(torch.where(self.tmp_total_domain_sel == i)[0])
                    #     print(self.tmp_total_domain_sel == j)
                    #     print(torch.where(self.tmp_total_domain_sel == j)[0])
                    #     print(new_u.shape)
                    index_i = torch.where(self.tmp_total_domain_sel == i)[0]
                    index_j = torch.where(self.tmp_total_domain_sel == j)[0]
                    M = ot.dist(new_u[index_i], new_u[index_j])
                    a = np.ones(tmp_size) / tmp_size
                    b = np.ones(tmp_size) / tmp_size
                    Wd = ot.emd2(a, b, M)
                    A[i][j] = Wd
                    A[j][i] = Wd
                except Exception as e:
                    # TODO: assign a large value on it?
                    if self.epoch >= 1:
                        print(i, j)
                        print(self.tmp_total_domain_sel)
                        print(self.tmp_total_domain_sel == i)
                        print(self.tmp_total_domain_sel == j)
                        exit(0)
                    A[i][j] = LARGE_NUM
                    A[j][i] = LARGE_NUM
                    pass
        factor = int(self.tmp_num_domain ** 2 / 3) 
        bound = np.sort(A.flatten())[factor]
        # print(f"{bound}\n********before************\n{A}\n", file=open(f"./check_{self.opt.online}.txt", 'a'))
        A_dis = A
        # generate self.A
        self.A = (A < bound)
        # print(f"{bound}\n********after************\n{self.A}\n", file=open(f"./check_{self.opt.online}.txt", 'a'))

        # calculate the beta seq
        mu_beta = self.embedding.fit_transform(A_dis)
        mu_beta = torch.from_numpy(mu_beta).to(self.device)
        mu_beta = F.normalize(mu_beta)
        # print(mu_beta)
        # exit(0)

        return mu_beta

    def __loss_D_dann__(self, d_seq):
        # this is for DANN
        loss = F.nll_loss(flat(d_seq), flat(self.domain_seq), reduction='none')  # , self.u_seq.mean(1)
        loss *= flat(self.domain_sel_mask)
        return loss.mean()

    def __loss_D_cida__(self, d_seq):
        # this is for CIDA
        # use L1 instead of L2
        return F.l1_loss(flat(d_seq),
                         flat(self.u_seq.detach()))  # , self.u_seq.mean(1)

    def __loss_D_grda__(self, d_seq):
        # this is for GRDA
        A = self.A
        
        criterion = nn.BCEWithLogitsLoss()
        # print(d_seq.shape, self.domain_sel_mask.shape)
        d = d_seq * self.domain_sel_mask.unsqueeze(-1)
        # random pick subchain and optimize the D
        # balance coefficient is calculate by pos/neg ratio
        # A is the adjancency matrix
        if self.opt.online:
            # ZIYANTODO: fix it
            my_sample = min(self.tmp_num_domain-1, self.opt.sample_v)
        else:
            my_sample=self.opt.sample_v

        sub_graph = self.__sub_graph__(my_sample_v=my_sample, A=A)
        # print(A.shape,len(sub_graph),sub_graph)
        # print(my_sample, sub_graph)
        # exit(0)

        errorD_connected = torch.zeros((1, )).to(self.device)  # .double()
        errorD_disconnected = torch.zeros((1, )).to(self.device)  # .double()

        count_connected = 0
        count_disconnected = 0

        for i in range(my_sample):
            v_i = sub_graph[i]
            # print(sub_graph, i)
            # no self loop version!!
            for j in range(i + 1, my_sample):
                # print(sub_graph, i, j)
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
