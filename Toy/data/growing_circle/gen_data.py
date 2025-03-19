#!/usr/bin/python3
import matplotlib
import collections
import csv
import shutil
import os
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom
import re

import pickle

def read_pickle(name):
    with open(name, "rb") as f:
        data = pickle.load(f)
    return data


def write_pickle(data, name):
    with open(name, "wb") as f:
        pickle.dump(data, f)

# generate data given the mean/std, radius and number
def generate_data(mean, std, radius, num):
    dim = mean.shape[0]
    m_data = np.random.randn(num, dim)
    print('bingo', m_data.shape)
    m_data *= std[None, :]
    m_radius = m_data[:, 0] ** 2 + m_data[:, 1] ** 2
    m_data += mean[None, :]
    m_data = m_data[m_radius <= radius ** 2, :]
    print('num of data points within radius', radius, ':', m_data.shape[0])
    return m_data

# generate label for circle-shape data
def generate_label(m_data, radius):
    m_radius = m_data[:, 0] ** 2 + m_data[:, 1] ** 2
    m_label = np.zeros((m_data.shape[0],))
    m_label[m_radius > radius ** 2] = 1
    return m_label

# create dataset, circle-shape domain manifold
def create_toy_data(t=0, radius_large=5):
    fname = f"./data/growing_circle/data/toy_d30_pi_{t}.pkl"
    l_angle = list(np.linspace(0, np.pi, 30)) # pi/2,15
    # radius_large = radius_large
    radius_small = 1
    std_small = 1

    lm_data = []
    l_domain = []
    for i, angle in enumerate(l_angle):
        mean = np.array([np.cos(angle), np.sin(angle)]) * radius_large
        std = np.ones((2,)) * std_small
        m_data = generate_data(mean, std, radius_small, 300)
        lm_data.append(m_data)
        l_domain.append(np.ones(m_data.shape[0],) * i)
    data_all = np.concatenate(lm_data, axis=0)
    domain_all = np.concatenate(l_domain, axis=0)
    label_all = generate_label(data_all, radius_large)

    d_pkl = dict()
    d_pkl['data'] = data_all
    d_pkl['label'] = label_all
    d_pkl['domain'] = domain_all
    write_pickle(d_pkl, fname)

    if t == 60 or t == 50:
        l_style = ['k*', 'r*', 'b*', 'y*', 'k.', 'r.']
        for i in range(2):
            data_sub = data_all[label_all == i, :]
            plt.plot(data_sub[:, 0], data_sub[:, 1], l_style[i])
        plt.title(f"Circle dataset with time frame is {t}.")
        plt.show()
        plt.savefig(f"./data/growing_circle/test_{t}.png")
        plt.clf()

def create_growing_toy_data(T=100):
    boudnary, neg, pos  = [], [], []
    for t in range(T):
        this_r = np.cos(t / np.pi) * 2 + 6
        create_toy_data(t=t, radius_large=this_r)
        boudnary.append(this_r)
        neg.append(this_r - 0.5)
        pos.append(this_r + 0.5)

    plt.plot(np.arange(t+1), boudnary, color='black')
    plt.plot(np.arange(t+1), pos, color='red', linestyle='--')
    plt.plot(np.arange(t+1), neg, color='blue', linestyle='--')
    plt.show()
    plt.savefig(f"./data/growing_circle/test_all.png")
    plt.clf()

# generate label for ellipse-shape data
def generate_label_ellipse(m_data, radius_x, radius_y):
    m_radius = m_data[:, 0] ** 2 / radius_x ** 2 + m_data[:, 1] ** 2 / radius_y ** 2
    m_label = np.zeros((m_data.shape[0],))
    m_label[m_radius > 1] = 1
    return m_label

# create dataset, circle-shape domain manifold, biased boundary
def create_toy_data_biased_circle():
    fname = 'toy_d15_biased_circle.pkl'
    l_angle = list(np.linspace(0, np.pi / 1, 15))
    radius_large_data = 5
    radius_large_x = 5
    radius_large_y = 4.5
    radius_small = 1.0
    std_small = 1.0

    lm_data = []
    l_domain = []
    for i, angle in enumerate(l_angle):
        mean = np.array([np.cos(angle), np.sin(angle)]) * radius_large_data
        std = np.ones((2,)) * std_small
        m_data = generate_data(mean, std, radius_small, 300)
        lm_data.append(m_data)
        l_domain.append(np.ones(m_data.shape[0],) * i)
    data_all = np.concatenate(lm_data, axis=0)
    domain_all = np.concatenate(l_domain, axis=0)
    label_all = generate_label_ellipse(data_all, radius_large_x, radius_large_y)

    d_pkl = dict()
    d_pkl['data'] = data_all
    d_pkl['label'] = label_all
    d_pkl['domain'] = domain_all
    # write_pickle(d_pkl, fname)

    l_style = ['k*', 'r*', 'b*', 'y*', 'k.', 'r.']
    for i in range(2):
        data_sub = data_all[label_all == i, :]
        plt.plot(data_sub[:, 0], data_sub[:, 1], l_style[i])
    plt.show()

# generate data for sin-shape dataset given A, w, phi of sin function, std, radius, and start point
def generate_data_sin(A, w, phi, std, radius, start, width, num):
    y = np.random.randn(num, 1) * std
    y = y[np.abs(y[:, 0]) < radius, :]
    x = np.random.rand(y.shape[0], 1) * width + start
    y = y + A * np.sin(w * x + phi)
    m_data = np.concatenate((x, y), axis=1)
    return m_data

# generate label for sin-shape dataset given A, w, phi of sin, x, and y
def generate_label_sin(m_data, A, w, phi):
    m_boundary = A * np.sin(w * m_data[:, 0] + phi)
    m_label = np.zeros((m_data.shape[0],))
    m_label[m_data[:, 1] > m_boundary] = 1
    return m_label

# create dataset, sin-shape domain manifold
def create_toy_data_sin():
    fname = 'toy_sin_d24_2pi.pkl'
    A = 1
    w = 3
    phi = 0
    start = 0
    width = np.pi / 9 # np.pi/6
    num_domain = 24 #15
    num_per_domain = 300
    std = 1
    radius = 1
    total_len = 2 * np.pi / 3 * 4 #2*pi

    l_start = list(np.linspace(0, total_len - width, num_domain))

    lm_data = []
    l_domain = []
    for i, start in enumerate(l_start):
        m_data = generate_data_sin(A, w, phi, std, radius, start, width, num_per_domain)
        lm_data.append(m_data)
        l_domain.append(np.ones(m_data.shape[0],) * i)
    data_all = np.concatenate(lm_data, axis=0)
    domain_all = np.concatenate(l_domain, axis=0)
    label_all = generate_label_sin(data_all, A, w, phi)

    d_pkl = dict()
    d_pkl['data'] = data_all
    d_pkl['label'] = label_all
    d_pkl['domain'] = domain_all
    write_pickle(d_pkl, fname)

    l_style = ['k*', 'r*', 'b*', 'y*', 'k.', 'r.']
    for i in range(2):
        data_sub = data_all[label_all == i, :]
        plt.plot(data_sub[:, 0], data_sub[:, 1], l_style[i])
    plt.show()

def plot_toy_pkl(fname='toy_d3.pkl', color='label', l_domain=None, plt_show=True):
    m = read_pickle(fname)
    data_all = m['data']
    if l_domain != None:
        domain = l_domain[0]
        data_all = m['data'][m['domain'] == domain, :]
        label_all = m['label'][m['domain'] == domain]
        gt_all = m['gt'][m['domain'] == domain]
        domain_all = m['domain'][m['domain'] == domain]
    else:
        label_all = m['label']
        gt_all = m['gt']
        domain_all = m['domain'] if color == 'domain' else None
    if color == 'label':
        plot_data_and_label(data_all, label_all, offset=-1)
    elif color == 'gt':
        plot_data_and_label(data_all, gt_all, offset=-1)
    elif color == 'domain':
        plot_data_and_label(data_all, domain_all)
    if plt_show:
        plt.show()

def plot_toy_pkl_encoding(fname='toy_d3.pkl', color='label', l_domain=None):
    plt.figure(0)
    plot_toy_pkl(fname, color, l_domain, plt_show=False)
    plt.figure(1)

    m = read_pickle(fname)

    from sklearn import decomposition
    enc_all = m['encoding']
    pca = decomposition.PCA(n_components=2)
    pca.fit(enc_all)
    data_all = pca.transform(enc_all)

    if l_domain != None:
        domain = l_domain[0]
        data_all = m['data'][m['domain'] == domain, :]
        label_all = m['label'][m['domain'] == domain]
        domain_all = m['domain'][m['domain'] == domain]
        gt_all = m['gt'][m['domain'] == domain]
    else:
        label_all = m['label']
        gt_all = m['gt']
        domain_all = m['domain'] if color == 'domain' else None
    if color == 'label':
        plot_data_and_label(data_all, label_all, offset=-1)
    elif color == 'gt':
        plot_data_and_label(data_all, gt_all, offset=-1)
    elif color == 'domain':
        plot_data_and_label(data_all, domain_all)
    plt.show()

def plot_data_and_label(data_all, label_all, offset=0):
    cmap = matplotlib.cm.get_cmap('Spectral')
    l_style = ['k*', 'r*', 'b*', 'y*', 'c*', 'k.', 'r.', 'b.', 'y.', 'c.', 'k+', 'r+', 'b+', 'y+', 'c+']
    l_color = [cmap(ele)[:3] for ele in np.linspace(0, 1, 15)]
    num = int(np.max(label_all)) + 1
    for i in range(num):
        data_sub = data_all[label_all == i, :]
        #plt.plot(data_sub[:, 0], data_sub[:, 1], l_style[i % len(l_style)])
        if offset == -1:
            # binary case, change cmap to red and black
            l_color[0], l_color[1] = (0,0,0), (1,0,0)
            plt.plot(data_sub[:, 0], data_sub[:, 1], '.', color=l_color[i % len(l_color)])
        else:
            plt.plot(data_sub[:, 0], data_sub[:, 1], '.', color=l_color[(i+offset) % len(l_color)])
    #plt.show()

def center_pkl(fin='toy_d6.pkl', fout='toy_d6_centered.pkl'):
    m = read_pickle(fin)
    num_domain = int(np.max(m['domain'])) + 1
    for i in range(num_domain):
        mean = np.mean(m['data'][m['domain'] == i, :], axis=0, keepdims=True)
        m['data'][m['domain'] == i, :] -= mean
    write_pickle(m, fout)

def smoothen_domain(fin='toy_sin_d15.pkl', fout='toy_sin_d15_con.pkl'):
    m = read_pickle(fin)
    m['domain'] = m['data'][:, 0]
    write_pickle(m, fout)

def ensemble_pred(f1='pred_tmp.pkl1', f2='pred_tmp.pkl2', fout='pred_tmp.pkl'):
    m1 = read_pickle(f1)
    m2 = read_pickle(f2)
    prob_mean = (m1['prob'] + m2['prob']) / 2
    label = np.argmax(prob_mean, axis=1)
    m1['label'] = label
    write_pickle(m1, fout)

def dataset_cut_tail(fname='toy_sin_d18_2pi.pkl', fout='toy_sin_d18_2pi_cut.pkl', cut_size=200):
    m = read_pickle(fname)
    m_new = dict()
    datasize = m['data'].shape[0]
    num_domain = int(np.max(m['domain'])) + 1
    d = {i:0 for i in range(num_domain)}
    l_data = list()
    l_label = list()
    l_domain = list()
    for i in range(datasize):
        if d[m['domain'][i]] < cut_size:
            l_data.append(m['data'][i][None, :])
            l_label.append(m['label'][i])
            l_domain.append(m['domain'][i])
            d[m['domain'][i]] += 1
    data_all = np.concatenate(l_data, axis=0)
    #domain_all = np.concatenate(l_domain, axis=0)
    domain_all = np.array(l_domain).astype('int64')
    #label_all = np.concatenate(l_label, axis=0)
    label_all = np.array(l_label).astype('int64')
    m_new['data'] = data_all
    m_new['domain'] = domain_all
    m_new['label'] = label_all
    write_pickle(m_new, fout)
    print('d', d)


if __name__ == '__main__':
    create_toy_data()
    create_growing_toy_data()
    #plot_toy_pkl(fname='pred_tmp.pkl', color='gt')
    # plot_toy_pkl(fname='pred_tmp.pkl', color='label')
    #dataset_cut_tail(fname='toy_d30_pi.pkl', fout='toy_d30_pi_cut.pkl', cut_size=100)
    #dataset_cut_tail(fname='toy_sin_d18_2pi.pkl', fout='toy_sin_d18_2pi_cut.pkl', cut_size=190)
    #plot_toy_pkl_encoding(fname='pred_tmp.pkl', color='gt')
    #plot_toy_pkl_encoding(fname='pred_tmp.pkl', color='label')
    #plot_toy_pkl(fname='pred_baseline.pkl')
    #plot_toy_pkl(fname='pred_baseline_norm.pkl')
    #plot_toy_pkl(fname='pred_domainnet.pkl')
    #plot_toy_pkl(fname='toy_d15.pkl', color='label', l_domain=None)
    #center_pkl(fin='toy_d60.pkl', fout='toy_d60_centered.pkl')
    #create_toy_data_sin()
    # create_toy_data_biased_circle()
    #smoothen_domain(fin='toy_sin_d15.pkl', fout='toy_sin_d15_con.pkl')
    #ensemble_pred(f1='pred_tmp.pkl1', f2='pred_tmp.pkl2', fout='pred_tmp.pkl')
