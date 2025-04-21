import pickle
import numpy as np
import matplotlib.pyplot as plt

# with open("data/toy_d15_quarter_circle.pkl", "rb") as data_file:
#     ref_data_pkl = pickle.load(data_file)
#     data = ref_data_pkl['data']
#     data_mean = data.mean(0, keepdims=True)
#     data_std = data.std(0, keepdims=True)
#     ref_data = (ref_data_pkl['data'] - data_mean) / data_std
T1, T2, T3 = 50, 100, 150
with open(f"data/growing_circle/data/toy_d30_pi_{T1}.pkl", "rb") as data_file:
    curr_data_pkl = pickle.load(data_file)
    curr_data_0 = curr_data_pkl['data']
    # curr_data_0 = (curr_data_pkl['data'] - data_mean) / data_std

with open(f"data/growing_circle/data/toy_d30_pi_{T2}.pkl", "rb") as data_file:
    next_data_pkl = pickle.load(data_file)
    next_data = next_data_pkl['data']
    # next_data = (next_data_pkl['data'] - data_mean) / data_std

with open(f"data/growing_circle/data/toy_d30_pi_{T3}.pkl", "rb") as data_file:
    prev_data_pkl = pickle.load(data_file)
    prev_data = prev_data_pkl['data']
    # next_data = (next_data_pkl['data'] - data_mean) / data_std

l_style_self = ['k*', 'r*', 'b*', 'y*', 'k.', 'r.']
c_style_self = ['red', 'blue']
l_style_ref = ['k*', 'r*', 'b*', 'y*', 'k.', 'r.']
c_style_ref = ['green', 'yellow']
for i in range(2):
    data_sub = curr_data_0[curr_data_pkl['label'] == i, :]
    print(data_sub.shape[0], 'our')
    plt.plot(data_sub[:, 0], data_sub[:, 1], l_style_self[i], color=c_style_self[i], alpha=0.1)

    data_sub = next_data[next_data_pkl['label'] == i, :]
    print(data_sub.shape[0], 'ref')
    plt.plot(data_sub[:, 0], data_sub[:, 1], l_style_ref[i], color=c_style_ref[i], alpha=0.1)

    data_sub = prev_data[prev_data_pkl['label'] == i, :]
    print(data_sub.shape[0], 'prev')
    plt.plot(data_sub[:, 0], data_sub[:, 1], l_style_ref[i], color=c_style_ref[i], alpha=0.1)
plt.title(f"Circle dataset with time frame.")
plt.legend()
plt.show()
plt.savefig(f"./test_diff_T_{T1}_{T2}_{T3}.png")
plt.clf()


# 画图
fig, axs = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)
# 子图 1：t=50
label_mask_0, label_mask_1 = (curr_data_pkl['label'] == 0), (curr_data_pkl['label'] == 1)
source_mask_0 = (curr_data_pkl['domain'] < 6) & label_mask_0
source_mask_1 = (curr_data_pkl['domain'] < 6) & label_mask_1
target_mask_0 = (curr_data_pkl['domain'] >= 6) & label_mask_0
target_mask_1 = (curr_data_pkl['domain'] >= 6) & label_mask_1
axs[0].plot(curr_data_0[source_mask_0, 0], curr_data_0[source_mask_0, 1], 'o', color='red', markersize=3, 
            alpha=0.63)
axs[0].plot(curr_data_0[source_mask_1, 0], curr_data_0[source_mask_1, 1], 'o', color='green', markersize=3, 
            alpha=0.63)
axs[0].plot(curr_data_0[target_mask_0, 0], curr_data_0[target_mask_0, 1], 'o', color='black', markersize=3,
            alpha=0.63)
axs[0].plot(curr_data_0[target_mask_1, 0], curr_data_0[target_mask_1, 1], 'o', color='blue', markersize=3,
            alpha=0.63)
axs[0].axhline(y=5, color='gray', linestyle='--', linewidth=1)
axs[0].set_title(f"T={T1}", fontsize=12, va='center')
axs[0].set_xlim(-8, 8)
axs[0].set_ylim(0, 8)
# axs[0].legend(loc='upper right')
# axs[0].set_xticks([])
# axs[0].set_yticks([])

# 子图 2：t=60
label_mask_0, label_mask_1 = (next_data_pkl['label'] == 0), (next_data_pkl['label'] == 1)
source_mask_0 = (next_data_pkl['domain'] < 6) & label_mask_0
source_mask_1 = (next_data_pkl['domain'] < 6) & label_mask_1
target_mask_0 = (next_data_pkl['domain'] >= 6) & label_mask_0
target_mask_1 = (next_data_pkl['domain'] >= 6) & label_mask_1
axs[1].plot(next_data[source_mask_0, 0], next_data[source_mask_0, 1], 'o', color='red', markersize=3, 
            alpha=0.63)
axs[1].plot(next_data[source_mask_1, 0], next_data[source_mask_1, 1], 'o', color='green', markersize=3, 
            alpha=0.63)
axs[1].plot(next_data[target_mask_0, 0], next_data[target_mask_0, 1], 'o', color='black', markersize=3,
            alpha=0.63)
axs[1].plot(next_data[target_mask_1, 0], next_data[target_mask_1, 1], 'o', color='blue', markersize=3,
            alpha=0.63)
axs[1].axhline(y=5, color='gray', linestyle='--', linewidth=1)
axs[1].set_title(f"T={T2}", fontsize=12, va='center')
axs[1].set_xlim(-8, 8)
axs[1].set_ylim(0, 8)
# axs[1].legend(loc='upper right')
# axs[1].set_xticks([])
# axs[1].set_yticks([])

# 子图 2：t=60
label_mask_0, label_mask_1 = (prev_data_pkl['label'] == 0), (prev_data_pkl['label'] == 1)
source_mask_0 = (prev_data_pkl['domain'] < 6) & label_mask_0
source_mask_1 = (prev_data_pkl['domain'] < 6) & label_mask_1
target_mask_0 = (prev_data_pkl['domain'] >= 6) & label_mask_0
target_mask_1 = (prev_data_pkl['domain'] >= 6) & label_mask_1
axs[2].plot(prev_data[source_mask_0, 0], prev_data[source_mask_0, 1], 'o', color='red', markersize=3, 
            alpha=0.63)
axs[2].plot(prev_data[source_mask_1, 0], prev_data[source_mask_1, 1], 'o', color='green', markersize=3, 
            alpha=0.63)
axs[2].plot(prev_data[target_mask_0, 0], prev_data[target_mask_0, 1], 'o', color='black', markersize=3,
            alpha=0.63)
axs[2].plot(prev_data[target_mask_1, 0], prev_data[target_mask_1, 1], 'o', color='blue', markersize=3,
            alpha=0.63)
axs[2].axhline(y=5, color='gray', linestyle='--', linewidth=1)
axs[2].set_title(f"T={T3}", fontsize=12, va='center')
axs[2].set_xlim(-8, 8)
axs[2].set_ylim(0, 8)
# axs[2].legend(loc='upper right')

# 添加底部标签
fig.suptitle("Circle dataset with time frame.", fontsize=14)
# plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.subplots_adjust(hspace=0.4,top=0.3)
plt.show()
plt.savefig(f"./test_diff_T_{T1}_{T2}_{T3}.png")
plt.savefig(f"./test_diff_T_{T1}_{T2}_{T3}.pdf", format='pdf', bbox_inches = 'tight', pad_inches = 0)
plt.clf()

# for i in range(2):
#     data_sub = curr_data_0[curr_data_pkl['label'] == i, :]
#     print(data_sub.shape[0], 'our')
#     plt.plot(data_sub[:, 0], data_sub[:, 1], l_style_self[i], color=c_style_self[i], alpha=0.5, label=f"curr_{i}")

#     data_sub = ref_data[ref_data_pkl['label'] == i, :]
#     print(data_sub.shape[0], 'ref')
#     plt.plot(data_sub[:, 0], data_sub[:, 1], l_style_ref[i], color=c_style_ref[i], alpha=0.5, label=f"ref_{i}")
# plt.title(f"Circle dataset with time frame.")
# plt.legend()
# plt.show()
# plt.savefig(f"./test_self_1.png")
# plt.clf()
# exit(0)