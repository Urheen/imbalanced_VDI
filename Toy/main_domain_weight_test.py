
import numpy as np
import matplotlib.pyplot as plt
# 设置参数
num_classes = 30
alpha = np.ones(num_classes) * 1.0  # 对称 Dirichlet 分布
# alpha = np.array([2.0] * opt.k + [1.0] * (opt.num_domain - opt.k))
print(alpha)

# 采样 Dirichlet 分布
dirichlet_sample = np.random.dirichlet(alpha)
sorted_sample = np.sort(dirichlet_sample)[::-1]

# 验证是否归一化（总和应该为 1）
print(f"Sum of Dirichlet Sample: {np.sum(sorted_sample)}")

# 可视化
plt.figure(figsize=(12, 5))
plt.bar(np.arange(num_classes), sorted_sample)
plt.xlabel('Class Index')
plt.ylabel('Probability')
plt.title('Dirichlet Distribution Sample (30 Classes)')
plt.show()
plt.savefig('./test_dirichlet_1.png')
plt.clf()

alpha_vals = [0.1, 0.5, 1.0, 2.0, 5.0]
num_classes = 30
for alpha_val in alpha_vals:
    alpha = np.ones(num_classes) * alpha_val  # 对称 Dirichlet 分布
    # alpha = np.array([2.0] * opt.k + [1.0] * (opt.num_domain - opt.k))
    print(alpha)

    # 采样 Dirichlet 分布
    dirichlet_sample = np.random.dirichlet(alpha)
    # dirichlet_sample = np.round(dirichlet_sample).astype(int)
    # dirichlet_sample = dirichlet_sample / np.sum(dirichlet_sample)
    sorted_sample = np.sort(dirichlet_sample)[::-1]

    # 验证是否归一化（总和应该为 1）
    print(f"Sum of Dirichlet Sample: {np.sum(sorted_sample)}")

    # 可视化
    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(num_classes), sorted_sample)
    plt.xlabel('Class Index')
    plt.ylabel('Probability')
    plt.title('Dirichlet Distribution Sample (30 Classes)')
    plt.show()
    plt.savefig(f"./test_dirichlet_{alpha_val}.png")
    plt.clf()
exit(0)

# dataloader, test_loader = get_loader(opt, epoch, return_test_loader=True)
        # for warm_epoch in range(opt.warm_epoch):
        #     model.learn(warm_epoch, dataloader, domain_weights=torch.ones_like(domain_weights))
        #     test_flag = (warm_epoch + 1) % opt.test_interval == 0 or (warm_epoch + 1) == opt.num_epoch
        #     if test_flag:
        #         model.test(epoch, test_loader)
        # assert warm_epoch == opt.warm_epoch-1, f"Warm-up training is not finished with warm_epoch {warm_epoch}!!!"
        # print(f"warm up training DONE!")