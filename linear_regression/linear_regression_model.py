import random
import torch

import matplotlib.pyplot as plt


def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    # normal：返回一个张量，包含从指定均值means和标准差std的离散正态分布抽取的一组随机数
    x = torch.normal(0, 1, (num_examples, len(w)))  # 均值为0，方差为1
    # matmul：矩阵乘法
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))


def data_iter(lc_batch_size, lc_features, lc_labels):
    num_example = len(lc_features)
    indices = list(range(num_example))
    random.shuffle(indices)
    for i in range(0, num_example, lc_batch_size):
        batch_indices = torch.tensor(indices[i: min(i + lc_batch_size, num_example)])
        yield lc_features[batch_indices], lc_labels[batch_indices]


def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b


def squared_loss(y_pre, y):
    """均方损失"""
    return (y_pre - y.reshape(y_pre.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            # 因为pytorch不会自动将梯度设置为0，设置为零后下次计算就不会与上次相关了
            param.grad.zero_()


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

net = linreg
loss = squared_loss

batch_size = 50
epochs = 10
lr = 0.03

for epoch in range(epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        # 求和之后算梯度，这里backward()会对w和b进行求导
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数

    # net(features, w, b)：预测值
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
# print(f'b的估计误差: {true_b - b}')

# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break
# plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
# plt.show()
