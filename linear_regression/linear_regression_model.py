import random

import torch
import matplotlib.pyplot as plt

torch.manual_seed(1)


def synthetic_data(w, b, num_examples):
    """
    生成y=Xw+b+噪声
    """
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    num_example = len(features)
    indices = list(range(num_example))
    random.shuffle(indices)
    for i in (0, num_example, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_example)])
        yield features[batch_indices], labels[batch_indices]


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
# plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
# plt.show()
