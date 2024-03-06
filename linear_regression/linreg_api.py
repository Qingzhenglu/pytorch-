import torch
import torch.utils.data
from torch import nn
import random


def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    # 将传入的特征和标签作为list传到TensorDataset，里面得到一个pytorch的数据集
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)


"""初始化模型参数"""
# 第一个指定输入特征形状为2，第二个指定输出特征形状为单个标量为1
net = nn.Sequential(nn.Linear(2, 1))
# net[0] 表示第0层，.weight访问w，data就是w的值，下划线的意思是，使用正态分布，替换掉权重data的值
net[0].weight.data.normal_(0, 0.01)
# 使用填充0，data是偏置b，替换掉偏差data的值
net[0].bias.data.fill_(0)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化函数
trainer = torch.optim.SGD(net.parameters(), lr=0.03)


true_w = torch.tensor([2, -3.4])
true_b = 4.2
# 合成数据
features, labels = synthetic_data(true_w, true_b, 1000)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

num_epochs = 4
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y) # 因为net自带参数，所以和之前不同的是不需要再传进去w和b
        trainer.zero_grad()  # 梯度清零
        l.backward()
        trainer.step()  # 调用step函数进行更新
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
