# 本文件是使用GPU进行训练的

import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from model import *
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10(root="../dataset", train=True, transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10(root="../dataset", train=False, transform=torchvision.transforms.ToTensor(),download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练集长度为{}".format(train_data_size))
print("测试集长度为{}".format(test_data_size))

# 准备数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 运行的设备
device = torch.device("cuda:0")                   # 单显卡写 "cuda"或者"cuda:0"都是一样的

#  创建网络实例
tudui = Tudui()
tudui = tudui.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 训练次数
total_train_step = 0
# 测试次数
total_test_step = 10
# epoch
epoch = 50

# 添加tensorboard
writer = SummaryWriter("./logs_train")

for i in range(epoch):
    print("-----------------第{}轮训练开始------------------".format(i + 1))
    
    # 训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data                                                      # 从dataloader取出数据
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = tudui(imgs)                                                     #  神经网络前向推理
        loss = loss_fn(input=outputs, target=targets)                             #  计算真实值与目标之间的误差
        
        # 优化器调整模型参数
        optimizer.zero_grad()          # 将上次存留的梯度信息清零
        loss.backward()                      #   反向传播
        optimizer.step()                      #   修改权重参数
        
        # 打印训练结果
        total_train_step += 1
        if total_train_step % 100 == 0:              # 每训练100次打印一次结果
            print("训练次数:{} , Loss: {}".format(total_train_step, loss))
            writer.add_scalar("Train Loss", loss.item(), total_train_step)

    # 测试步骤开始：
    tudui.eval()
    total_accuracy = 0
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = tudui(imgs)
            loss = loss_fn(input=outputs, target=targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1)  == targets).sum()                                                                      # 逐行进行比较
            # print(accuracy)
            total_accuracy += accuracy.item()
            # print("total is ："+str(total_accuracy))
    
    writer.add_scalar("Validation Loss", total_test_loss, total_test_step)
    print("本epoch上整体验证集上的Loss:  {}".format(total_test_loss))
    writer.add_scalar("Validation Acc  ", total_accuracy/test_data_size, total_test_step)
    print("本epoch上整体验证集上的Acc  :  {}".format(total_accuracy/test_data_size))
    
    # 本epoch的测试结束，计步器++
    total_test_step += 1

torch.save(tudui, "CAFIR10_ckpt_e50.pth")
print("Model has been save")
writer.close()
