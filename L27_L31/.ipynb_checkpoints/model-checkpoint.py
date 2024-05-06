from torch import nn
import torch

# 搭建CAFIR10神经网络
class Tudui(nn.Module):
    
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5,  stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),                                         # 实际上就是把batch_size后面的三个维度全部展平  https://www.jb51.net/article/271950.html
            nn.Linear(in_features=64*4*4,out_features=64),        #  全连接层
            nn.Linear(in_features=64, out_features=10)            #  最后映射到10分类上
        )
        
    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    tudui = Tudui()
    input = torch.ones((64, 3, 32, 32))
    output = tudui(input)
    print(output.shape)
