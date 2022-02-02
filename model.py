import torch
import numpy as np
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

S = 7
B = 2
C = 20

input = torch.randn(100, 3, 448, 448) #batch_size, channels, height, width
#in_channels, out_channels, kernel_size, stride
m = nn.Conv2d(3, 64, 7, padding=3, stride=2, bias=False)(input) #224x224x64
m = nn.BatchNorm2d(64)(m)
m = nn.LeakyReLU(0.1)(m)
m = nn.MaxPool2d(2, stride=2)(m) # 112x112x64 #padding 사이즈를 어떻게 알지? 그냥 단순히 뒷 부분 사이즈를 보고 계산한건가?
m = nn.Conv2d(64, 192, 3, padding=1)(m) # 112x112x192
m = nn.BatchNorm2d(192)(m)
m = nn.LeakyReLU(0.1)(m)
m = nn.MaxPool2d(2, stride=2)(m) #56x56x192
m = nn.Conv2d(192, 128, 1, padding=0)(m) #56x56x128
m = nn.BatchNorm2d(128)(m)
m = nn.LeakyReLU(0.1)(m)
m = nn.Conv2d(128, 256, 3, padding=1)(m) #56x56x128
m = nn.BatchNorm2d(256)(m)
m = nn.LeakyReLU(0.1)(m)
m = nn.Conv2d(256, 256, 1, padding=0)(m) #56x56x256
m = nn.BatchNorm2d(256)(m)
m = nn.LeakyReLU(0.1)(m)
m = nn.Conv2d(256, 512, 3, padding=1)(m) #56x56x512
m = nn.BatchNorm2d(512)(m)
m = nn.LeakyReLU(0.1)(m)
m = nn.MaxPool2d(2, stride=2)(m) #28x28x512

m = nn.Conv2d(512, 256, 1, padding=0)(m) #28x28x256
m = nn.BatchNorm2d(256)(m)
m = nn.LeakyReLU(0.1)(m)
m = nn.Conv2d(256, 512, 3, padding=1)(m) #28x28x512
m = nn.BatchNorm2d(512)(m)
m = nn.LeakyReLU(0.1)(m)
m = nn.Conv2d(512, 256, 1, padding=0)(m) #28x28x256
m = nn.BatchNorm2d(256)(m)
m = nn.LeakyReLU(0.1)(m)
m = nn.Conv2d(256, 512, 3, padding=1)(m) #28x28x512
m = nn.BatchNorm2d(512)(m)
m = nn.LeakyReLU(0.1)(m)
m = nn.Conv2d(512, 256, 1, padding=0)(m) #28x28x256
m = nn.BatchNorm2d(256)(m)
m = nn.LeakyReLU(0.1)(m)
m = nn.Conv2d(256, 512, 3, padding=1)(m) #28x28x512
m = nn.BatchNorm2d(512)(m)
m = nn.LeakyReLU(0.1)(m)
m = nn.Conv2d(512, 256, 1, padding=0)(m) #28x28x256
m = nn.BatchNorm2d(256)(m)
m = nn.LeakyReLU(0.1)(m)
m = nn.Conv2d(256, 512, 3, padding=1)(m) #28x28x512
m = nn.BatchNorm2d(512)(m)
m = nn.LeakyReLU(0.1)(m)
m = nn.Conv2d(512, 512, 1, padding=0)(m) #28x28x512
m = nn.BatchNorm2d(512)(m)
m = nn.LeakyReLU(0.1)(m)
m = nn.Conv2d(512, 1024, 3, padding=1)(m) #28x28x1024
m = nn.BatchNorm2d(1024)(m)
m = nn.LeakyReLU(0.1)(m)
m = nn.MaxPool2d(2, stride=2)(m) #14x14x1024

m = nn.Conv2d(1024, 512, 1, padding=0)(m) #14x14x512
m = nn.BatchNorm2d(512)(m)
m = nn.LeakyReLU(0.1)(m)
m = nn.Conv2d(512, 1024, 3, padding=1)(m) #14x14x1024
m = nn.BatchNorm2d(1024)(m)
m = nn.LeakyReLU(0.1)(m)
m = nn.Conv2d(1024, 512, 1, padding=0)(m) #14x14x512
m = nn.BatchNorm2d(512)(m)
m = nn.LeakyReLU(0.1)(m)
m = nn.Conv2d(512, 1024, 3, padding=1)(m) #14x14x1024
m = nn.BatchNorm2d(1024)(m)
m = nn.LeakyReLU(0.1)(m)
m = nn.Conv2d(1024, 1024, 3, padding=1)(m) #14x14x1024
m = nn.BatchNorm2d(1024)(m)
m = nn.LeakyReLU(0.1)(m)
m = nn.Conv2d(1024, 1024, 3, padding=1, stride=2)(m) #7x7x1024
m = nn.BatchNorm2d(1024)(m)
m = nn.LeakyReLU(0.1)(m)

m = nn.Conv2d(1024, 1024, 3, padding=1)(m) #7x7x1024
m = nn.BatchNorm2d(1024)(m)
m = nn.LeakyReLU(0.1)(m)
m = nn.Conv2d(1024, 1024, 3, padding=1)(m) #7x7x1024
m = nn.BatchNorm2d(1024)(m)
m = nn.LeakyReLU(0.1)(m)

m = torch.flatten(m, start_dim=1) #50176
m = nn.Linear(50176, 4096)(m) #4096
m = nn.Linear(4096, S**2 * (C+B*5))(m)
m = torch.reshape(m, (-1, S, S, (C+B*5))) #7x7x30(30=2(bounding box 갯수)x5(x,y,w,h,confidence prediction)+20(class))

print(m.shape)