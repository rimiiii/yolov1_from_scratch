"""
input: Batch, S, S, B*5+C : model을 거쳐서 나온 
target: Batch, S, S, B*5+C
mse = torch.nn.MSELoss(reduction='sum')
loss = mse(x,y에 관한)+mse(w, h에 관한)+mse(c에 관한)+mse(c에 관한)+mse(p에 관한)
"""

import torch
import numpy as np
import torch.nn as nn
from utils import intersection_over_union

S = 2
B = 2
C = 20
lambda_coord = 5
lambda_noobj = 0.5

pred_label = torch.abs(torch.randn(2, S, S, C+B*5)) #c1,c2..,c20,c,x,y,w,h,c,x,y,w,h
true_label = torch.abs(torch.randn(2, S, S, C+B*5)) #c1,c2..,c20,c,x,y,w,h,0,0,0,0,0

#box별로 IOU 계산
#box1, box2 중에 IOU가 높은 box, IOU 추출
iou1 = intersection_over_union(pred_label[..., 21:25], true_label[..., 21:25]) # box x,y,w,h
iou2 = intersection_over_union(pred_label[..., 26:30], true_label[..., 21:25])
print(iou1, iou2)
#print(iou1.unsqueeze(-1).shape)
print(torch.cat((iou1.unsqueeze(-1), iou2.unsqueeze(-1)), -1))
print(torch.argmax(torch.cat((iou1.unsqueeze(-1), iou2.unsqueeze(-1)), -1), dim=-1))
print(torch.argmax(torch.cat((iou1.unsqueeze(-1), iou2.unsqueeze(-1)), -1), dim=-1).shape)
max_iou_index = torch.argmax(torch.cat((iou1.unsqueeze(-1), iou2.unsqueeze(-1)), -1), dim=-1)

# iou가 큰 box 찾기


# ======================== #
#   FOR BOX COORDINATES    #
# ======================== #

# ==================== #
#   FOR OBJECT LOSS    #
# ==================== #

# ======================= #
#   FOR NO OBJECT LOSS    #
# ======================= #

# ================== #
#   FOR CLASS LOSS   #
# ================== #