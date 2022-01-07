"""
input: Batch, S, S, B*5+C : model을 거쳐서 나온 
target: Batch, S, S, B*5+C
mse = torch.nn.MSELoss(reduction='sum')
loss = mse(x,y에 관한)+mse(w, h에 관한)+mse(c에 관한)+mse(c에 관한)+mse(p에 관한)
"""

import torch
import numpy as np
import torch.nn as nn


S = 7
B = 2
C = 20
lambda_coord = 5
lambda_noobj = 0.5

pred_label = torch.randn(100, S, S, C+B*5) #c1,c2..,c20,x,y,w,h,C,x,y,w,h,C
true_label = torch.randn(100, S, S, C+B*5) #c1,c2..,c20,x,y,w,h,C,x,y,w,h,C

#x,y 뽑아서 box별로 IOU 계산
#box1, box2 중에 IOU가 높은 box, IOU 추출
pred_box1 = pred_label[..., 21:25] # box1 x,y,w,h
pred_box2 = pred_label[..., 26:30] # box2 x,y,w,h
true_box1 = pred_label[..., 21:25] # box1 x,y,w,h
pre_box2 = pred_label[..., 26:30] # box2 x,y,w,h



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

"""
#print(pred_label[..., 21:25])

#import numpy as np
#test = np.arange(100*S*S*(B*5+C)).reshape((100, S, S, B*5+C))
#print(test[..., 21:25])
# input 
true_x1, true_x2, true_y1, true_y2 = 23, 35, 47, 62 
pred_x1, pred_x2, pred_y1, pred_y2 = 30, 38, 32, 50 

union = (true_x2-true_x1) * (true_y2-true_y1) + (pred_x2 - pred_x1) * (pred_y2-pred_y1) - (abs(pred_x1 - true_x1) * abs(pred_y1 - true_y1))

x = range(1, 7)
y = range(8, 20)
xs = set(x)
#print(len(xs.intersection(y)))
#print(xs.union(y))

true_x = range(true_x1, true_x2+1)
pred_x = range(pred_x1, pred_x2+1)
true_y = range(true_y1, true_y2+1)
pred_y = range(pred_x1, pred_y2+1)

true_xs = set(true_x)
true_ys = set(true_y)

intersection_x = sorted(list(true_xs.intersection(pred_x)))
intersection_y = sorted(list(true_ys.intersection(pred_y)))

xy_intersection = (intersection_x[-1]-intersection_x[0]) * (intersection_y[-1]-intersection_y[0])
xy_union = (true_x2 - true_x1) * (true_y2 - true_y1) + (pred_x2-pred_x1) * (pred_y2 - pred_y1) - xy_intersection

intersection_over_union = xy_intersection / xy_union
print(intersection_over_union)
print(xy_intersection)


"""