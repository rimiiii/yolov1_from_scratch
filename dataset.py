# 데이터 label 만들기
# c1, c2, ..., c20, c, x, y, w, h, 0, 0, 0, 0, 0
import torch
import os
import pandas as pd
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = label.split()
                # string이라서 소수는 소수 int는 int로 변환
                class_label, x, y, width, height = int(class_label), float(x), float(y), float(width), float(height)
                boxes.append([class_label, x, y, width, height]) 
                """
                x,y,w,h는 물체 크기에서 상대적인 위치임
                grid로 나눴을 때 어느곳에 있을지는 cell의 크기를 각각 1로 봤을때 x*S를 하면 일의자리수로 cell의 위치 파악
                예를 들어서 S가 4이고 x, y의 위치가 가로 2, 세로 3번째 칸의 물체라면 대략 (0.28, 0.6)
                S를 곱한다는 의미는 물체의 크기를 1에서 S만큼 키운다는 의미이고 x, y의 상대적인 위치도 S만큼 커진 걸로 변화함
                예시로 본다면 (0.28, 0.6) * 4 -> (1.12, 2.4)로 i = 1, j = 2(0부터 시작함)
                """
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        
        if self.transform:
            image, boxes = self.transfrom(image, boxes)
            
        label_matrix = torch.zeros((S, S, 20+B*5))

        # 같은 cell 안에 object가 여러개인 경우 결국 하나만 들어간다
        # 먼저 들어온 것만 추출하고 그 뒤로 들어오는 건 들어오지 못함
        # i, j 구하기 
        for box in boxes:
            if label_matrix[i, j, 20]  == 0: # 만일 [i, j, 20]이 0일때만 가능, 1일땐 object가 이미 있으므로 넣지 않음
                label_matrix[i, j, 20] = 1
                
                i, j = int(box[1]*S), int(box[2]*S)
                cell_x, cell_y = box[1]*S - j, box[2]*S - i # cell에서의 x, y 구하기 S*주어진 x-i
                cell_w, cell_h = box[3]*S, box[4]*S
                label_matrix[i, j][21:25] = torch.tensor([cell_x, cell_y, cell_w, cell_h]) #bounding box[21:25]에 x,y,w,h 넣기
                label_matrix[i, j, box[0]] = 1
        
        return image, label_matrix