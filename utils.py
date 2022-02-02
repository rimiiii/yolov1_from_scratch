import torch

def intersection_over_union(pred_box, true_box):
    # 각 box의 넓이 w*h
    pred_box_area = pred_box[...,2]*pred_box[...,3]
    true_box_area = true_box[...,2]*true_box[...,3]
    # 교집합 영역의 넓이
    # 1. 교집합 영역이 있는지 파악
    #   true, pred의 x 중 큰 값 x1: x-1/2w, 작은 값 x2: x+1/2w
    #   위에서 구한 x2 < x1 이라면 교집합 없음
    # 교집합 영역이 있다면 0, 없다면 다음 단계
    # 2. true, pred의 x 중 큰 값: max_x-1/2w, 작은 값: min_x+1/2w 작은값에서 큰값 뺴기
    #   min_x - max_x + 1/2w(max_x) + 1/2w(min_x)

    max_x = torch.max(pred_box[...,0], true_box[...,0])
    min_x = torch.min(pred_box[...,0], true_box[...,0])
    intersection_width = min_x - max_x + pred_box[...,2]/2 + true_box[...,2]/2
    intersection_width[(intersection_width<0)] = 0

    max_y = torch.max(pred_box[...,1], true_box[...,1])
    min_y = torch.min(pred_box[...,1], true_box[...,1])
    intersection_height = min_y - max_y + pred_box[...,3]/2 + true_box[...,3]/2
    intersection_height[(intersection_height<0)] = 0
    intersection_area = intersection_width * intersection_height

    union_area = pred_box[...,2]*pred_box[...,3] + true_box[...,2]*true_box[...,3] - intersection_area
    return intersection_area / union_area