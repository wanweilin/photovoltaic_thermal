import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    
    C = tensor.size(1)        #获得图像的维数
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))     
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)                 #将维数的数据转换到第一位
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)              


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-5

    def forward(self, output, target):
        assert output.size() == target.size(), "'input' and 'target' must have the same shape"
        # output = F.softmax(output, dim=1)
        output = flatten(output)
        target = flatten(target)
        # intersect = (output * target).sum(-1).sum() + self.epsilon
        # denominator = ((output + target).sum(-1)).sum() + self.epsilon

        intersect = (output * target).sum(-1)
        denominator = (output + target).sum(-1)
        # print(intersect, denominator)
        dice = intersect / denominator
        dice = torch.mean(dice)
        print('Dice Loss:', dice)
        return 1 - 2. * dice

if __name__ == "__main__":
    dice_loss = DiceLoss()
    a = torch.Tensor(np.zeros((8,1,16,16)))
    b = torch.Tensor(np.zeros((8,1,16,16)))
    a[0,0,0,0] = 1.0
    b[0,0,0,0] = 1.0
    print(dice_loss(a,b))