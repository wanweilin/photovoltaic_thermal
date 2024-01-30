import os
import glob
from tqdm import tqdm
import torch
import torchvision
import sys 
from PIL import Image
from pathlib import Path
current_folder = Path(__file__).absolute().parent 
father_folder = str(current_folder.parent) 
sys.path.append(father_folder)
from models.unet3plus import UNet_3Plus

# # 加载PyTorch模型
# model = UNet_3Plus(in_channels=1)
# model_weights = torch.load("./ckpt/unet_demo_7/best.pth")["model_state_dict"]
# model.load_state_dict(model_weights)

# # 将模型转换为ONNX格式
# input_shape = (1, 1, 256, 256)  # 输入图片的大小
# input_names = ["input"]  # 模型输入的名称
# output_names = ["output"]  # 模型输出的名称
# dummy_input = torch.randn(input_shape)  # 创建虚拟输入
# torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, input_names=input_names, output_names=output_names)


# if __name__ == "__main__":
    # with open('vis/thresholds', 'r') as f:
    #     data = f.readlines()
    # print(len(data))
    # best = 0
    # btp, bfp, bth = 0, 0, 0
    # for line in tqdm(data):
    #     figs = line.split('_')
    #     fp = float(figs[0])
    #     tp = float(figs[1])
    #     th = float(figs[2])
    #     if tp - fp > best:
    #         best = tp - fp
    #         btp = tp
    #         bfp = fp
    #         bth = th
    # print(btp, bfp, bth, best)
            