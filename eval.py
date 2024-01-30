import argparse
import os
import torch
import cv2
import torch.nn as nn
import joblib
import pickle
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import yaml
from loss.iouLoss import IOU
from models.unet3plus import UNet_3Plus
from utils.eval_utils import draw_roc_curve
from dataloaders.dataloader import Photovoltaic_dataset
from utils.eval_utils import save_evaluation_curves
from sklearn.metrics import auc, roc_auc_score, roc_curve, average_precision_score, accuracy_score, precision_score, recall_score, f1_score

def evaluate(config, ckpt_path, suffix):
    dataset_name = config["dataset_name"]
    dataset_base_dir = config["dataset_base_dir"]
    device = config["device"]
    num_workers = config["num_workers"]
    spatial_size = config["model_paras"]["spatial_size"]

    eval_dir = os.path.join(config["eval_root"], config["exp_name"])
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    
    model = UNet_3Plus(in_channels=1).to(device).eval()
    model_weights = torch.load(ckpt_path)["model_state_dict"]
    model.load_state_dict(model_weights)

    score_func = nn.BCELoss(size_average=True)
    iou = IOU()
    dataset_test = Photovoltaic_dataset(config, os.path.join(config['dataset_base_dir'], config['dataset_name']), spatial_size=spatial_size, mode="test")
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=1, num_workers=num_workers, shuffle=False)

    total_pixel_scores = np.zeros((spatial_size * spatial_size * len(dataset_test)))
    total_gt_pixel_scores = np.zeros((spatial_size * spatial_size * len(dataset_test)))
    mask_cnt = 0
    iou_scores = []
    for test_data in tqdm(dataloader_test, desc="Eval: ", total=len(dataloader_test)):
        thermal_img, mask_img, check = test_data
        thermal_img = thermal_img.to(device)
        mask_img = mask_img.to(device)

        pred_mask_img = model(thermal_img)

        iou_score = iou(mask_img.detach(), pred_mask_img.detach())
        iou_scores.append(iou_score.cpu().numpy())

        pred_mask_cv = pred_mask_img.detach().cpu().numpy()[0, :, :, :].transpose((1,2,0))
        mask_cv = mask_img.detach().cpu().numpy()[0, :, :, :].transpose((1,2,0))
        # print(np.mean(pred_mask_cv), np.mean(mask_cv))
        flat_mask_cv = mask_cv.flatten()
        flat_pred_mask_cv = pred_mask_cv.flatten()
        total_pixel_scores[mask_cnt * spatial_size * spatial_size:(mask_cnt + 1) * spatial_size * spatial_size] = flat_pred_mask_cv
        total_gt_pixel_scores[mask_cnt * spatial_size * spatial_size:(mask_cnt + 1) * spatial_size * spatial_size] = flat_mask_cv
        mask_cnt += 1
    
    total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
    total_gt_pixel_scores = total_gt_pixel_scores[:spatial_size * spatial_size * mask_cnt]
    total_pixel_scores = total_pixel_scores[:spatial_size * spatial_size * mask_cnt]

    print(len(total_pixel_scores), len(total_gt_pixel_scores))
    print(np.sum(total_pixel_scores), np.sum(total_gt_pixel_scores))
    print(len(iou_scores))
    print('mean iou score:', np.mean(iou_scores))

    

    auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
    ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
    p_pixel = precision_score(total_gt_pixel_scores, np.around(total_pixel_scores+0.4888, 0).astype(int))
    acc_pixel = accuracy_score(total_gt_pixel_scores, np.around(total_pixel_scores+0.4888, 0).astype(int))
    recall_pixel = recall_score(total_gt_pixel_scores, np.around(total_pixel_scores+0.4888, 0).astype(int))
    f1_pixel = f1_score(total_gt_pixel_scores, np.around(total_pixel_scores+0.4888, 0).astype(int))
    # fpr, tpr, roc_thresholds = roc_curve(total_gt_pixel_scores, total_pixel_scores, pos_label=1)
    # draw_roc_curve(fpr, tpr, auroc_pixel, './vis')
    # with open('vis/thresholds', 'w') as f:
    #     for fp, tp, th in zip(fpr, tpr, roc_thresholds):
    #         f.write('%.4f_%.4f_%.4f\n' %(fp, tp, th))
    print("AUC Pixel:  " +str(auroc_pixel))
    print("AP Pixel:  " +str(ap_pixel))
    print("P Pixel:  " +str(p_pixel))
    print("ACC Pixel:  " +str(acc_pixel))
    print("RECALL Pixel:  " +str(recall_pixel))
    print("F1 Pixel:  " +str(f1_pixel))
    print("IOU score:" +str(iou_score))

    return auroc_pixel, ap_pixel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_path", type=str,
                        default="./ckpt/unet_demo_7/best.pth",
                        help='path to pretrained weights')
    parser.add_argument("--cfg_file", type=str,
                        default="./cfgs/cfg.yaml",
                        help='path to pretrained model configs')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.cfg_file))
    dataset_name = config["dataset_name"]
    dataset_base_dir = config["dataset_base_dir"]
    config['device'] = 'cuda:1'
    os.makedirs(os.path.join("./eval", config["exp_name"]), exist_ok=True)
    with torch.no_grad():
        auc, ap = evaluate(config, args.model_save_path, suffix="best")
        print(auc, ap)
        