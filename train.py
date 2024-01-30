import gc
import os
import torch
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter
import torch.nn as nn
import numpy as np
import yaml
import shutil
from eval import evaluate
from tqdm import tqdm

from loss.bceLoss import BCE_loss
from loss.iouLoss import IOU
from loss.diceLoss import DiceLoss
from loss.msssimLoss import MSSSIM
from loss.loss import SSIM
from dataloaders.dataloader import Photovoltaic_dataset, img_batch_tensor2numpy
from models.unet3plus import UNet_3Plus

from utils.initialization_utils import weights_init_kaiming
from utils.vis_utils import visualize_sequences
from utils.model_utils import loader, saver, only_model_saver

import wandb

def train(config):
    paths = dict(log_dir="%s/%s" % (config["log_root"], config["exp_name"]),
                 ckpt_dir="%s/%s" % (config["ckpt_root"], config["exp_name"]))

    os.makedirs(paths["ckpt_dir"], exist_ok=True)

    batch_size = config["batchsize"]
    epochs = config["num_epochs"]
    num_workers = config["num_workers"]
    device = config["device"]
    lr = config["lr"]
    spatial_size = config["model_paras"]["spatial_size"]

    mile_stones = config["mile_stones"]
    gamma = config["gamma"]

    # loss functions
    bce_loss = nn.BCELoss(size_average=True)
    iou_loss = IOU(size_average=True)
    ssim_loss = SSIM(device=device)
    dice_coeff = DiceLoss()

    model = UNet_3Plus(in_channels=1).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=mile_stones, gamma=gamma)

    step = 0
    epoch_last = 0
    if not config["pretrained"]:
        model.apply(weights_init_kaiming)
    else:
        assert (config["pretrained"] is not None)
        model_state_dict, optimizer_state_dict, step = loader(config["pretrained"])
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
    
    writer = SummaryWriter(paths["log_dir"])
    shutil.copyfile("./cfgs/cfg.yaml", os.path.join(config["log_root"], config["exp_name"], "cfg.yaml"))

    best_auc = -1
    for epoch in range(epoch_last, epochs + epoch_last):
        dataset = Photovoltaic_dataset(config, os.path.join(config['dataset_base_dir'], config['dataset_name']), spatial_size=spatial_size, mode="train")
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        for idx, train_data in tqdm(enumerate(dataloader),
                                    desc="Training Epoch %d" % (epoch + 1), total=len(dataloader)):
            model.train()
            thermal_img, mask_img, check = train_data
            thermal_img = thermal_img.to(device)
            mask_img = mask_img.to(device)

            pred_mask_img = model(thermal_img)

            loss_bce = bce_loss(pred_mask_img, mask_img)
            loss_iou = iou_loss(pred_mask_img, mask_img)
            loss_ssim = ssim_loss(pred_mask_img, mask_img)
            loss_dice = dice_coeff(pred_mask_img, mask_img)

            loss_all = config['lam_bce'] * loss_bce + \
                       config['lam_ssim'] * loss_ssim + \
                       config['lam_dice'] * loss_dice

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            if step % config["logevery"] == config["logevery"] - 1:
                print("[Step: {}/ Epoch: {}]: Loss: {:.4f} ".format(step + 1, epoch + 1, loss_all))

                writer.add_scalar('loss_total/train', loss_all, global_step=step + 1)
                writer.add_scalar('loss_bce/train', config["lam_bce"] * loss_bce, global_step=step + 1)
                writer.add_scalar('loss_dice/train', config["lam_dice"] * loss_dice, global_step=step + 1)
                writer.add_scalar('loss_iou/train', config["lam_iou"] * loss_iou, global_step=step + 1)
                writer.add_scalar('loss_ssim/train', config["lam_ssim"] * loss_ssim, global_step=step + 1)

                num_vis = 6
                writer.add_figure("img/pre_mask",
                                  visualize_sequences(img_batch_tensor2numpy(
                                  pred_mask_img.detach().cpu()[:num_vis, :, :, :]),
                                  seq_len=1,
                                  return_fig=True),
                                  global_step=step + 1)
                writer.add_figure("img/gt_mask",
                                  visualize_sequences(img_batch_tensor2numpy(
                                  mask_img.detach().cpu()[:num_vis, :, :, :]),
                                  seq_len=1,
                                  return_fig=True),
                                  global_step=step + 1)
                writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], global_step=step + 1)
            
            step += 1
        scheduler.step()

        if epoch % config["saveevery"] == config["saveevery"] - 1:
            model_save_path = os.path.join(paths["ckpt_dir"], config["model_savename"])
            saver(model.state_dict(), optimizer.state_dict(), model_save_path, epoch + 1, step, max_to_save=5)

            # computer training stats
            stats_save_path = os.path.join(paths["ckpt_dir"], "training_stats.npy-%d" % (epoch + 1))

            with torch.no_grad():
                auc, ap = evaluate(config, model_save_path + "-%d" % (epoch + 1), suffix=str(epoch + 1))
                if auc > best_auc:
                    best_auc = auc
                    only_model_saver(model.state_dict(), os.path.join(paths["ckpt_dir"], "best.pth"))

                writer.add_scalar("auc", auc, global_step=epoch + 1)
                writer.add_scalar("ap", ap, global_step=epoch + 1)

    print("================ Best AUC %.4f ================" % best_auc)


if __name__ == '__main__':
    config = yaml.safe_load(open("./cfgs/cfg.yaml"))
    dataset_name = config["dataset_name"]
    dataset_base_dir = config["dataset_base_dir"]

    train(config)