import torch
import numpy as np
import cv2
from collections import OrderedDict
import os
import math
import glob
import yaml
import scipy.io as sio
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import joblib


def get_inputs(file_addr):
    file_format = file_addr.split('.')[-1]
    if file_format == 'mat':
        return sio.loadmat(file_addr, verify_compressed_data_integrity=False)['uv']
    elif file_format == 'npy':
        return np.load(file_addr)
    else:
        return cv2.imread(file_addr)


def img_tensor2numpy(img):
    # mutual transformation between ndarray-like imgs and Tensor-like images
    # both intensity and rgb images are represented by 3-dim data
    if isinstance(img, np.ndarray):
        return torch.from_numpy(np.transpose(img, [2, 0, 1]))
    else:
        return np.transpose(img, [1, 2, 0]).numpy()


def img_batch_tensor2numpy(img_batch):
    # both intensity and rgb image batch are represented by 4-dim data
    if isinstance(img_batch, np.ndarray):
        if len(img_batch.shape) == 4:
            return torch.from_numpy(np.transpose(img_batch, [0, 3, 1, 2]))
        else:
            return torch.from_numpy(np.transpose(img_batch, [0, 1, 4, 2, 3]))
    else:
        if len(img_batch.numpy().shape) == 4:
            return np.transpose(img_batch, [0, 2, 3, 1]).numpy()
        else:
            return np.transpose(img_batch, [0, 1, 3, 4, 2]).numpy()


transform = transforms.Compose([
    transforms.ToTensor(),
])

class Photovoltaic_dataset(Dataset):
    def __init__(self, config, data_dir, transform=transform, spatial_size=256, mode="train"):
        super(Photovoltaic_dataset, self).__init__()

        self.data_dir = data_dir
        self.spatial_size = spatial_size
        self.img_mask = np.load(os.path.join(data_dir, 'imgs_mask.npy'), allow_pickle=True)
        self.img_thermal = np.load(os.path.join(data_dir, 'imgs_temp.npy'), allow_pickle=True)
        self.img_check = np.load(os.path.join(data_dir, 'imgs_check.npy'), allow_pickle=True)
        if mode == "train":
            self.img_mask = self.img_mask[:800]
            self.img_thermal = self.img_thermal[:800]
            self.img_check = self.img_check[:800]
        else:
            self.img_mask = self.img_mask[800:]
            self.img_thermal = self.img_thermal[800:]
            self.img_check = self.img_check[800:]

        self.img_thermal = cv2.normalize(self.img_thermal, None, 0, 255, cv2.NORM_MINMAX)

        self.transform = transform
    
    def __len__(self):
        return len(self.img_check)
    
    def resize(self, sample):
        return np.array(cv2.resize(sample, (self.spatial_size, self.spatial_size)))
    
    def __getitem__(self, indice):
        thermal_img = self.img_thermal[indice] / 255.0
        mask_img = self.img_mask[indice]
        check_img = self.img_check[indice]

        thermal_img = self.resize(thermal_img)[:,:,None]
        mask_img = self.resize(mask_img)[None,:,:]

        return self.transform(thermal_img), mask_img.astype(np.float32), check_img

        # appearance = self.chunked_samples_appearance[indice] / 255.0
        # flow = self.chunked_samples_motion[indice]
        # pose = self.chunked_samples_pose[indice] / 255.0
        # bbox = self.chunked_samples_bbox[indice]
        # pred_frame = self.chunked_samples_pred_frame[indice]

        # # flow * root(area)
        # flow = flow / np.sqrt(self.scale(bbox)) * 50.0

        # if self.spatial_size != 32:
        #     appearance = self.resize(appearance)
        #     pose = self.resize(pose)
        #     flow = self.resize(flow)
        
        # # random mask
        # if self.mode=="train" and torch.rand(1)[0].item() < self.mask_prob:
        #     appearance[-2, :, :, :] = torch.rand(appearance[-2, :, :, :].shape).numpy() 

        # x = appearance
        # x = np.transpose(x, [1, 2, 0, 3]).astype(np.float32)
        # x = np.reshape(x, (x.shape[0], x.shape[1], -1))

        # y = pose if not self.last_pose else pose[-1:] 
        # y = np.transpose(y, [1, 2, 0, 3]).astype(np.float32)
        # y = np.reshape(y, (y.shape[0], y.shape[1], -1))

        # z = flow if not self.last_flow else flow[-1:] 
        # z = np.transpose(z, [1, 2, 0, 3]).astype(np.float32)
        # z = np.reshape(z, (z.shape[0], z.shape[1], -1))


        # return self.transform(x), self.transform(y), self.transform(z), \
        #        bbox.astype(np.float32), pred_frame, indice

class Chunked_sample_dataset(Dataset):
    def __init__(self, config, chunk_file, last_flow=False, last_pose=False, transform=transform, context=4, spatial_size=32, mode="train"):
        super(Chunked_sample_dataset, self).__init__()

        self.mask_prob = 0.0
        self.mode = mode

        self.chunk_file = chunk_file
        self.last_flow = last_flow
        self.last_pose = last_pose
        self.context = context
        self.spatial_size=spatial_size
        # dict(sample_id=[], appearance=[], motion=[], bbox=[], pred_frame=[], pose=[])
        self.chunked_samples = joblib.load(self.chunk_file)

        self.chunked_samples_appearance = self.chunked_samples["appearance"]
        self.chunked_samples_motion = self.chunked_samples["motion"]
        self.chunked_samples_pose = self.chunked_samples["pose"]
        self.chunked_samples_bbox = self.chunked_samples["bbox"]
        self.chunked_samples_pred_frame = self.chunked_samples["pred_frame"]
        self.chunked_samples_id = self.chunked_samples["sample_id"]

        self.transform = transform

    def __len__(self):
        return len(self.chunked_samples_id)
    
    def resize(self, samples):
        return np.array([cv2.resize(sample, (self.spatial_size, self.spatial_size)) for sample in samples])
    
    def get_area(self, box):
        return (box[3]-box[1])*(box[2]-box[0])
    
    def scale(self, box):
        area = self.get_area(box)
        if box[2]-box[0] > box[3]-box[1]:
            scale = min(area, 50*50)
        else:
            scale = area
        return scale


    def __getitem__(self, indice):
        appearance = self.chunked_samples_appearance[indice] / 255.0
        flow = self.chunked_samples_motion[indice]
        pose = self.chunked_samples_pose[indice] / 255.0
        bbox = self.chunked_samples_bbox[indice]
        pred_frame = self.chunked_samples_pred_frame[indice]

        # flow * root(area)
        flow = flow / np.sqrt(self.scale(bbox)) * 50.0

        if self.spatial_size != 32:
            appearance = self.resize(appearance)
            pose = self.resize(pose)
            flow = self.resize(flow)
        
        # random mask
        if self.mode=="train" and torch.rand(1)[0].item() < self.mask_prob:
            appearance[-2, :, :, :] = torch.rand(appearance[-2, :, :, :].shape).numpy() 

        x = appearance
        x = np.transpose(x, [1, 2, 0, 3]).astype(np.float32)
        x = np.reshape(x, (x.shape[0], x.shape[1], -1))

        y = pose if not self.last_pose else pose[-1:] 
        y = np.transpose(y, [1, 2, 0, 3]).astype(np.float32)
        y = np.reshape(y, (y.shape[0], y.shape[1], -1))

        z = flow if not self.last_flow else flow[-1:] 
        z = np.transpose(z, [1, 2, 0, 3]).astype(np.float32)
        z = np.reshape(z, (z.shape[0], z.shape[1], -1))


        return self.transform(x), self.transform(y), self.transform(z), \
               bbox.astype(np.float32), pred_frame, indice


if __name__ == "__main__":
    config = yaml.safe_load(open("./cfgs/cfg.yaml"))
    data_dir = os.path.join(config['dataset_base_dir'], config['dataset_name'])
    batch_size = config['batchsize']
    num_workers = config['num_workers']


    dataset = Photovoltaic_dataset(config, data_dir)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    for idx, train_data in tqdm(enumerate(dataloader), total=len(dataloader)):
        x, y, check = train_data
        print(x.shape, y.shape, check.shape)