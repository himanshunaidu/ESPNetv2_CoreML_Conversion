"""
This script converts a ESPNetv2 PyTorch model to CoreML format for semantic segmentation.
Currently, it uses the old TorchScript method for conversion.
It is recommended to shift to the torch.export method for better performance.
"""
import argparse
import os.path as osp
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import functional as F
import torchvision
import json
import cv2
from PIL import Image
from print_utils import *

from espnetv2 import espnetv2_seg

import coremltools as ct

class ToTensor(object):
    '''
    mean and std should be of the channel order 'bgr'
    '''
    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, rgb_img, label_img=None):
        rgb_img = F.to_tensor(rgb_img) # convert to tensor (values between 0 and 1)
        rgb_img = F.normalize(rgb_img, self.mean, self.std) # normalize the tensor
        label_img = torch.LongTensor(np.array(label_img).astype(np.int64))
        return rgb_img, label_img

# Normalization PARAMETERS for the IMAGENET dataset
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class WrappedESPNetv2(nn.Module):
    def __init__(self, args):
        super(WrappedESPNetv2, self).__init__()
        self.model = espnetv2_seg(args=args)
        self.model.load_state_dict(torch.load(args.weight_path, map_location=torch.device('cpu')), strict=False)
        self.model.eval()

    def forward(self, x):
        res = self.model(x)
        # print('res shape:', res.shape)
        out = torch.argmax(res, dim=1, keepdim=True).float()
        # print('out shape:', out.shape)
        # out = out.float() / 255
        return out
    
if __name__ == '__main__':
    segmentation_datasets = ['pascal', 'city', 'edge_mapping', 'coco_stuff']

    parser = argparse.ArgumentParser()
    # general details
    parser.add_argument('--weight-path', default='', help='Pretrained weights directory.') # model/semantic_segmentation/model_zoo/espnetv2/espnetv2_s_2.0_city_512x256.pth
    parser.add_argument('--s', default=2.0, type=float, help='scale')
    # dataset details
    parser.add_argument('--dataset', default='city', choices=segmentation_datasets, help='Dataset name')
    # input details
    parser.add_argument('--im-size', type=int, nargs="+", default=[512, 256], help='Image size for testing (W x H)')
    parser.add_argument('--channels', default=3, type=int, help='Input channels')
    parser.add_argument('--num-classes', default=1000, type=int,
                        help='ImageNet classes. Required for loading the base network')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--outpath', dest='out_pth', type=str,
            default='./model_zoo/')
    parser.add_argument('--img-path', dest='img_path', type=str, default='./data/test.jpg',)
    args = parser.parse_args()

    args.weights = ''

    if args.dataset == 'city':
        CITYSCAPE_CLASS_LIST = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
                        'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                        'bicycle', 'background']
        seg_classes = len(CITYSCAPE_CLASS_LIST)
    else:
        print_error_message('{} dataset not yet supported'.format(args.dataset))
        exit(-1)
    args.classes = seg_classes

    # Prepare data
    to_tensor = ToTensor(
        # mean=(0.3257, 0.3690, 0.3223), # city, rgb
        # std=(0.2112, 0.2148, 0.2115),
        mean=(0.0, 0.0, 0.0), # placeholder
        std=(1.0, 1.0, 1.0),
    )
    scale = 1/(0.226*255.0)
    bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]
    print('Loading image:', args.img_path)
    im = cv2.imread(args.img_path)#[:, :, ::-1]
    # Resize
    im = cv2.resize(im, args.im_size)
    empty_label = np.zeros(im.shape[:2], dtype=np.int64)
    im, label = to_tensor(rgb_img=im, label_img=empty_label)
    im = im.unsqueeze(0)
    
    # Prepare model
    torch_model = WrappedESPNetv2(args=args)
    torch_model = torch_model.to('cpu')
    torch_model.eval()
    # torch_model(im)
    # torch_model = torch_model.to('mps')
    # exit()
    traced_model = torch.jit.trace(torch_model, im)
    # scripted_model = torch.jit.script(torch_model)

    ml_model = ct.convert(
        traced_model,
        # scripted_model,
        inputs=[ct.ImageType(name="input", shape=im.shape, scale=scale, bias=bias)],
        outputs=[ct.ImageType(name="output", color_layout=ct.colorlayout.GRAYSCALE)],
        # compute_precision=ct.precision.FLOAT16
        # minimum_deployment_target=ct.target.iOS13,
        # compute_units=ct.ComputeUnit.CPU_AND_GPU
    )

    ml_model_path = osp.join(args.out_pth, 'espnetv2_{}_{}_{}_{}.mlpackage'.format(args.dataset, args.num_classes, args.im_size[0], args.im_size[1]))
    ml_model.save(ml_model_path)
    print(f"Saved the model to {ml_model_path}")