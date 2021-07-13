# coding: utf-8

__author__ = 'cleardusk'

import os.path as osp
import time
import numpy as np
import cv2
import torch
from torchvision.transforms import Compose
import torch.backends.cudnn as cudnn
import sys
sys.path.append("/media/thimabru/ssd/Perse/3dpc-net/Depth_3DDFA_V2") # Adds higher directory to python modules path.

from Depth_3DDFA_V2 import models
# import models
# from bfm import BFMModel
from Depth_3DDFA_V2.bfm import BFMModel
from .utils.io import _load
from .utils.functions import (
    crop_img, parse_roi_box_from_bbox, parse_roi_box_from_landmark,
)
from .utils.tddfa_util import (
    load_model, _parse_param, similar_transform,
    ToTensorGjz, NormalizeGjz
)

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)


class TDDFA(object):
    """TDDFA: named Three-D Dense Face Alignment (TDDFA)"""

    def __init__(self, **kvs):
        torch.set_grad_enabled(False)

        # load BFM
        self.bfm = BFMModel(
            bfm_fp=kvs.get('bfm_fp', make_abs_path('./configs/bfm_noneck_v3.pkl')),
            shape_dim=kvs.get('shape_dim', 40),
            exp_dim=kvs.get('exp_dim', 10)
        )
        self.tri = self.bfm.tri

        # config
        self.gpu_mode = kvs.get('gpu_mode', False)
        self.gpu_id = kvs.get('gpu_id', 0)
        self.size = kvs.get('size', 120)

        param_mean_std_fp = kvs.get(
            'param_mean_std_fp', make_abs_path(f'configs/param_mean_std_62d_{self.size}x{self.size}.pkl')
        )

        # load model, default output is dimension with length 62 = 12(pose) + 40(shape) +10(expression)
        model = getattr(models, kvs.get('arch'))(
            num_classes=kvs.get('num_params', 62),
            widen_factor=kvs.get('widen_factor', 1),
            size=self.size,
            mode=kvs.get('mode', 'small')
        )
        print(make_abs_path(kvs.get('checkpoint_fp')))
        model = load_model(model, make_abs_path(kvs.get('checkpoint_fp')))

        if self.gpu_mode:
            cudnn.benchmark = True
            model = model.cuda(device=self.gpu_id)

        self.model = model
        self.model.eval()  # eval mode, fix BN

        # data normalization
        transform_normalize = NormalizeGjz(mean=127.5, std=128)
        transform_to_tensor = ToTensorGjz()
        transform = Compose([transform_to_tensor, transform_normalize])
        self.transform = transform

        # params normalization config
        r = _load(param_mean_std_fp)
        self.param_mean = r.get('mean')
        self.param_std = r.get('std')

        # print('param_mean and param_srd', self.param_mean, self.param_std)

    def __call__(self, img_ori, objs, **kvs):
        """The main call of TDDFA, given image and box / landmark, return 3DMM params and roi_box
        :param img_ori: the input image
        :param objs: the list of box or landmarks
        :param kvs: options
        :return: param list and roi_box list
        """
        # Crop image, forward to get the param
        param_lst = []
        roi_box_lst = []

        crop_policy = kvs.get('crop_policy', 'box')
        for obj in objs:
            if crop_policy == 'box':
                # by face box
                roi_box = parse_roi_box_from_bbox(obj)
            elif crop_policy == 'landmark':
                # by landmarks
                roi_box = parse_roi_box_from_landmark(obj)
            else:
                raise ValueError(f'Unknown crop policy {crop_policy}')

            roi_box_lst.append(roi_box)
            self.img_crop = crop_img(img_ori, roi_box)
            img = cv2.resize(self.img_crop, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)
            inp = self.transform(img).unsqueeze(0)

            if self.gpu_mode:
                inp = inp.cuda(device=self.gpu_id)

            # if kvs.get('timer_flag', False):
            if True:
                end = time.time()
                param = self.model(inp)
                elapse = f'Inference: {(time.time() - end) * 1000:.1f}ms'
                print(elapse)
            else:
                param = self.model(inp)

            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
            param = param * self.param_std + self.param_mean  # re-scale
            # print('output', param)
            param_lst.append(param)

        return param_lst, roi_box_lst

    def recon_vers(self, param_lst, roi_box_lst, **kvs):
        dense_flag = kvs.get('dense_flag', False)
        size = self.size

        ver_lst = []
        for param, roi_box in zip(param_lst, roi_box_lst):
            if dense_flag:
                R, offset, alpha_shp, alpha_exp = _parse_param(param)
                pts3d = R @ (self.bfm.u + self.bfm.w_shp @ alpha_shp + self.bfm.w_exp @ alpha_exp). \
                    reshape(3, -1, order='F') + offset
                pts3d = similar_transform(pts3d, roi_box, size)
            else:
                R, offset, alpha_shp, alpha_exp = _parse_param(param)
                pts3d = R @ (self.bfm.u_base + self.bfm.w_shp_base @ alpha_shp + self.bfm.w_exp_base @ alpha_exp). \
                    reshape(3, -1, order='F') + offset
                pts3d = similar_transform(pts3d, roi_box, size)

            ver_lst.append(pts3d)

        return ver_lst

# Test inference of DFA3D
# import yaml
# from FaceBoxes import FaceBoxes
# import sys
# from open3d import read_point_cloud, draw_geometries
# from .utils.serialization import ser_to_ply

if __name__=='__main__':
    cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    # if args.onnx:
    #     import os
    #     os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    #     os.environ['OMP_NUM_THREADS'] = '4'

    #     from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
    #     from TDDFA_ONNX import TDDFA_ONNX

    #     face_boxes = FaceBoxes_ONNX()
    #     tddfa = TDDFA_ONNX(**cfg)
    # else:
    gpu_mode = True
    tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
    face_boxes = FaceBoxes()

    # Given a still image path and load to BGR channel
    img_path = '/media/thimabru/ssd/Perse/3dpc-net/data/OULU_processed/Protocol_3/Division_2/Train/Live/1_1_01_1_4.jpg'
    img = cv2.imread(img_path)

    # Detect faces, get 3DMM params and roi boxes
    boxes = face_boxes(img)
    n = len(boxes)
    if n == 0:
        print(f'No face detected, exit')
        sys.exit(-1)
    print(f'Detect {n} faces')

    print(boxes)
    param_lst, roi_box_lst = tddfa(img, boxes)
    param_lst, roi_box_lst = tddfa(img, boxes)
    img_crop = tddfa.img_crop

    cv2.imwrite('../data/img_crop_v2.jpg', img_crop)

    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)

    ply_path = '../data/test_downsampled_v2.ply'

    print(f'Ver lst len: {len(ver_lst)}')
    ser_to_ply(ver_lst, tddfa.tri, height=img.shape[0], wfp=ply_path)
    print(ver_lst[0].shape)

    # ply_path = wfp
    cloud = read_point_cloud(ply_path) # Read the point cloud
    print(cloud)
    draw_geometries([cloud]) # Visualize the point cloud