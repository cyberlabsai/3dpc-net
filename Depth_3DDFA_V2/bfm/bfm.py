# coding: utf-8

__author__ = 'cleardusk'

import sys
# sys.path.append("/media/thimabru/ssd/Perse/3dpc-net/Depth_3DDFA_V2/bfm") # Adds higher directory to python modules path.
# sys.path.append('..')

import os.path as osp
import numpy as np
# from utils.io import _load

import pickle

# ! Both functions copied and pasted from utils.io to handle python imports
def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]
    
def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        # ! print(fp)
        return pickle.load(open(fp, 'rb'))

# ! ----------------------------------------------------

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)


def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr


class BFMModel(object):
    def __init__(self, bfm_fp, shape_dim=40, exp_dim=10):
        # bfm = _load("/media/thimabru/ssd/Perse/3dpc-net/Depth_3DDFA_V2/" + bfm_fp)
        bfm = _load(make_abs_path("../" + bfm_fp))
        self.u = bfm.get('u').astype(np.float32)  # fix bug
        self.w_shp = bfm.get('w_shp').astype(np.float32)[..., :shape_dim]
        self.w_exp = bfm.get('w_exp').astype(np.float32)[..., :exp_dim]
        if osp.split(bfm_fp)[-1] == 'bfm_noneck_v3.pkl':
            self.tri = _load(make_abs_path('../configs/tri.pkl'))  # this tri/face is re-built for bfm_noneck_v3
            # self.tri = _load('../configs/tri.pkl')  # this tri/face is re-built for bfm_noneck_v3
        else:
            self.tri = bfm.get('tri')

        self.tri = _to_ctype(self.tri.T).astype(np.int32)
        self.keypoints = bfm.get('keypoints').astype(np.long)  # fix bug
        w = np.concatenate((self.w_shp, self.w_exp), axis=1)
        self.w_norm = np.linalg.norm(w, axis=0)

        self.u_base = self.u[self.keypoints].reshape(-1, 1)
        self.w_shp_base = self.w_shp[self.keypoints]
        self.w_exp_base = self.w_exp[self.keypoints]
