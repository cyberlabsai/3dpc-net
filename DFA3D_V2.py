from Depth_3DDFA_V2.TDDFA import TDDFA
from Depth_3DDFA_V2.utils.serialization import ser_to_ply
from open3d import read_point_cloud, draw_geometries
import cv2
import numpy as np
import pandas as pd
import yaml
import time
import sys
from retinaface.retinaface import RetinaFace


'''
    This is an abstraction of 3DDFA repository to make easier the generation of labels
    in 3DCPC-Net. The class instantiates the model, crop a face, does the inference 
    and save vertices.
'''

class DFA3D():
    def __init__(self, gpu=True, onnx=True):
        cfg = yaml.load(open('Depth_3DDFA_V2/configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)

        # Init FaceBoxes and TDDFA, recommend using onnx flag
        if onnx:
            print('Using ONNX model')
            import os
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['OMP_NUM_THREADS'] = '4'

            # from Depth_3DDFA_V2.FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
            from Depth_3DDFA_V2.TDDFA_ONNX import TDDFA_ONNX

            # self.face_boxes = FaceBoxes_ONNX()
            self.model = TDDFA_ONNX(**cfg)
        else:
            hardware = {True: 'GPU', False: 'CPU'}
            print(f'Not using ONNX. Using {hardware[gpu]}')
            self.model = TDDFA(gpu_mode=gpu, **cfg)
            # self.face_boxes = FaceBoxes()

        self.tri = self.model.tri

    def __call__(self, img, roi_box):

        vertices = self.model(img, roi_box)
        self.img_crop = self.model.img_crop
        self.img_shape0 = img.shape[0]
        return vertices

    def save_vertices(self, vertices, save_path):
        # print(f"Vertices shape: {vertices.shape}")
        np.save(save_path, vertices)

    def save_ply(self, vertices, image_height, save_path):
        ser_to_ply(vertices, self.tri, height=image_height, wfp=save_path)



# Test inference of DFA3D
if __name__=='__main__':
    # Load 3DDFA model
    dfa3d = DFA3D()

    # img_path = '/media/thimabru/ssd/Perse/3dpc-net/5_1_06_1_17.jpg'
    img_path = '/media/thimabru/ssd/Perse/3dpc-net/data/Thiago2.jpeg'
    img = cv2.imread(img_path)
    
    retina = RetinaFace('retinaface/mnet.25', 0, 0, 'net3')

    # Detect faces, get 3DMM params and roi boxes
    start = time.time()
    boxes, _ = retina.detect(img, threshold=0.5)
    end = time.time()
    print(f"Retina inference took: {end-start}")
    # print('[Here]')
    # print(bbs)
    
    print(boxes.shape)
    n = len(boxes)
    # n = len(bbs)
    
    if n == 0:
        print(f'No face detected, exit')
        sys.exit(-1)
    else:
        scores = boxes[:, -1]
        bbs = boxes[:, :-1]
        i = np.argmax(scores)
        best_score = scores[i]
        best_box = bbs[i]
        # print(best_score)
        # print(best_box)
        
    print(f'Detect {n} faces')

    # print(boxes)
    # print(boxes[0])
    # print([boxes[0][:-1]])
    # param_lst, roi_box_lst = dfa3d(img, [boxes[0][:-1]])
    param_lst, roi_box_lst = dfa3d(img, [best_box])

    # param_lst, roi_box_lst = dfa3d(img, boxes)
    
    img_crop = dfa3d.img_crop

    cv2.imwrite('data/img_crop_v2.jpg', img_crop)
    
    ver_lst = dfa3d.model.recon_vers(param_lst, roi_box_lst, dense_flag=True)
    
    vertices = ver_lst[0]

    print(f'Vertices shape: {vertices.shape}')
    print(vertices.transpose((1, 0)).shape)

    # Downsample vertices

    vertices_df = pd.DataFrame(vertices.transpose((1, 0)))

    print(vertices_df)

    vertices_downsampled = vertices_df.sample(n=10000)

    vertices_downsampled = np.array(vertices_downsampled).transpose((1, 0))
    print(vertices_downsampled)

    # 0 z axis for spoof cases
    # vertices_downsampled[2, :] = 0

    dfa3d.save_vertices(vertices, 'data/test_v2.npy')

    dfa3d.save_ply([vertices], 'data/test_v2.ply')

    dfa3d.save_ply([vertices_downsampled], 'data/test_downsampled_v2.ply')

    # Read the ply file and plot the 3D Point Cloud
    ply_path = 'data/test_v2.ply'   
    # ply_path = 'data/test_downsampled_v2.ply'
    cloud = read_point_cloud(ply_path) # Read the point cloud
    print(cloud)
    draw_geometries([cloud]) # Visualize the point cloud