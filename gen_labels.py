import argparse
from DFA3D_V2 import DFA3D
from retinaface.retinaface import RetinaFace
import os
import cv2
# import dlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
# import threading as th

'''
    Generate Point Cloud label for a specific subsset (Train, Dev, Test)
'''
def downsample(vertices, n_samples=10000):
    '''
        Downsample point cloud vertices randomly
    '''
    vertices_df = pd.DataFrame(vertices.transpose((1, 0)))

    vertices_downsampled = vertices_df.sample(n=n_samples)

    vertices_downsampled = np.array(vertices_downsampled).transpose((1, 0))
    return vertices_downsampled

parser = argparse.ArgumentParser()
parser.add_argument("-dp", "--dataset_path", type=str,
                    default='/media/thimabru/ssd/Key_DATASETS/OULU/output/train_images',
                    help="Dataset path")
parser.add_argument("--dfa3d_path", type=str, 
                    default='Depth_3DDFA/models/phase1_wpdc_vdc.pth.tar',
                    help="Path to 3DDFA model's path")
parser.add_argument("--dlib_path", type=str, 
                    default='Depth_3DDFA/models/shape_predictor_68_face_landmarks.dat',
                    help="Path to Dlib model's path")
parser.add_argument("-sp", "--save_path", type=str, required=True,
                    help="Dataset path")
parser.add_argument("--protocol", type=str, required=True,
                    help="OULU protocol wanted to preprocess")
parser.add_argument("--division", type=str,
                    help="OULU division wanted to preprocess if needed \
                    (Applied only to protocol 3 and 4)")
parser.add_argument("--subset", type=str, required=True,
                    help="Select the subset folder to preprocess")
parser.add_argument("--onnx", action='store_true',
                    help="Choose whether DFA3D model use onnx to speed inference")
parser.add_argument("--gpu", action='store_true',
                    help="(Available only if onnx if false) Choose whether use gpu in DFA3D. Otherwise, uses cpu.")

args = parser.parse_args()

if args.protocol == '3' or args.protocol == '4':
    if not args.division:
        print('Please choose a division to preprocess')
        sys.exit(1)

if args.division:
    protocol_div_path = os.path.join(f'Protocol_{args.protocol}', f'Division_{args.division}')
else:
    protocol_div_path = f'Protocol_{args.protocol}'

# Load 3DDFA
dfa3d = DFA3D(args.gpu, args.onnx)

retina = RetinaFace('retinaface/mnet.25', 0, 0, 'net3')

# for label in ['Live', 'Print_Attack', 'Video_Replay_Attack']:
for label in ['Print_Attack', 'Video_Replay_Attack']:
    label_path = os.path.join(args.dataset_path, protocol_div_path,  args.subset, label)

    frames = os.listdir(label_path)

    for frame in tqdm(frames, desc=f"{args.subset}/{label}"):
        frame_path = os.path.join(label_path, frame)
        img = cv2.imread(frame_path)

        boxes, _ = retina.detect(img, threshold=0.5)

        if len(boxes) == 0:
            print(f"Didn't find any face for {frame_path}" )
            continue
        else:
            scores = boxes[:, -1]
            bbs = boxes[:, :-1]
            i = np.argmax(scores)
            best_score = scores[i]
            best_box = bbs[i]

        # img_crop = dfa3d.crop_face(img, [best_box])
        param_lst, roi_box_lst = dfa3d(img, [best_box])
        
        # img_crop = dfa3d.img_crop
    
        ver_lst = dfa3d.model.recon_vers(param_lst, roi_box_lst, dense_flag=True)
        
        vertices = ver_lst[0]

        vertices_downsampled = downsample(vertices)

        if label == 'Print_Attack' or label == 'Video_Replay_Attack':
            vertices_downsampled[2, :] = 0

        frame_id = frame.split('.')[0]
        save_path = os.path.join(args.save_path, protocol_div_path, args.subset)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save cropped image
        left, top, right, bottom = best_box # left, top, right, bottom = x1, y1, x2, y2
        img_crop = img[int(top):int(bottom), int(left):int(right)]
        cv2.imwrite(os.path.join(save_path, frame), img_crop)

        dfa3d.save_vertices(vertices_downsampled, os.path.join(save_path, frame_id + '.npy'))

        # dfa3d.save_ply(vertices_downsampled, os.path.join(save_path, frame_number + '.ply'))
