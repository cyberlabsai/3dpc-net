import numpy as np
from DFA3D_V2 import DFA3D
from open3d import read_point_cloud, draw_geometries
from PIL import Image
import torch

def Normalize_label(points):
    axis_distances = []
    medium_point = np.zeros(points.shape)
    # print('here')
    # print(medium_point.shape)
    # print(points.shape[0])
    # Axis: 0->x; 1->y; 2->z 
    # points.shape = (3, 2500)
    for axis in range(points.shape[0]):
        max_ax = points[axis, :].max()
        min_ax = points[axis, :].min()
        axis_distances.append(max_ax - min_ax)
        medium_point[axis, :] = (max_ax + min_ax)/2

    max_dist = max(axis_distances)
    # Formula got from 3DPC-Net paper
    points_normalized = (points - medium_point)/(2*max_dist) + 0.5
    return points_normalized

# Live
# img_path = '/media/thimabru/ssd/Perse/3dpc-net/data/OULU_labels/Protocol_1/Train/1_1_01_1_3.jpg'
# vertices_path = '/media/thimabru/ssd/Perse/3dpc-net/data/OULU_labels/Protocol_1/Train/1_1_01_1_3.npy'

# Spoof
img_path = '/media/thimabru/ssd/Perse/3dpc-net/data/OULU_labels/Protocol_1/Train/1_1_01_2_0.jpg'
vertices_path = '/media/thimabru/ssd/Perse/3dpc-net/data/OULU_labels/Protocol_1/Train/1_1_01_2_0.npy'

# Load image
img = Image.open(img_path)
width, img_height = img.size

# Load vertices
vertices = np.load(vertices_path)
vertices = Normalize_label(vertices)
print(f"Vertices shape: {vertices.shape}")
print(vertices.max(axis=1))
print(vertices.min(axis=1))
print(vertices.max(axis=1) - vertices.min(axis=1))
# print(vertices.max() - vertices.min())
# print(vertices.max() - vertices.min())

print(vertices[2, :].shape)
score = np.mean(vertices[2, :], 0)
print(score)
print(f"Score shape: {score.shape}")
print('-----------------------------------')

# batch_vertices = np.expand_dims(vertices, axis=0)
# print(batch_vertices.shape)
torch_vertices = np.stack([vertices, vertices], axis=0)
print(torch_vertices.shape)
score = torch.mean(torch.tensor(torch_vertices), 2)[:, 2]
print(score)
print(f"Score shape: {score.shape}")

print("-------------------------------------")
torch_vertices = np.stack([vertices, vertices], axis=0)
print(torch_vertices.shape)
print(torch_vertices[:, 2, :].shape)
score = torch.mean(torch.tensor(torch_vertices[:, 2, :]), 1)# [:, 1]
print(score)
print(f"Score shape: {score.shape}")

# Load model
dfa3d = DFA3D(gpu=False)

dfa3d.save_ply([vertices], img_height, 'data/vertices_inspection.ply')

# Read the ply file and plot the 3D Point Cloud
ply_path = 'data/vertices_inspection.ply'   
# ply_path = 'data/test_downsampled_v2.ply'
cloud = read_point_cloud(ply_path) # Read the point cloud
print(cloud)
draw_geometries([cloud]) # Visualize the point cloud