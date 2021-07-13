import numpy as np
from DFA3D_V2 import DFA3D
import cv2
from PIL import Image
from open3d import read_point_cloud, draw_geometries
import random

def visualization():
    dfa3d_path = 'Depth_3DDFA/models/phase1_wpdc_vdc.pth.tar'

    dfa3d = DFA3D(dfa3d_path)

    vertices_path = '/home/fischer/Desktop/Fischer/Cyberlabs/Research/anti-spoofing/3dpc-net/data/Train/1_2_11_1_33.npy'
    vertices = np.load(vertices_path)

    print(vertices.shape)

    dfa3d.save_ply(vertices, 'data/test_downsampled2.ply')

    # Read the ply file and plot the 3D Point Cloud       
    ply_path = 'data/test_downsampled2.ply'
    cloud = read_point_cloud(ply_path) # Read the point cloud
    print(cloud)
    draw_geometries([cloud]) # Visualize the point cloud
#========================================================
def rot_pointcloud(pointcloud,image):  # Random rotation of a PointCloud in Z axis
        rotated_data = np.zeros(pointcloud.shape, dtype=np.float32)
        pointcloud= np.transpose(pointcloud)
        #for k in range(pointcloud.shape[0]): 
        rotation_angle = np.random.uniform(low=-1.0,high=1.0) * np.pi/6
        cosval = np.cos(rotation_angle) 
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
                                    
        print(rotation_angle*(180/np.pi))
        rotated_img=image.rotate(rotation_angle*(180/np.pi))
        rotated_data = np.dot(pointcloud, rotation_matrix)
        rotated_data=np.transpose(rotated_data)

        return rotated_img , rotated_data
#========================================================
def visualize_rotation():
    dfa3d_path = 'Depth_3DDFA/models/phase1_wpdc_vdc.pth.tar'

    dfa3d = DFA3D(dfa3d_path)

    #vertices_path = '/home/fischer/Desktop/Fischer/Cyberlabs/Research/anti-spoofing/3dpc-net/data/Train/1_2_11_1_33.npy'
    #image_path = '/home/fischer/Desktop/Fischer/Cyberlabs/Research/anti-spoofing/3dpc-net/data/Train/1_2_11_1_33.jpg'
    
    image_path='/home/fischer/Desktop/Fischer/Cyberlabs/Research/anti-spoofing/3dpc-net/data/problem_img/5_2_12_3_148.jpg'
    vertices_path='/home/fischer/Desktop/Fischer/Cyberlabs/Research/anti-spoofing/3dpc-net/data/problem_img/5_2_12_3_148.npy'

    image=Image.open(image_path)
    width,height=image.size
    vertices = np.load(vertices_path)
    rot_img, rot_vertices=rot_pointcloud(vertices,image)


    print(vertices.shape)
    print(rot_vertices.shape)


    rot_img.show()
    
    dfa3d.save_ply([rot_vertices],height,'data/test_downsampled_rot.ply')
    #dfa3d.save_ply([vertices],height,'data/test_downsampled2.ply')

    # Read the ply file and plot the 3D Point Cloud       
    ply_path = 'data/test_downsampled2.ply'
    ply_path2 = 'data/test_downsampled_rot.ply'
    cloud = read_point_cloud(ply_path) # Read the point cloud
    cloud2 = read_point_cloud(ply_path2) # Read the point cloud
    #print(cloud)
    draw_geometries([cloud2]) # Visualize the point cloud
#========================================================================
def vis_uniform_crop():
    #img_path= '/home/fischer/Desktop/Fischer/Cyberlabs/Research/anti-spoofing/3dpc-net/data/Train/1_2_11_1_33.jpg'
    img_path='/home/fischer/Desktop/Fischer/Cyberlabs/Research/anti-spoofing/3dpc-net/data/problem_img/5_2_12_3_148.jpg'
    #img_path = '/media/thimabru/ssd/Perse/3dpc-net/data/OULU_labels/Protocol_3/Division_2/Train/1_1_01_1_4.jpg'
    img = Image.open(img_path)
    width, img_height = img.size
    
    #vertices_path = '/home/fischer/Desktop/Fischer/Cyberlabs/Research/anti-spoofing/3dpc-net/data/Train/1_2_11_1_33.npy'
    vertices_path = '/home/fischer/Desktop/Fischer/Cyberlabs/Research/anti-spoofing/3dpc-net/data/problem_img/5_2_12_3_148.npy'
    #vertices_path = '/media/thimabru/ssd/Perse/3dpc-net/data/OULU_labels/Protocol_3/Division_2/Train/1_1_01_1_4.npy'
    vertices = np.load(vertices_path)

    print(f"Vertices shape: {vertices.shape}")
    
    dfa3d = DFA3D(gpu=False, onnx=False)

    dfa3d.save_ply([vertices], img_height, 'data/test_UnifCrop.ply')

    # Read the ply file and plot the 3D Point Cloud       
    ply_path = 'data/test_UnifCrop.ply'
    cloud1 = read_point_cloud(ply_path) # Read the point cloud
    # # print(cloud)
    # draw_geometries([cloud1])
    
    print(f'Image original size: {img.size}')
    
    points = vertices
    
    """
    # X axis
    iw = img.size[0]
    print(points.shape)
    # tw = points.shape[1]
    #tw = iw - 300
    tw=224
    lx = points[0, :].max() # 0 for x axis
    sx = points[0, :].min()
    dx = lx - sx
    # Como obter tw ??? A target image seria a imagem cropada? Se sim tw seria o width que gostariamos q o crop fosse ?
    print(iw - tw)
    rx = random.randint(0, iw - tw)
    sx_crop = sx + (rx/iw * dx)
    lx_crop = lx + ((rx + tw)/iw * dx)
    
    # Y axis
    ih = img.size[1]
    print(points.shape)
    # tw = points.shape[1]
    #th = ih - 300
    th=224
    ly = points[1, :].max() # 1 for y axis
    sy = points[1, :].min()
    dy = ly - sy
    # Como obter tw ??? A target image seria a imagem cropada? Se sim tw seria o width que gostariamos q o crop fosse ?
    print(ih - th)
    ry = random.randint(0, ih - th)
    sy_crop = sy + (ry/ih * dy)
    ly_crop = lx + ((ry + th)/ih * dy)

    print(f"dx : {dx}")
    print(f"iw: {iw}")
    print(f"dy : {dy}")
    print(f"ih: {ih}")
    """

    target_x=220
    target_y=220

    lx = points[0, :].max() # 0 for x axis
    sx = points[0, :].min()

    ly = points[1, :].max() # 1 for y axis
    sy = points[1, :].min()

    lz = points[2, :].max() # 0 for z axis
    sz = points[2, :].min()

    dx=random.randint(20,target_x)
    dy=random.randint(20,target_y)

    sx_crop = sx + dx
    lx_crop = lx - dx
    sy_crop = sy + dy
    ly_crop = ly - dy

    #limits
    percentage=0.85
    iw = img.size[0]
    ih = img.size[1]
    limit_x= iw*percentage
    limit_y= ih*percentage

    if(ly_crop < limit_y ): ly_crop=limit_y
    if(lx_crop < limit_x ): lx_crop=limit_x



    # new_x = random.randint(sx_crop, lx_crop)
    # new_y = random.randint(sy_crop, ly_crop)

    
    
    # new_points = points.copy()
    new_points = []
    # (3, 10k)

    print(f"Limites no eixo X: Min {sx}, Max {lx}")
    print(f"Limites no eixo Y: Min {sy}, Max {ly}")
    print()
    print(f"Limites Crop no eixo X: Min {sx_crop}, Max {lx_crop}")
    print(f"Limites Crop no eixo Y: Min {sy_crop}, Max {ly_crop}")
    print(f"Limites Crop no eixo Z: Min {sz}, Max {lz}")

    #sx_crop=400.0
    #sy_crop=300.0
    #lx_crop=750.0
    #ly_crop=800.0
    lz_crop=250 #250
    sz_crop=10 #50
    croped_points=0
    for i in range(points.shape[1]):
        point = points[:, i]
        # print(point)
        # print(point[0])
        # print(point.shape)

        if abs(point[2]) < abs(lz_crop) and abs(point[2]) > abs(sz_crop):
            if point[0] < lx_crop and point[0] > sx_crop:
                if point[1] < ly_crop and point[1] > sy_crop:
                    new_points.append(point)
                    continue
        croped_points+=1            
            
    # for axis in range(points.shape[0]):
    #     for point in range(points.shape[1]):
    #         if points[axis, point] < lx_crop and points[axis, point] > sx_crop:
    #         # else:
    #         #     new_points[axis, point] = -1
    new_points_npy = np.array(new_points)
    print(f'New points shape: {new_points_npy.shape}')
    print(f'Croped_points: {croped_points}')
    
    img_npy = np.array(img)
    cropped_img = img_npy[int(sy_crop):int(ly_crop), int(sx_crop):int(ly_crop)]
    cv2.imwrite('data/Img_UnifCrop_cropped.jpg', cropped_img)
    print(f'Cropped img shape: {cropped_img.shape}')
    img_height2, width, _ = cropped_img.shape
    
    dfa3d.save_ply([np.transpose(new_points_npy)], img_height2, 'data/test_UnifCrop_Cropped.ply')

    # Read the ply file and plot the 3D Point Cloud       
    ply_path = 'data/test_UnifCrop_Cropped.ply'
    cloud2 = read_point_cloud(ply_path) # Read the point cloud
    # print(cloud)
    draw_geometries([cloud2])
    draw_geometries([cloud1])
    
    # cropped_img = img.crop((left, top, right, bottom))
    return new_points_npy, cropped_img

if __name__=='__main__':
    #visualization()
    visualize_rotation()
    #vis_uniform_crop()