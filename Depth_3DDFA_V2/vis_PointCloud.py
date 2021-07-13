import numpy as np
from open3d import *    

def main():
    ply_path = '/media/thimabru/ssd/Perse/3DDFA_V2/examples/results/emma_ply.ply'
    cloud = read_point_cloud(ply_path) # Read the point cloud
    print(cloud)
    draw_geometries([cloud]) # Visualize the point cloud     

if __name__ == "__main__":
    main()