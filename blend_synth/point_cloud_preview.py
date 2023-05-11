import numpy as np
import open3d as o3d
from tqdm import tqdm
import blender_data

def combine_preview():
    # Merges together a few frames into a single 3D point cloud
    # using inverse projection methods

    xyzs = []
    rgbs = []
    
    start_frame = 1
    end_frame = 4
    for i in tqdm(list(range(start_frame,end_frame+1))):
        
        intr_matrix, extr_matrix, rgb, depth, xyz, normals_world = blender_data.load_frame(i)
        xyzs.append(xyz.reshape((1024*1024, 3)))
        rgbs.append(rgb.reshape((1024*1024, 3)))
        
    xyz = np.concatenate(xyzs, axis=0)
    rgb = np.concatenate(rgbs, axis=0)
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[::5])
    pcd.colors = o3d.utility.Vector3dVector(rgb[::5]/255)
    o3d.io.write_point_cloud(f'preview.pcd', pcd)
    print('done')

if __name__ == '__main__':
    combine_preview()