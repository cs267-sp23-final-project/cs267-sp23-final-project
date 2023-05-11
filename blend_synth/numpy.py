
import blender_data
import numpy as np
import argparse
import time

from pathlib import Path
import skimage.io
import matplotlib.pyplot as plt
import scipy

from tqdm import tqdm

# NUM_FRAMES = 2

def colormap_easy(arr, colormap=plt.cm.gray, vmin=None, vmax=None):

    if vmin is None:
        vmin = np.min(arr)
    if vmax is None:
        vmax = np.max(arr)

    arr = (arr - vmin) / (vmax - vmin)
    return (colormap(arr)[...,:3] * 255).astype(np.uint8)

def tonemap_reinhard(x):
    return x / (1+x)

def load_all_data():

    intr_matrices = []
    extr_matrices = []
    rgbs = []
    depths = []
    normals = []
    xyzs = []

    # Load all of the data
    print('Loading data...')
    for i in range(1,NUM_FRAMES+1):
        intr_matrix, extr_matrix, rgb, depth, xyz, normals_world = blender_data.load_frame(i, data_dir='/home/eliang/Documents/cs267_final_project/blender_data')
        intr_matrices.append(intr_matrix)
        extr_matrices.append(extr_matrix)
        rgbs.append(rgb)
        depths.append(depth)
        normals.append(normals_world)
        xyzs.append(xyz)

    intr_matrices = np.stack(intr_matrices)
    extr_matrices = np.stack(extr_matrices)
    rgbs = np.stack(rgbs)
    depths = np.stack(depths)[...,None]
    normals = np.stack(normals)
    xyzs = np.stack(xyzs)

    return intr_matrices, extr_matrices, rgbs, depths, normals, xyzs

def simple_downsample(arr, factor):
    return arr[:, ::factor,::factor]

def main():

    output_dir = Path('output')
    
    # Time data loading
    data_start = time.perf_counter()

    intr_matrices, extr_matrices, rgbs, depths, normals, xyzs = load_all_data()

    data_end = time.perf_counter()
    data_time = data_end - data_start

    if True:
        factor = 1
        intr_matrices[:,:2] /= factor
        rgbs = rgbs[:, ::factor,::factor]
        depths = depths[:, ::factor,::factor]
        normals = normals[:, ::factor,::factor]
        xyzs = xyzs[:, ::factor,::factor]

    camera_centers = extr_matrices[:,:3,3]

    width = rgbs.shape[1]

    num_cameras = len(intr_matrices)

    # Displacement
    #displacements = xyzs - camera_centers[:,None,None,:]

    # Camera distance
    #distances_sq = np.sum(np.square(displacements), axis=-1, keepdims=True)

    # TODO: This can be computed just using the camera matrices
    #distances = np.sqrt(distances_sq)
    #dir_to_point = displacements / distances

    # Normal dot prod
    #normal_dot_prod = np.sum(-dir_to_point * normals, axis=-1, keepdims=True)

    # Illuminate every image given only the single camera
    #single_irradiances = normal_dot_prod * 1/distances_sq

    # Measure total irradiance
    total_irradiances = np.zeros((num_cameras, width, width, 1))

    print('Running...')

    run_start = time.perf_counter()

    # Have every camera also illuminate every other 
    for cam_me in tqdm(list(range(num_cameras))):
        for cam_other in range(num_cameras):

            # Project onto the other camera [W, W, 3]
            points_on_other_xyd = blender_data.xyz_to_pixel(
                intr_matrices[cam_other], extr_matrices[cam_other], 
                xyzs[cam_me].reshape((width*width, 3))).reshape((width, width, 3))

            points_other_xy = points_on_other_xyd[...,:2]
            points_other_d = points_on_other_xyd[...,2,None]

            # Where to sample the other camera's depth map
            depth_samples_xy_int = np.round(points_other_xy).astype(int)

            # Check for out-of-bounds
            depth_samples_oob = np.logical_or(np.any(depth_samples_xy_int < 0, axis=-1), np.any(depth_samples_xy_int >= width, axis=-1))

            # Hack: clip for now
            samples_xy_clipped = np.clip(depth_samples_xy_int, 0, width - 1)
            samples_xy_clipped_vec = samples_xy_clipped.reshape((width*width, 2))
            occluder_other_d = depths[cam_other][samples_xy_clipped_vec[:,1],samples_xy_clipped_vec[:,0]].reshape((width, width, 1))

            # Check shadows.
            # If the occluder has a smaller depth than the sample, then it is in shade
            # Negative values are occluded, positive are not occluded
            compare_depth = (occluder_other_d - points_other_d)

            # Shadow bias
            visibility = np.logical_and(compare_depth > -0.01, points_other_d > 0)

            # Handle out-of-range
            visibility[depth_samples_oob] = 0

            # Handle 

            # Direction to the other camera
            displacement_other = xyzs[cam_me] - camera_centers[cam_other,None,None,:]
            distance_other_sq = np.sum(np.square(displacement_other), axis=-1, keepdims=True)
            distance_other = np.sqrt(distance_other_sq)
            direction_other = displacement_other / distance_other

            output_dir.mkdir(exist_ok=True, parents=True)
            # skimage.io.imsave(output_dir / f'{cam_me+1}_test.png', ((direction_other + 1) * (255/2)).astype(np.uint8))

            dot_prod = -direction_other * normals[cam_me]
            dot_prod = np.sum(dot_prod, axis=-1, keepdims=True)
            # skimage.io.imsave(output_dir / f'{cam_me+1}_test2.png', colormap_easy(dot_prod[...,0], vmin=-1, vmax=1, colormap=plt.cm.PiYG))
            # print(dot_prod.shape)
            cosine_law = np.maximum(dot_prod, 0)

            # skimage.io.imsave(output_dir / f'{cam_me+1}_test3.png', colormap_easy(cosine_law[...,0], vmin=-1, vmax=1, colormap=plt.cm.PiYG))

            total_irradiances[cam_me] += visibility * cosine_law * (1 / distance_other_sq)


    prob = 1/total_irradiances
    prob[prob > 1000] = 1000

    run_end = time.perf_counter()
    run_time = run_end - run_start

    # Log info
    with open(output_dir / '{}_times.txt'.format(NUM_FRAMES), 'w+') as f:
        f.write('Data time: {}\nRun time: {}'.format(data_time, run_time))
    return  # skip saving for now
    # Save
    print('Saving...')
    for cam in range(num_cameras):

        val = prob[cam,...,0]
        skimage.io.imsave(output_dir / f'{cam+1}_prob.png', colormap_easy(val, vmin=0, vmax=np.median(val)*8, colormap=plt.cm.magma))
        val = np.log(val)
        skimage.io.imsave(output_dir / f'{cam+1}_prob_log.png', colormap_easy(val, vmin=np.median(val)-2, vmax=np.median(val)+2, colormap=plt.cm.magma))

        val = total_irradiances[cam,...,0]
        np.save(output_dir / f'{cam+1}_irr.npy', val)
        output_dir.mkdir(exist_ok=True, parents=True)
        skimage.io.imsave(output_dir / f'{cam+1}_irr_turbo.png', colormap_easy(val, vmin=0, vmax=np.median(val)*8, colormap=plt.cm.turbo))
        skimage.io.imsave(output_dir / f'{cam+1}_irr.png', colormap_easy(tonemap_reinhard(val), vmin=0, vmax=None))

    return
    viz_heatmap(output_dir, intr_matrices, xyzs, prob, 'prob')
    viz_heatmap(output_dir, intr_matrices, xyzs, None, 'naive')

def viz_heatmap(output_dir, intr_matrices, xyzs, weights=None, suffix=''):

    num_cameras = len(intr_matrices)
    
    width = xyzs.shape[1]


    print('Making heatmap...')
    # Compute a heatmap of points
    res = 100
    xyzs_flat = xyzs.reshape(num_cameras * width * width, 3)
    min_x, min_y, min_z = np.floor(np.min(xyzs_flat, axis=0)).astype(int)
    max_x, max_y, max_z = np.ceil(np.max(xyzs_flat, axis=0)).astype(int)
    bin_edges = [
        np.linspace(min_x, max_x, res * (max_x - min_x)),
        np.linspace(min_y, max_y, res * (max_y - min_y)),
        np.linspace(min_z, max_z, res * (max_z - min_z)),
    ]

    if weights is not None:
        weights = weights.reshape(num_cameras * width * width)

    H, edges = np.histogramdd(xyzs_flat, bins=bin_edges, weights=weights)

    # Prefilter to avoid aliasing
    H = scipy.ndimage.gaussian_filter(H, sigma=0.02 * res, mode='constant')
    #H = scipy.ndimage.maximum_filter(H, size=res//20)

    def midpoints(x):
        return (x[:-1] + x[1:])/2

    interp = scipy.interpolate.RegularGridInterpolator((midpoints(x) for x in edges), H, bounds_error=False, fill_value=0)

    # Measure total heat
    heat = interp(xyzs)[...,None]

    # Save
    print('Saving...')
    for cam in range(num_cameras):

        output_dir.mkdir(exist_ok=True, parents=True)
        val = heat[cam,...,0]
        skimage.io.imsave(output_dir / f'{cam+1}_heat{suffix}.png', colormap_easy(val, vmin=0, vmax=np.median(val)*8, colormap=plt.cm.turbo))

    # print(H)
    # print(edges[0][:-1] - edges[0][1:])
    # print(edges[1])
    # print(edges[2])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='get number of input frames.')
    parser.add_argument('NUM_FRAMES', type=int, choices=range(1, 101))
    args = parser.parse_args()
    NUM_FRAMES = args.NUM_FRAMES
    main() 
