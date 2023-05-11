from blender_data_taichi import *
import taichi as ti
import taichi.math as tim
import numpy as np
from pathlib import Path
import skimage
import matplotlib.pyplot as plt

def colormap_easy(arr, colormap=plt.cm.gray, vmin=None, vmax=None):

    if vmin is None:
        vmin = np.min(arr)
    if vmax is None:
        vmax = np.max(arr)

    arr = (arr - vmin) / (vmax - vmin)
    return (colormap(arr)[...,:3] * 255).astype(np.uint8)

def tonemap_reinhard(x):
    return x / (1+x)

def save_as_images():

    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True, parents=True)
    total_irradiances_np = total_irradiances.to_numpy()
    
    prob = 1/total_irradiances_np
    prob[prob > 1000] = 1000
    # Save
    print('Saving...')
    for cam in range(NUM_FRAMES):

        val = prob[cam,...]
        skimage.io.imsave(output_dir / f'{cam+1}_prob.png', colormap_easy(val, vmin=0, vmax=np.median(val)*8, colormap=plt.cm.magma))
        val = np.log(val)
        skimage.io.imsave(output_dir / f'{cam+1}_prob_log.png', colormap_easy(val, vmin=np.median(val)-2, vmax=np.median(val)+2, colormap=plt.cm.magma))

        val = total_irradiances_np[cam,...]
        np.save(output_dir / f'{cam+1}_irr.npy', val)
        output_dir.mkdir(exist_ok=True, parents=True)
        skimage.io.imsave(output_dir / f'{cam+1}_irr_turbo.png', colormap_easy(val, vmin=0, vmax=np.median(val)*8, colormap=plt.cm.turbo))
        skimage.io.imsave(output_dir / f'{cam+1}_irr.png', colormap_easy(tonemap_reinhard(val), vmin=0, vmax=None))


data_dir="blender_data"
load_all_data(data_dir)
print('Image2World...')
image2world()
print('Irradiance...')
compute_radiance()
print('Saving...')
np.save("irr.npy", total_irradiances.to_numpy()) 
print('Making images...')
save_as_images()
print('Done!')