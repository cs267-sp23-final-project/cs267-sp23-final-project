Download Blender data here:
https://drive.google.com/drive/folders/1UnuKK0_iMa-HLGhoEeY8YXnRqWUGjFFE?usp=sharing 

Unzip as follows:
```
+ blend_synth
    + blender_data
        + 1_data_depth.npz
        + 2_data_depth.npz
        + ...
    + blender_data.py
    + point_cloud_preview.py
    + ...
```

Run `point_cloud_preview.py` to generate a 3D point cloud from the RGBD data. This also shows how to use the included functions for transforming between RGBD and world-space XYZ. Use Open3D to view the point cloud.
