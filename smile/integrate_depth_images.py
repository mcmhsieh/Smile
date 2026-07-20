"""
Integrate depth images into a TSDF volume for a set of synthetic view
poses based on each key frame's camera pose.
Fit a surface mesh to each integrated TSDF volume by optimising a loss function.

This is one stage of the processing pipeline for https://github.com/mcmhsieh/Smile

SPDX-FileCopyrightText: 2026 Mark Hsieh
SPDX-License-Identifier: MIT
"""

import sys
import builtins
import warnings
import time
import datetime
import pathlib
import shutil
import pickle
import shelve
import fractions
import collections
import itertools

import numpy as np
import scipy
import sklearn.cluster
import sklearn.mixture
import skimage
import networkx
import cv2
import open3d as o3d
import torch

import IPython
spyder_ide = IPython.get_ipython().__class__.__name__ == 'SpyderShell'

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
if not spyder_ide:
    # TODO: check whether the system display scaling setting needs to be taken into account
    plt.rcParams['figure.dpi'] = 80.0
    if hasattr(IPython.get_ipython(), 'run_line_magic'):
        IPython.get_ipython().run_line_magic('matplotlib', 'qt')

from image_filtering import rgb_to_gray
from weighting_functions import cauchy, gamma_softplus
from fig_paging import setup_new_fig_page, stash_fig_page

from pipeline_server import start_pipeline_server, post_to_pipeline_server, get_queue_from_pipeline_server


if __name__ == '__main__':

    torch.set_default_device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # %%

    working_subdir_config_path = pathlib.Path(r'../pipeline-workspace/working_subdir.txt')
    with open(working_subdir_config_path, 'r') as config_file:
        working_subdir = config_file.read().rstrip('\n')

    workspace_dirpath = pathlib.Path(r'../pipeline-workspace') / working_subdir
    image_source_dirpath = workspace_dirpath / 'calc_sequential_flow_and_blur'
    input_source_dirpath = workspace_dirpath / 'compute_depth_images'
    output_dirpath = workspace_dirpath / 'integrate_depth_images'

    if output_dirpath.exists():
        shutil.rmtree(output_dirpath)

    start_pipeline_server()
    post_to_pipeline_server((f'{working_subdir} / {output_dirpath.name}', 'waiting'))
    while True:
        pipeline_queue = get_queue_from_pipeline_server()
        print(pipeline_queue)
        if f'{working_subdir} / {input_source_dirpath.name}' not in pipeline_queue:
            break
        time.sleep(10)
    post_to_pipeline_server((f'{working_subdir} / {output_dirpath.name}', 'running'))
    print(get_queue_from_pipeline_server())

    # %%

    input_path = workspace_dirpath / 'stitch_key_frames' / 'stitched_key_frames.pickle'
    with open(input_path, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
        key_frame_indices = data['key_frame_indices']
        key_frame_motion_blurs = data['key_frame_motion_blurs']
        triangulated_idxs_weights = data['triangulated_idxs_weights']
        key_frame_image_sample_points = data['key_frame_image_sample_points']
        key_frame_image_triangulated_point_idxs = data['key_frame_image_triangulated_point_idxs']
        cross_stitch_disparity_confidence_maps = data['cross_stitch_disparity_confidence_maps']
        camera_extrinsics = data['camera_extrinsics']
        camera_intrinsic = data['camera_intrinsic']
        model_triangulated_points = data['model_triangulated_points']
        post_optimise_triangulated_idxs_mask = data['post_optimise_triangulated_idxs_mask']

    # %%

    frame_images = []
    filtered_frame_images = []
    frame_img_masks = []
    for frame_index, frame_time in key_frame_indices:
        filename = f'{frame_time.strftime("%Y%m%d-%H%M%S%f")}.{frame_index:03d}.resized.png'
        frame_images.append(cv2.cvtColor(cv2.imread(image_source_dirpath / filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB))

        filename = f'{frame_time.strftime("%Y%m%d-%H%M%S%f")}.{frame_index:03d}.filtered.png'
        filtered_frame_images.append(cv2.cvtColor(cv2.imread(image_source_dirpath / filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB))

        filename = f'{frame_time.strftime("%Y%m%d-%H%M%S%f")}.{frame_index:03d}.mask.png'
        frame_img_masks.append(cv2.imread(image_source_dirpath / filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(bool))

    image_sizes = set([img.shape[1::-1] for img in frame_images])
    assert len(image_sizes) == 1
    image_size = image_sizes.pop()

    # %%

    input_path = input_source_dirpath / 'depth_images'
    with shelve.open(input_path) as depth_images:
        depth_image_frame_idxs = set(map(int, depth_images.keys()))

    for rgbd_frame_idx in depth_image_frame_idxs:
        with shelve.open(input_path) as depth_images:
            frame_depth_images = depth_images[str(rgbd_frame_idx)]
        for depth_img, normal_img, confidence_map in frame_depth_images.values():
            assert np.sum(depth_img <= 0) == 0
            assert np.sum(depth_img > 30) == 0

    # %%

    def visualise_geometries(geometries):
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1024, height=768, left=200, top=200)

        axes_geometry = o3d.geometry.LineSet(o3d.utility.Vector3dVector([[0, 0, 0],
                                                                         [1, 0, 0],
                                                                         [0, 1, 0],
                                                                         [0, 0, 1]]),
                                             o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]]))
        axes_geometry.scale(20, [0, 0, 0])
        axes_geometry.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        vis.add_geometry(axes_geometry, reset_bounding_box=True)

        ctr = vis.get_view_control()
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])
        # vector from the lookat point to the camera
        # make gaze from the camera to lookat point left-ward and down-ward
        ctr.set_front([0.5, -0.5, -0.5])
        ctr.set_zoom(1.0)
        ctr.set_constant_z_far(200.0)

        for prev_camera_extrinsic, camera_extrinsic in zip([None] + camera_extrinsics[:-1], camera_extrinsics):
            # The extrinsic matrix transforms from world coordinates to camera coordinates
            camera_lines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=image_size[0], view_height_px=image_size[1],
                                                                            intrinsic=camera_intrinsic,
                                                                            extrinsic=camera_extrinsic)
            camera_lines.paint_uniform_color([0, 0.5, 1])
            if prev_camera_extrinsic is not None:
                camera_lines.points = o3d.utility.Vector3dVector(np.vstack([camera_lines.points,
                                                                            np.linalg.inv(prev_camera_extrinsic)[:3, 3],
                                                                            np.linalg.inv(camera_extrinsic)[:3, 3]]))
                camera_lines.lines = o3d.utility.Vector2iVector(np.vstack([camera_lines.lines,
                                                                           [len(camera_lines.points) - 2, len(camera_lines.points) - 1]]))
                camera_lines.colors = o3d.utility.Vector3dVector(np.vstack([camera_lines.colors,
                                                                            [1, 0, 0.5]]))
            vis.add_geometry(camera_lines, reset_bounding_box=False)

        for geometry in geometries:
            if isinstance(geometry, (o3d.geometry.PointCloud, o3d.geometry.TriangleMesh)):
                geometry = geometry.crop(o3d.geometry.AxisAlignedBoundingBox([-100, -100, -100], [100, 100, 100]))
            vis.add_geometry(geometry, reset_bounding_box=False)

        view_status = vis.get_view_status()
        view_status_time = time.time()
        visualisation_idle_timeout = 60 if IPython.get_ipython() is not None else 10
        while True:
            close_vis = not vis.poll_events()
            vis.update_renderer()
            new_view_status = vis.get_view_status()
            if new_view_status != view_status:
                view_status = new_view_status
                view_status_time = time.time()
            elif time.time() > view_status_time + visualisation_idle_timeout:
                close_vis = True
            if close_vis:
                break

        vis.destroy_window()

    # %%

    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(#voxel_length=0.2, sdf_trunc=1.5,
                                                          voxel_length=0.3, sdf_trunc=3.0,
                                                          color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for rgbd_frame_idx in depth_image_frame_idxs:
        with shelve.open(input_path) as depth_images:
            frame_depth_images = depth_images[str(rgbd_frame_idx)]
        for depth_img, normal_img, confidence_map in frame_depth_images.values():
            ref_img = filtered_frame_images[rgbd_frame_idx]
            # Depth values larger than depth_trunc are truncated to 0
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(ref_img),
                                                                            o3d.geometry.Image(depth_img),
                                                                            depth_scale=1.0, depth_trunc=np.inf,
                                                                            convert_rgb_to_intensity=False)
            volume.integrate(rgbd_image,
                             o3d.camera.PinholeCameraIntrinsic(np.array(rgbd_image.depth).shape[1],
                                                               np.array(rgbd_image.depth).shape[0],
                                                               camera_intrinsic),
                             camera_extrinsics[rgbd_frame_idx])

    visualise_geometries([volume.extract_triangle_mesh()])
    #visualise_geometries([volume.extract_point_cloud()])
    #visualise_geometries([volume.extract_voxel_point_cloud()])

    # %%

    rgbd_pcds = []
    for rgbd_frame_idx in depth_image_frame_idxs:
        camera_extrinsic = camera_extrinsics[rgbd_frame_idx]
        with shelve.open(input_path) as depth_images:
            frame_depth_images = depth_images[str(rgbd_frame_idx)]
        for depth_img, normal_img, confidence_map in frame_depth_images.values():
            ref_img = filtered_frame_images[rgbd_frame_idx]
            # Depth values larger than depth_trunc are truncated to 0
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(ref_img),
                                                                            o3d.geometry.Image(depth_img),
                                                                            depth_scale=1.0, depth_trunc=np.inf,
                                                                            convert_rgb_to_intensity=False)
            rgbd_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                                      o3d.camera.PinholeCameraIntrinsic(np.array(rgbd_image.depth).shape[1],
                                                                                                        np.array(rgbd_image.depth).shape[0],
                                                                                                        camera_intrinsic),
                                                                      camera_extrinsic)
            rgbd_pcd = rgbd_pcd.voxel_down_sample(voxel_size=0.5)
            rgbd_pcds.append(rgbd_pcd)

    visualise_geometries(rgbd_pcds)

    # %%

    rgbd_combined_pcd = o3d.geometry.PointCloud()
    for rgbd_frame_idx in depth_image_frame_idxs:
        camera_extrinsic = camera_extrinsics[rgbd_frame_idx]
        with shelve.open(input_path) as depth_images:
            frame_depth_images = depth_images[str(rgbd_frame_idx)]
        for depth_img, normal_img, confidence_map in frame_depth_images.values():
            ref_img = filtered_frame_images[rgbd_frame_idx]
            # Depth values larger than depth_trunc are truncated to 0
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(ref_img),
                                                                            o3d.geometry.Image(depth_img),
                                                                            depth_scale=1.0, depth_trunc=np.inf,
                                                                            convert_rgb_to_intensity=False)
            rgbd_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                                      o3d.camera.PinholeCameraIntrinsic(np.array(rgbd_image.depth).shape[1],
                                                                                                        np.array(rgbd_image.depth).shape[0],
                                                                                                        camera_intrinsic),
                                                                      camera_extrinsic)
            rgbd_combined_pcd += rgbd_pcd.voxel_down_sample(voxel_size=0.5)

    visualise_geometries([rgbd_combined_pcd])

    # When max_depth=9, leaf node size = 0.412109375
    octree = o3d.geometry.Octree(max_depth=9, origin=[0, 0, 0], size=200)
    octree.convert_from_point_cloud(rgbd_combined_pcd, size_expand=0)

    visualise_geometries([octree])

    dense_nodes = []
    density_threshold = 3

    def traverse_fn(node, node_info):
        assert isinstance(node, (o3d.geometry.OctreeInternalPointNode, o3d.geometry.OctreePointColorLeafNode))

        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            # early stopping: if True, traversal of children of the current node will be skipped
            return len(node.indices) < density_threshold
        else:
            assert node_info.depth == octree.max_depth
            if len(node.indices) >= density_threshold:
                dense_nodes.append(tuple(node_info.origin) + (len(node.indices),))

        return False

    octree.traverse(traverse_fn)
    dense_nodes = np.array(dense_nodes)

    density_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dense_nodes[:, :3]))
    normed_densities = matplotlib.colors.Normalize(vmin=density_threshold, vmax=np.percentile(dense_nodes[:, 3], 90))(dense_nodes[:, 3])
    density_pcd.colors = o3d.utility.Vector3dVector(matplotlib.colormaps['viridis'](normed_densities)[:, :3])
    visualise_geometries([density_pcd])

    plt.figure(figsize=(16, 10))
    plt.hist(dense_nodes[:, 3], bins=np.arange(np.amax(dense_nodes[:, 3]) + 2) - 0.5, log=True)
    """

    # %%

    '''
    target_depth_stack_disparity_point_count = 3

    merged_rgbd_images = {}
    for rgbd_frame_idx in depth_image_frame_idxs:
        with shelve.open(input_path) as depth_images:
            frame_depth_images = depth_images[str(rgbd_frame_idx)]
        depth_imgs = []
        for depth_img, normal_img, confidence_map in frame_depth_images.values():
            depth_imgs.append(depth_img)

        # Stack the frame's depth images and filter depth outliers
        depth_img_stack = np.array(np.stack(depth_imgs))

        for outlier_threshold in [3.0, 1.0]:
            # Circumnavigate RuntimeWarning: All-NaN slice encountered
            insufficient_depth_points_mask = np.sum(np.isfinite(depth_img_stack), axis=0) < target_depth_stack_disparity_point_count
            depth_img_stack[:, insufficient_depth_points_mask] = 0
            depth_img_median = np.nanmedian(depth_img_stack, axis=0)
            depth_img_median[insufficient_depth_points_mask] = np.nan
            depth_img_stack[:, insufficient_depth_points_mask] = np.nan

            depth_img_stack[np.abs(depth_img_stack - depth_img_median) > outlier_threshold] = np.nan

        # Circumnavigate RuntimeWarning: All-NaN slice encountered
        insufficient_depth_points_mask = np.sum(np.isfinite(depth_img_stack), axis=0) < target_depth_stack_disparity_point_count
        depth_img_stack[:, insufficient_depth_points_mask] = 0
        depth_img_mean = np.nanmean(depth_img_stack, axis=0)
        depth_img_mean[insufficient_depth_points_mask] = np.nan
        depth_img_stack[:, insufficient_depth_points_mask] = np.nan

        # Depth values larger than depth_trunc are truncated to 0
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(filtered_frame_images[rgbd_frame_idx]),
                                                                        o3d.geometry.Image(np.require(depth_img_mean, dtype=np.float32)),
                                                                        depth_scale=1.0, depth_trunc=np.inf,
                                                                        convert_rgb_to_intensity=False)

        merged_rgbd_images[rgbd_frame_idx] = rgbd_image

    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=0.3, sdf_trunc=3.0,
                                                          color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for rgbd_frame_idx, rgbd_image in merged_rgbd_images.items():
        volume.integrate(rgbd_image,
                         o3d.camera.PinholeCameraIntrinsic(np.array(rgbd_image.depth).shape[1],
                                                           np.array(rgbd_image.depth).shape[0],
                                                           camera_intrinsic),
                         camera_extrinsics[rgbd_frame_idx])

    integrated_merged_rgbd_images_mesh = volume.extract_triangle_mesh()
    visualise_geometries([integrated_merged_rgbd_images_mesh])
    '''

    # %%

    # TODO: Optimise with e.g. PyTorch, CuPy, JAX, PyCUDA, Numba. An example approach to consider:
    #  - https://github.com/andyzeng/tsdf-fusion-python/blob/master/fusion.py
    #  - https://github.com/andyzeng/tsdf-fusion-python/blob/master/demo.py

    def integrate_tsdf_rgbd(vox_coords_pcd, tsdf_weight_colour, tsdf_threshold, rgb_img, depth_imgs, confidence_maps,
                            camera_intrinsic, camera_extrinsic):
        # TODO: speed up by calculating corners of coarse grid?
        pcd = o3d.geometry.PointCloud(vox_coords_pcd)
        pcd.transform(np.block([[camera_intrinsic, np.zeros((3, 1))], [0, 0, 0, 1]]) @ camera_extrinsic)

        xyz = np.array(pcd.points, dtype=np.float32).T
        xyz[:, xyz[2, :] <= 0] = np.nan
        uv = xyz[:2, :] / xyz[2, :]

        h, w = depth_imgs.shape[1:]
        view_idxs = np.where((uv[0, :] > -0.5) & (uv[0, :] < w - 0.5) & (uv[1, :] > -0.5) & (uv[1, :] < h - 0.5))[0]

        view_uv = np.round(uv[:, view_idxs]).astype(np.int32)
        view_z = xyz[2, view_idxs]

        for depth_img, confidence_map in zip(depth_imgs, confidence_maps):
            view_uv_depth = depth_img[view_uv[1, :], view_uv[0, :]]
            # Interior values are negative, exterior values are positive
            signed_distance = (view_uv_depth - view_z) / tsdf_threshold

            # The TSDF weight is normally distributed with tail values of np.exp(-4.5) = 0.011 at [-tsdf_threshold, +tsdf_threshold]
            # The TSDF signed distance value is linear from [1, -1] between [-tsdf_threshold, +tsdf_threshold]
            tsdf_mask = (np.abs(signed_distance) < 1) & (view_uv_depth > 0)
            tsdf_idxs = view_idxs[tsdf_mask]
            tsdf_uv = view_uv[:, tsdf_mask]
            tsdf_dist = signed_distance[tsdf_mask]
            tsdf_weight = np.exp(-4.5 * np.power(tsdf_dist, 2))

            obs_tsdf_weight = confidence_map[tsdf_uv[1, :], tsdf_uv[0, :]] * tsdf_weight
            obs_tsdf_weight[~np.isfinite(obs_tsdf_weight)] = 0

            tsdf_weight_colour[:, tsdf_idxs] += np.vstack([obs_tsdf_weight * tsdf_dist,
                                                           obs_tsdf_weight,
                                                           obs_tsdf_weight * rgb_img[tsdf_uv[1, :], tsdf_uv[0, :], :].T])

    # %%

    # Perspective distortion is directly associated with camera to object proximity, and is not directly connected to
    # the values of fx & fy (the pinhole camera focal lengths). However, when fx & fy have smaller values, the zoom is lower
    # and angle of view is wider, so the camera is typically positioned closer to the object,
    # which means that perspective distortion tends to be is indirectly influenced by the values of fx & fy.
    # Reduce the perspective distortion in the synthetic view by positioning the camera further away from the target,
    # and preserve the field of view of the camera as it recedes (akin to a dolly zoom).
    # As the camera gets further away, projection lines become more parallel and the view approaches isometric projection.
    camera_pull_back_z = 10
    synthetic_camera_zoom = {(640, 360): 0.5, (480, 640): 1.0}[image_size] * (camera_pull_back_z / 8 + 1)
    camera_intrinsic_synthetic = np.block([[camera_intrinsic[:2, :2] * synthetic_camera_zoom, camera_intrinsic[:2, 2:] + np.array(image_size)[:, None]], [0, 0, 1]])

    # %%

    # TSDF distance threshold
    tsdf_threshold = 5.0

    frustum_sample_zs = np.arange(0, 30, 0.5) + camera_pull_back_z

    integrated_weighted_depth_kernels = []
    for ref_frame_idx in range(len(key_frame_indices)):
        print('Integrating TSDF ref_frame_idx', ref_frame_idx)

        synthetic_camera_extrinsic = np.block([[np.identity(3), np.array([0, 0, camera_pull_back_z])[:, None]], [0, 0, 0, 1]]) @ camera_extrinsics[ref_frame_idx]

        ref_img = frame_images[ref_frame_idx]

        h, w = ref_img.shape[:2]
        hj, wj = h * 3, w * 3

        grid_step = 8
        camera_intrinsic_stepped = np.block([[camera_intrinsic_synthetic[:2, :2] / grid_step, (camera_intrinsic_synthetic[:2, 2:] + 0.5) / grid_step - 0.5], [0, 0, 1]])

        assert wj % grid_step == 0 and hj % grid_step == 0
        ws, hs = wj // grid_step, hj // grid_step

        uvs = np.vstack([uv.flatten() for uv in np.mgrid[0:hs, 0:ws][::-1]])
        uvcs = uvs - camera_intrinsic_stepped[:2, 2:]
        xys = np.linalg.inv(camera_intrinsic_stepped[:2, :2]) @ uvcs
        xyzs = np.vstack([xys, np.ones(xys.shape[1],)])
        frustum_sample_points = np.hstack([xyzs * z for z in frustum_sample_zs]).T

        tsdf_vox_coords_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(frustum_sample_points))
        tsdf_vox_coords_pcd.transform(np.linalg.inv(synthetic_camera_extrinsic))

        # Initialize voxel volume array for accumulating the observations (TSDF, weight, R, G, B) over each voxel
        tsdf_weight_colour = np.zeros((5, frustum_sample_points.shape[0]), dtype=np.float32)
        # Interior values are negative, exterior values are positive
        tsdf_weight_colour[0, :] = 1e-3
        tsdf_weight_colour[1, :] = 1e-6

        for rgbd_frame_idx in depth_image_frame_idxs:
            camera_transform = camera_extrinsics[rgbd_frame_idx] @ np.linalg.inv(synthetic_camera_extrinsic)
            rvec, _ = cv2.Rodrigues(camera_transform[:3, :3])
            # Ignore rotation around the z-axis (in the primary frame of reference)
            rotation_magnitude = np.linalg.norm(rvec[:2])
            frame_weight = cauchy(rotation_magnitude, np.pi / 4)
            print(rgbd_frame_idx, np.round(np.rad2deg(rotation_magnitude), 1), np.round(frame_weight, 3))
            if frame_weight > 0.01:
                ref_img = filtered_frame_images[rgbd_frame_idx]
                depth_imgs = []
                confidence_maps = []
                with shelve.open(input_path) as depth_images:
                    frame_depth_images = depth_images[str(rgbd_frame_idx)]
                for depth_img, normal_img, confidence_map in frame_depth_images.values():
                    # Depth values larger than depth_trunc are truncated to 0
                    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(ref_img),
                                                                                    o3d.geometry.Image(depth_img),
                                                                                    depth_scale=1.0, depth_trunc=np.inf,
                                                                                    convert_rgb_to_intensity=False)
                    rgbd_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                                              o3d.camera.PinholeCameraIntrinsic(depth_img.shape[1],
                                                                                                                depth_img.shape[0],
                                                                                                                camera_intrinsic),
                                                                              camera_extrinsics[rgbd_frame_idx])
                    rgbd_points = np.full(depth_img.shape + (3,), fill_value=np.nan, dtype=np.float32)
                    rgbd_points[np.isfinite(depth_img), :] = np.array(rgbd_pcd.points)

                    rgbd_normals = (np.linalg.inv(camera_extrinsics[rgbd_frame_idx])[:3, :3] @ normal_img[:, :, :, None])[:, :, :, 0]

                    ref_camera_origin = np.linalg.inv(synthetic_camera_extrinsic)[:3, 3]
                    ref_camera_rays = rgbd_points - ref_camera_origin
                    ref_camera_rays = ref_camera_rays / np.clip(np.linalg.norm(ref_camera_rays, axis=-1), 1e-6, np.inf)[:, :, None]

                    if True:
                        ray_normal_angles = np.arccos(np.clip(np.sum(-rgbd_normals * ref_camera_rays, axis=-1), -1, 1))

                        ray_normal_weight = 0.5 * (1 - scipy.special.erf((ray_normal_angles - 7 / 12 * np.pi) / (np.pi / 12)))

                        depth_imgs.append(np.nan_to_num(depth_img, nan=0))
                        confidence_maps.append(confidence_map * frame_weight * ray_normal_weight)

                    else:
                        normal_ray_alignment = np.sum(ref_camera_rays * rgbd_normals, axis=-1)
                        camera_ray_to_object_plane = ref_camera_rays / np.clip(-normal_ray_alignment, 1e-8, 1)[:, :, None]
                        camera_ray_to_image_plane = ref_camera_rays * -rgbd_normals[:, :, 2:] / ref_camera_rays[:, :, 2:]

                        object_to_image_ratio = (np.linalg.norm(normal_img + camera_ray_to_object_plane, axis=-1)
                                                 / np.clip(np.linalg.norm(normal_img + camera_ray_to_image_plane, axis=-1), 1e-8, np.inf))
                        perspective_distortion = 1 - np.min(np.stack([object_to_image_ratio,
                                                                      np.clip(1 / object_to_image_ratio, 1e-8, np.inf)]), axis=0)

                        perspective_distortion_scores = 0.5 * (1 - scipy.special.erf((perspective_distortion - 0.4) / 0.15))
                        perspective_distortion_scores[normal_ray_alignment >= 0] = 0

                        depth_imgs.append(np.nan_to_num(depth_img, nan=0))
                        confidence_maps.append(confidence_map * frame_weight * perspective_distortion_scores)

                depth_imgs = np.array(depth_imgs)
                confidence_maps = np.array(confidence_maps)

                # Integrate observations into voxel volume
                integrate_tsdf_rgbd(tsdf_vox_coords_pcd, tsdf_weight_colour, tsdf_threshold,
                                    np.array(ref_img, dtype=np.float32), depth_imgs, confidence_maps,
                                    camera_intrinsic, camera_extrinsics[rgbd_frame_idx])

        depth_kernels = []
        depth_img = []
        color_img = []
        for pix_idx in range(ws * hs):
            tsdf = tsdf_weight_colour[0, pix_idx::ws*hs]
            weights = tsdf_weight_colour[1, pix_idx::ws*hs]
            colors = tsdf_weight_colour[2:, pix_idx::ws*hs] / weights
            # Interior values are negative, exterior values are positive
            surface_idxs = np.where((tsdf[:-1] >= 0) & (tsdf[1:] < 0) & (weights[:-1] >= 0.5))[0]
            surface_weights = weights[surface_idxs]
            interp_fraction = -tsdf[surface_idxs + 1] / (tsdf[surface_idxs] - tsdf[surface_idxs + 1])
            interp_zs = interp_fraction * frustum_sample_zs[surface_idxs] + (1 - interp_fraction) * frustum_sample_zs[surface_idxs + 1]
            for z, weight in zip(interp_zs, surface_weights):
                depth_kernels.append((pix_idx % ws, pix_idx // ws, z, weight))
            if len(surface_idxs) > 0:
                sort_idxs = np.argsort(surface_weights)
                if len(sort_idxs) == 1 or surface_weights[sort_idxs[-1]] >= surface_weights[sort_idxs[-2]] * 2:
                    depth_img.append(interp_zs[sort_idxs[-1]])
                    color_img.append(colors[:, surface_idxs[sort_idxs[-1]]])
                else:
                    depth_img.append(np.nan)
                    color_img.append(np.full((3,), fill_value=np.nan))
            else:
                depth_img.append(np.nan)
                color_img.append(np.full((3,), fill_value=np.nan))
        depth_img = np.array(depth_img, dtype=np.float32).reshape((hs, ws))
        color_img = np.clip(np.array(color_img, dtype=np.float32).reshape((hs, ws, -1)), 0, 255)

        integrated_weighted_depth_kernels.append((synthetic_camera_extrinsic, camera_intrinsic_stepped, depth_kernels, color_img, depth_img))

    # %%

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1024, height=768, left=200, top=200)

    axes_geometry = o3d.geometry.LineSet(o3d.utility.Vector3dVector([[0, 0, 0],
                                                                     [1, 0, 0],
                                                                     [0, 1, 0],
                                                                     [0, 0, 1]]),
                                         o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]]))
    axes_geometry.scale(20, [0, 0, 0])
    axes_geometry.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    vis.add_geometry(axes_geometry, reset_bounding_box=True)

    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    # vector from the lookat point to the camera
    # make gaze from the camera to lookat point left-ward and down-ward
    ctr.set_front([0.5, -0.5, -0.5])
    ctr.set_zoom(1.0)
    ctr.set_constant_z_far(200.0)

    view_status = vis.get_view_status()
    view_status_time = time.time()
    visualisation_idle_timeout = 60 if IPython.get_ipython() is not None else 10
    while True:
        close_vis = False
        geometries = []
        for rgbd_frame_idx, (prev_camera_extrinsic, camera_extrinsic) in enumerate(zip([None] + camera_extrinsics[:-1], camera_extrinsics)):
            # The extrinsic matrix transforms from world coordinates to camera coordinates
            camera_lines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=image_size[0], view_height_px=image_size[1],
                                                                            intrinsic=camera_intrinsic,
                                                                            extrinsic=camera_extrinsic)
            camera_lines.paint_uniform_color([0, 0.5, 1])
            if prev_camera_extrinsic is not None:
                camera_lines.points = o3d.utility.Vector3dVector(np.vstack([camera_lines.points,
                                                                            np.linalg.inv(prev_camera_extrinsic)[:3, 3],
                                                                            np.linalg.inv(camera_extrinsic)[:3, 3]]))
                camera_lines.lines = o3d.utility.Vector2iVector(np.vstack([camera_lines.lines,
                                                                           [len(camera_lines.points) - 2, len(camera_lines.points) - 1]]))
                camera_lines.colors = o3d.utility.Vector3dVector(np.vstack([camera_lines.colors,
                                                                            [1, 0, 0.5]]))
            vis.add_geometry(camera_lines, reset_bounding_box=False)
            geometries.append(camera_lines)

            synthetic_camera_extrinsic, camera_intrinsic_stepped, depth_kernels, color_img, depth_img = integrated_weighted_depth_kernels[rgbd_frame_idx]

            # Depth values larger than depth_trunc are truncated to 0
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(np.nan_to_num(color_img, nan=127).astype(np.uint8)),
                                                                            o3d.geometry.Image(depth_img),
                                                                            depth_scale=1.0, depth_trunc=np.inf,
                                                                            convert_rgb_to_intensity=False)

            rgbd_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                                      o3d.camera.PinholeCameraIntrinsic(np.array(rgbd_image.depth).shape[1],
                                                                                                        np.array(rgbd_image.depth).shape[0],
                                                                                                        camera_intrinsic_stepped),
                                                                      synthetic_camera_extrinsic)

            vis.add_geometry(rgbd_pcd, reset_bounding_box=False)

            start_time = time.time()
            while True:
                close_vis = not vis.poll_events()
                vis.update_renderer()
                new_view_status = vis.get_view_status()
                if new_view_status != view_status:
                    view_status = new_view_status
                    view_status_time = time.time()
                elif time.time() > view_status_time + visualisation_idle_timeout:
                    close_vis = True
                if close_vis or time.time() > start_time + 0.05:
                    break

            vis.remove_geometry(rgbd_pcd, reset_bounding_box=False)

            if close_vis:
                break

        if close_vis:
            break

        for geometry in geometries:
            vis.remove_geometry(geometry, reset_bounding_box=False)

    vis.destroy_window()

    # %%

    plt.close('Optimised integrated weighted depth')

    model_depth_images = []

    for ref_frame_idx, (synthetic_camera_extrinsic, camera_intrinsic_stepped, depth_kernels, color_img, depth_img) in enumerate(integrated_weighted_depth_kernels):
        print('Optimising integrated weighted depth ref_frame_idx', ref_frame_idx)

        model_base_depth = torch.tensor(0, dtype=torch.float32, requires_grad=True)
        #model_depth = torch.tensor(np.nan_to_num(depth_img, nan=np.max(frustum_sample_zs)), dtype=torch.float32, requires_grad=True)
        model_depth = torch.tensor(np.full(depth_img.shape, fill_value=np.mean(frustum_sample_zs)), dtype=torch.float32, requires_grad=True)

        depth_kernel_uv = torch.tensor(np.array(depth_kernels)[:, :2], dtype=torch.int32)
        depth_kernel_zw = torch.tensor(np.array(depth_kernels)[:, 2:], dtype=torch.float32)

        gamma_softplus_alpha = torch.tensor(2, dtype=torch.float32)

        def loss_fn(temperature, gravity):
            depth_samples = model_base_depth + model_depth[depth_kernel_uv[:, 1], depth_kernel_uv[:, 0]]

            depth_err = depth_samples - depth_kernel_zw[:, 0]

            threshold = 3.0 * (1 - temperature) + 30.0 * temperature
            depth_err_losses = gamma_softplus(depth_err, threshold=threshold, alpha=gamma_softplus_alpha, relative_outer_gradient=0.01)
            #depth_err_losses = 1 - 1 / torch.sqrt(1 + torch.pow(depth_err * 3.0 / threshold, 2))

            #grad_ud = torch.abs(model_depth[:-1, :] - model_depth[1:, :])
            #grad_lr = torch.abs(model_depth[:, :-1] - model_depth[:, 1:])
            grad_ud = torch.pow(model_depth[:-1, :] - model_depth[1:, :], 2)
            grad_lr = torch.pow(model_depth[:, :-1] - model_depth[:, 1:], 2)
            # torch.nn.functional.pad() pad parameter is specified as left, right pairs in from the last dimension to the first dimension
            grad_ud = torch.nn.functional.pad(grad_ud, (0, 0, 0, 1)) + torch.nn.functional.pad(grad_ud, (0, 0, 1, 0))
            grad_lr = torch.nn.functional.pad(grad_lr, (0, 1, 0, 0)) + torch.nn.functional.pad(grad_lr, (1, 0, 0, 0))

            return (torch.sum(depth_err_losses * depth_kernel_zw[:, 1]) / torch.sum(depth_kernel_zw[:, 1])
                    + 0.5 * torch.mean(grad_ud + grad_lr)
                    - gravity * (model_base_depth + torch.mean(model_depth)))

        lr = 0.005
        num_steps = 3000
        convergence_criterion = {'rtol': 1e-4, 'window_size': 100, 'min_num_steps': 300}
        optimiser = torch.optim.Adam([model_base_depth, model_depth], lr=lr)
        lr_lambda = lambda epoch: np.sin(min((epoch + 1) / convergence_criterion['min_num_steps'], 0.5) * np.pi)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lr_lambda)
        losses = []
        for optim_step in range(num_steps):
            optimiser.zero_grad()
            temperature = (1 + np.cos(np.pi * min(optim_step / convergence_criterion['min_num_steps'], 1))) / 2
            loss = loss_fn(temperature, 0)
            loss.backward()
            optimiser.step()
            scheduler.step()
            losses.append(loss.numpy(force=True))
            loss_window_std = np.std(losses[-convergence_criterion['window_size']:])
            if optim_step % 100 == 0:
                print('optim_step', optim_step, 'loss', losses[-1], 'loss_window_std', loss_window_std,
                      'learning rate', np.round(scheduler.get_last_lr()[0], 6), 'temperature', np.round(temperature, 3))
            if optim_step >= convergence_criterion['min_num_steps'] and loss_window_std < convergence_criterion['rtol'] * losses[-1]:
                break

        print('len(losses)', len(losses))
        print('initial, first, final loss', losses[0], losses[convergence_criterion['min_num_steps']], losses[-1])
        print('highest, lowest loss', np.max(losses[convergence_criterion['min_num_steps']:]), np.min(losses[convergence_criterion['min_num_steps']:]))

        """
        plt.figure(figsize=(16, 10))
        plt.suptitle('losses')
        ax = plt.subplot(4, 1, 1)
        plt.plot(losses)
        ax = plt.subplot(4, 1, 2, sharex=ax)
        plt.semilogy(losses)
        ax = plt.subplot(4, 1, 3, sharex=ax)
        plt.plot(np.arange(convergence_criterion['min_num_steps'], len(losses)), losses[convergence_criterion['min_num_steps']:])
        ax = plt.subplot(4, 1, 4, sharex=ax)
        plt.semilogy(np.arange(convergence_criterion['min_num_steps'], len(losses)), losses[convergence_criterion['min_num_steps']:])
        plt.tight_layout()
        """

        model_depth_img = model_base_depth.numpy(force=True) + model_depth.numpy(force=True)

        displacement_drop = 0.1

        with torch.no_grad():
            model_base_depth += displacement_drop

        lr = 0.005
        num_steps = 300
        gravity = 1e-3
        optimiser = torch.optim.Adam([model_base_depth, model_depth], lr=lr)
        for optim_step in range(num_steps):
            optimiser.zero_grad()
            loss = loss_fn(0, gravity)
            loss.backward()
            optimiser.step()

        model_depth_displacement = model_base_depth.numpy(force=True) + model_depth.numpy(force=True) - model_depth_img
        model_depth_confidence = np.clip(1 - model_depth_displacement / displacement_drop, 0, 1)

        model_depth_images.append((model_depth_img, model_depth_confidence))

        plt.figure('Optimised integrated weighted depth', figsize=(16, 10))
        setup_new_fig_page()
        plt.suptitle(f'frame idx: {ref_frame_idx}')
        ax = plt.subplot(2, 3, 1)
        plt.imshow(depth_img, vmin=camera_pull_back_z, vmax=camera_pull_back_z+30)
        plt.title('integrated depth')
        ax = plt.subplot(2, 3, 2, sharex=ax, sharey=ax)
        plt.imshow(model_depth_img, vmin=camera_pull_back_z, vmax=camera_pull_back_z+30)
        plt.title('optimised depth')
        ax = plt.subplot(2, 3, 3, sharex=ax, sharey=ax)
        ax.set_facecolor('grey')
        plt.imshow(model_depth_img - depth_img, vmin=-30, vmax=30, cmap='seismic')
        plt.title('optimised - integrated depth')
        ax = plt.subplot(2, 3, 4, sharex=ax, sharey=ax)
        plt.imshow(model_depth_confidence, vmin=0, vmax=1)
        plt.title('optimised depth confidence')
        ax = plt.subplot(2, 3, 5, sharex=ax, sharey=ax)
        plt.imshow(color_img / 255)
        plt.title('integrated colour image')
        plt.tight_layout()
        stash_fig_page()

    # %%

    plt.close('Optimised integrated weighted masked depth')

    masked_model_depth_images = []
    for ref_frame_idx, (model_depth_img, model_depth_confidence) in enumerate(model_depth_images):
        model_depth_mask = skimage.filters.apply_hysteresis_threshold(model_depth_confidence, low=0.1, high=0.9)
        masked_model_depth_img = np.array(model_depth_img)
        masked_model_depth_img[~model_depth_mask] = np.nan
        masked_model_depth_images.append(masked_model_depth_img)

        plt.figure('Optimised integrated weighted masked depth', figsize=(16, 10))
        setup_new_fig_page()
        plt.suptitle(f'frame idx: {ref_frame_idx}')
        ax = plt.subplot(2, 2, 1)
        plt.imshow(model_depth_img, vmin=camera_pull_back_z, vmax=camera_pull_back_z+30)
        plt.title('optimised depth')
        ax = plt.subplot(2, 2, 2, sharex=ax, sharey=ax)
        plt.imshow(model_depth_confidence, vmin=0, vmax=1)
        plt.title('optimised depth confidence')
        ax = plt.subplot(2, 2, 3, sharex=ax, sharey=ax)
        plt.imshow(model_depth_mask)
        plt.title('depth mask')
        ax = plt.subplot(2, 2, 4, sharex=ax, sharey=ax)
        plt.imshow(masked_model_depth_img, vmin=camera_pull_back_z, vmax=camera_pull_back_z+30)
        plt.title('masked optimised depth')
        plt.tight_layout()
        stash_fig_page()

    # %%

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1024, height=768, left=200, top=200)

    axes_geometry = o3d.geometry.LineSet(o3d.utility.Vector3dVector([[0, 0, 0],
                                                                     [1, 0, 0],
                                                                     [0, 1, 0],
                                                                     [0, 0, 1]]),
                                         o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]]))
    axes_geometry.scale(20, [0, 0, 0])
    axes_geometry.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    vis.add_geometry(axes_geometry, reset_bounding_box=True)

    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    # vector from the lookat point to the camera
    # make gaze from the camera to lookat point left-ward and down-ward
    ctr.set_front([0.5, -0.5, -0.5])
    ctr.set_zoom(1.0)
    ctr.set_constant_z_far(200.0)

    view_status = vis.get_view_status()
    view_status_time = time.time()
    visualisation_idle_timeout = 60 if IPython.get_ipython() is not None else 10
    while True:
        close_vis = False
        geometries = []
        for rgbd_frame_idx, (prev_camera_extrinsic, camera_extrinsic) in enumerate(zip([None] + camera_extrinsics[:-1], camera_extrinsics)):
            # The extrinsic matrix transforms from world coordinates to camera coordinates
            camera_lines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=image_size[0], view_height_px=image_size[1],
                                                                            intrinsic=camera_intrinsic,
                                                                            extrinsic=camera_extrinsic)
            camera_lines.paint_uniform_color([0, 0.5, 1])
            if prev_camera_extrinsic is not None:
                camera_lines.points = o3d.utility.Vector3dVector(np.vstack([camera_lines.points,
                                                                            np.linalg.inv(prev_camera_extrinsic)[:3, 3],
                                                                            np.linalg.inv(camera_extrinsic)[:3, 3]]))
                camera_lines.lines = o3d.utility.Vector2iVector(np.vstack([camera_lines.lines,
                                                                           [len(camera_lines.points) - 2, len(camera_lines.points) - 1]]))
                camera_lines.colors = o3d.utility.Vector3dVector(np.vstack([camera_lines.colors,
                                                                            [1, 0, 0.5]]))
            vis.add_geometry(camera_lines, reset_bounding_box=False)
            geometries.append(camera_lines)

            synthetic_camera_extrinsic, camera_intrinsic_stepped, depth_kernels, color_img, depth_img = integrated_weighted_depth_kernels[rgbd_frame_idx]

            model_depth_img = masked_model_depth_images[rgbd_frame_idx]

            # Depth values larger than depth_trunc are truncated to 0
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(np.nan_to_num(color_img, nan=127).astype(np.uint8)),
                                                                            o3d.geometry.Image(model_depth_img),
                                                                            depth_scale=1.0, depth_trunc=np.inf,
                                                                            convert_rgb_to_intensity=False)

            rgbd_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                                      o3d.camera.PinholeCameraIntrinsic(np.array(rgbd_image.depth).shape[1],
                                                                                                        np.array(rgbd_image.depth).shape[0],
                                                                                                        camera_intrinsic_stepped),
                                                                      synthetic_camera_extrinsic)

            vis.add_geometry(rgbd_pcd, reset_bounding_box=False)

            start_time = time.time()
            while True:
                close_vis = not vis.poll_events()
                vis.update_renderer()
                new_view_status = vis.get_view_status()
                if new_view_status != view_status:
                    view_status = new_view_status
                    view_status_time = time.time()
                elif time.time() > view_status_time + visualisation_idle_timeout:
                    close_vis = True
                if close_vis or time.time() > start_time + 0.05:
                    break

            vis.remove_geometry(rgbd_pcd, reset_bounding_box=False)

            if close_vis:
                break

        if close_vis:
            break

        for geometry in geometries:
            vis.remove_geometry(geometry, reset_bounding_box=False)

    vis.destroy_window()

    # %%

    integrated_weighted_canvas_meshes = []

    for ref_frame_idx, ((synthetic_camera_extrinsic, camera_intrinsic_stepped, depth_kernels, color_img, depth_img),
                        model_depth_img) in enumerate(zip(integrated_weighted_depth_kernels, masked_model_depth_images)):
        hs, ws = model_depth_img.shape
        uv_grids = np.mgrid[0:hs, 0:ws][::-1]
        uvs = np.vstack([uv.flatten() for uv in uv_grids])
        uvcs = uvs - camera_intrinsic_stepped[:2, 2:]
        xys = np.linalg.inv(camera_intrinsic_stepped[:2, :2]) @ uvcs

        vertices = np.vstack([xys, np.ones(xys.shape[1],)]) * model_depth_img.flatten()

        # Anticlockwise ordering
        vertex_idxs = uv_grids[0, :, :] + ws * uv_grids[1, :, :]
        upper_triangle_idxs = np.vstack([vertex_idxs[:-1, :-1].flatten(), vertex_idxs[1:, :-1].flatten(), vertex_idxs[:-1, 1:].flatten()])
        lower_triangle_idxs = np.vstack([vertex_idxs[1:, 1:].flatten(), vertex_idxs[:-1, 1:].flatten(), vertex_idxs[1:, :-1].flatten()])
        triangle_idxs = np.hstack([upper_triangle_idxs, lower_triangle_idxs])

        canvas_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices.T),
                                                o3d.utility.Vector3iVector(triangle_idxs.T))
        canvas_mesh.vertex_colors = o3d.utility.Vector3dVector(np.nan_to_num(color_img, nan=127).reshape((-1, 3)) / 255)
        canvas_mesh.compute_vertex_normals()
        # Orient normals towards the camera
        vertex_normals = np.array(canvas_mesh.vertex_normals).T
        reoriented_vertex_normals = vertex_normals * np.sign(np.sum(-vertex_normals * vertices, axis=0))
        canvas_mesh.vertex_normals = o3d.utility.Vector3dVector(reoriented_vertex_normals.T)

        canvas_mesh.transform(np.linalg.inv(synthetic_camera_extrinsic))

        integrated_weighted_canvas_meshes.append(canvas_mesh)

    # %%

    vertex_normal_lines = []
    for mesh in integrated_weighted_canvas_meshes:
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)

        line_length = 1.0
        points = np.vstack([vertices, vertices + line_length * normals])
        lines = np.arange(points.shape[0]).reshape((2, -1)).T
        colors = np.array([1, 0, 0]) * np.ones((lines.shape[0], 1))

        normal_lines = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        normal_lines.colors = o3d.utility.Vector3dVector(colors)

        vertex_normal_lines.append(normal_lines)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1024, height=768, left=200, top=200)

    axes_geometry = o3d.geometry.LineSet(o3d.utility.Vector3dVector([[0, 0, 0],
                                                                     [1, 0, 0],
                                                                     [0, 1, 0],
                                                                     [0, 0, 1]]),
                                         o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]]))
    axes_geometry.scale(20, [0, 0, 0])
    axes_geometry.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    vis.add_geometry(axes_geometry, reset_bounding_box=True)

    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    # vector from the lookat point to the camera
    # make gaze from the camera to lookat point left-ward and down-ward
    ctr.set_front([0.5, -0.5, -0.5])
    ctr.set_zoom(1.0)
    ctr.set_constant_z_far(200.0)

    view_status = vis.get_view_status()
    view_status_time = time.time()
    visualisation_idle_timeout = 60 if IPython.get_ipython() is not None else 10
    while True:
        close_vis = False
        geometries = []
        for rgbd_frame_idx, (prev_camera_extrinsic, camera_extrinsic) in enumerate(zip([None] + camera_extrinsics[:-1], camera_extrinsics)):
            # The extrinsic matrix transforms from world coordinates to camera coordinates
            camera_lines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=image_size[0], view_height_px=image_size[1],
                                                                            intrinsic=camera_intrinsic,
                                                                            extrinsic=camera_extrinsic)
            camera_lines.paint_uniform_color([0, 0.5, 1])
            if prev_camera_extrinsic is not None:
                camera_lines.points = o3d.utility.Vector3dVector(np.vstack([camera_lines.points,
                                                                            np.linalg.inv(prev_camera_extrinsic)[:3, 3],
                                                                            np.linalg.inv(camera_extrinsic)[:3, 3]]))
                camera_lines.lines = o3d.utility.Vector2iVector(np.vstack([camera_lines.lines,
                                                                           [len(camera_lines.points) - 2, len(camera_lines.points) - 1]]))
                camera_lines.colors = o3d.utility.Vector3dVector(np.vstack([camera_lines.colors,
                                                                            [1, 0, 0.5]]))
            vis.add_geometry(camera_lines, reset_bounding_box=False)
            geometries.append(camera_lines)

            synthetic_camera_extrinsic, camera_intrinsic_stepped, depth_kernels, color_img, depth_img = integrated_weighted_depth_kernels[rgbd_frame_idx]

            mesh = o3d.geometry.TriangleMesh(integrated_weighted_canvas_meshes[rgbd_frame_idx])
            mesh.vertex_normals = o3d.utility.Vector3dVector()
            normal_lines = vertex_normal_lines[rgbd_frame_idx]

            vis.add_geometry(mesh, reset_bounding_box=False)
            vis.add_geometry(normal_lines, reset_bounding_box=False)

            start_time = time.time()
            while True:
                close_vis = not vis.poll_events()
                vis.update_renderer()
                new_view_status = vis.get_view_status()
                if new_view_status != view_status:
                    view_status = new_view_status
                    view_status_time = time.time()
                elif time.time() > view_status_time + visualisation_idle_timeout:
                    close_vis = True
                if close_vis or time.time() > start_time + 0.05:
                    break

            vis.remove_geometry(mesh, reset_bounding_box=False)
            vis.remove_geometry(normal_lines, reset_bounding_box=False)

            if close_vis:
                break

        if close_vis:
            break

        for geometry in geometries:
            vis.remove_geometry(geometry, reset_bounding_box=False)

    vis.destroy_window()

    # %%

    '''
    if False:
        # Generate and combine TSDF empty space voxel point clouds over all merged RGBD images
        space_pcd = o3d.geometry.PointCloud()
        for rgbd_frame_idx, rgbd_image in merged_rgbd_images.items():
            volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=0.2, sdf_trunc=10.0,
                                                                  color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
            volume.integrate(rgbd_image,
                             o3d.camera.PinholeCameraIntrinsic(np.array(rgbd_image.depth).shape[1],
                                                               np.array(rgbd_image.depth).shape[0],
                                                               camera_intrinsic),
                             camera_extrinsics[rgbd_frame_idx])
            voxel_pcd = volume.extract_voxel_point_cloud()
            space_pcd += voxel_pcd.select_by_index(np.where(np.mean(np.array(voxel_pcd.colors), axis=1) > 0.6)[0])

        visualise_geometries([space_pcd])

        # Filter the combined empty space voxel point cloud
        filtered_space_pcds = [space_pcd]
        for iter_idx in range(1):
            pcd, _ = filtered_space_pcds[-1].remove_radius_outlier(nb_points=30, radius=0.25)
            filtered_space_pcds.append(pcd)
        filtered_space_pcd = filtered_space_pcds[-1]

        visualise_geometries([filtered_space_pcd])

        # Use the filtered empty space voxel point cloud to carve each RGBD merged image
        carved_merged_rgbd_images = {}
        merged_rgbd_combined_pcd = o3d.geometry.PointCloud()
        carved_merged_rgbd_combined_pcd = o3d.geometry.PointCloud()
        for rgbd_frame_idx, rgbd_image in merged_rgbd_images.items():

            rgbd_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                                      o3d.camera.PinholeCameraIntrinsic(np.array(rgbd_image.depth).shape[1],
                                                                                                        np.array(rgbd_image.depth).shape[0],
                                                                                                        camera_intrinsic),
                                                                      camera_extrinsics[rgbd_frame_idx])

            depth_idxs = np.vstack(np.where(np.asarray(rgbd_image.depth) > 0)).T
            """
            pcd = o3d.geometry.PointCloud(rgbd_pcd)
            pcd.transform(camera_extrinsics[rgbd_frame_idx])
            assert np.allclose(np.asarray(pcd.points)[:, 2], np.asarray(rgbd_image.depth)[depth_idxs[:, 0], depth_idxs[:, 1]])
            """

            space_distances = rgbd_pcd.compute_point_cloud_distance(filtered_space_pcd)
            space_idxs = np.where(np.asarray(space_distances) < 0.25)[0]

            # Depth values larger than depth_trunc are truncated to 0
            carved_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgbd_image.color),
                                                                                   o3d.geometry.Image(rgbd_image.depth),
                                                                                   depth_scale=1.0, depth_trunc=np.inf,
                                                                                   convert_rgb_to_intensity=False)
            depth = np.asarray(carved_rgbd_image.depth)
            depth[depth_idxs[space_idxs, 0], depth_idxs[space_idxs, 1]] = np.nan

            carved_merged_rgbd_images[rgbd_frame_idx] = carved_rgbd_image

            carved_rgbd_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(carved_rgbd_image,
                                                                             o3d.camera.PinholeCameraIntrinsic(np.array(carved_rgbd_image.depth).shape[1],
                                                                                                               np.array(carved_rgbd_image.depth).shape[0],
                                                                                                               camera_intrinsic),
                                                                             camera_extrinsics[rgbd_frame_idx])

            merged_rgbd_combined_pcd += rgbd_pcd.voxel_down_sample(voxel_size=0.1)
            carved_merged_rgbd_combined_pcd += carved_rgbd_pcd.voxel_down_sample(voxel_size=0.1)

        visualise_geometries([merged_rgbd_combined_pcd])
        visualise_geometries([carved_merged_rgbd_combined_pcd])

    else:
        carved_merged_rgbd_images = dict(merged_rgbd_images)
    '''

    # %%

    '''
    for rgbd_frame_idx, rgbd_image in merged_rgbd_images.items():
        depth = np.array(rgbd_image.depth)
        assert np.sum(depth <= 0) == 0
        assert np.sum(depth > 30) == 0

    for rgbd_frame_idx, rgbd_image in carved_merged_rgbd_images.items():
        depth = np.array(rgbd_image.depth)
        assert np.sum(depth <= 0) == 0
        assert np.sum(depth > 30) == 0
    '''

    # %%

    '''
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1024, height=768, left=200, top=200)

    axes_geometry = o3d.geometry.LineSet(o3d.utility.Vector3dVector([[0, 0, 0],
                                                                     [1, 0, 0],
                                                                     [0, 1, 0],
                                                                     [0, 0, 1]]),
                                         o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]]))
    axes_geometry.scale(20, [0, 0, 0])
    axes_geometry.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    vis.add_geometry(axes_geometry, reset_bounding_box=True)

    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    # vector from the lookat point to the camera
    # make gaze from the camera to lookat point left-ward and down-ward
    ctr.set_front([0.5, -0.5, -0.5])
    ctr.set_zoom(1.0)
    ctr.set_constant_z_far(200.0)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(model_triangulated_points[:3, post_optimise_triangulated_idxs_mask].T))
    pcd.colors = o3d.utility.Vector3dVector(model_triangulated_points[3:, post_optimise_triangulated_idxs_mask].T / 255)
    vis.add_geometry(pcd.crop(o3d.geometry.AxisAlignedBoundingBox([-100, -100, -100], [100, 100, 100])), reset_bounding_box=False)

    view_status = vis.get_view_status()
    view_status_time = time.time()
    visualisation_idle_timeout = 60 if IPython.get_ipython() is not None else 10
    while True:
        close_vis = False
        geometries = []
        for rgbd_frame_idx, (prev_camera_extrinsic, camera_extrinsic) in enumerate(zip([None] + camera_extrinsics[:-1], camera_extrinsics)):
            # The extrinsic matrix transforms from world coordinates to camera coordinates
            camera_lines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=image_size[0], view_height_px=image_size[1],
                                                                            intrinsic=camera_intrinsic,
                                                                            extrinsic=camera_extrinsic)
            camera_lines.paint_uniform_color([0, 0.5, 1])
            if prev_camera_extrinsic is not None:
                camera_lines.points = o3d.utility.Vector3dVector(np.vstack([camera_lines.points,
                                                                            np.linalg.inv(prev_camera_extrinsic)[:3, 3],
                                                                            np.linalg.inv(camera_extrinsic)[:3, 3]]))
                camera_lines.lines = o3d.utility.Vector2iVector(np.vstack([camera_lines.lines,
                                                                           [len(camera_lines.points) - 2, len(camera_lines.points) - 1]]))
                camera_lines.colors = o3d.utility.Vector3dVector(np.vstack([camera_lines.colors,
                                                                            [1, 0, 0.5]]))
            vis.add_geometry(camera_lines, reset_bounding_box=False)
            geometries.append(camera_lines)

            if rgbd_frame_idx in carved_merged_rgbd_images:
                rgbd_image = carved_merged_rgbd_images[rgbd_frame_idx]
                rgbd_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                                          o3d.camera.PinholeCameraIntrinsic(np.array(rgbd_image.depth).shape[1],
                                                                                                            np.array(rgbd_image.depth).shape[0],
                                                                                                            camera_intrinsic),
                                                                          camera_extrinsic)
                pcd = rgbd_pcd.crop(o3d.geometry.AxisAlignedBoundingBox([-100, -100, -100], [100, 100, 100]))
                vis.add_geometry(pcd, reset_bounding_box=False)
            else:
                pcd = None

            start_time = time.time()
            while True:
                close_vis = not vis.poll_events()
                vis.update_renderer()
                new_view_status = vis.get_view_status()
                if new_view_status != view_status:
                    view_status = new_view_status
                    view_status_time = time.time()
                elif time.time() > view_status_time + visualisation_idle_timeout:
                    close_vis = True
                if close_vis or time.time() > start_time + 0.05:
                    break

            if pcd is not None:
                vis.remove_geometry(pcd, reset_bounding_box=False)

            if close_vis:
                break

        if close_vis:
            break

        for geometry in geometries:
            vis.remove_geometry(geometry, reset_bounding_box=False)

    vis.destroy_window()
    '''

    # %%

    """
    # Note: attempting to use o3d.visualization.rendering.OffscreenRenderer results in an "EGL Headless is not supported on this platform." error

    rgbd_meshes = {}
    for rgbd_frame_idx in depth_image_frame_idxs:
        camera_extrinsic = camera_extrinsics[frame_idx]

        img = filtered_frame_images[frame_idx]
        h, w = img.shape[:2]

        volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=0.5, sdf_trunc=5.0,
                                                              color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(integrated_carved_merged_rgbd_images_mesh))

        intrinsic_matrix = camera_intrinsic + np.array([[0, 0, w / 2], [0, 0, h / 2], [0, 0, 0]])
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(intrinsic_matrix=intrinsic_matrix,
                                                                  extrinsic_matrix=camera_extrinsic,
                                                                  width_px=w*2, height_px=h*2)

        casted_rays = scene.cast_rays(rays)
        depth_img = casted_rays['t_hit'].numpy()
        depth_img[~np.isfinite(depth_img)] = np.nan
        depth_img[depth_img >= 30] = np.nan

        triangle_idxs = casted_rays['primitive_ids'].numpy()
        mask = triangle_idxs != o3d.t.geometry.RaycastingScene.INVALID_ID

        vertex_idxs = np.array(integrated_carved_merged_rgbd_images_mesh.triangles)[triangle_idxs[mask], :]
        triangle_vertex_colours = np.array(integrated_carved_merged_rgbd_images_mesh.vertex_colors)[vertex_idxs, :]

        uv = casted_rays['primitive_uvs'].numpy()[mask]
        wuv = np.hstack([1 - np.sum(uv, axis=1)[:, None], uv])

        ray_colours = np.sum(triangle_vertex_colours * wuv[:, :, None], axis=1)

        ray_img = np.full(triangle_idxs.shape + (3,), fill_value=127, dtype=np.uint8)
        ray_img[mask, :] = np.clip(ray_colours * 255, 0, 255).astype(np.uint8)

        # Depth values larger than depth_trunc are truncated to 0
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(ray_img),
                                                                        o3d.geometry.Image(np.require(depth_img, dtype=np.float32)),
                                                                        depth_scale=1.0, depth_trunc=np.inf,
                                                                        convert_rgb_to_intensity=False)

        volume.integrate(rgbd_image,
                         o3d.camera.PinholeCameraIntrinsic(np.array(rgbd_image.depth).shape[1],
                                                           np.array(rgbd_image.depth).shape[0],
                                                           intrinsic_matrix),
                         camera_extrinsic)

        with shelve.open(input_path) as depth_images:
            frame_depth_images = depth_images[str(rgbd_frame_idx)]
        for depth_img, normal_img, confidence_map in frame_depth_images.values():
            ref_img = filtered_frame_images[rgbd_frame_idx]
            # Depth values larger than depth_trunc are truncated to 0
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(ref_img),
                                                                            o3d.geometry.Image(depth_img),
                                                                            depth_scale=1.0, depth_trunc=np.inf,
                                                                            convert_rgb_to_intensity=False)
            volume.integrate(rgbd_image,
                             o3d.camera.PinholeCameraIntrinsic(np.array(rgbd_image.depth).shape[1],
                                                               np.array(rgbd_image.depth).shape[0],
                                                               camera_intrinsic),
                             camera_extrinsic)

        rgbd_mesh = volume.extract_triangle_mesh()
        if len(rgbd_mesh.triangles) > 0:
            rgbd_meshes[frame_idx] = rgbd_mesh

    # %%

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1024, height=768, left=200, top=200)

    axes_geometry = o3d.geometry.LineSet(o3d.utility.Vector3dVector([[0, 0, 0],
                                                                     [1, 0, 0],
                                                                     [0, 1, 0],
                                                                     [0, 0, 1]]),
                                         o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]]))
    axes_geometry.scale(20, [0, 0, 0])
    axes_geometry.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    vis.add_geometry(axes_geometry, reset_bounding_box=True)

    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    # vector from the lookat point to the camera
    # make gaze from the camera to lookat point left-ward and down-ward
    ctr.set_front([0.5, -0.5, -0.5])
    ctr.set_zoom(1.0)
    ctr.set_constant_z_far(200.0)

    view_status = vis.get_view_status()
    view_status_time = time.time()
    visualisation_idle_timeout = 60 if IPython.get_ipython() is not None else 10
    while True:
        close_vis = False
        geometries = []
        for rgbd_frame_idx, (prev_camera_extrinsic, camera_extrinsic) in enumerate(zip([None] + camera_extrinsics[:-1], camera_extrinsics)):
            # The extrinsic matrix transforms from world coordinates to camera coordinates
            camera_lines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=image_size[0], view_height_px=image_size[1],
                                                                            intrinsic=camera_intrinsic,
                                                                            extrinsic=camera_extrinsic)
            camera_lines.paint_uniform_color([0, 0.5, 1])
            if prev_camera_extrinsic is not None:
                camera_lines.points = o3d.utility.Vector3dVector(np.vstack([camera_lines.points,
                                                                            np.linalg.inv(prev_camera_extrinsic)[:3, 3],
                                                                            np.linalg.inv(camera_extrinsic)[:3, 3]]))
                camera_lines.lines = o3d.utility.Vector2iVector(np.vstack([camera_lines.lines,
                                                                           [len(camera_lines.points) - 2, len(camera_lines.points) - 1]]))
                camera_lines.colors = o3d.utility.Vector3dVector(np.vstack([camera_lines.colors,
                                                                            [1, 0, 0.5]]))
            vis.add_geometry(camera_lines, reset_bounding_box=False)
            geometries.append(camera_lines)

            if rgbd_frame_idx in rgbd_meshes:
                mesh = rgbd_meshes[rgbd_frame_idx].crop(o3d.geometry.AxisAlignedBoundingBox([-100, -100, -100], [100, 100, 100]))
                vis.add_geometry(mesh, reset_bounding_box=False)
            else:
                mesh = None

            start_time = time.time()
            while True:
                close_vis = not vis.poll_events()
                vis.update_renderer()
                new_view_status = vis.get_view_status()
                if new_view_status != view_status:
                    view_status = new_view_status
                    view_status_time = time.time()
                elif time.time() > view_status_time + visualisation_idle_timeout:
                    close_vis = True
                if close_vis or time.time() > start_time + 0.05:
                    break

            if mesh is not None:
                vis.remove_geometry(mesh, reset_bounding_box=False)

            if close_vis:
                break

        if close_vis:
            break

        for geometry in geometries:
            vis.remove_geometry(geometry, reset_bounding_box=False)

    vis.destroy_window()
    """

    # %%

    output_path = output_dirpath / 'integrated_depth_images.pickle'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as pickle_file:
        pickle.dump({'camera_pull_back_z': camera_pull_back_z,
                     'synthetic_camera_zoom': synthetic_camera_zoom,
                     'camera_intrinsic_synthetic': camera_intrinsic_synthetic,
                     'integrated_weighted_canvas_meshes': [{'vertices': np.array(canvas_mesh.vertices, dtype=np.float32),
                                                            'triangles': np.array(canvas_mesh.triangles),
                                                            'vertex_normals': np.array(canvas_mesh.vertex_normals, dtype=np.float32),
                                                            'vertex_colors': np.array(canvas_mesh.vertex_colors, dtype=np.float32)}
                                                           for canvas_mesh in integrated_weighted_canvas_meshes],
                     'integrated_weighted_depth_kernels': integrated_weighted_depth_kernels},
                    pickle_file)

    # %%

    post_to_pipeline_server((f'{working_subdir} / {output_dirpath.name}', None))
