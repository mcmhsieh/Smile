"""
Compare the projection errors between each frame by casting camera rays onto
their respective canvas/surface meshes.
Select target frames for view synthesis by optimising a loss function designed
to converge on a subset of frames that represent clusters of similar views.
Generate synthetic views by optimising the projection and intensity mapping
of each synthesised image point.

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
import fractions
import collections
import itertools

import numpy as np
import scipy
import sklearn.cluster
import sklearn.mixture
import rsatoolbox
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

from image_filtering import rgb_to_gray, nan_gaussian_filter
from weighting_functions import cauchy
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
    input_source_dirpath = workspace_dirpath / 'integrate_depth_images'
    output_dirpath = workspace_dirpath / 'view_synthesis'

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

    input_path = input_source_dirpath / 'integrated_depth_images.pickle'
    with open(input_path, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
        camera_pull_back_z = data['camera_pull_back_z']
        synthetic_camera_zoom = data['synthetic_camera_zoom']
        camera_intrinsic_synthetic = data['camera_intrinsic_synthetic']
        integrated_weighted_canvas_meshes = []
        for canvas_mesh_data in data['integrated_weighted_canvas_meshes']:
            canvas_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(canvas_mesh_data['vertices']),
                                                    o3d.utility.Vector3iVector(canvas_mesh_data['triangles']))
            canvas_mesh.vertex_normals = o3d.utility.Vector3dVector(canvas_mesh_data['vertex_normals'])
            canvas_mesh.vertex_colors = o3d.utility.Vector3dVector(canvas_mesh_data['vertex_colors'])
            integrated_weighted_canvas_meshes.append(canvas_mesh)
        integrated_weighted_depth_kernels = data['integrated_weighted_depth_kernels']

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

    '''
    integrated_carved_merged_rgbd_images_signed_distances = {}
    for rgbd_frame_idx, rgbd_image in carved_merged_rgbd_images.items():
        volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=0.3, sdf_trunc=10.0,
                                                              color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        volume.integrate(rgbd_image,
                         o3d.camera.PinholeCameraIntrinsic(np.array(rgbd_image.depth).shape[1],
                                                           np.array(rgbd_image.depth).shape[0],
                                                           camera_intrinsic),
                         camera_extrinsics[rgbd_frame_idx])
        voxel_pcd = volume.extract_voxel_point_cloud()
        kd_tree = scipy.spatial.KDTree(np.array(voxel_pcd.points))
        signed_distances = np.mean(np.array(voxel_pcd.colors), axis=1) - 0.5
        integrated_carved_merged_rgbd_images_signed_distances[rgbd_frame_idx] = (kd_tree, signed_distances)

    # %%

    interframe_incoherence = np.full((len(key_frame_indices), len(key_frame_indices), 2), fill_value=np.nan, dtype=np.float32)
    for rgbd_frame_idx, rgbd_image in carved_merged_rgbd_images.items():
        rgbd_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                                  o3d.camera.PinholeCameraIntrinsic(np.array(rgbd_image.depth).shape[1],
                                                                                                    np.array(rgbd_image.depth).shape[0],
                                                                                                    camera_intrinsic),
                                                                  camera_extrinsics[rgbd_frame_idx])
        rgbd_down_pcd = rgbd_pcd.voxel_down_sample(voxel_size=0.15)

        for frame_idx, (kd_tree, signed_distances) in integrated_carved_merged_rgbd_images_signed_distances.items():
            nn_distances, nn_point_idxs = kd_tree.query(np.array(rgbd_down_pcd.points), k=1, distance_upper_bound=1.0)
            rgbd_frame_signed_distances = np.full(nn_point_idxs.shape, fill_value=np.nan, dtype=np.float32)
            valid_idxs = nn_point_idxs < kd_tree.n
            interframe_incoherence[frame_idx, rgbd_frame_idx, 1] = np.mean(valid_idxs)
            if np.sum(valid_idxs) > 0:
                rgbd_frame_signed_distances[valid_idxs] = signed_distances[nn_point_idxs[valid_idxs]]
                interframe_incoherence[frame_idx, rgbd_frame_idx, 0] = np.sqrt(np.nanmean(np.power(rgbd_frame_signed_distances, 2)))
            print(frame_idx, rgbd_frame_idx, interframe_incoherence[frame_idx, rgbd_frame_idx])
    '''

    # %%

    '''
    geometries = []
    for rgbd_frame_idx in [10, 11]:
        rgbd_image = carved_merged_rgbd_images[rgbd_frame_idx]
        rgbd_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                                  o3d.camera.PinholeCameraIntrinsic(np.array(rgbd_image.depth).shape[1],
                                                                                                    np.array(rgbd_image.depth).shape[0],
                                                                                                    camera_intrinsic),
                                                                  camera_extrinsics[rgbd_frame_idx])
        geometries.append(rgbd_pcd)
    visualise_geometries(geometries)
    '''

    # %%

    '''
    plt.close('RGBD normal weights')

    carved_merged_rgbd_normal_weights = {}
    for rgbd_frame_idx, rgbd_image in carved_merged_rgbd_images.items():
        rgbd_depth = np.array(rgbd_image.depth)
        normal_weights = np.full(rgbd_depth.shape, fill_value=np.nan, dtype=np.float32)
        rgbd_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                                  o3d.camera.PinholeCameraIntrinsic(rgbd_depth.shape[1],
                                                                                                    rgbd_depth.shape[0],
                                                                                                    camera_intrinsic),
                                                                  np.identity(4))

        if np.sum(np.isfinite(rgbd_image.depth)) > 30:
            rgbd_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
            rgbd_pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))
            #rgbd_pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, -1]))
            #rgbd_pcd.orient_normals_consistent_tangent_plane(**{'k': 30, 'lambda': 0.0, 'cos_alpha_tol': 1.0})
            #if np.mean(np.array(rgbd_pcd.normals)[:, 2]) > 0:
            #    rgbd_pcd.normals = o3d.utility.Vector3dVector(-np.array(rgbd_pcd.normals))
            assert np.allclose(np.linalg.norm(np.array(rgbd_pcd.normals), axis=1), 1)

            triangulated_points = np.array(rgbd_pcd.points).T
            triangulated_normals = np.array(rgbd_pcd.normals).T
            camera_rays = triangulated_points / np.clip(np.linalg.norm(triangulated_points, axis=0), 1e-6, np.inf)
            normal_ray_alignment = np.sum(camera_rays * triangulated_normals, axis=0)
            camera_ray_to_object_plane = camera_rays / np.clip(-normal_ray_alignment, 1e-8, 1)
            camera_ray_to_image_plane = camera_rays * -triangulated_normals[2, :] / camera_rays[2, :]

            object_to_image_ratio = (np.linalg.norm(triangulated_normals + camera_ray_to_object_plane, axis=0)
                                     / np.clip(np.linalg.norm(triangulated_normals + camera_ray_to_image_plane, axis=0), 1e-8, np.inf))
            perspective_distortion = 1 - np.min(np.stack([object_to_image_ratio,
                                                          np.clip(1 / object_to_image_ratio, 1e-8, np.inf)]), axis=0)

            triangulated_normal_weights = 0.5 * (1 - scipy.special.erf((perspective_distortion - 0.4) / 0.15))
            triangulated_normal_weights[normal_ray_alignment >= 0] = 0

            y, x = np.where(np.isfinite(rgbd_depth))
            normal_weights[y, x] = triangulated_normal_weights

        carved_merged_rgbd_normal_weights[rgbd_frame_idx] = normal_weights

        plt.figure('RGBD normal weights', figsize=(16, 10))
        setup_new_fig_page()
        plt.suptitle(f'rgbd frame idx: {rgbd_frame_idx}')
        ax = plt.subplot(2, 2, 1)
        plt.imshow(rgbd_depth)
        plt.title('rgbd_depth')
        ax = plt.subplot(2, 2, 3, sharex=ax, sharey=ax)
        plt.imshow(normal_weights)
        plt.title('normal_weights')
        rgbd_pcd_down = rgbd_pcd.voxel_down_sample(voxel_size=0.2)
        rgbd_pcd_down.transform(np.linalg.inv(camera_extrinsics[rgbd_frame_idx]))
        ax3 = plt.subplot(1, 2, 2, projection='3d')
        ax3.scatter(*np.array(rgbd_pcd_down.points).T, s=2, c=np.array(rgbd_pcd_down.colors))
        ax3.set_xlim((-20, 20))
        ax3.set_ylim((-20, 20))
        ax3.set_zlim((0, 40))
        ax3.set_aspect('equal', adjustable='datalim')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.view_init(elev=-135, azim=-90, roll=0)
        plt.tight_layout()
        stash_fig_page()
    '''

    # %%

    '''
    # Compute an inter frame incoherence metric based on the difference in depth of projected points.
    # Each point comparison is weighted by their depth and the alignment of their surface normals to their camera rays.
    # The inter frame metric weight is based on the proportion of overlapping points, each weighted by the alignment
    # of their surface normals to their camera rays.
    interframe_incoherence = np.full((len(key_frame_indices), len(key_frame_indices), 2), fill_value=np.nan, dtype=np.float32)
    for rgbd_frame_idx, rgbd_image in carved_merged_rgbd_images.items():
        rgbd_depth = np.array(rgbd_image.depth)
        rgbd_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                                  o3d.camera.PinholeCameraIntrinsic(rgbd_depth.shape[1],
                                                                                                    rgbd_depth.shape[0],
                                                                                                    camera_intrinsic),
                                                                  camera_extrinsics[rgbd_frame_idx])
        y, x = np.where(np.isfinite(rgbd_depth))
        rgbd_normal_weights = carved_merged_rgbd_normal_weights[rgbd_frame_idx][y, x]

        for secondary_frame_idx, secondary_rgbd_image in carved_merged_rgbd_images.items():
            secondary_rgbd_depth = np.array(secondary_rgbd_image.depth)
            secondary_rgbd_normal_weights = carved_merged_rgbd_normal_weights[secondary_frame_idx]

            pcd = o3d.geometry.PointCloud(rgbd_pcd)
            pcd.transform(camera_extrinsics[secondary_frame_idx])
            projected_points = camera_intrinsic @ np.array(pcd.points).T
            inlier_idxs = np.where(projected_points[2, :] > 0)[0]
            projected_points = np.round(projected_points[:2, inlier_idxs] / projected_points[2, inlier_idxs])
            projected_inlier_idxs = ((projected_points[0, :] >= 0) & (projected_points[0, :] <= secondary_rgbd_depth.shape[1] - 1)
                                     & (projected_points[1, :] >= 0) & (projected_points[1, :] <= secondary_rgbd_depth.shape[0] - 1))

            primary_inlier_depth = np.array(pcd.points)[inlier_idxs[projected_inlier_idxs], 2]
            primary_inlier_normal_weights = rgbd_normal_weights[inlier_idxs[projected_inlier_idxs]]
            assert np.all(np.isfinite(primary_inlier_depth))
            assert np.all(np.isfinite(primary_inlier_normal_weights))

            secondary_inlier_depth = secondary_rgbd_depth[projected_points[1, projected_inlier_idxs].astype(np.int32),
                                                          projected_points[0, projected_inlier_idxs].astype(np.int32)]
            secondary_inlier_normal_weights = secondary_rgbd_normal_weights[projected_points[1, projected_inlier_idxs].astype(np.int32),
                                                                            projected_points[0, projected_inlier_idxs].astype(np.int32)]

            if np.nansum(rgbd_normal_weights) > 0:
                normal_weights = primary_inlier_normal_weights * secondary_inlier_normal_weights
                weight = np.nansum(normal_weights) / np.nansum(np.power(rgbd_normal_weights, 2))
                interframe_incoherence[rgbd_frame_idx, secondary_frame_idx, 1] = weight
                if weight > 0:
                    # TODO: should incoherence be based on inverse depth?
                    if True:
                        incoherence = 1 - cauchy(secondary_inlier_depth - primary_inlier_depth,
                                                 np.sqrt(primary_inlier_depth * secondary_inlier_depth) / 3)
                        incoherence = np.nansum(incoherence * normal_weights) / np.nansum(normal_weights)
                    elif True:
                        incoherence = np.nanmean(1 - cauchy(secondary_inlier_depth - primary_inlier_depth,
                                                            3.0 * np.linalg.norm(np.stack([primary_inlier_depth, secondary_inlier_depth]), axis=0)))
                        #incoherence = np.sqrt(incoherence)
                    elif False:
                        incoherence = np.sqrt(np.nanmean(np.power((secondary_inlier_depth - primary_inlier_depth)
                                                                  / np.linalg.norm(np.stack([primary_inlier_depth, secondary_inlier_depth]), axis=0), 2)))
                    else:
                        incoherence = np.sqrt(np.nanmean(np.power((1 / secondary_inlier_depth - 1 / primary_inlier_depth)
                                                                  * np.linalg.norm(np.stack([primary_inlier_depth, secondary_inlier_depth]), axis=0), 2)))
                    interframe_incoherence[rgbd_frame_idx, secondary_frame_idx, 0] = incoherence
            print(rgbd_frame_idx, secondary_frame_idx, interframe_incoherence[rgbd_frame_idx, secondary_frame_idx])
    '''

    # %%

    '''
    mesh_frame_dissimilarity_threshold = 0.3
    mesh_frame_dissimilarity_boundary = 2 * mesh_frame_dissimilarity_threshold
    '''

    interframe_angles = []
    for primary_frame_camera_extrinsic in camera_extrinsics:
        interframe_angles.append([])
        for secondary_frame_camera_extrinsic in camera_extrinsics:
            camera_transform = secondary_frame_camera_extrinsic @ np.linalg.inv(primary_frame_camera_extrinsic)
            rvec, _ = cv2.Rodrigues(camera_transform[:3, :3])
            # Ignore rotation around the z-axis (in the primary frame of reference)
            interframe_angles[-1].append(np.linalg.norm(rvec[:2]))
    interframe_angles = np.array(interframe_angles)

    # TODO: some of the interframe_confidences may need to be recalculated depending on whether
    #       the original camera transform estimate, reprojection of points, epipolar lines or rectification
    #       changed significantly after optimisation
    interframe_confidences = np.full((len(key_frame_indices),) * 2, fill_value=np.nan, dtype=np.float32)
    interframe_num_disparities = np.full((len(key_frame_indices),) * 2, fill_value=np.nan, dtype=np.float32)
    for (cross_frame_idx, current_frame_idx), (img_size_trim, rect_proximity,
                                               min_disparity, max_disparity, num_disparities,
                                               disparity_confidence_map, filtered_disparity_confidence_map,
                                               cross_triangulated_image_points) in cross_stitch_disparity_confidence_maps.items():

        pad_width = 16
        image_mask = np.full(np.array(filtered_disparity_confidence_map.shape[:2]) + pad_width * 2, fill_value=0, dtype=np.uint8)
        image_points = np.round(cross_triangulated_image_points).astype(int) + pad_width
        image_mask[image_points[:, 1], image_points[:, 0]] = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        image_mask = cv2.morphologyEx(image_mask, op=cv2.MORPH_CLOSE, kernel=kernel,
                                      iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)

        image_mask_idxs = np.where(image_mask[pad_width:-pad_width, pad_width:-pad_width])

        if np.array(image_mask_idxs).size > 0:
            filtered_confidence_values = filtered_disparity_confidence_map[*image_mask_idxs]
            interframe_confidences[cross_frame_idx, current_frame_idx] = np.nansum(filtered_confidence_values) / len(filtered_confidence_values)
            interframe_confidences[current_frame_idx, cross_frame_idx] = np.nansum(filtered_confidence_values) / len(filtered_confidence_values)

        interframe_num_disparities[cross_frame_idx, current_frame_idx] = num_disparities
        interframe_num_disparities[current_frame_idx, cross_frame_idx] = num_disparities

    '''
    interframe_incoherence_values = interframe_incoherence[~np.identity(interframe_incoherence.shape[0], dtype=bool), 0]
    interframe_incoherence_values = interframe_incoherence_values[np.isfinite(interframe_incoherence_values)]
    gm = sklearn.mixture.GaussianMixture(n_components=2, covariance_type='diag', random_state=0)
    gm.fit(interframe_incoherence_values[:, None])
    component_idxs = np.argsort(gm.means_[:, 0])
    print('outlier component P(X = mesh_frame_dissimilarity_threshold)',
          gm.weights_[component_idxs[1]] * scipy.stats.norm.pdf(mesh_frame_dissimilarity_threshold,
                                                                loc=gm.means_[component_idxs[1], 0],
                                                                scale=np.sqrt(gm.covariances_[component_idxs[1], 0])))
    print('outlier component P(X < mesh_frame_dissimilarity_threshold)',
          gm.weights_[component_idxs[1]] * scipy.stats.norm.cdf(mesh_frame_dissimilarity_threshold,
                                                                loc=gm.means_[component_idxs[1], 0],
                                                                scale=np.sqrt(gm.covariances_[component_idxs[1], 0])))
    print('outlier component P(X < mesh_frame_dissimilarity_threshold | outlier)',
          scipy.stats.norm.cdf(mesh_frame_dissimilarity_threshold,
                               loc=gm.means_[component_idxs[1], 0],
                               scale=np.sqrt(gm.covariances_[component_idxs[1], 0])))
    print('inlier component P(X = mesh_frame_dissimilarity_boundary)',
          gm.weights_[component_idxs[0]] * scipy.stats.norm.pdf(mesh_frame_dissimilarity_boundary,
                                                                loc=gm.means_[component_idxs[0], 0],
                                                                scale=np.sqrt(gm.covariances_[component_idxs[0], 0])))
    print('inlier component P(X > mesh_frame_dissimilarity_boundary)',
          gm.weights_[component_idxs[0]] * (1 - scipy.stats.norm.cdf(mesh_frame_dissimilarity_boundary,
                                                                     loc=gm.means_[component_idxs[0], 0],
                                                                     scale=np.sqrt(gm.covariances_[component_idxs[0], 0]))))
    print('inlier component P(X > mesh_frame_dissimilarity_boundary | inlier)',
          1 - scipy.stats.norm.cdf(mesh_frame_dissimilarity_boundary,
                                   loc=gm.means_[component_idxs[0], 0],
                                   scale=np.sqrt(gm.covariances_[component_idxs[0], 0])))

    plt.figure('Interframe dissimilarities', figsize=(16, 10))
    plt.clf()
    ax = plt.subplot(2, 3, 1)
    plt.imshow(interframe_incoherence[:, :, 0], cmap='jet', interpolation='none')
    plt.title('interframe_incoherence [value]')
    ax = plt.subplot(2, 3, 2, sharex=ax, sharey=ax)
    plt.imshow(interframe_incoherence[:, :, 1], cmap='jet', interpolation='none')
    plt.title('interframe_incoherence [weight]')
    ax = plt.subplot(2, 3, 4, sharex=ax, sharey=ax)
    plt.imshow(interframe_angles)
    plt.title('interframe_angles')
    ax = plt.subplot(2, 3, 5, sharex=ax, sharey=ax)
    ax.set_facecolor('grey')
    plt.imshow(interframe_confidences, vmin=-1, vmax=1, cmap='seismic', interpolation='none')
    plt.title('interframe_confidences')
    ax = plt.subplot(2, 3, 6, sharex=ax, sharey=ax)
    plt.imshow(interframe_num_disparities, cmap='jet', interpolation='none')
    plt.title('interframe_num_disparities')
    plt.subplot(2, 3, 3)
    bins = np.linspace(0, 1, 51)
    plt.hist(interframe_incoherence_values, bins=bins, density=True)
    for component_idx in np.argsort(gm.means_[:, 0]):
        plt.plot(bins, gm.weights_[component_idx] * scipy.stats.norm.pdf(bins, loc=gm.means_[component_idx, 0],
                                                                         scale=np.sqrt(gm.covariances_[component_idx, 0])))
    plt.axvline(mesh_frame_dissimilarity_threshold, color='black', linestyle='--')
    plt.axvline(mesh_frame_dissimilarity_boundary, color='red', linestyle='--')
    plt.plot(mesh_frame_dissimilarity_threshold,
             gm.weights_[component_idxs[1]] * scipy.stats.norm.pdf(mesh_frame_dissimilarity_threshold,
                                                                   loc=gm.means_[component_idxs[1], 0],
                                                                   scale=np.sqrt(gm.covariances_[component_idxs[1], 0])),
             'ko')
    plt.plot(mesh_frame_dissimilarity_boundary,
             gm.weights_[component_idxs[0]] * scipy.stats.norm.pdf(mesh_frame_dissimilarity_boundary,
                                                                   loc=gm.means_[component_idxs[0], 0],
                                                                   scale=np.sqrt(gm.covariances_[component_idxs[0], 0])),
             'ro')
    plt.title('interframe_incoherence [value]')
    plt.tight_layout()
    '''

    # %%

    '''
    for n_components in range(3, len(key_frame_indices) + 1):
        mds = sklearn.manifold.MDS(n_components=n_components, metric=False, n_init=10, max_iter=10000, random_state=0, dissimilarity='precomputed')
        # When metric_mds=False (i.e. non-metric MDS), dissimilarities with 0 are considered as missing values.
        X = interframe_incoherence[:, :, 0] + interframe_incoherence[:, :, 0].T
        X[~np.isfinite(X)] = 0
        mds.fit(X)

        # The final value of the stress (sum of squared distance of the disparities and the distances for all constrained points).
        # If normalized_stress=True, and metric=False returns Stress-1.
        # A value of 0 indicates “perfect” fit, 0.025 excellent, 0.05 good, 0.1 fair, and 0.2 poor.
        print(n_components, mds.stress_)

        if n_components == 3:
            plt.figure(figsize=(16, 10))
            ax3 = plt.subplot(1, 1, 1, projection='3d')
            ax3.scatter(*mds.embedding_.T)
            ax3.set_aspect('equal', adjustable='datalim')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            plt.tight_layout()

    # %%

    interframe_distances = sklearn.metrics.pairwise_distances(mds.embedding_)

    plt.figure(figsize=(16, 10))
    plt.imshow(mds.dissimilarity_matrix_, cmap='jet', interpolation='none')
    plt.figure(figsize=(16, 10))
    plt.imshow(interframe_distances, cmap='jet', interpolation='none')

    # %%

    # which clustering algorithm can minimise the intra cluster diameter?

    gm = sklearn.mixture.GaussianMixture(n_components=3, covariance_type='diag', random_state=0)
    gm.fit(mds.embedding_)

    print(gm.converged_)
    print(gm.weights_)
    predictions = gm.predict(mds.embedding_)
    proba = gm.predict_proba(mds.embedding_)
    plt.figure(figsize=(16, 10))
    ax = plt.subplot(2, 1, 1)
    plt.plot(predictions)
    ax = plt.subplot(2, 1, 2, sharex=ax)
    plt.plot(proba)
    '''

    # %%

    '''
    # Combine inter frame incoherence and inter frame angles into a weighted dissimilarity metric.
    # Apply weighted MDS to fit embedding vectors to the weighted dissimilarity metric,
    # and compute inter frame costs from the distances between embedding vectors.

    # TODO: if the dissimilarities matrix is sparse, e.g. when not combining with interframe_angles,
    # check that embedding nodes are all interconnected (i.e. as one group, not multiple or isolated groups)
    if True:
        interframe_incoherence_stacked = np.array(np.stack([interframe_incoherence, interframe_incoherence.transpose(1, 0, 2)]))
        valid_mask = np.any(interframe_incoherence_stacked[:, :, :, 1] > 0, axis=0)
        #embedding_frame_idxs = np.where(np.sum(valid_mask, axis=0) > 1)[0]
        interframe_incoherence_stacked[:, ~valid_mask, :] = 0
        dissimilarities = np.nan_to_num(np.nanmean(interframe_incoherence_stacked[:, :, :, 0], axis=0))
        weights = np.nan_to_num(np.nanmean(interframe_incoherence_stacked[:, :, :, 1], axis=0))
    else:
        symmetric_interframe_incoherence = 0.5 * (interframe_incoherence + np.transpose(interframe_incoherence, axes=(1, 0, 2)))
        #embedding_frame_idxs = np.where(np.sum(symmetric_interframe_incoherence[:, :, 1] > 0, axis=0) > 1)[0]
        dissimilarities = np.nan_to_num(symmetric_interframe_incoherence[:, :, 0])
        weights = np.nan_to_num(symmetric_interframe_incoherence[:, :, 1])

    if True:
        # TODO: Could try using interframe_num_disparities instead of interframe_angles?
        if False:
            # Apply interframe angle dissimilarity weighting when frustums overlap (i.e. where disparity has been computed)
            # and the disparity confidence is low
            interframe_angles_dissimilarity = mesh_frame_dissimilarity_boundary * (1 - cauchy(interframe_angles, np.pi / 4))
            angle_weight = 1.0 + 1.0 * np.nan_to_num(1 - interframe_confidences)
        elif True:
            # If the disparity confidence is high, then suppress incoherence dissimilarity more heavily especially with low inter frame angle
            # If the disparity confidence is low, then suppress incoherence dissimilarity from lightly to moderately for decreasing inter frame angle
            #                            Dissimilarity     Weight
            #                               Angle:         Angle:
            #                             low   high     low   high
            # Disparity confidence: low   low   high     0.5   0.01
            #                       high  v.low mid      1.0   0.2
            # interframe_angles_dissimilarity = 0.5 * mesh_frame_dissimilarity_boundary when interframe_angles = np.pi / 12
            #interframe_angles_dissimilarity = mesh_frame_dissimilarity_boundary * (1 - cauchy(interframe_angles, np.pi / 4))
            normalised_interframe_confidences = np.nan_to_num(interframe_confidences + 1) / 2
            interframe_angles_dissimilarity = mesh_frame_dissimilarity_boundary * (1 - cauchy(interframe_angles, np.pi / 4 * (1 + normalised_interframe_confidences)))
            #angle_weight = 0.1 + (0.4 + 0.5 * np.nan_to_num(interframe_confidences)) * cauchy(interframe_angles, np.pi / 8)
            angle_weight = ((0.5 + 0.5 * normalised_interframe_confidences) * cauchy(interframe_angles, np.pi / 8)
                            + (0.01 + 0.19 * normalised_interframe_confidences) * (1 - cauchy(interframe_angles, np.pi / 8)))
        elif True:
            interframe_angles_dissimilarity = 0.2 * (1 - cauchy(interframe_angles, np.pi / 4))
            angle_weight = cauchy(interframe_angles, np.pi / 4) * (0.3 + 0.3 * np.nan_to_num(1 - interframe_confidences))
        dissimilarities = (dissimilarities * weights + interframe_angles_dissimilarity * angle_weight) / np.maximum(weights + angle_weight, 1e-8)
        weights += angle_weight
        #embedding_frame_idxs = np.arange(len(key_frame_indices))

    embedding_frame_idxs = np.where(np.sum(weights, axis=0) > 1)[0]
    embedding_dissimilarities = dissimilarities[embedding_frame_idxs[:, None], embedding_frame_idxs]
    embedding_weights = weights[embedding_frame_idxs[:, None], embedding_frame_idxs]

    # Higher n_components should converge to an embedding that fits dissimilarities with lower residual errors / stress,
    # but may not generalise well to missing dissimilarities?
    # Also the variance of distances (such as Euclidean or Manhattan) between points converges to zero as the number of
    # dimensions increases https://towardsdatascience.com/curse-of-dimensionality-an-intuitive-exploration-1fbf155e1411/
    for n_components in [2, 3, 6]:
        embedding, stress, n_iter = rsatoolbox.util.vis_utils.smacof(embedding_dissimilarities, metric=True, n_components=n_components, n_init=8, n_jobs=-1,
                                                                     max_iter=300, random_state=0, return_n_iter=True, weight=embedding_weights)

        interframe_costs = np.full((len(key_frame_indices), len(key_frame_indices)), fill_value=np.nan, dtype=np.float32)
        interframe_costs[embedding_frame_idxs[:, None], embedding_frame_idxs] = sklearn.metrics.euclidean_distances(embedding)

        interframe_cost_values = interframe_costs[~np.identity(interframe_costs.shape[0], dtype=bool)]
        interframe_cost_values = interframe_cost_values[np.isfinite(interframe_cost_values)]

        gm = sklearn.mixture.GaussianMixture(n_components=2, covariance_type='diag', random_state=0)
        gm.fit(interframe_cost_values[:, None])
        component_idxs = np.argsort(gm.means_[:, 0])

        # https://stackoverflow.com/questions/22579434/python-finding-the-intersection-point-of-two-gaussian-curves
        a = 1 / (2 * gm.covariances_[component_idxs[0], 0]) - 1 / (2 * gm.covariances_[component_idxs[1], 0])
        b = (gm.means_[component_idxs[1], 0] / gm.covariances_[component_idxs[1], 0]
             - gm.means_[component_idxs[0], 0] / gm.covariances_[component_idxs[0], 0])
        c = (np.power(gm.means_[component_idxs[0], 0], 2) / (2 * gm.covariances_[component_idxs[0], 0])
             - np.power(gm.means_[component_idxs[1], 0], 2) / (2 * gm.covariances_[component_idxs[1], 0])
             - np.log(gm.covariances_[component_idxs[1], 0] / gm.covariances_[component_idxs[0], 0]) / 2
             - np.log(gm.weights_[component_idxs[0]] / gm.weights_[component_idxs[1]]))
        gm_intersections = np.roots([a, b, c])
        inter_idxs = np.where((gm.means_[component_idxs[0], 0] < gm_intersections) & (gm_intersections < gm.means_[component_idxs[1], 0]))[0]
        assert len(inter_idxs) == 1
        mesh_frame_dissimilarity_cluster_intersection = gm_intersections[inter_idxs[0]]

        if True:
            #distance_threshold = mesh_frame_dissimilarity_cluster_intersection
            distance_threshold = 1.5 * mesh_frame_dissimilarity_threshold
            #distance_threshold = mesh_frame_dissimilarity_boundary
            clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=None, linkage='complete',
                                                                 distance_threshold=distance_threshold)
            clustering.fit(embedding)
            embedding_cluster_count = clustering.n_clusters_
        elif False:
            #bandwidth = sklearn.cluster.estimate_bandwidth(embedding, quantile=0.3, n_samples=500)
            bandwidth = 0.75 * mesh_frame_dissimilarity_threshold
            clustering = sklearn.cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
            clustering.fit(embedding)
            embedding_cluster_count = len(set(clustering.labels_))
        else:
            preference = -15 * np.power(mesh_frame_dissimilarity_threshold, 2)
            clustering = sklearn.cluster.AffinityPropagation(preference=preference, random_state=0)
            clustering.fit(embedding)
            embedding_cluster_count = len(set(clustering.labels_))

        embedding_cluster_labels = np.full((len(key_frame_indices),), fill_value=-1, dtype=np.int32)
        embedding_cluster_labels[embedding_frame_idxs] = clustering.labels_

        print(n_components, stress, n_iter, mesh_frame_dissimilarity_cluster_intersection, embedding_cluster_count)

        print('outlier component P(X = mesh_frame_dissimilarity_threshold)',
              gm.weights_[component_idxs[1]] * scipy.stats.norm.pdf(mesh_frame_dissimilarity_threshold,
                                                                    loc=gm.means_[component_idxs[1], 0],
                                                                    scale=np.sqrt(gm.covariances_[component_idxs[1], 0])))
        print('outlier component P(X < mesh_frame_dissimilarity_threshold)',
              gm.weights_[component_idxs[1]] * scipy.stats.norm.cdf(mesh_frame_dissimilarity_threshold,
                                                                    loc=gm.means_[component_idxs[1], 0],
                                                                    scale=np.sqrt(gm.covariances_[component_idxs[1], 0])))
        print('outlier component P(X < mesh_frame_dissimilarity_threshold | outlier)',
              scipy.stats.norm.cdf(mesh_frame_dissimilarity_threshold,
                                   loc=gm.means_[component_idxs[1], 0],
                                   scale=np.sqrt(gm.covariances_[component_idxs[1], 0])))
        print('inlier component P(X = mesh_frame_dissimilarity_boundary)',
              gm.weights_[component_idxs[0]] * scipy.stats.norm.pdf(mesh_frame_dissimilarity_boundary,
                                                                    loc=gm.means_[component_idxs[0], 0],
                                                                    scale=np.sqrt(gm.covariances_[component_idxs[0], 0])))
        print('inlier component P(X > mesh_frame_dissimilarity_boundary)',
              gm.weights_[component_idxs[0]] * (1 - scipy.stats.norm.cdf(mesh_frame_dissimilarity_boundary,
                                                                         loc=gm.means_[component_idxs[0], 0],
                                                                         scale=np.sqrt(gm.covariances_[component_idxs[0], 0]))))
        print('inlier component P(X > mesh_frame_dissimilarity_boundary | inlier)',
              1 - scipy.stats.norm.cdf(mesh_frame_dissimilarity_boundary,
                                       loc=gm.means_[component_idxs[0], 0],
                                       scale=np.sqrt(gm.covariances_[component_idxs[0], 0])))

        colour_sequence = itertools.cycle(matplotlib.color_sequences['tab10'])
        marker_sequence = itertools.cycle([marker for marker in matplotlib.lines.Line2D.filled_markers if marker not in ['.']])
        if n_components == 2:
            plt.figure('Interframe dissimilarity embedding', figsize=(16, 10))
            setup_new_fig_page()
            plt.suptitle(f'n_components: {n_components}')
            ax = plt.subplot(1, 1, 1)
            plt.plot(*embedding.T, alpha=0.5)
            for cluster_idx, colour, marker in zip(range(embedding_cluster_count), colour_sequence, marker_sequence):
                plt.scatter(*embedding[clustering.labels_ == cluster_idx].T, s=200, color=colour, marker=marker, alpha=0.5)
            for frame_idx, xy in zip(embedding_frame_idxs, embedding):
                plt.text(*xy, f'{frame_idx}')
            ax.set_aspect('equal', adjustable='datalim')
            plt.tight_layout()
            stash_fig_page()
        elif n_components >= 3:
            if n_components > 3:
                mds = sklearn.manifold.MDS(n_components=3, metric=True, random_state=0)
                mds.fit(embedding)
                embedding_3d = mds.embedding_
            else:
                embedding_3d = embedding
            plt.figure('Interframe dissimilarity embedding', figsize=(16, 10))
            setup_new_fig_page()
            plt.suptitle(f'n_components: {n_components}')
            ax3 = plt.subplot(1, 1, 1, projection='3d')
            ax3.plot(*embedding_3d.T, alpha=0.5)
            for cluster_idx, colour, marker in zip(range(embedding_cluster_count), colour_sequence, marker_sequence):
                ax3.scatter(*embedding_3d[clustering.labels_ == cluster_idx].T, s=200, color=colour, marker=marker, alpha=0.5)
            for frame_idx, xyz in zip(embedding_frame_idxs, embedding_3d):
                ax3.text(*xyz, f'{frame_idx}')
            ax3.set_aspect('equal', adjustable='datalim')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            plt.tight_layout()
            stash_fig_page()

        plt.figure('Interframe costs', figsize=(16, 10))
        setup_new_fig_page()
        plt.suptitle(f'n_components: {n_components}')
        ax = plt.subplot(2, 2, 1)
        plt.imshow(dissimilarities, cmap='jet', interpolation='none')
        plt.title('dissimilarities')
        ax = plt.subplot(2, 2, 2, sharex=ax, sharey=ax)
        plt.imshow(interframe_costs, cmap='jet', interpolation='none')
        plt.title('interframe_costs')
        ax = plt.subplot(2, 2, 3, sharex=ax, sharey=ax)
        plt.imshow(interframe_costs / np.maximum(dissimilarities, 1e-8), vmin=0.2, vmax=5, cmap='jet', interpolation='none')
        plt.title('interframe_costs / dissimilarities')
        plt.subplot(2, 2, 4)
        bins = np.linspace(0, 1, 51)
        plt.hist(interframe_cost_values, bins=bins, density=True)
        for component_idx in np.argsort(gm.means_[:, 0]):
            plt.plot(bins, gm.weights_[component_idx] * scipy.stats.norm.pdf(bins, loc=gm.means_[component_idx, 0],
                                                                             scale=np.sqrt(gm.covariances_[component_idx, 0])))
        plt.axvline(mesh_frame_dissimilarity_threshold, color='black', linestyle='--')
        plt.axvline(mesh_frame_dissimilarity_boundary, color='red', linestyle='--')
        plt.axvline(mesh_frame_dissimilarity_cluster_intersection, color='blue', linestyle='--')
        plt.plot(mesh_frame_dissimilarity_threshold,
                 gm.weights_[component_idxs[1]] * scipy.stats.norm.pdf(mesh_frame_dissimilarity_threshold,
                                                                       loc=gm.means_[component_idxs[1], 0],
                                                                       scale=np.sqrt(gm.covariances_[component_idxs[1], 0])),
                 'ko')
        plt.plot(mesh_frame_dissimilarity_boundary,
                 gm.weights_[component_idxs[0]] * scipy.stats.norm.pdf(mesh_frame_dissimilarity_boundary,
                                                                       loc=gm.means_[component_idxs[0], 0],
                                                                       scale=np.sqrt(gm.covariances_[component_idxs[0], 0])),
                 'ro')
        plt.plot(mesh_frame_dissimilarity_cluster_intersection,
                 gm.weights_[component_idxs[0]] * scipy.stats.norm.pdf(mesh_frame_dissimilarity_cluster_intersection,
                                                                       loc=gm.means_[component_idxs[0], 0],
                                                                       scale=np.sqrt(gm.covariances_[component_idxs[0], 0])),
                 'bo')
        plt.title('interframe_cost_values')
        plt.tight_layout()
        stash_fig_page()
    '''

    # %%

    '''
    """
    mesh_frame_idxs = {}
    for ref_frame_idx in embedding_frame_idxs:
        rgbd_frame_idxs = []
        for rgbd_frame_idx in carved_merged_rgbd_images:
            camera_transform = camera_extrinsics[rgbd_frame_idx] @ np.linalg.inv(camera_extrinsics[ref_frame_idx])
            rvec, _ = cv2.Rodrigues(camera_transform[:3, :3])
            # Ignore rotation around the z-axis (in the primary frame of reference)
            rotation_magnitude = np.linalg.norm(rvec[:2])
            if rotation_magnitude <= np.pi / 4:
                rgbd_frame_idxs.append(rgbd_frame_idx)
        mesh_frame_idxs[ref_frame_idx] = tuple(rgbd_frame_idxs)
    """

    # Use the inter frame costs to determine which frames to include in the volumetric integration for each frame's mesh.
    mesh_frame_idxs = {ref_frame_idx: tuple(carved_merged_rgbd_images.keys()
                                            & tuple(np.where(interframe_costs[ref_frame_idx, :] < mesh_frame_dissimilarity_threshold)[0]))
                       for ref_frame_idx in embedding_frame_idxs}

    integrated_carved_merged_rgbd_images_meshes = {}
    def integrate_carved_merged_rgbd_images(ref_frame_idx):
        rgbd_frame_idxs = mesh_frame_idxs[ref_frame_idx]
        if rgbd_frame_idxs not in integrated_carved_merged_rgbd_images_meshes:
            volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=0.3, sdf_trunc=3.0,
                                                                  color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
            for rgbd_frame_idx in rgbd_frame_idxs:
                rgbd_image = carved_merged_rgbd_images[rgbd_frame_idx]
                volume.integrate(rgbd_image,
                                 o3d.camera.PinholeCameraIntrinsic(np.array(rgbd_image.depth).shape[1],
                                                                   np.array(rgbd_image.depth).shape[0],
                                                                   camera_intrinsic),
                                 camera_extrinsics[rgbd_frame_idx])
            # TODO: remove isolated / disconnected small mesh components?
            integrated_carved_merged_rgbd_images_meshes[rgbd_frame_idxs] = volume.extract_triangle_mesh()
        return integrated_carved_merged_rgbd_images_meshes[rgbd_frame_idxs]

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

            if rgbd_frame_idx in mesh_frame_idxs:
                mesh = integrate_carved_merged_rgbd_images(rgbd_frame_idx)
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
    '''

    # %%

    # Select target view synthesis frames by maximising a function of the frame mesh ray cast grid mapping score
    # and the relative surface area of the frame mesh.

    ray_cast_cache = {}
    def ray_cast_grid_points(ref_frame_idx, camera_extrinsic, camera_intrinsic, w, h):
        #rgbd_frame_idxs = mesh_frame_idxs[ref_frame_idx]
        #cache_key = (rgbd_frame_idxs, tuple(map(tuple, camera_extrinsic)), tuple(map(tuple, camera_intrinsic)), w, h)
        cache_key = (ref_frame_idx, tuple(map(tuple, camera_extrinsic)), tuple(map(tuple, camera_intrinsic)), w, h)

        if cache_key not in ray_cast_cache:
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(integrated_weighted_canvas_meshes[ref_frame_idx]))

            rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(intrinsic_matrix=camera_intrinsic,
                                                                      extrinsic_matrix=camera_extrinsic,
                                                                      width_px=w, height_px=h)

            casted_rays = scene.cast_rays(rays)
            depth_img = casted_rays['t_hit'].numpy()
            no_intersection_mask = ~np.isfinite(depth_img)
            depth_img[no_intersection_mask] = np.nan

            surface_normals = casted_rays['primitive_normals'].numpy()
            surface_normals[no_intersection_mask, :] = np.nan
            assert np.allclose(np.linalg.norm(surface_normals[~no_intersection_mask], axis=-1), 1)

            ray_directions = rays.numpy()[:, :, 3:]
            ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=-1, keepdims=True)

            normal_coincidence_img = -np.sum(ray_directions * surface_normals, axis=-1)

            ray_cast_cache[cache_key] = (depth_img, normal_coincidence_img)

            """
            while len(ray_cast_cache) > 100:
                del ray_cast_cache[next(iter(ray_cast_cache))]
            """

        depth_img, normal_coincidence_img = ray_cast_cache[cache_key]
        depth_predictions = depth_img.flatten()

        yg, xg = np.mgrid[0:h, 0:w].reshape((2, -1)).astype(int)
        uvg = np.vstack([xg, yg]) - camera_intrinsic[:2, 2:]
        xyg = np.linalg.inv(camera_intrinsic[:2, :2]) @ uvg
        xy1g = np.vstack([xyg, np.ones(xyg.shape[1],)])
        return xy1g * depth_predictions, xy1g, xg, yg, depth_img, normal_coincidence_img

    path_distances = -np.log(np.clip(interframe_confidences, 1e-2, 1))
    path_distances[~np.isfinite(path_distances)] = -np.log(1e-2)

    dist_matrix, predecessors = scipy.sparse.csgraph.shortest_path(path_distances, directed=False, return_predecessors=True)

    median_dist_matrix = np.median(dist_matrix, axis=0)

    # TODO: The interframe misalignment varies spatially across the synthetic image, depending on the
    #       camera planes / rays and the surface normals.
    #       If possible, iterate / adapt / optimise the surface or ray depth to minimise discontinuities between frames
    #interframe_misalignments = np.power(dist_matrix, 2) * (1 - cauchy(interframe_angles, np.pi / 4))
    #interframe_misalignments = np.sqrt(dist_matrix * (1 - cauchy(interframe_angles, np.pi / 4)))
    #interframe_misalignments = (1 - cauchy(dist_matrix, 0.5)) * (1 - cauchy(interframe_angles, np.pi / 4))
    #interframe_misalignments = (1 - cauchy(dist_matrix, 0.5)) * np.power(interframe_angles / (np.pi / 9), 2)
    #interframe_misalignments = (1 - cauchy(dist_matrix, 0.5)) * np.power(interframe_angles / (np.pi / 2), 2)
    #interframe_misalignments = np.exp(3.0 * dist_matrix) - 1
    interframe_misalignments = 0.5 * (1 - cauchy(dist_matrix, 0.5)) + 0.5 * (1 - cauchy(interframe_angles, np.pi / 4))

    # Compute the projection errors between camera frames and their associated canvas meshes to serve as a cost
    # measure between them when optimising the mapped frame regions and their boundaries for view synthesis

    # Cast rays from each reference frame's camera onto the canvas meshes of the reference frame and comparison frame.
    # Project the ray canvas intersection points onto the comparison frame's imaging plane.
    # Calculate the projection errors between the two sets of projected points.
    interframe_projection_errors = np.zeros((len(key_frame_indices),) * 2, dtype=np.float32)
    for ref_frame_idx in range(len(key_frame_indices)):
        print('Calculating interframe projection errors ref_frame_idx', ref_frame_idx)

        img = frame_images[ref_frame_idx]

        h, w = img.shape[:2]
        grid_step = 5

        assert w % grid_step == 0 and h % grid_step == 0
        ws, hs = w // grid_step, h // grid_step

        camera_intrinsic_stepped = np.block([[camera_intrinsic[:2, :2] / grid_step, (camera_intrinsic[:2, 2:] + 0.5) / grid_step - 0.5], [0, 0, 1]])

        ref_xyzsg, _, _, _, _, _ = ray_cast_grid_points(ref_frame_idx, camera_extrinsics[ref_frame_idx], camera_intrinsic_stepped, ws, hs)

        for frame_idx in range(len(key_frame_indices)):

            xyzsg, _, _, _, _, _ = ray_cast_grid_points(frame_idx, camera_extrinsics[ref_frame_idx], camera_intrinsic_stepped, ws, hs)

            camera_transform = camera_extrinsics[frame_idx] @ np.linalg.inv(camera_extrinsics[ref_frame_idx])

            ref_projected_points = camera_intrinsic @ (camera_transform[:3, :3] @ ref_xyzsg + camera_transform[:3, 3:])
            ref_projected_points = ref_projected_points[:2, :] / ref_projected_points[2, :]

            projected_points = camera_intrinsic @ (camera_transform[:3, :3] @ xyzsg + camera_transform[:3, 3:])
            projected_points = projected_points[:2, :] / projected_points[2, :]

            projection_errors = np.linalg.norm(projected_points - ref_projected_points, axis=0)

            inlier_mask = ((ref_projected_points[0, :] > -0.5) & (ref_projected_points[0, :] < w - 0.5)
                           & (ref_projected_points[1, :] > -0.5) & (ref_projected_points[1, :] < h - 0.5)
                           & (projected_points[0, :] > -0.5) & (projected_points[0, :] < w - 0.5)
                           & (projected_points[1, :] > -0.5) & (projected_points[1, :] < h - 0.5))
            projection_errors[~inlier_mask] = np.nan

            if np.any(np.isfinite(projection_errors)):
                interframe_projection_errors[ref_frame_idx, frame_idx] = np.percentile(np.nan_to_num(projection_errors), 80)
            else:
                interframe_projection_errors[ref_frame_idx, frame_idx] = np.nan

    #interframe_distances = 1 - cauchy(interframe_projection_errors + interframe_projection_errors.T, 10)
    #interframe_distances = np.log(1 + interframe_projection_errors + interframe_projection_errors.T)

    #interframe_costs = 1 - np.cos(interframe_angles) + interframe_distances
    #interframe_costs = 0.5 * np.power(interframe_angles / (np.pi / 2), 2) + interframe_distances
    #interframe_costs = 0.5 * interframe_misalignments + 0.5 * interframe_distances

    symmetric_interframe_projection_errors = np.stack([interframe_projection_errors, interframe_projection_errors.T])
    symmetric_interframe_projection_errors[:, np.all(~np.isfinite(symmetric_interframe_projection_errors), axis=0)] = 0
    interframe_costs = 1 - cauchy(np.nanmean(symmetric_interframe_projection_errors, axis=0), 10)

    plt.figure('Interframe costs', figsize=(16, 10))
    plt.clf()
    ax = plt.subplot(3, 3, 1)
    plt.imshow(interframe_angles)
    plt.title('interframe_angles')
    ax = plt.subplot(3, 3, 2, sharex=ax, sharey=ax)
    ax.set_facecolor('grey')
    plt.imshow(interframe_confidences, vmin=-1, vmax=1, cmap='seismic', interpolation='none')
    plt.title('interframe_confidences')
    plt.subplot(3, 3, 3, sharex=ax, sharey=ax)
    plt.imshow(path_distances)
    plt.title('path_distances')
    plt.subplot(3, 3, 4, sharex=ax, sharey=ax)
    plt.imshow(dist_matrix)
    plt.title('dist_matrix')
    plt.subplot(3, 3, 5, sharex=ax)
    plt.plot(median_dist_matrix)
    plt.title('median_dist_matrix')
    plt.subplot(3, 3, 6, sharex=ax, sharey=ax)
    plt.imshow(interframe_misalignments)
    plt.title('interframe_misalignments')
    plt.subplot(3, 3, 7, sharex=ax, sharey=ax)
    plt.imshow(interframe_projection_errors)
    plt.title('interframe_projection_errors')
    plt.subplot(3, 3, 8, sharex=ax, sharey=ax)
    plt.imshow(interframe_costs)
    plt.title('interframe_costs')
    plt.tight_layout()

    # %%

    '''
    # Cast rays from each reference frame's camera onto the canvas meshes of the reference and comparison frames.
    # Calculate the depth errors between the intersection points.
    interframe_depth_errors = np.full((len(key_frame_indices),) * 2, fill_value=np.nan, dtype=np.float32)
    interframe_depth_error_weights = np.full((len(key_frame_indices),) * 2, fill_value=np.nan, dtype=np.float32)
    for ref_frame_idx in range(len(key_frame_indices)):
        print('Calculating interframe depth errors ref_frame_idx', ref_frame_idx)

        synthetic_camera_extrinsic = np.block([[np.identity(3), np.array([0, 0, camera_pull_back_z])[:, None]], [0, 0, 0, 1]]) @ camera_extrinsics[ref_frame_idx]

        img = frame_images[ref_frame_idx]

        h, w = img.shape[:2]
        hj, wj = h * 3, w * 3

        grid_step = 5
        camera_intrinsic_stepped = np.block([[camera_intrinsic_synthetic[:2, :2] / grid_step, (camera_intrinsic_synthetic[:2, 2:] + 0.5) / grid_step - 0.5], [0, 0, 1]])

        assert wj % grid_step == 0 and hj % grid_step == 0
        ws, hs = wj // grid_step, hj // grid_step
        ref_xyzsg, _, _, _, _, _ = ray_cast_grid_points(ref_frame_idx, synthetic_camera_extrinsic, camera_intrinsic_stepped, ws, hs)

        for frame_idx in range(len(key_frame_indices)):

            xyzsg, _, _, _, _, _ = ray_cast_grid_points(frame_idx, synthetic_camera_extrinsic, camera_intrinsic_stepped, ws, hs)

            depth_errors = 1 - cauchy(xyzsg[2, :] - ref_xyzsg[2, :], np.sqrt(xyzsg[2, :] * ref_xyzsg[2, :]) / 5)

            if np.any(np.isfinite(depth_errors)):
                interframe_depth_errors[ref_frame_idx, frame_idx] = np.sqrt(np.nanmean(np.power(depth_errors, 2)))

            interframe_depth_error_weights[ref_frame_idx, frame_idx] = (np.sum(np.isfinite(xyzsg[2, :]) & np.isfinite(ref_xyzsg[2, :],))
                                                                        / np.sum(np.isfinite(xyzsg[2, :]) | np.isfinite(ref_xyzsg[2, :],)))

    plt.figure('Interframe depth errors', figsize=(16, 10))
    plt.clf()
    ax = plt.subplot(1, 2, 1)
    plt.imshow(interframe_depth_errors)
    plt.title('interframe_depth_errors')
    ax = plt.subplot(1, 2, 2, sharex=ax, sharey=ax)
    plt.imshow(interframe_depth_error_weights)
    plt.title('interframe_depth_error_weights')
    plt.tight_layout()
    '''

    # %%

    # Compute the projection errors between synthetic camera frames and their associated canvas meshes to serve as a dissimilarity
    # measure between them when selecting target frames for view synthesis

    # Cast rays from each reference frame's synthetic camera onto the canvas meshes of the reference frame and comparison frame.
    # Project the ray canvas intersection points onto the comparison frame's synthetic camera imaging plane.
    # Calculate the projection errors between the two sets of projected points.

    synthetic_camera_extrinsics = [np.block([[np.identity(3), np.array([0, 0, camera_pull_back_z])[:, None]], [0, 0, 0, 1]]) @ camera_extrinsic
                                   for camera_extrinsic in camera_extrinsics]

    interframe_canvas_projection_errors = np.zeros((len(key_frame_indices),) * 2, dtype=np.float32)
    interframe_canvas_projection_error_weights = np.zeros((len(key_frame_indices),) * 2, dtype=np.float32)
    #frame_synth_projection_score_histograms = []
    #frame_synth_projection_score_data = []
    frame_synth_projection_score_images = []
    for ref_frame_idx in range(len(key_frame_indices)):
        print('Calculating interframe synthetic camera projection errors ref_frame_idx', ref_frame_idx)

        img = frame_images[ref_frame_idx]

        h, w = img.shape[:2]
        hj, wj = h * 3, w * 3

        grid_step = 15
        camera_intrinsic_stepped = np.block([[camera_intrinsic_synthetic[:2, :2] / grid_step, (camera_intrinsic_synthetic[:2, 2:] + 0.5) / grid_step - 0.5], [0, 0, 1]])

        assert wj % grid_step == 0 and hj % grid_step == 0
        ws, hs = wj // grid_step, hj // grid_step
        ref_xyzsg, _, xsg, ysg, _, _ = ray_cast_grid_points(ref_frame_idx, synthetic_camera_extrinsics[ref_frame_idx], camera_intrinsic_stepped, ws, hs)

        projection_error_images = []
        for frame_idx in range(len(key_frame_indices)):

            xyzsg, _, _, _, _, _ = ray_cast_grid_points(frame_idx, synthetic_camera_extrinsics[ref_frame_idx], camera_intrinsic_stepped, ws, hs)

            camera_transform = synthetic_camera_extrinsics[frame_idx] @ np.linalg.inv(synthetic_camera_extrinsics[ref_frame_idx])

            ref_projected_points = camera_intrinsic_synthetic @ (camera_transform[:3, :3] @ ref_xyzsg + camera_transform[:3, 3:])
            ref_projected_points = ref_projected_points[:2, :] / ref_projected_points[2, :]

            projected_points = camera_intrinsic_synthetic @ (camera_transform[:3, :3] @ xyzsg + camera_transform[:3, 3:])
            projected_points = projected_points[:2, :] / projected_points[2, :]

            projection_errors = np.linalg.norm(projected_points - ref_projected_points, axis=0)

            inlier_mask = ((ref_projected_points[0, :] > -0.5) & (ref_projected_points[0, :] < wj - 0.5)
                           & (ref_projected_points[1, :] > -0.5) & (ref_projected_points[1, :] < hj - 0.5)
                           & (projected_points[0, :] > -0.5) & (projected_points[0, :] < wj - 0.5)
                           & (projected_points[1, :] > -0.5) & (projected_points[1, :] < hj - 0.5))
            projection_errors[~inlier_mask] = np.nan

            if np.any(np.isfinite(projection_errors)):
                interframe_canvas_projection_errors[ref_frame_idx, frame_idx] = np.sqrt(np.nanmean(np.power(projection_errors, 2)))
            else:
                interframe_canvas_projection_errors[ref_frame_idx, frame_idx] = max(hj, wj)

            interframe_canvas_projection_error_weights[ref_frame_idx, frame_idx] = (np.sum(np.isfinite(projected_points) & np.isfinite(ref_projected_points))
                                                                                    / np.sum(np.isfinite(projected_points) | np.isfinite(ref_projected_points)))

            camera_transform = camera_extrinsics[frame_idx] @ np.linalg.inv(synthetic_camera_extrinsics[ref_frame_idx])

            ref_projected_points = camera_intrinsic @ (camera_transform[:3, :3] @ ref_xyzsg + camera_transform[:3, 3:])
            ref_projected_points = ref_projected_points[:2, :] / ref_projected_points[2, :]

            projected_points = camera_intrinsic @ (camera_transform[:3, :3] @ xyzsg + camera_transform[:3, 3:])
            projected_points = projected_points[:2, :] / projected_points[2, :]

            """
            projection_error_image = np.full((hs, ws), fill_value=np.nan, dtype=np.float32)
            projection_error_image[ysg, xsg] = np.linalg.norm(projected_points - ref_projected_points, axis=0)

            inlier_mask = ((ref_projected_points[0, :] > -0.5) & (ref_projected_points[0, :] < w - 0.5)
                           & (ref_projected_points[1, :] > -0.5) & (ref_projected_points[1, :] < h - 0.5)
                           & (projected_points[0, :] > -0.5) & (projected_points[0, :] < w - 0.5)
                           & (projected_points[1, :] > -0.5) & (projected_points[1, :] < h - 0.5))
            projection_error_image[ysg[~inlier_mask], xsg[~inlier_mask]] = np.nan
            """

            primary_mesh_cross_flow = ref_projected_points.T.reshape((hs, ws, 2))
            secondary_mesh_cross_flow = projected_points.T.reshape((hs, ws, 2))

            cross_flow_inlier = ((primary_mesh_cross_flow[:, :, 0] > -0.5) & (primary_mesh_cross_flow[:, :, 0] < w - 0.5)
                                 & (primary_mesh_cross_flow[:, :, 1] > -0.5) & (primary_mesh_cross_flow[:, :, 1] < h - 0.5))
            inlier_scores = nan_gaussian_filter(cross_flow_inlier, ksize=(3, 3))
            #inlier_scores[~cross_flow_inlier] = 0
            inlier_scores[~cross_flow_inlier] = np.nan

            flow_discrepancy_score = cauchy(np.linalg.norm(secondary_mesh_cross_flow - primary_mesh_cross_flow, axis=-1), 10)

            flow_gradient_x = np.gradient(primary_mesh_cross_flow, grid_step, edge_order=2, axis=1)
            flow_gradient_y = np.gradient(primary_mesh_cross_flow, grid_step, edge_order=2, axis=0)

            flow_grid_scaling_aspect = np.linalg.norm(flow_gradient_x, axis=-1) / np.linalg.norm(flow_gradient_y, axis=-1)
            flow_grid_scaling_aspect_score = cauchy(np.clip(np.max(np.stack([flow_grid_scaling_aspect, 1 / flow_grid_scaling_aspect]) - 1, axis=0), 0, np.inf), 2.0)

            flow_grid_gradient_cross = np.cross(flow_gradient_x, flow_gradient_y, axis=-1)
            flow_grid_gradient_cross[flow_grid_gradient_cross <= 0] = np.nan
            # Note that the acute/obtuse angle ambiguity between flow_gradient_x and flow_gradient_y is not
            # relevant providing we only consider the magnitude of the offset from np.pi / 2
            assert ~np.any(np.abs(flow_grid_gradient_cross) > (1 + 1e-6) * np.linalg.norm(flow_gradient_x, axis=-1) * np.linalg.norm(flow_gradient_y, axis=-1))
            flow_grid_orthogonality_score = cauchy(np.pi / 2 - np.arcsin(np.clip(flow_grid_gradient_cross / np.linalg.norm(flow_gradient_x, axis=-1) / np.linalg.norm(flow_gradient_y, axis=-1), -1, 1)),
                                                   #np.pi / 2 * cauchy(interframe_angles[primary_frame_idx, secondary_frame_idx], np.pi / 4))
                                                   np.pi / 4)

            flow_laplacian_x = np.gradient(flow_gradient_x, grid_step, edge_order=2, axis=1)
            flow_laplacian_x_transverse_normed = np.abs(np.sum(np.stack([-flow_laplacian_x[:, :, 1], flow_laplacian_x[:, :, 0]], axis=-1) * flow_gradient_x, axis=-1)) / np.sum(np.power(flow_gradient_x, 2), axis=-1)
            flow_laplacian_y = np.gradient(flow_gradient_y, grid_step, edge_order=2, axis=0)
            flow_laplacian_y_transverse_normed = np.abs(np.sum(np.stack([-flow_laplacian_y[:, :, 0], flow_laplacian_y[:, :, 1]], axis=-1) * flow_gradient_y, axis=-1)) / np.sum(np.power(flow_gradient_y, 2), axis=-1)
            flow_laplacian_score = cauchy(np.linalg.norm(np.stack([flow_laplacian_x_transverse_normed, flow_laplacian_y_transverse_normed]), axis=0), 0.1)

            if True:
                mapping_score = (inlier_scores
                                 * flow_discrepancy_score
                                 * flow_grid_scaling_aspect_score
                                 * flow_grid_orthogonality_score
                                 * flow_laplacian_score
                                 * cauchy(key_frame_motion_blurs[frame_idx], 8.0))
            elif True:
                mapping_score = (inlier_scores
                                 * nan_gaussian_filter(flow_discrepancy_score, ksize=(5, 5), unfiltered_point_value=np.nan)
                                 * nan_gaussian_filter(flow_grid_scaling_aspect_score, ksize=(5, 5), unfiltered_point_value=np.nan)
                                 * nan_gaussian_filter(flow_grid_orthogonality_score, ksize=(5, 5), unfiltered_point_value=np.nan)
                                 * nan_gaussian_filter(flow_laplacian_score, ksize=(5, 5), unfiltered_point_value=np.nan)
                                 * cauchy(key_frame_motion_blurs[frame_idx], 8.0))
            else:
                mapping_score = np.power(6 / (1 / np.maximum(inlier_scores, 1e-8)
                                              + 1 / np.maximum(nan_gaussian_filter(flow_discrepancy_score, ksize=(5, 5), unfiltered_point_value=np.nan), 1e-8)
                                              + 1 / np.maximum(nan_gaussian_filter(flow_grid_scaling_aspect_score, ksize=(5, 5), unfiltered_point_value=np.nan), 1e-8)
                                              + 1 / np.maximum(nan_gaussian_filter(flow_grid_orthogonality_score, ksize=(5, 5), unfiltered_point_value=np.nan), 1e-8)
                                              + 1 / np.maximum(nan_gaussian_filter(flow_laplacian_score, ksize=(5, 5), unfiltered_point_value=np.nan), 1e-8)
                                              + 1 / np.maximum(cauchy(key_frame_motion_blurs[frame_idx], 8.0), 1e-8)), 2)

            projection_error_images.append(mapping_score)

        projection_error_images = np.array(projection_error_images)

        valid_mask = np.any(np.isfinite(projection_error_images), axis=0)
        #frame_synth_projection_score_image = cauchy(np.min(np.nan_to_num(projection_error_images, nan=max(w, h)), axis=0), 10)
        frame_synth_projection_score_image = np.max(np.nan_to_num(projection_error_images, nan=0), axis=0)
        frame_synth_projection_score_image[~valid_mask] = np.nan

        frame_synth_projection_score_images.append(frame_synth_projection_score_image)

        """
        ref_xyzs_grid = ref_xyzsg.T.reshape((hs, ws, 3))
        ref_xyzs_grid_surface_areas = np.linalg.norm(np.cross(np.gradient(ref_xyzs_grid, edge_order=2, axis=0),
                                                              np.gradient(ref_xyzs_grid, edge_order=2, axis=1)), axis=-1)
        """

        synthetic_camera_extrinsic, camera_intrinsic_stepped, depth_kernels, color_img, depth_img = integrated_weighted_depth_kernels[ref_frame_idx]

        """
        mask = cv2.resize(np.isfinite(depth_img).astype(np.float32), dsize=(ws, hs), fx=0, fy=0, interpolation=cv2.INTER_LINEAR) >= 0.5
        frame_synth_projection_score_image[~mask] = np.nan
        """

        """
        samples = frame_synth_projection_score_image[np.isfinite(frame_synth_projection_score_image)]
        hist, bin_edges = np.histogram(samples, bins=10, range=(0, 1), density=False)
        frame_synth_projection_score_histograms.append(hist)
        """

        """
        frame_synth_projection_score_data.append((np.nansum(frame_synth_projection_score_image * ref_xyzs_grid_surface_areas),
                                                  np.nansum(ref_xyzs_grid_surface_areas)))
        """

        plt.figure('Interframe synthetic camera projection errors', figsize=(16, 10))
        setup_new_fig_page()
        plt.suptitle(f'ref_frame_idx: {ref_frame_idx}')
        ax = plt.subplot(2, 2, 1)
        xys_extent = (-0.5, ws - 0.5, hs - 0.5, -0.5)
        # Apply np.require() as a workaround for ValueError: arrays must be of dtype byte, short, float32 or float64
        # https://github.com/matplotlib/matplotlib/issues/28448
        plt.imshow(np.require(color_img, dtype=np.float32) / 255, extent=xys_extent)
        plt.title('integrated colour image')
        ax = plt.subplot(2, 2, 2, sharex=ax, sharey=ax)
        plt.imshow(depth_img, vmin=camera_pull_back_z, vmax=camera_pull_back_z+30, extent=xys_extent)
        plt.title('integrated depth')
        ax = plt.subplot(2, 2, 3, sharex=ax, sharey=ax)
        plt.imshow(frame_synth_projection_score_image, cmap='jet', vmin=0, vmax=1)
        plt.title('frame_synth_projection_score_image')
        """
        plt.subplot(2, 2, 4)
        plt.bar(bin_edges[:-1], hist / np.sum(hist), width=np.diff(bin_edges), align='edge')
        plt.ylim((0, 1))
        plt.title('frame_synth_projection_score_image histogram')
        """
        """
        ax = plt.subplot(2, 2, 4, sharex=ax, sharey=ax)
        plt.imshow(ref_xyzs_grid_surface_areas, cmap='jet')
        plt.title('ref_xyzs_grid_surface_areas')
        """
        plt.tight_layout()
        stash_fig_page()

    """
    frame_synth_projection_score_data = np.array(frame_synth_projection_score_data)
    """

    """
    frame_synth_projection_scores = []
    for hist in frame_synth_projection_score_histograms:
        hist = hist / np.sum(hist)
        # Divide by np.log(len(samples)) to calculate the entropy relative to that of a uniform distribution
        frame_synth_projection_scores.append(1 + np.sum(np.log(np.maximum(hist, 1e-6)) * hist) / np.log(len(samples)))
    frame_synth_projection_scores = np.array(frame_synth_projection_scores)

    num_samples_median = np.median([sum(hist) for hist in frame_synth_projection_score_histograms]).astype(int)
    frame_synth_projection_score_bounds = []
    for samples in [(scipy.special.erf(np.linspace(-3, 3, num_samples_median)) + 1) / 2,
                    (scipy.special.erf(np.linspace(0, 3, num_samples_median)) + 1) / 2]:
        hist, bin_edges = np.histogram(samples, bins=10, range=(0, 1), density=False)
        hist = hist / np.sum(hist)
        # Divide by np.log(len(samples)) to calculate the entropy relative to that of a uniform distribution
        frame_synth_projection_score_bounds.append(1 + np.sum(np.log(np.maximum(hist, 1e-6)) * hist) / np.log(len(samples)))
    print('frame_synth_projection_score_bounds', frame_synth_projection_score_bounds)
    """

    """
    frame_synth_projection_scores = frame_synth_projection_score_data[:, 0] / np.median(frame_synth_projection_score_data[:, 1])

    frame_synth_projection_score_bounds = [0.3, 0.7]

    frame_synth_projection_weights = (scipy.special.erf((frame_synth_projection_scores - np.mean(frame_synth_projection_score_bounds))
                                                        / np.diff(frame_synth_projection_score_bounds) * 3) + 1) / 2
    """

    interframe_synth_projection_scores = np.zeros((len(key_frame_indices),) * 2, dtype=np.float32)
    for ref_frame_idx in range(len(key_frame_indices)):
        print('Calculating interframe synthetic camera projection scores ref_frame_idx', ref_frame_idx)

        img = frame_images[ref_frame_idx]

        h, w = img.shape[:2]
        hj, wj = h * 3, w * 3

        grid_step = 15
        camera_intrinsic_stepped = np.block([[camera_intrinsic_synthetic[:2, :2] / grid_step, (camera_intrinsic_synthetic[:2, 2:] + 0.5) / grid_step - 0.5], [0, 0, 1]])

        assert wj % grid_step == 0 and hj % grid_step == 0
        ws, hs = wj // grid_step, hj // grid_step
        ref_xyzsg, _, xsg, ysg, _, _ = ray_cast_grid_points(ref_frame_idx, synthetic_camera_extrinsics[ref_frame_idx], camera_intrinsic_stepped, ws, hs)

        for frame_idx in range(len(key_frame_indices)):

            camera_transform = synthetic_camera_extrinsics[frame_idx] @ np.linalg.inv(synthetic_camera_extrinsics[ref_frame_idx])

            ref_projected_points = camera_intrinsic_stepped @ (camera_transform[:3, :3] @ ref_xyzsg + camera_transform[:3, 3:])
            ref_projected_points = ref_projected_points[:2, :] / ref_projected_points[2, :]

            primary_mesh_cross_flow = ref_projected_points.T.reshape((hs, ws, 2)).astype(np.float32)

            projection_score_image_warp = cv2.remap(frame_synth_projection_score_images[frame_idx], primary_mesh_cross_flow, None, cv2.INTER_LINEAR,
                                                    borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)

            projection_score_image_diff = projection_score_image_warp - frame_synth_projection_score_images[ref_frame_idx]
            if np.any(np.isfinite(projection_score_image_diff)):
                interframe_synth_projection_scores[ref_frame_idx, frame_idx] = np.cbrt(np.nanmean(np.power(projection_score_image_diff, 3)))

            """
            plt.figure('Interframe synthetic projection scores', figsize=(16, 10))
            setup_new_fig_page()
            plt.suptitle(f'ref_frame_idx, frame_idx: {ref_frame_idx}, {frame_idx}')
            ax = plt.subplot(2, 2, 1)
            plt.imshow(frame_synth_projection_score_images[ref_frame_idx], cmap='jet', vmin=0, vmax=1)
            ax = plt.subplot(2, 2, 2, sharex=ax, sharey=ax)
            plt.imshow(frame_synth_projection_score_images[frame_idx], cmap='jet', vmin=0, vmax=1)
            ax = plt.subplot(2, 2, 3, sharex=ax, sharey=ax)
            plt.imshow(projection_score_image_warp, cmap='jet', vmin=0, vmax=1)
            ax = plt.subplot(2, 2, 4, sharex=ax, sharey=ax)
            plt.imshow(projection_score_image_diff)
            plt.tight_layout()
            stash_fig_page()
            """

    # %%

    frame_synth_projection_score_image_means = [np.nanmean(frame_synth_projection_score_image)
                                                for frame_synth_projection_score_image in frame_synth_projection_score_images]

    interframe_support = cauchy(interframe_canvas_projection_errors, 20) * interframe_canvas_projection_error_weights
    model_interframe_support = torch.tensor(interframe_support, dtype=torch.float32)
    model_interframe_synth_projection_scores = torch.tensor(interframe_synth_projection_scores, dtype=torch.float32)
    model_frame_synth_projection_weights = torch.tensor(frame_synth_projection_score_image_means, dtype=torch.float32, requires_grad=True)
    model_frame_synth_projection_score_scaling = torch.tensor(1, dtype=torch.float32, requires_grad=True)

    def loss_fn():
        loss_components = {}

        model_frame_synth_projection_weights_sigmoid = torch.nn.functional.sigmoid(4 * (model_frame_synth_projection_weights - 0.5))
        model_interframe_synth_projection_weight_targets = model_frame_synth_projection_weights[:, None] + model_frame_synth_projection_score_scaling * model_interframe_synth_projection_scores
        model_interframe_synth_projection_weight_targets_sigmoid = torch.nn.functional.sigmoid(4 * (model_interframe_synth_projection_weight_targets - 0.5))
        model_frame_synth_projection_weight_errors = model_frame_synth_projection_weights_sigmoid - model_interframe_synth_projection_weight_targets_sigmoid

        loss_components['errors'] = model_interframe_support * torch.pow(model_frame_synth_projection_weight_errors, 2)

        loss_components['mean'] = 1e-2 * torch.pow(torch.mean(model_frame_synth_projection_weights) - np.mean(frame_synth_projection_score_image_means), 2)
        #exp_factor = 100
        #weights_apex = (torch.logsumexp(exp_factor * model_frame_synth_projection_weights, dim=0) - np.log(model_frame_synth_projection_weights.shape[0])) / exp_factor
        # The ceiling loss has a minimum at weights_apex = 1.0 where d(ceiling loss)/d(weights_apex) = 0
        #loss_components['ceiling'] = 1e-3 * (-34 * weights_apex + torch.pow(weights_apex, 2) + torch.pow(weights_apex, 32))
        loss_components['floor'] = 2e-3 / model_frame_synth_projection_weights_sigmoid
        loss_components['ceiling'] = 1e-3 / (1 - model_frame_synth_projection_weights_sigmoid)
        loss_components['scaling'] = 2e-2 * -torch.log(model_frame_synth_projection_score_scaling)

        return sum(torch.mean(loss_component) for loss_component in loss_components.values()), loss_components

    if False:
        lr = 0.005
        num_steps = 3000
        convergence_criterion = {'atol': 1e-5, 'window_size': 100, 'min_num_steps': 300}
        optimiser = torch.optim.Adam([model_frame_synth_projection_weights, model_frame_synth_projection_score_scaling], lr=lr)
        lr_lambda = lambda epoch: (np.sin(min((epoch + 1) / convergence_criterion['min_num_steps'], 0.5) * np.pi)
                                   * np.power(0.1, max(epoch - 0.5 * convergence_criterion['min_num_steps'], 0) / num_steps))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lr_lambda)
        losses = []
        loss_components = {}
        for optim_step in range(num_steps):
            optimiser.zero_grad()
            loss, loss_components = loss_fn()
            loss.backward()
            optimiser.step()
            scheduler.step()
            losses.append(loss.numpy(force=True))
            loss_window_std = np.std(losses[-convergence_criterion['window_size']:])
            if optim_step % 100 == 0:
                print('optim_step', optim_step, 'loss', losses[-1], 'loss_window_std', loss_window_std, 'learning rate', scheduler.get_last_lr()[0])
            if optim_step >= convergence_criterion['min_num_steps'] and loss_window_std < convergence_criterion['atol']:
                break

        print('len(losses)', len(losses))
        print('initial, first, final loss', losses[0], losses[convergence_criterion['min_num_steps']], losses[-1])
        print('highest, lowest loss', np.max(losses[convergence_criterion['min_num_steps']:]), np.min(losses[convergence_criterion['min_num_steps']:]))
    else:
        lr = 0.005
        num_steps = 3000
        convergence_criterion = {'atol': 1e-5, 'window_size': 100, 'min_num_steps': 300}
        optimiser = torch.optim.Adam([model_frame_synth_projection_weights, model_frame_synth_projection_score_scaling], lr=lr)
        losses = []
        loss_components = {}
        for optim_step in range(num_steps):
            optimiser.zero_grad()
            loss, loss_components = loss_fn()
            loss.backward()
            optimiser.step()
            losses.append(loss.numpy(force=True))
            loss_window_std = np.std(losses[-convergence_criterion['window_size']:])
            if optim_step % 100 == 0:
                print('optim_step', optim_step, 'loss', losses[-1], 'loss_window_std', loss_window_std)
            if optim_step >= convergence_criterion['min_num_steps'] and loss_window_std < convergence_criterion['atol']:
                break

    frame_synth_projection_weights = torch.nn.functional.sigmoid(4 * (model_frame_synth_projection_weights - 0.5)).numpy(force=True)

    plt.figure('Interframe canvas projection errors', figsize=(16, 10))
    plt.clf()
    ax = plt.subplot(2, 2, 1)
    plt.imshow(interframe_canvas_projection_errors)
    plt.title('interframe_canvas_projection_errors')
    ax = plt.subplot(2, 2, 2, sharex=ax, sharey=ax)
    plt.imshow(interframe_canvas_projection_error_weights)
    plt.title('interframe_canvas_projection_error_weights')
    """
    ax = plt.subplot(2, 2, 3)
    plt.plot(frame_synth_projection_scores)
    plt.title('frame_synth_projection_scores')
    plt.subplot(2, 2, 4, sharex=ax)
    plt.plot(frame_synth_projection_weights)
    plt.ylim((-0.05, 1.05))
    plt.title('frame_synth_projection_weights')
    """
    ax = plt.subplot(2, 2, 3, sharex=ax, sharey=ax)
    vmaxabs = np.max(np.abs(interframe_synth_projection_scores))
    plt.imshow(interframe_synth_projection_scores, cmap='seismic', vmin=-vmaxabs, vmax=vmaxabs)
    plt.title('interframe_synth_projection_scores')
    plt.subplot(2, 2, 4)
    plt.plot(frame_synth_projection_weights)
    plt.plot(frame_synth_projection_score_image_means, ':')
    plt.plot(np.sum(interframe_synth_projection_scores * interframe_support, axis=0) / np.sum(interframe_support, axis=0), '-.')
    plt.title('frame_synth_projection_weights')
    plt.tight_layout()

    # %%

    # Select target frames by optimising a regularised interframe support loss function

    model_interframe_support = torch.tensor(interframe_support, dtype=torch.float32)
    model_frame_synth_projection_weights = torch.tensor(frame_synth_projection_weights, dtype=torch.float32)
    model_target_frames = torch.tensor(np.zeros_like(frame_synth_projection_weights), dtype=torch.float32, requires_grad=True)
    #model_target_frames = torch.tensor(np.random.uniform(low=-0.1, high=0.1, size=frame_synth_projection_weights.shape), dtype=torch.float32, requires_grad=True)

    def loss_fn(temperature):
        target_frame_activations = torch.nn.functional.sigmoid(model_target_frames)
        prob_target_frame_activations = torch.nn.functional.softmax(target_frame_activations, dim=0)
        loss_components = {}

        threshold = 0.9
        #target_frame_exclusions = 1 / (1 + torch.pow(target_frame_activations[:, None] * model_interframe_support / (threshold / 3), 2))
        #target_frame_exclusions = torch.exp(-0.5 * torch.pow(target_frame_activations[:, None] * model_interframe_support / (threshold / 3), 2))
        #loss_components['exclusion'] = 1 / torch.mean(1 / target_frame_exclusions, dim=0)
        #loss_components['exclusion'] = torch.exp(-0.5 * torch.sum(torch.pow(target_frame_activations[:, None] * model_interframe_support / (threshold / 3), 2), dim=0))
        #exp_factor = 99 * (1 - temperature) + 1
        exp_factor = 100
        activated_frame_support = (torch.logsumexp(exp_factor * target_frame_activations[:, None] * model_interframe_support, dim=0) - np.log(target_frame_activations.shape[0])) / exp_factor
        #loss_components['exclusion'] = torch.exp(-0.5 * torch.pow(activated_frame_support / (threshold / 3), 2))
        loss_components['exclusion'] = torch.exp(-3.0 * activated_frame_support / threshold) * model_frame_synth_projection_weights

        loss_components['activations'] = 10.0 * np.exp(-3.0 * temperature) * target_frame_activations * torch.exp(-3.0 * model_frame_synth_projection_weights)

        # Divide by np.log(prob_target_frame_activations.shape[0]) to calculate the entropy relative to that of a uniform distribution
        loss_components['entropy'] = 1.0 * -torch.sum(torch.log(torch.clamp(prob_target_frame_activations, min=1e-4)) * prob_target_frame_activations) / np.log(prob_target_frame_activations.shape[0])

        loss_components['information'] = 0.5 * torch.log(1 + 10 * torch.mean(target_frame_activations))

        return sum(torch.mean(loss_component) for loss_component in loss_components.values()), loss_components

    if False:
        lr = 0.005
        num_steps = 20000
        convergence_criterion = {'atol': 1e-5, 'window_size': 200, 'min_num_steps': 2000}
        optimiser = torch.optim.Adam([model_target_frames], lr=lr)
        lr_lambda = lambda epoch: (np.sin(min((epoch + 1) / convergence_criterion['min_num_steps'], 0.5) * np.pi)
                                   * np.power(0.1, max(epoch - 0.5 * convergence_criterion['min_num_steps'], 0) / num_steps))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lr_lambda)
        losses = []
        loss_components = {}
        for optim_step in range(num_steps):
            optimiser.zero_grad()
            loss, loss_components = loss_fn()
            loss.backward()
            optimiser.step()
            scheduler.step()
            losses.append(loss.numpy(force=True))
            loss_window_std = np.std(losses[-convergence_criterion['window_size']:])
            if optim_step % 100 == 0:
                print('optim_step', optim_step, 'loss', losses[-1], 'loss_window_std', loss_window_std, 'learning rate', scheduler.get_last_lr()[0])
            if optim_step >= convergence_criterion['min_num_steps'] and loss_window_std < convergence_criterion['atol']:
                break

        print('len(losses)', len(losses))
        print('initial, first, final loss', losses[0], losses[convergence_criterion['min_num_steps']], losses[-1])
        print('highest, lowest loss', np.max(losses[convergence_criterion['min_num_steps']:]), np.min(losses[convergence_criterion['min_num_steps']:]))
    elif True:
        lr = 0.005
        num_steps = 20000
        convergence_criterion = {'atol': 1e-5, 'window_size': 200, 'min_num_steps': 5000}
        optimiser = torch.optim.Adam([model_target_frames], lr=lr)
        losses = []
        loss_components = {}
        for optim_step in range(num_steps):
            optimiser.zero_grad()
            temperature = (1 + np.cos(np.pi * min(optim_step / convergence_criterion['min_num_steps'], 1))) / 2
            loss, loss_components = loss_fn(temperature)
            loss.backward()
            optimiser.step()
            losses.append(loss.numpy(force=True))
            loss_window_std = np.std(losses[-convergence_criterion['window_size']:])
            if optim_step % 100 == 0:
                print('optim_step', optim_step, 'loss', losses[-1], 'loss_window_std', loss_window_std, 'temperature', np.round(temperature, 3))
            if optim_step >= convergence_criterion['min_num_steps'] and loss_window_std < convergence_criterion['atol']:
                break
    else:
        lr = 0.005
        num_steps = 20000
        convergence_criterion = {'atol': 1e-3, 'window_size': 200, 'min_num_steps': 5000}
        optimiser = torch.optim.Adam([model_target_frames], lr=lr)
        losses = []
        loss_components = {}
        target_frame_activations_history = []
        for optim_step in range(num_steps):
            optimiser.zero_grad()
            temperature = (1 + np.cos(np.pi * min(optim_step / convergence_criterion['min_num_steps'], 1))) / 2
            loss, loss_components = loss_fn(temperature)
            loss.backward()
            optimiser.step()
            losses.append(loss.numpy(force=True))
            loss_window_std = np.std(losses[-convergence_criterion['window_size']:])
            target_frame_activations = scipy.special.expit(model_target_frames.numpy(force=True))
            target_frame_activations_history.append(target_frame_activations)
            target_frame_activations_window_std = np.sqrt(np.mean(np.var(target_frame_activations_history[-convergence_criterion['window_size']:], axis=0)))
            if optim_step % 100 == 0:
                print('optim_step', optim_step, 'loss', losses[-1], 'loss_window_std', loss_window_std,
                      'target_frame_activations_window_std', target_frame_activations_window_std, 'temperature', np.round(temperature, 3))
            if optim_step >= convergence_criterion['min_num_steps'] and target_frame_activations_window_std < convergence_criterion['atol']:
                break

    target_frame_activations = scipy.special.expit(model_target_frames.numpy(force=True))

    target_frame_idxs = np.where(target_frame_activations >= min(0.25, np.max(target_frame_activations)))[0]
    print('target_frame_idxs', target_frame_idxs)

    target_frame_support_cluster_labels = target_frame_idxs[np.argmax(interframe_support[target_frame_idxs, :], axis=0)]

    plt.figure('Target frame selection', figsize=(16, 10))
    plt.clf()
    ax = plt.subplot(2, 2, 1)
    plt.plot(target_frame_activations)
    plt.title('target_frame_activations')
    plt.subplot(2, 2, 2, sharex=ax)
    plt.plot(loss_components['exclusion'].numpy(force=True))
    plt.ylim((-0.05, 1.05))
    plt.title('frame exclusion loss')
    plt.subplot(2, 2, 3, sharex=ax)
    plt.plot(frame_synth_projection_weights)
    plt.plot(target_frame_idxs, frame_synth_projection_weights[target_frame_idxs], 'o')
    plt.ylim((-0.05, 1.05))
    plt.title('frame_synth_projection_weights')
    plt.subplot(2, 2, 4, sharex=ax)
    for target_frame_idx in target_frame_idxs:
        plt.plot(interframe_support[target_frame_idx, :], label=target_frame_idx)
    plt.ylim((-0.05, 1.05))
    plt.legend()
    plt.title('interframe_support[target_frame_idxs]')
    plt.tight_layout()

    # %%

    interframe_dissimilarities = 1 - cauchy(0.5 * (interframe_canvas_projection_errors + interframe_canvas_projection_errors.T), 50)
    interframe_dissimilarity_weights = 0.5 * (interframe_canvas_projection_error_weights + interframe_canvas_projection_error_weights.T)

    embedding_frame_idxs = np.where(np.sum(interframe_dissimilarity_weights, axis=0) > 1)[0]
    embedding_dissimilarities = interframe_dissimilarities[embedding_frame_idxs[:, None], embedding_frame_idxs]
    embedding_weights = interframe_dissimilarity_weights[embedding_frame_idxs[:, None], embedding_frame_idxs]

    # Higher n_components should converge to an embedding that fits dissimilarities with lower residual errors / stress,
    # but may not generalise well to missing dissimilarities?
    # Also the variance of distances (such as Euclidean or Manhattan) between points converges to zero as the number of
    # dimensions increases https://towardsdatascience.com/curse-of-dimensionality-an-intuitive-exploration-1fbf155e1411/
    for n_components in [2, 3, 6]:
        embedding, stress, n_iter = rsatoolbox.util.vis_utils.smacof(embedding_dissimilarities, metric=True, n_components=n_components, n_init=8, n_jobs=-1,
                                                                     max_iter=300, random_state=0, return_n_iter=True, weight=embedding_weights)

        embedding_costs = np.full((len(key_frame_indices), len(key_frame_indices)), fill_value=np.nan, dtype=np.float32)
        embedding_costs[embedding_frame_idxs[:, None], embedding_frame_idxs] = sklearn.metrics.euclidean_distances(embedding)

        embedding_cost_values = embedding_costs[~np.identity(embedding_costs.shape[0], dtype=bool)]
        embedding_cost_values = embedding_cost_values[np.isfinite(embedding_cost_values)]

        gm = sklearn.mixture.GaussianMixture(n_components=2, covariance_type='diag', random_state=0)
        gm.fit(embedding_cost_values[:, None])
        component_idxs = np.argsort(gm.means_[:, 0])

        # Locate the intersection point between the two Gaussian Mixture component means (if it exists).
        # There may be no such intersection point if the distribution is fundamentally unimodal - this
        # commonly arises when the number of key frames is small.
        # https://stackoverflow.com/questions/22579434/python-finding-the-intersection-point-of-two-gaussian-curves
        a = 1 / (2 * gm.covariances_[component_idxs[0], 0]) - 1 / (2 * gm.covariances_[component_idxs[1], 0])
        b = (gm.means_[component_idxs[1], 0] / gm.covariances_[component_idxs[1], 0]
             - gm.means_[component_idxs[0], 0] / gm.covariances_[component_idxs[0], 0])
        c = (np.power(gm.means_[component_idxs[0], 0], 2) / (2 * gm.covariances_[component_idxs[0], 0])
             - np.power(gm.means_[component_idxs[1], 0], 2) / (2 * gm.covariances_[component_idxs[1], 0])
             - np.log(gm.covariances_[component_idxs[1], 0] / gm.covariances_[component_idxs[0], 0]) / 2
             - np.log(gm.weights_[component_idxs[0]] / gm.weights_[component_idxs[1]]))
        gm_intersections = np.roots([a, b, c])
        inter_idxs = np.where((gm.means_[component_idxs[0], 0] < gm_intersections) & (gm_intersections < gm.means_[component_idxs[1], 0]))[0]
        gm_component_intersection = gm_intersections[inter_idxs[0]] if len(inter_idxs) == 1 else np.nan

        print('GMM n_components, stress, n_iter, gm_component_intersection',
              n_components, stress, n_iter, gm_component_intersection)

        colour_sequence = itertools.cycle(matplotlib.color_sequences['tab10'])
        marker_sequence = itertools.cycle([marker for marker in matplotlib.lines.Line2D.filled_markers if marker not in ['.']])
        if n_components == 2:
            plt.figure('Interframe dissimilarity embedding', figsize=(16, 10))
            setup_new_fig_page()
            plt.suptitle(f'n_components: {n_components}')
            ax = plt.subplot(1, 1, 1)
            plt.plot(*embedding.T, alpha=0.5)
            marker_sizes = np.maximum(frame_synth_projection_weights[embedding_frame_idxs], 0) * 180 + 20
            for target_frame_idx, colour, marker in zip(target_frame_idxs, colour_sequence, marker_sequence):
                cluster_mask = target_frame_support_cluster_labels[embedding_frame_idxs] == target_frame_idx
                plt.scatter(*embedding[cluster_mask, :].T,
                            s=marker_sizes[cluster_mask], color=colour, marker=marker, alpha=0.5, label=target_frame_idx)
                plt.scatter(*embedding[embedding_frame_idxs == target_frame_idx, :].T, s=4000,
                            edgecolor=colour, facecolor='none', marker=marker, alpha=0.5)
            for frame_idx, xy in zip(embedding_frame_idxs, embedding):
                plt.text(*xy, f'{frame_idx}')
            ax.set_aspect('equal', adjustable='datalim')
            plt.legend()
            plt.tight_layout()
            stash_fig_page()
        elif n_components >= 3:
            if n_components > 3:
                mds = sklearn.manifold.MDS(n_components=3, metric=True, random_state=0)
                mds.fit(embedding)
                embedding_3d = mds.embedding_
            else:
                embedding_3d = embedding
            plt.figure('Interframe dissimilarity embedding', figsize=(16, 10))
            setup_new_fig_page()
            plt.suptitle(f'n_components: {n_components}')
            ax3 = plt.subplot(1, 1, 1, projection='3d')
            ax3.plot(*embedding_3d.T, alpha=0.5)
            marker_sizes = np.maximum(frame_synth_projection_weights[embedding_frame_idxs], 0) * 180 + 20
            for target_frame_idx, colour, marker in zip(target_frame_idxs, colour_sequence, marker_sequence):
                cluster_mask = target_frame_support_cluster_labels[embedding_frame_idxs] == target_frame_idx
                ax3.scatter(*embedding_3d[cluster_mask, :].T,
                            s=marker_sizes[cluster_mask], color=colour, marker=marker, alpha=0.5, label=target_frame_idx)
                ax3.scatter(*embedding_3d[embedding_frame_idxs == target_frame_idx, :].T, s=4000,
                            edgecolor=colour, facecolor='none', marker=marker, alpha=0.5)
            for frame_idx, xyz in zip(embedding_frame_idxs, embedding_3d):
                ax3.text(*xyz, f'{frame_idx}')
            ax3.set_aspect('equal', adjustable='datalim')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            plt.legend()
            plt.tight_layout()
            stash_fig_page()

        plt.figure('Embedding costs', figsize=(16, 10))
        setup_new_fig_page()
        plt.suptitle(f'n_components: {n_components}')
        ax = plt.subplot(2, 3, 1)
        plt.imshow(interframe_dissimilarities, cmap='jet', interpolation='none')
        plt.title('interframe_dissimilarities')
        ax = plt.subplot(2, 3, 2, sharex=ax, sharey=ax)
        plt.imshow(interframe_dissimilarity_weights, cmap='jet', interpolation='none')
        plt.title('interframe_dissimilarity_weights')
        ax = plt.subplot(2, 3, 4, sharex=ax, sharey=ax)
        plt.imshow(embedding_costs, cmap='jet', interpolation='none')
        plt.title('embedding_costs')
        ax = plt.subplot(2, 3, 5, sharex=ax, sharey=ax)
        plt.imshow(embedding_costs / np.maximum(interframe_dissimilarities, 1e-8), vmin=0.2, vmax=5, cmap='jet', interpolation='none')
        plt.title('embedding_costs / interframe_dissimilarities')
        plt.subplot(2, 3, 6)
        bins = np.linspace(0, 1, 51)
        plt.hist(embedding_cost_values, bins=bins, density=True)
        for component_idx in np.argsort(gm.means_[:, 0]):
            plt.plot(bins, gm.weights_[component_idx] * scipy.stats.norm.pdf(bins, loc=gm.means_[component_idx, 0],
                                                                             scale=np.sqrt(gm.covariances_[component_idx, 0])))
        plt.axvline(gm_component_intersection, color='blue', linestyle='--')
        plt.plot(gm_component_intersection,
                 gm.weights_[component_idxs[0]] * scipy.stats.norm.pdf(gm_component_intersection,
                                                                       loc=gm.means_[component_idxs[0], 0],
                                                                       scale=np.sqrt(gm.covariances_[component_idxs[0], 0])),
                 'bo')
        plt.title('embedding_cost_values')
        plt.tight_layout()
        stash_fig_page()

    # %%

    '''
    interframe_dissimilarities = 1 - cauchy(0.5 * (interframe_canvas_projection_errors + interframe_canvas_projection_errors.T), 50)
    interframe_dissimilarity_weights = 0.5 * (interframe_canvas_projection_error_weights + interframe_canvas_projection_error_weights.T)

    embedding_frame_idxs = np.where(np.sum(interframe_dissimilarity_weights, axis=0) > 1)[0]
    embedding_dissimilarities = interframe_dissimilarities[embedding_frame_idxs[:, None], embedding_frame_idxs]
    embedding_weights = interframe_dissimilarity_weights[embedding_frame_idxs[:, None], embedding_frame_idxs]

    # Higher n_components should converge to an embedding that fits dissimilarities with lower residual errors / stress,
    # but may not generalise well to missing dissimilarities?
    # Also the variance of distances (such as Euclidean or Manhattan) between points converges to zero as the number of
    # dimensions increases https://towardsdatascience.com/curse-of-dimensionality-an-intuitive-exploration-1fbf155e1411/
    for n_components in [2, 3, 6]:
        embedding, stress, n_iter = rsatoolbox.util.vis_utils.smacof(embedding_dissimilarities, metric=True, n_components=n_components, n_init=8, n_jobs=-1,
                                                                     max_iter=300, random_state=0, return_n_iter=True, weight=embedding_weights)

        embedding_costs = np.full((len(key_frame_indices), len(key_frame_indices)), fill_value=np.nan, dtype=np.float32)
        embedding_costs[embedding_frame_idxs[:, None], embedding_frame_idxs] = sklearn.metrics.euclidean_distances(embedding)

        embedding_cost_values = embedding_costs[~np.identity(embedding_costs.shape[0], dtype=bool)]
        embedding_cost_values = embedding_cost_values[np.isfinite(embedding_cost_values)]

        gm = sklearn.mixture.GaussianMixture(n_components=2, covariance_type='diag', random_state=0)
        gm.fit(embedding_cost_values[:, None])
        component_idxs = np.argsort(gm.means_[:, 0])

        # https://stackoverflow.com/questions/22579434/python-finding-the-intersection-point-of-two-gaussian-curves
        a = 1 / (2 * gm.covariances_[component_idxs[0], 0]) - 1 / (2 * gm.covariances_[component_idxs[1], 0])
        b = (gm.means_[component_idxs[1], 0] / gm.covariances_[component_idxs[1], 0]
             - gm.means_[component_idxs[0], 0] / gm.covariances_[component_idxs[0], 0])
        c = (np.power(gm.means_[component_idxs[0], 0], 2) / (2 * gm.covariances_[component_idxs[0], 0])
             - np.power(gm.means_[component_idxs[1], 0], 2) / (2 * gm.covariances_[component_idxs[1], 0])
             - np.log(gm.covariances_[component_idxs[1], 0] / gm.covariances_[component_idxs[0], 0]) / 2
             - np.log(gm.weights_[component_idxs[0]] / gm.weights_[component_idxs[1]]))
        gm_intersections = np.roots([a, b, c])
        inter_idxs = np.where((gm.means_[component_idxs[0], 0] < gm_intersections) & (gm_intersections < gm.means_[component_idxs[1], 0]))[0]
        assert len(inter_idxs) == 1
        mesh_frame_dissimilarity_cluster_intersection = gm_intersections[inter_idxs[0]]

        if True:
            #distance_threshold = mesh_frame_dissimilarity_cluster_intersection
            distance_threshold = 1.5 * mesh_frame_dissimilarity_threshold
            #distance_threshold = mesh_frame_dissimilarity_boundary
            clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=None, linkage='complete',
                                                                 distance_threshold=distance_threshold)
            clustering.fit(embedding)
            embedding_cluster_count = clustering.n_clusters_
        elif False:
            #bandwidth = sklearn.cluster.estimate_bandwidth(embedding, quantile=0.3, n_samples=500)
            bandwidth = 0.75 * mesh_frame_dissimilarity_threshold
            clustering = sklearn.cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
            clustering.fit(embedding)
            embedding_cluster_count = len(set(clustering.labels_))
        else:
            preference = -15 * np.power(mesh_frame_dissimilarity_threshold, 2)
            clustering = sklearn.cluster.AffinityPropagation(preference=preference, random_state=0)
            clustering.fit(embedding)
            embedding_cluster_count = len(set(clustering.labels_))

        embedding_cluster_labels = np.full((len(key_frame_indices),), fill_value=-1, dtype=np.int32)
        embedding_cluster_labels[embedding_frame_idxs] = clustering.labels_

        print(n_components, stress, n_iter, mesh_frame_dissimilarity_cluster_intersection, embedding_cluster_count)

        print('outlier component P(X = mesh_frame_dissimilarity_threshold)',
              gm.weights_[component_idxs[1]] * scipy.stats.norm.pdf(mesh_frame_dissimilarity_threshold,
                                                                    loc=gm.means_[component_idxs[1], 0],
                                                                    scale=np.sqrt(gm.covariances_[component_idxs[1], 0])))
        print('outlier component P(X < mesh_frame_dissimilarity_threshold)',
              gm.weights_[component_idxs[1]] * scipy.stats.norm.cdf(mesh_frame_dissimilarity_threshold,
                                                                    loc=gm.means_[component_idxs[1], 0],
                                                                    scale=np.sqrt(gm.covariances_[component_idxs[1], 0])))
        print('outlier component P(X < mesh_frame_dissimilarity_threshold | outlier)',
              scipy.stats.norm.cdf(mesh_frame_dissimilarity_threshold,
                                   loc=gm.means_[component_idxs[1], 0],
                                   scale=np.sqrt(gm.covariances_[component_idxs[1], 0])))
        print('inlier component P(X = mesh_frame_dissimilarity_boundary)',
              gm.weights_[component_idxs[0]] * scipy.stats.norm.pdf(mesh_frame_dissimilarity_boundary,
                                                                    loc=gm.means_[component_idxs[0], 0],
                                                                    scale=np.sqrt(gm.covariances_[component_idxs[0], 0])))
        print('inlier component P(X > mesh_frame_dissimilarity_boundary)',
              gm.weights_[component_idxs[0]] * (1 - scipy.stats.norm.cdf(mesh_frame_dissimilarity_boundary,
                                                                         loc=gm.means_[component_idxs[0], 0],
                                                                         scale=np.sqrt(gm.covariances_[component_idxs[0], 0]))))
        print('inlier component P(X > mesh_frame_dissimilarity_boundary | inlier)',
              1 - scipy.stats.norm.cdf(mesh_frame_dissimilarity_boundary,
                                       loc=gm.means_[component_idxs[0], 0],
                                       scale=np.sqrt(gm.covariances_[component_idxs[0], 0])))

        colour_sequence = itertools.cycle(matplotlib.color_sequences['tab10'])
        marker_sequence = itertools.cycle([marker for marker in matplotlib.lines.Line2D.filled_markers if marker not in ['.']])
        if n_components == 2:
            plt.figure('Interframe dissimilarity embedding', figsize=(16, 10))
            setup_new_fig_page()
            plt.suptitle(f'n_components: {n_components}')
            ax = plt.subplot(1, 1, 1)
            plt.plot(*embedding.T, alpha=0.5)
            for cluster_idx, colour, marker in zip(range(embedding_cluster_count), colour_sequence, marker_sequence):
                plt.scatter(*embedding[clustering.labels_ == cluster_idx].T, s=200, color=colour, marker=marker, alpha=0.5)
            for frame_idx, xy in zip(embedding_frame_idxs, embedding):
                plt.text(*xy, f'{frame_idx}')
            ax.set_aspect('equal', adjustable='datalim')
            plt.tight_layout()
            stash_fig_page()
        elif n_components >= 3:
            if n_components > 3:
                mds = sklearn.manifold.MDS(n_components=3, metric=True, random_state=0)
                mds.fit(embedding)
                embedding_3d = mds.embedding_
            else:
                embedding_3d = embedding
            plt.figure('Interframe dissimilarity embedding', figsize=(16, 10))
            setup_new_fig_page()
            plt.suptitle(f'n_components: {n_components}')
            ax3 = plt.subplot(1, 1, 1, projection='3d')
            ax3.plot(*embedding_3d.T, alpha=0.5)
            for cluster_idx, colour, marker in zip(range(embedding_cluster_count), colour_sequence, marker_sequence):
                ax3.scatter(*embedding_3d[clustering.labels_ == cluster_idx].T, s=200, color=colour, marker=marker, alpha=0.5)
            for frame_idx, xyz in zip(embedding_frame_idxs, embedding_3d):
                ax3.text(*xyz, f'{frame_idx}')
            ax3.set_aspect('equal', adjustable='datalim')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            plt.tight_layout()
            stash_fig_page()

        plt.figure('Embedding costs', figsize=(16, 10))
        setup_new_fig_page()
        plt.suptitle(f'n_components: {n_components}')
        ax = plt.subplot(2, 3, 1)
        plt.imshow(interframe_dissimilarities, cmap='jet', interpolation='none')
        plt.title('interframe_dissimilarities')
        ax = plt.subplot(2, 3, 2, sharex=ax, sharey=ax)
        plt.imshow(interframe_dissimilarity_weights, cmap='jet', interpolation='none')
        plt.title('interframe_dissimilarity_weights')
        ax = plt.subplot(2, 3, 4, sharex=ax, sharey=ax)
        plt.imshow(embedding_costs, cmap='jet', interpolation='none')
        plt.title('embedding_costs')
        ax = plt.subplot(2, 3, 5, sharex=ax, sharey=ax)
        plt.imshow(embedding_costs / np.maximum(interframe_dissimilarities, 1e-8), vmin=0.2, vmax=5, cmap='jet', interpolation='none')
        plt.title('embedding_costs / interframe_dissimilarities')
        plt.subplot(2, 3, 6)
        bins = np.linspace(0, 1, 51)
        plt.hist(embedding_cost_values, bins=bins, density=True)
        for component_idx in np.argsort(gm.means_[:, 0]):
            plt.plot(bins, gm.weights_[component_idx] * scipy.stats.norm.pdf(bins, loc=gm.means_[component_idx, 0],
                                                                             scale=np.sqrt(gm.covariances_[component_idx, 0])))
        plt.axvline(mesh_frame_dissimilarity_threshold, color='black', linestyle='--')
        plt.axvline(mesh_frame_dissimilarity_boundary, color='red', linestyle='--')
        plt.axvline(mesh_frame_dissimilarity_cluster_intersection, color='blue', linestyle='--')
        plt.plot(mesh_frame_dissimilarity_threshold,
                 gm.weights_[component_idxs[1]] * scipy.stats.norm.pdf(mesh_frame_dissimilarity_threshold,
                                                                       loc=gm.means_[component_idxs[1], 0],
                                                                       scale=np.sqrt(gm.covariances_[component_idxs[1], 0])),
                 'ko')
        plt.plot(mesh_frame_dissimilarity_boundary,
                 gm.weights_[component_idxs[0]] * scipy.stats.norm.pdf(mesh_frame_dissimilarity_boundary,
                                                                       loc=gm.means_[component_idxs[0], 0],
                                                                       scale=np.sqrt(gm.covariances_[component_idxs[0], 0])),
                 'ro')
        plt.plot(mesh_frame_dissimilarity_cluster_intersection,
                 gm.weights_[component_idxs[0]] * scipy.stats.norm.pdf(mesh_frame_dissimilarity_cluster_intersection,
                                                                       loc=gm.means_[component_idxs[0], 0],
                                                                       scale=np.sqrt(gm.covariances_[component_idxs[0], 0])),
                 'bo')
        plt.title('embedding_cost_values')
        plt.tight_layout()
        stash_fig_page()
    '''

    # %%

    '''
    # A GaussianMixture model fits to the sample density of the data set.
    # This should jointly select a set of frames that have a maximal number of points in minimal
    # neighbourhood areas (i.e. maximal density) and also where the combined neighbourhoods have
    # maximal coverage over the entire population of frames.
    # But sample density might not be what we are after because the camera might just be hovering around
    # a particular location.

    gm = sklearn.mixture.GaussianMixture(n_components=3, covariance_type='spherical', random_state=0)
    gm.fit(embedding)

    print(gm.converged_)
    print(gm.weights_)
    print(gm.aic(embedding), gm.bic(embedding), gm.score(embedding))
    print(np.sqrt(gm.covariances_))
    predictions = gm.predict(embedding)
    proba = gm.predict_proba(embedding)
    plt.figure(figsize=(16, 10))
    ax = plt.subplot(2, 1, 1)
    plt.plot(embedding_frame_idxs, predictions, 'o:')
    ax = plt.subplot(2, 1, 2, sharex=ax)
    plt.plot(embedding_frame_idxs, proba, 'o:')

    # %%

    # alternatively: embedding_frame_idxs[np.argmax(gm.score_samples(embedding) + np.log(np.maximum(proba, 1e-32)).T, axis=1)]
    target_frame_idxs = embedding_frame_idxs[np.argmin(np.linalg.norm(embedding[:, :, None] - gm.means_.T, axis=1), axis=0)]
    sort_idxs = np.argsort(target_frame_idxs)
    target_frame_idxs = target_frame_idxs[sort_idxs]
    #target_secondary_frame_idxs = [embedding_frame_idxs[proba[:, idx] > 0.2] for idx in sort_idxs]

    geometries = [integrated_merged_rgbd_images_mesh]

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

    colour_sequence = matplotlib.color_sequences['tab10']
    colour_sequence = colour_sequence * int(np.ceil(len(target_frame_idxs) / len(colour_sequence)))
    for frame_idx, (prev_camera_extrinsic, camera_extrinsic) in enumerate(zip([None] + camera_extrinsics[:-1], camera_extrinsics)):
        # The extrinsic matrix transforms from world coordinates to camera coordinates
        camera_lines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=image_size[0], view_height_px=image_size[1],
                                                                        intrinsic=camera_intrinsic,
                                                                        extrinsic=camera_extrinsic)
        if frame_idx in embedding_frame_idxs:
            weights = proba[frame_idx == embedding_frame_idxs, :].T[sort_idxs]
            assert weights.shape[1] == 1
            colour = np.sum(weights * np.array(colour_sequence[:weights.shape[0]]), axis=0) / np.sum(weights)
        else:
            colour = [0, 0, 0]
        camera_lines.paint_uniform_color(colour)
        if prev_camera_extrinsic is not None:
            camera_lines.points = o3d.utility.Vector3dVector(np.vstack([camera_lines.points,
                                                                        np.linalg.inv(prev_camera_extrinsic)[:3, 3],
                                                                        np.linalg.inv(camera_extrinsic)[:3, 3]]))
            camera_lines.lines = o3d.utility.Vector2iVector(np.vstack([camera_lines.lines,
                                                                       [len(camera_lines.points) - 2, len(camera_lines.points) - 1]]))
            camera_lines.colors = o3d.utility.Vector3dVector(np.vstack([camera_lines.colors,
                                                                        [1, 0, 0.5]]))
        vis.add_geometry(camera_lines, reset_bounding_box=False)

    for frame_idx, colour in zip(target_frame_idxs, colour_sequence):
        sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
        sphere_mesh.translate(np.linalg.inv(camera_extrinsics[frame_idx])[:3, 3], relative=False)
        sphere_mesh.paint_uniform_color(colour)
        vis.add_geometry(sphere_mesh, reset_bounding_box=False)

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
    '''

    # %%

    '''
    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=0.3, sdf_trunc=3.0,
                                                          color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for rgbd_frame_idx, rgbd_image in carved_merged_rgbd_images.items():
        volume.integrate(rgbd_image,
                         o3d.camera.PinholeCameraIntrinsic(np.array(rgbd_image.depth).shape[1],
                                                           np.array(rgbd_image.depth).shape[0],
                                                           camera_intrinsic),
                         camera_extrinsics[rgbd_frame_idx])
    # TODO: remove isolated / disconnected small mesh components?
    mesh = volume.extract_triangle_mesh()
    total_surface_area = mesh.get_surface_area()

    frame_mesh_relative_surface_areas = np.array([integrate_carved_merged_rgbd_images(frame_idx).get_surface_area()
                                                  if frame_idx in mesh_frame_idxs else 0
                                                  for frame_idx in range(len(key_frame_indices))]) / total_surface_area
    '''

    # %%

    '''
    frame_mesh_mapping_scores = []
    for primary_frame_idx in range(len(key_frame_indices)):

        if primary_frame_idx in mesh_frame_idxs:
            synthetic_camera_extrinsic = np.block([[np.identity(3), np.array([0, 0, camera_pull_back_z])[:, None]], [0, 0, 0, 1]]) @ camera_extrinsics[primary_frame_idx]

            img = frame_images[primary_frame_idx]

            h, w = img.shape[:2]
            hj, wj = h * 3, w * 3

            grid_step = 5
            camera_intrinsic_stepped = np.block([[camera_intrinsic_synthetic[:2, :2] / grid_step, (camera_intrinsic_synthetic[:2, 2:] + 0.5) / grid_step - 0.5], [0, 0, 1]])

            assert wj % grid_step == 0 and hj % grid_step == 0
            ws, hs = wj // grid_step, hj // grid_step
            primary_mesh_xyzsg, xy1sg, _, _, _, _ = ray_cast_grid_points(primary_frame_idx, synthetic_camera_extrinsic, camera_intrinsic_stepped, ws, hs)

            projected_points = camera_intrinsic_synthetic @ xy1sg
            projected_points = projected_points[:2, :] / projected_points[2, :]
            xys_extent = (np.amin(projected_points[0, :]) - grid_step / 2, np.amax(projected_points[0, :]) + grid_step / 2,
                          np.amax(projected_points[1, :]) + grid_step / 2, np.amin(projected_points[1, :]) - grid_step / 2)
            assert np.allclose(xys_extent, (-0.5, wj - 0.5, hj - 0.5, -0.5))

            camera_transform = camera_extrinsics[primary_frame_idx] @ np.linalg.inv(synthetic_camera_extrinsic)

            primary_mesh_projected_points = camera_intrinsic @ (camera_transform[:3, :3] @ primary_mesh_xyzsg + camera_transform[:3, 3:])
            primary_mesh_projected_points = primary_mesh_projected_points[:2, :] / primary_mesh_projected_points[2, :]

            primary_mesh_cross_flow = primary_mesh_projected_points.T.reshape((hs, ws, 2))

            flow_gradient_x = np.gradient(primary_mesh_cross_flow, grid_step, edge_order=2, axis=1)
            flow_gradient_y = np.gradient(primary_mesh_cross_flow, grid_step, edge_order=2, axis=0)

            flow_grid_scaling_aspect = np.linalg.norm(flow_gradient_x, axis=-1) / np.linalg.norm(flow_gradient_y, axis=-1)
            flow_grid_scaling_aspect_score = cauchy(np.clip(np.max(np.stack([flow_grid_scaling_aspect, 1 / flow_grid_scaling_aspect]) - 1, axis=0), 0, np.inf), 2.0)

            flow_grid_gradient_cross = np.cross(flow_gradient_x, flow_gradient_y, axis=-1)
            flow_grid_gradient_cross[flow_grid_gradient_cross <= 0] = np.nan
            # Note that the acute/obtuse angle ambiguity between flow_gradient_x and flow_gradient_y is not
            # relevant providing we only consider the magnitude of the offset from np.pi / 2
            assert ~np.any(np.abs(flow_grid_gradient_cross) > (1 + 1e-6) * np.linalg.norm(flow_gradient_x, axis=-1) * np.linalg.norm(flow_gradient_y, axis=-1))
            flow_grid_orthogonality_score = cauchy(np.pi / 2 - np.arcsin(np.clip(flow_grid_gradient_cross / np.linalg.norm(flow_gradient_x, axis=-1) / np.linalg.norm(flow_gradient_y, axis=-1), -1, 1)),
                                                   #np.pi / 2)
                                                   np.pi / 4)

            flow_laplacian_x = np.gradient(flow_gradient_x, grid_step, edge_order=2, axis=1)
            flow_laplacian_x_transverse_normed = np.abs(np.sum(np.stack([-flow_laplacian_x[:, :, 1], flow_laplacian_x[:, :, 0]], axis=-1) * flow_gradient_x, axis=-1)) / np.sum(np.power(flow_gradient_x, 2), axis=-1)
            flow_laplacian_y = np.gradient(flow_gradient_y, grid_step, edge_order=2, axis=0)
            flow_laplacian_y_transverse_normed = np.abs(np.sum(np.stack([-flow_laplacian_y[:, :, 0], flow_laplacian_y[:, :, 1]], axis=-1) * flow_gradient_y, axis=-1)) / np.sum(np.power(flow_gradient_y, 2), axis=-1)
            flow_laplacian_score = cauchy(np.linalg.norm(np.stack([flow_laplacian_x_transverse_normed, flow_laplacian_y_transverse_normed]), axis=0), 0.1)

            if True:
                mapping_score = (nan_gaussian_filter(flow_grid_scaling_aspect_score, unfiltered_point_value=np.nan)
                                 * nan_gaussian_filter(flow_grid_orthogonality_score, unfiltered_point_value=np.nan)
                                 * nan_gaussian_filter(flow_laplacian_score, unfiltered_point_value=np.nan))
            else:
                mapping_score = np.power(3 / (1 / np.maximum(nan_gaussian_filter(flow_grid_scaling_aspect_score, unfiltered_point_value=np.nan), 1e-8)
                                              + 1 / np.maximum(nan_gaussian_filter(flow_grid_orthogonality_score, unfiltered_point_value=np.nan), 1e-8)
                                              + 1 / np.maximum(nan_gaussian_filter(flow_laplacian_score, unfiltered_point_value=np.nan), 1e-8)), 2)

            # TODO: weight mapping scores by depth?
            #frame_mesh_mapping_scores.append((np.sum(np.isfinite(mapping_score)) / mapping_score.size, np.nanmean(mapping_score)))
            frame_mesh_mapping_scores.append((frame_mesh_relative_surface_areas[primary_frame_idx],
                                              np.nanmean(mapping_score)))
        else:
            frame_mesh_mapping_scores.append((0, 0))
    frame_mesh_mapping_scores = np.array(frame_mesh_mapping_scores)

    def frame_mesh_scores(mapping_scores):
        #return (1 - cauchy(mapping_scores[..., 0], 2 / 3)) * mapping_scores[..., 1]
        return 0.5 * (1 + scipy.special.erf((mapping_scores[..., 0] - 1 / 3) * 3)) * mapping_scores[..., 1]

    plt.figure('Frame mesh mapping scores', figsize=(16, 10))
    plt.clf()
    ax = plt.subplot(2, 2, 1)
    plt.plot(frame_mesh_mapping_scores[:, 0])
    colour_sequence = itertools.cycle(matplotlib.color_sequences['tab10'])
    marker_sequence = itertools.cycle([marker for marker in matplotlib.lines.Line2D.filled_markers if marker not in ['.']])
    for cluster_idx, colour, marker in zip(range(embedding_cluster_count), colour_sequence, marker_sequence):
        cluster_frame_idxs = np.where(embedding_cluster_labels == cluster_idx)[0]
        plt.scatter(cluster_frame_idxs, frame_mesh_mapping_scores[cluster_frame_idxs, 0], s=50, color=colour, marker=marker, alpha=0.5)
    plt.title('mapping_score surface area')
    ax = plt.subplot(2, 2, 2, sharex=ax)
    plt.plot(frame_mesh_mapping_scores[:, 1])
    colour_sequence = itertools.cycle(matplotlib.color_sequences['tab10'])
    marker_sequence = itertools.cycle([marker for marker in matplotlib.lines.Line2D.filled_markers if marker not in ['.']])
    for cluster_idx, colour, marker in zip(range(embedding_cluster_count), colour_sequence, marker_sequence):
        cluster_frame_idxs = np.where(embedding_cluster_labels == cluster_idx)[0]
        plt.scatter(cluster_frame_idxs, frame_mesh_mapping_scores[cluster_frame_idxs, 1], s=50, color=colour, marker=marker, alpha=0.5)
    plt.title('mapping_score mean')
    ax = plt.subplot(2, 2, 3, sharex=ax)
    plt.plot(frame_mesh_scores(frame_mesh_mapping_scores))
    colour_sequence = itertools.cycle(matplotlib.color_sequences['tab10'])
    marker_sequence = itertools.cycle([marker for marker in matplotlib.lines.Line2D.filled_markers if marker not in ['.']])
    for cluster_idx, colour, marker in zip(range(embedding_cluster_count), colour_sequence, marker_sequence):
        cluster_frame_idxs = np.where(embedding_cluster_labels == cluster_idx)[0]
        plt.scatter(cluster_frame_idxs, frame_mesh_scores(frame_mesh_mapping_scores[cluster_frame_idxs, :]), s=50, color=colour, marker=marker, alpha=0.5)
    plt.title('frame mesh score')
    plt.subplot(2, 2, 4)
    colour_sequence = itertools.cycle(matplotlib.color_sequences['tab10'])
    marker_sequence = itertools.cycle([marker for marker in matplotlib.lines.Line2D.filled_markers if marker not in ['.']])
    for cluster_idx, colour, marker in zip(range(embedding_cluster_count), colour_sequence, marker_sequence):
        cluster_frame_idxs = np.where(embedding_cluster_labels == cluster_idx)[0]
        plt.scatter(*frame_mesh_mapping_scores[cluster_frame_idxs, :].T, s=50, color=colour, marker=marker, alpha=0.5)
    for frame_idx, xy in enumerate(frame_mesh_mapping_scores):
        plt.text(*xy, f'{frame_idx}')
    x = np.arange(*np.percentile(frame_mesh_mapping_scores[frame_mesh_mapping_scores[:, 0] > 0, 0], [0, 100]), 0.01)
    y = np.arange(*np.percentile(frame_mesh_mapping_scores[frame_mesh_mapping_scores[:, 0] > 0, 1], [0, 100]), 0.01)
    z = frame_mesh_scores(np.stack(np.meshgrid(x, y), axis=-1))
    levels = np.linspace(*np.percentile(frame_mesh_scores(frame_mesh_mapping_scores[frame_mesh_mapping_scores[:, 0] > 0, :]), [0, 100]), 5)
    plt.contour(x, y, z, levels)
    xlim = plt.xlim()
    plt.xlim(0, xlim[1])
    ylim = plt.ylim()
    plt.ylim(0, ylim[1])
    plt.xlabel('mapping_score surface area')
    plt.ylabel('mapping_score mean')
    plt.tight_layout()
    '''

    # %%

    '''
    # Compute an inter frame projection dissimilarity metric
    interframe_projection_dissimilarities = np.full((len(key_frame_indices), len(key_frame_indices)), fill_value=np.nan, dtype=np.float32)
    for rgbd_frame_idx in range(len(key_frame_indices)):
        img = frame_images[rgbd_frame_idx]

        for secondary_frame_idx in range(len(key_frame_indices)):
            hs, ws = np.round(np.array(img.shape[:2]) / 10).astype(int) + 1
            uvs = np.vstack([uv.flatten() for uv in np.mgrid[0:img.shape[0]-1:hs*1j, 0:img.shape[1]-1:ws*1j][::-1]])
            uvcs = uvs - camera_intrinsic[:2, 2:]
            xys = np.linalg.inv(camera_intrinsic[:2, :2]) @ uvcs
            xyzs = np.vstack([xys, np.ones(xys.shape[1],)])

            nominal_depth = 8
            sample_points = xyzs * nominal_depth

            camera_transform = camera_extrinsics[rgbd_frame_idx] @ np.linalg.inv(camera_extrinsics[secondary_frame_idx])

            projected_points = camera_intrinsic @ (camera_transform[:3, :3] @ sample_points + camera_transform[:3, 3:])
            projected_points = projected_points[:2, :] / projected_points[2, :]

            sigma = np.mean(img.shape[:2]) * 2 / 3
            bhattacharyya_coefficient = (np.sum(np.sqrt(scipy.stats.norm.pdf(uvs, loc=camera_intrinsic[:2, 2:], scale=sigma)
                                                        * scipy.stats.norm.pdf(projected_points, loc=camera_intrinsic[:2, 2:], scale=sigma)))
                                         / np.sqrt(np.sum(scipy.stats.norm.pdf(uvs, loc=camera_intrinsic[:2, 2:], scale=sigma)))
                                         / np.sqrt(np.sum(scipy.stats.norm.pdf(projected_points, loc=camera_intrinsic[:2, 2:], scale=sigma))))

            # (1 - bhattacharyya_coefficient) is also known as the squared Hellinger distance
            interframe_projection_dissimilarities[rgbd_frame_idx, secondary_frame_idx] = max(1 - bhattacharyya_coefficient
                                                                                             * np.cos(interframe_angles[rgbd_frame_idx, secondary_frame_idx]), 0)
            print(rgbd_frame_idx, secondary_frame_idx, interframe_projection_dissimilarities[rgbd_frame_idx, secondary_frame_idx])

    dissimilarities = 0.5 * (interframe_projection_dissimilarities + interframe_projection_dissimilarities.T)

    # Higher n_components should converge to an embedding that fits dissimilarities with lower residual errors / stress,
    # but may not generalise well to missing dissimilarities?
    # Also the variance of distances (such as Euclidean or Manhattan) between points converges to zero as the number of
    # dimensions increases https://towardsdatascience.com/curse-of-dimensionality-an-intuitive-exploration-1fbf155e1411/
    for n_components in [2, 3, 6]:
        embedding, stress, n_iter = rsatoolbox.util.vis_utils.smacof(dissimilarities, metric=True, n_components=n_components, n_init=8, n_jobs=-1,
                                                                     max_iter=300, random_state=0, return_n_iter=True, weight=None)

        print(n_components, stress, n_iter)

        kde = sklearn.neighbors.KernelDensity(bandwidth=0.05, kernel='gaussian')
        kde.fit(embedding)
        embedding_density = np.exp(kde.score_samples(embedding))
        embedding_density /= np.mean(embedding_density)

        interframe_projection_distances = sklearn.metrics.euclidean_distances(embedding)
        interframe_projection_embedding_weights = 1 / embedding_density

        if n_components == 2:
            plt.figure('Interframe projection dissimilarity embedding', figsize=(16, 10))
            setup_new_fig_page()
            plt.suptitle(f'n_components: {n_components}')
            ax = plt.subplot(1, 1, 1)
            plt.plot(*embedding.T, alpha=0.5)
            scatter = plt.scatter(*embedding.T, s=500*np.power(embedding_density, 2), c=embedding_density, cmap='jet', alpha=0.5)
            for frame_idx, xy in enumerate(embedding):
                plt.text(*xy, f'{frame_idx}')
            ax.set_aspect('equal', adjustable='datalim')
            plt.colorbar(scatter)
            plt.tight_layout()
            stash_fig_page()
        elif n_components >= 3:
            if n_components > 3:
                mds = sklearn.manifold.MDS(n_components=3, metric=True, random_state=0)
                mds.fit(embedding)
                embedding_3d = mds.embedding_
            else:
                embedding_3d = embedding
            plt.figure('Interframe projection dissimilarity embedding', figsize=(16, 10))
            setup_new_fig_page()
            plt.suptitle(f'n_components: {n_components}')
            ax3 = plt.subplot(1, 1, 1, projection='3d')
            ax3.plot(*embedding_3d.T, alpha=0.5)
            scatter = ax3.scatter(*embedding_3d.T, s=500*np.power(embedding_density, 2), c=embedding_density, cmap='jet', alpha=0.5)
            for frame_idx, xyz in enumerate(embedding_3d):
                ax3.text(*xyz, f'{frame_idx}')
            ax3.set_aspect('equal', adjustable='datalim')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            plt.colorbar(scatter)
            plt.tight_layout()
            stash_fig_page()

        plt.figure('Interframe projection distances', figsize=(16, 10))
        setup_new_fig_page()
        plt.suptitle(f'n_components: {n_components}')
        ax = plt.subplot(2, 2, 1)
        plt.imshow(interframe_projection_dissimilarities, cmap='jet', interpolation='none')
        plt.title('interframe_projection_dissimilarities')
        ax = plt.subplot(2, 2, 2, sharex=ax, sharey=ax)
        plt.imshow(dissimilarities, cmap='jet', interpolation='none')
        plt.title('dissimilarities')
        ax = plt.subplot(2, 2, 3, sharex=ax, sharey=ax)
        plt.imshow(interframe_projection_distances, cmap='jet', interpolation='none')
        plt.title('interframe_projection_distances')
        ax = plt.subplot(2, 2, 4, sharex=ax, sharey=ax)
        plt.imshow(interframe_projection_distances / np.maximum(dissimilarities, 1e-8), vmin=0.2, vmax=5, cmap='jet', interpolation='none')
        plt.title('interframe distances / dissimilarities')
        plt.tight_layout()
        stash_fig_page()
    '''

    # %%

    """
    for samples in [[1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0.5, 0.5, 1, 1],
                    [0, 0, 1, 1, 1, 1],
                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    np.linspace(0.05, 0.95, 10),
                    np.linspace(0.05, 0.95, 12)]:
        hist, bin_edges = np.histogram(samples, bins=10, range=(0, 1), density=False)
        hist = hist / np.sum(hist)
        # Divide by np.log(len(samples)) to calculate the entropy relative to that of a uniform distribution
        print(samples, 1 + np.sum(np.log(np.maximum(hist, 1e-6)) * hist) / np.log(len(samples)))
    """

    '''
    canvas_projection_scores = []
    for ref_frame_idx in range(len(key_frame_indices)):

        canvas_mesh = o3d.geometry.TriangleMesh(integrated_weighted_canvas_meshes[ref_frame_idx])

        vertices = np.array(canvas_mesh.vertices).T
        vertex_normals = np.array(canvas_mesh.vertex_normals).T
        valid_idxs = np.where(np.all(np.isfinite(vertices), axis=0))[0]
        vertices = vertices[:, valid_idxs]
        vertex_normals = vertex_normals[:, valid_idxs]

        frame_perspective_distortions = []
        for frame_idx in range(len(key_frame_indices)):
            camera_extrinsic = camera_extrinsics[frame_idx]

            object_points = camera_extrinsic[:3, :3] @ vertices + camera_extrinsic[:3, 3:]
            object_normals = camera_extrinsic[:3, :3] @ vertex_normals

            camera_rays = object_points / np.clip(np.linalg.norm(object_points, axis=0), 1e-6, np.inf)
            normal_ray_alignment = np.sum(camera_rays * object_normals, axis=0)
            camera_ray_to_object_plane = camera_rays / np.clip(-normal_ray_alignment, 1e-8, 1)
            camera_ray_to_image_plane = camera_rays * -object_normals[2, :] / camera_rays[2, :]

            object_to_image_ratio = (np.linalg.norm(object_normals + camera_ray_to_object_plane, axis=0)
                                     / np.clip(np.linalg.norm(object_normals + camera_ray_to_image_plane, axis=0), 1e-8, np.inf))
            perspective_distortion = 1 - np.min(np.stack([object_to_image_ratio,
                                                          np.clip(1 / object_to_image_ratio, 1e-8, np.inf)]), axis=0)

            perspective_distortion[normal_ray_alignment >= 0] = np.nan

            projected_points = camera_intrinsic @ object_points
            projected_points = projected_points[:2, :] / projected_points[2, :]

            img = frame_images[frame_idx]
            h, w = img.shape[:2]
            inlier_mask = ((projected_points[0, :] > -0.5) & (projected_points[0, :] < w - 0.5)
                           & (projected_points[1, :] > -0.5) & (projected_points[1, :] < h - 0.5))
            perspective_distortion[~inlier_mask] = np.nan

            frame_perspective_distortions.append(perspective_distortion)

        frame_perspective_distortions = np.array(frame_perspective_distortions)

        valid_mask = np.any(np.isfinite(frame_perspective_distortions), axis=0)
        vertex_perspective_distortions = np.min(np.nan_to_num(frame_perspective_distortions, nan=1), axis=0)
        vertex_perspective_distortions[~valid_mask] = np.nan

        samples = vertex_perspective_distortions[valid_mask]
        hist, bin_edges = np.histogram(samples, bins=10, range=(0, 1), density=False)
        hist = hist / np.sum(hist)
        # Divide by np.log(len(samples)) to calculate the entropy relative to that of a uniform distribution
        canvas_projection_scores.append(1 + np.sum(np.log(np.maximum(hist, 1e-6)) * hist) / np.log(len(samples)))

        synthetic_camera_extrinsic, camera_intrinsic_stepped, depth_kernels, color_img, depth_img = integrated_weighted_depth_kernels[ref_frame_idx]

        vertex_perspective_distortions_image = np.full((len(canvas_mesh.vertices),), fill_value=np.nan)
        vertex_perspective_distortions_image[valid_idxs] = vertex_perspective_distortions
        vertex_perspective_distortions_image = vertex_perspective_distortions_image.reshape(depth_img.shape)

        plt.figure('Vertex projections scores', figsize=(16, 10))
        setup_new_fig_page()
        plt.suptitle(f'ref_frame_idx: {ref_frame_idx}')
        ax = plt.subplot(2, 2, 1)
        plt.imshow(color_img / 255)
        plt.title('integrated colour image')
        ax = plt.subplot(2, 2, 2, sharex=ax, sharey=ax)
        plt.imshow(depth_img, vmin=camera_pull_back_z, vmax=camera_pull_back_z+30)
        plt.title('integrated depth')
        ax = plt.subplot(2, 2, 3, sharex=ax, sharey=ax)
        plt.imshow(vertex_perspective_distortions_image, cmap='jet', vmin=0, vmax=1)
        plt.title('vertex_perspective_distortions_image')
        plt.subplot(2, 2, 4)
        plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge')
        plt.ylim((0, 1))
        plt.title('vertex_perspective_distortions histogram')
        plt.tight_layout()
        stash_fig_page()

    canvas_projection_scores = np.array(canvas_projection_scores)
    '''

    # %%

    '''
    target_frame_idxs = []
    intra_cluster_rms_interframe_depth_errors = np.full((len(key_frame_indices),), fill_value=np.nan)
    for cluster_idx in range(embedding_cluster_count):
        cluster_frame_idxs = np.where(embedding_cluster_labels == cluster_idx)[0]
        cluster_interframe_depth_errors = interframe_depth_errors[cluster_frame_idxs[:, None], cluster_frame_idxs]
        cluster_interframe_depth_errors = cluster_interframe_depth_errors[~np.identity(cluster_interframe_depth_errors.shape[0], dtype=bool)].reshape((cluster_interframe_depth_errors.shape[0], -1))
        cluster_interframe_depth_error_weights = interframe_depth_error_weights[cluster_frame_idxs[:, None], cluster_frame_idxs]
        cluster_interframe_depth_error_weights = cluster_interframe_depth_error_weights[~np.identity(cluster_interframe_depth_error_weights.shape[0], dtype=bool)].reshape((cluster_interframe_depth_error_weights.shape[0], -1))
        cluster_rms_interframe_depth_errors = np.sqrt(np.sum(np.power(cluster_interframe_depth_errors, 2) * cluster_interframe_depth_error_weights, axis=1)
                                                      / np.sum(cluster_interframe_depth_error_weights, axis=1))
        cluster_canvas_projection_scores = canvas_projection_scores[cluster_frame_idxs]
        intra_cluster_rms_interframe_depth_errors[cluster_frame_idxs] = cluster_rms_interframe_depth_errors
        target_frame_idxs.append(cluster_frame_idxs[np.argmax(cluster_canvas_projection_scores / cluster_rms_interframe_depth_errors)])

    target_frame_idxs = sorted(target_frame_idxs)
    print('target_frame_idxs', target_frame_idxs)

    plt.figure('Target frames', figsize=(16, 10))
    plt.clf()
    ax = plt.subplot(2, 2, 1)
    plt.plot(canvas_projection_scores)
    colour_sequence = itertools.cycle(matplotlib.color_sequences['tab10'])
    marker_sequence = itertools.cycle([marker for marker in matplotlib.lines.Line2D.filled_markers if marker not in ['.']])
    for cluster_idx, colour, marker in zip(range(embedding_cluster_count), colour_sequence, marker_sequence):
        cluster_frame_idxs = np.where(embedding_cluster_labels == cluster_idx)[0]
        plt.scatter(cluster_frame_idxs, canvas_projection_scores[cluster_frame_idxs], s=50, color=colour, marker=marker, alpha=0.5)
    plt.title('canvas_projection_scores')
    ax = plt.subplot(2, 2, 2, sharex=ax)
    plt.plot(intra_cluster_rms_interframe_depth_errors)
    colour_sequence = itertools.cycle(matplotlib.color_sequences['tab10'])
    marker_sequence = itertools.cycle([marker for marker in matplotlib.lines.Line2D.filled_markers if marker not in ['.']])
    for cluster_idx, colour, marker in zip(range(embedding_cluster_count), colour_sequence, marker_sequence):
        cluster_frame_idxs = np.where(embedding_cluster_labels == cluster_idx)[0]
        plt.scatter(cluster_frame_idxs, intra_cluster_rms_interframe_depth_errors[cluster_frame_idxs], s=50, color=colour, marker=marker, alpha=0.5)
    plt.title('intra_cluster_rms_interframe_depth_errors')
    ax = plt.subplot(2, 2, 3, sharex=ax)
    target_frame_scores = canvas_projection_scores / intra_cluster_rms_interframe_depth_errors
    plt.plot(target_frame_scores)
    colour_sequence = itertools.cycle(matplotlib.color_sequences['tab10'])
    marker_sequence = itertools.cycle([marker for marker in matplotlib.lines.Line2D.filled_markers if marker not in ['.']])
    for cluster_idx, colour, marker in zip(range(embedding_cluster_count), colour_sequence, marker_sequence):
        cluster_frame_idxs = np.where(embedding_cluster_labels == cluster_idx)[0]
        plt.scatter(cluster_frame_idxs, target_frame_scores[cluster_frame_idxs], s=50, color=colour, marker=marker, alpha=0.5)
    plt.title('target_frame_scores')
    plt.tight_layout()
    '''

    # %%

    '''
    target_frame_idxs = []
    intra_cluster_rms_interframe_canvas_projection_errors = np.full((len(key_frame_indices),), fill_value=np.nan)
    for cluster_idx in range(embedding_cluster_count):
        cluster_frame_idxs = np.where(embedding_cluster_labels == cluster_idx)[0]
        cluster_interframe_canvas_projection_errors = interframe_canvas_projection_errors[cluster_frame_idxs[:, None], cluster_frame_idxs]
        cluster_interframe_canvas_projection_errors = cluster_interframe_canvas_projection_errors[~np.identity(cluster_interframe_canvas_projection_errors.shape[0], dtype=bool)].reshape((cluster_interframe_canvas_projection_errors.shape[0], -1))
        cluster_interframe_canvas_projection_error_weights = interframe_canvas_projection_error_weights[cluster_frame_idxs[:, None], cluster_frame_idxs]
        cluster_interframe_canvas_projection_error_weights = cluster_interframe_canvas_projection_error_weights[~np.identity(cluster_interframe_canvas_projection_error_weights.shape[0], dtype=bool)].reshape((cluster_interframe_canvas_projection_error_weights.shape[0], -1))
        cluster_rms_interframe_canvas_projection_errors = np.sqrt(np.sum(np.power(cluster_interframe_canvas_projection_errors, 2) * cluster_interframe_canvas_projection_error_weights, axis=1)
                                                                  / np.sum(cluster_interframe_canvas_projection_error_weights, axis=1))
        cluster_synth_projection_scores = frame_synth_projection_scores[cluster_frame_idxs]
        intra_cluster_rms_interframe_canvas_projection_errors[cluster_frame_idxs] = cluster_rms_interframe_canvas_projection_errors
        target_frame_idxs.append(cluster_frame_idxs[np.argmax(cluster_synth_projection_scores / cluster_rms_interframe_canvas_projection_errors)])

    target_frame_idxs = sorted(target_frame_idxs)
    print('target_frame_idxs', target_frame_idxs)

    plt.figure('Target frames', figsize=(16, 10))
    plt.clf()
    ax = plt.subplot(2, 2, 1)
    plt.plot(frame_synth_projection_scores)
    colour_sequence = itertools.cycle(matplotlib.color_sequences['tab10'])
    marker_sequence = itertools.cycle([marker for marker in matplotlib.lines.Line2D.filled_markers if marker not in ['.']])
    for cluster_idx, colour, marker in zip(range(embedding_cluster_count), colour_sequence, marker_sequence):
        cluster_frame_idxs = np.where(embedding_cluster_labels == cluster_idx)[0]
        plt.scatter(cluster_frame_idxs, frame_synth_projection_scores[cluster_frame_idxs], s=50, color=colour, marker=marker, alpha=0.5)
    plt.title('frame_synth_projection_scores')
    ax = plt.subplot(2, 2, 2, sharex=ax)
    plt.plot(intra_cluster_rms_interframe_canvas_projection_errors)
    colour_sequence = itertools.cycle(matplotlib.color_sequences['tab10'])
    marker_sequence = itertools.cycle([marker for marker in matplotlib.lines.Line2D.filled_markers if marker not in ['.']])
    for cluster_idx, colour, marker in zip(range(embedding_cluster_count), colour_sequence, marker_sequence):
        cluster_frame_idxs = np.where(embedding_cluster_labels == cluster_idx)[0]
        plt.scatter(cluster_frame_idxs, intra_cluster_rms_interframe_canvas_projection_errors[cluster_frame_idxs], s=50, color=colour, marker=marker, alpha=0.5)
    plt.title('intra_cluster_rms_interframe_canvas_projection_errors')
    ax = plt.subplot(2, 2, 3, sharex=ax)
    target_frame_scores = frame_synth_projection_scores / intra_cluster_rms_interframe_canvas_projection_errors
    plt.plot(target_frame_scores)
    colour_sequence = itertools.cycle(matplotlib.color_sequences['tab10'])
    marker_sequence = itertools.cycle([marker for marker in matplotlib.lines.Line2D.filled_markers if marker not in ['.']])
    for cluster_idx, colour, marker in zip(range(embedding_cluster_count), colour_sequence, marker_sequence):
        cluster_frame_idxs = np.where(embedding_cluster_labels == cluster_idx)[0]
        plt.scatter(cluster_frame_idxs, target_frame_scores[cluster_frame_idxs], s=50, color=colour, marker=marker, alpha=0.5)
    plt.title('target_frame_scores')
    plt.tight_layout()
    '''

    # %%

    '''
    # Find the furthest distance of any node to the nearest target frame for all target frame combinations.
    # This tends to hide granularity as the furthest node from one target frame will remain the furthest
    # node for a substantial set of combinations of target frames.
    n_components = 3
    furthest_distances_to_closest_target = []
    for target_idxs in itertools.combinations(embedding_frame_idxs, n_components):
        closest_distances = np.min(interframe_costs[np.array(target_idxs)[:, None], embedding_frame_idxs], axis=0)
        furthest_distances_to_closest_target.append((target_idxs,
                                                     np.max(closest_distances),
                                                     frame_mesh_scores(frame_mesh_mapping_scores[np.array(target_idxs), :])))

    # TODO: add a penalty term for target frames being too physically close
    def target_frame_sort_func(args):
        target_idxs, furthest_distance_to_closest_target, mesh_scores = args
        return furthest_distance_to_closest_target - 0.2 * np.mean(mesh_scores)
    sorted_target_frame_idxs = sorted(furthest_distances_to_closest_target, key=target_frame_sort_func)
    print(sorted_target_frame_idxs[:10])

    target_frame_idxs = sorted_target_frame_idxs[0][0]
    '''

    # %%

    '''
    # Find the Cube-Root Mean Cube distance of nodes to their nearest target frames for all target frame combinations.
    # If this was RMS instead of CRMC, this could be notionally similar to modelling with a Gaussian Mixture
    # having fixed & tied spherical covariances.
    n_components = 3
    crmc_distances_to_closest_target = []
    for target_idxs in itertools.combinations(embedding_frame_idxs, n_components):
        closest_distances = np.min(interframe_costs[np.array(target_idxs)[:, None], embedding_frame_idxs], axis=0)
        crmc_distances_to_closest_target.append((target_idxs,
                                                 np.power(np.mean(np.power(closest_distances, 3)), 1 / 3),
                                                 frame_mesh_scores(frame_mesh_mapping_scores[np.array(target_idxs), :])))

    # TODO: add a penalty term for target frames being too physically close
    def target_frame_sort_func(args):
        target_idxs, crmc_distance_to_closest_target, mesh_scores = args
        #return crmc_distance_to_closest_target - 1.0 * np.mean(mesh_scores)
        #return crmc_distance_to_closest_target / np.mean(mesh_scores)
        return crmc_distance_to_closest_target * np.mean(1 / np.maximum(mesh_scores, 1e-8))
    sorted_target_frame_idxs = sorted(crmc_distances_to_closest_target, key=target_frame_sort_func)
    print(sorted_target_frame_idxs[:10])

    target_frame_idxs = sorted_target_frame_idxs[0][0]
    '''

    # %%

    '''
    # Find the Cube-Root Mean Cube distance of nodes to their nearest target frames for all target frame combinations.
    # If this was RMS instead of CRMC, this could be notionally similar to modelling with a Gaussian Mixture
    # having fixed & tied spherical covariances.
    n_components = 3
    crmc_distances_to_closest_target = []
    for target_idxs in itertools.combinations(range(interframe_projection_distances.shape[0]), n_components):
        closest_distances = np.min(interframe_projection_distances[target_idxs, :], axis=0)
        #inter_target_distances = interframe_projection_distances[np.ix_(target_idxs, target_idxs)][np.tril_indices(n_components, k=-1)]
        #inter_target_distances = len(inter_target_distances) / np.sum(1 / np.maximum(inter_target_distances, 1e-8))
        crmc_distances_to_closest_target.append((target_idxs,
                                                 np.power(np.sum(interframe_projection_embedding_weights * np.power(closest_distances, 3)) / np.sum(interframe_projection_embedding_weights), 1 / 3),
                                                 frame_mesh_scores(frame_mesh_mapping_scores[np.array(target_idxs), :])))

    # TODO: add a penalty term for target frames being too physically close
    def target_frame_sort_func(args):
        target_idxs, crmc_distance_to_closest_target, mesh_scores = args
        #return crmc_distance_to_closest_target - 1.0 * np.mean(mesh_scores)
        #return crmc_distance_to_closest_target / np.mean(mesh_scores)
        return crmc_distance_to_closest_target * np.mean(1 / np.maximum(mesh_scores, 1e-8))
    sorted_target_frame_idxs = sorted(crmc_distances_to_closest_target, key=target_frame_sort_func)
    print(sorted_target_frame_idxs[:10])

    target_frame_idxs = sorted_target_frame_idxs[0][0]
    '''

    # %%

    '''
    # Find the Cube-Root Mean Cube distance of nodes to their nearest target frames for all target frame combinations.
    # If this was RMS instead of CRMC, this could be notionally similar to modelling with a Gaussian Mixture
    # having fixed & tied spherical covariances.
    crmc_distances_to_closest_target = []
    for target_idxs in itertools.product(*[np.where(embedding_cluster_labels == cluster_idx)[0]
                                          for cluster_idx in range(embedding_cluster_count)]):
        closest_distances = np.min(interframe_costs[np.array(target_idxs)[:, None], embedding_frame_idxs], axis=0)
        crmc_distances_to_closest_target.append((target_idxs,
                                                 np.power(np.mean(np.power(closest_distances, 3)), 1 / 3),
                                                 frame_mesh_scores(frame_mesh_mapping_scores[np.array(target_idxs), :])))

    # TODO: add a penalty term for target frames being too physically close
    def target_frame_sort_func(args):
        target_idxs, crmc_distance_to_closest_target, mesh_scores = args
        #return crmc_distance_to_closest_target - 1.0 * np.mean(mesh_scores)
        #return crmc_distance_to_closest_target / np.mean(mesh_scores)
        return crmc_distance_to_closest_target * np.mean(1 / np.maximum(mesh_scores, 1e-8))
    sorted_target_frame_idxs = sorted(crmc_distances_to_closest_target, key=target_frame_sort_func)
    print(sorted_target_frame_idxs[:10])

    target_frame_idxs = sorted_target_frame_idxs[0][0]
    '''

    # %%

    '''
    target_frame_idxs = []
    for cluster_idx in range(embedding_cluster_count):
        cluster_frame_idxs = np.where(embedding_cluster_labels == cluster_idx)[0]
        cluster_interframe_costs = interframe_costs[cluster_frame_idxs[:, None], cluster_frame_idxs]
        cluster_interframe_costs = cluster_interframe_costs[~np.identity(cluster_interframe_costs.shape[0], dtype=bool)].reshape((cluster_interframe_costs.shape[0], -1))
        rms_cluster_interframe_costs = np.sqrt(np.mean(np.power(cluster_interframe_costs, 2), axis=1))
        cluster_frame_mesh_scores = frame_mesh_scores(frame_mesh_mapping_scores[cluster_frame_idxs, :])
        target_frame_idxs.append(cluster_frame_idxs[np.argmax(cluster_frame_mesh_scores / rms_cluster_interframe_costs)])

    target_frame_idxs = sorted(target_frame_idxs)
    print('target_frame_idxs', target_frame_idxs)
    '''

    # %%

    '''
    target_frame_idxs = []
    for cluster_idx in range(embedding_cluster_count):
        cluster_frame_idxs = np.where(embedding_cluster_labels == cluster_idx)[0]
        target_frame_idxs.append(cluster_frame_idxs[np.argmax(canvas_projection_scores[cluster_frame_idxs])])

    target_frame_idxs = sorted(target_frame_idxs)
    print('target_frame_idxs', target_frame_idxs)
    '''

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

    colour_sequence = matplotlib.color_sequences['tab10']
    colour_sequence = colour_sequence * int(np.ceil(len(target_frame_idxs) / len(colour_sequence)))
    for frame_idx, (prev_camera_extrinsic, camera_extrinsic) in enumerate(zip([None] + camera_extrinsics[:-1], camera_extrinsics)):
        # The extrinsic matrix transforms from world coordinates to camera coordinates
        camera_lines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=image_size[0], view_height_px=image_size[1],
                                                                        intrinsic=camera_intrinsic,
                                                                        extrinsic=camera_extrinsic)
        if frame_idx in embedding_frame_idxs:
            weights = 1 / np.maximum(interframe_dissimilarities[frame_idx, target_frame_idxs], 1e-8)
            colour = np.clip(np.sum(weights[:, None] * np.array(colour_sequence[:len(weights)]), axis=0) / np.sum(weights), 0, 1)
        else:
            colour = [0, 0, 0]
        camera_lines.paint_uniform_color(colour)
        if prev_camera_extrinsic is not None:
            camera_lines.points = o3d.utility.Vector3dVector(np.vstack([camera_lines.points,
                                                                        np.linalg.inv(prev_camera_extrinsic)[:3, 3],
                                                                        np.linalg.inv(camera_extrinsic)[:3, 3]]))
            camera_lines.lines = o3d.utility.Vector2iVector(np.vstack([camera_lines.lines,
                                                                       [len(camera_lines.points) - 2, len(camera_lines.points) - 1]]))
            camera_lines.colors = o3d.utility.Vector3dVector(np.vstack([camera_lines.colors,
                                                                        [1, 0, 0.5]]))
        vis.add_geometry(camera_lines, reset_bounding_box=False)

    view_status = vis.get_view_status()
    view_status_time = time.time()
    visualisation_idle_timeout = 60 if IPython.get_ipython() is not None else 10
    while True:
        close_vis = False
        for target_frame_idx in target_frame_idxs:
            mesh = o3d.geometry.TriangleMesh(integrated_weighted_canvas_meshes[target_frame_idx])
            mesh.vertex_normals = o3d.utility.Vector3dVector()
            geometries = [mesh]

            for frame_idx, colour in zip(target_frame_idxs, colour_sequence[:len(target_frame_idxs)]):
                sphere_radius = 0.6 if frame_idx == target_frame_idx else 0.3
                sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
                sphere_mesh.translate(np.linalg.inv(camera_extrinsics[frame_idx])[:3, 3], relative=False)
                sphere_mesh.paint_uniform_color(colour)
                geometries.append(sphere_mesh)

            geometries = [geometry.crop(o3d.geometry.AxisAlignedBoundingBox([-100, -100, -100], [100, 100, 100]))
                          if isinstance(geometry, (o3d.geometry.PointCloud, o3d.geometry.TriangleMesh)) else geometry
                          for geometry in geometries]

            for geometry in geometries:
                vis.add_geometry(geometry, reset_bounding_box=False)

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
                if close_vis or time.time() > start_time + 1.0:
                    break

            for geometry in geometries:
                vis.remove_geometry(geometry, reset_bounding_box=False)

            if close_vis:
                break

        if close_vis:
            break

    vis.destroy_window()

    # %%

    '''
    frame_triangulated_point_normals = []
    for frame_idx in range(len(key_frame_indices)):
        image_to_triangulated_point_idxs = key_frame_image_triangulated_point_idxs[frame_idx]
        frame_triangulated_point_idxs = image_to_triangulated_point_idxs[image_to_triangulated_point_idxs >= 0]
        valid_mask = np.all(np.isfinite(model_triangulated_points[:3, frame_triangulated_point_idxs]), axis=0)
        frame_triangulated_point_idxs = frame_triangulated_point_idxs[valid_mask]

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(model_triangulated_points[:3, frame_triangulated_point_idxs].T))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=64))
        #pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, -1]))
        # https://github.com/isl-org/Open3D/blob/v0.18.0/cpp/open3d/geometry/PointCloud.h
        # Function to consistently orient estimated normals based on consistent tangent planes as described in Hoppe et al.,
        # "Surface Reconstruction from Unorganized Points", 1992.
        # Further details on parameters are described in Piazza, Valentini, Varetti, "Mesh Reconstruction from Point Cloud", 2023.
        #  - k: k nearest neighbour for graph reconstruction for normal propagation.
        #  - lambda: penalty constant on the distance of a point from the tangent plane
        #  - cos_alpha_tol: threshold that defines the amplitude of the cone spanned by the reference normal
        # pcd.orient_normals_consistent_tangent_plane tends to invert normals for points not fully aligned with the primary surface
        #pcd.orient_normals_consistent_tangent_plane(**{'k': 30, 'lambda': 0.0, 'cos_alpha_tol': 1.0})
        #if np.mean(np.array(pcd.normals)[:, 2]) > 0:
        #    pcd.normals = o3d.utility.Vector3dVector(-np.array(pcd.normals))
        pcd_points = np.array(pcd.points).T
        pcd_normals = np.array(pcd.normals).T
        normal_ray_alignment = np.zeros(frame_triangulated_point_idxs.shape, dtype=np.float32)
        for image_to_triangulated_point_idxs, camera_extrinsic in zip(key_frame_image_triangulated_point_idxs,
                                                                      camera_extrinsics):
            triangulated_image_idxs = np.where(image_to_triangulated_point_idxs >= 0)[0]
            triangulated_point_idxs = image_to_triangulated_point_idxs[triangulated_image_idxs]
            image_points_weights = np.zeros((model_triangulated_points.shape[1],), dtype=np.float32)
            image_points_weights[triangulated_point_idxs] = 1

            R = camera_extrinsic[:3, :3]
            t = camera_extrinsic[:3, 3:]
            object_points = R @ pcd_points + t
            object_normals = R @ pcd_normals

            object_point_ray = object_points / np.clip(np.linalg.norm(object_points, axis=0), 1e-6, np.inf)
            normal_ray_alignment += np.sum(object_point_ray * object_normals, axis=0) * image_points_weights[frame_triangulated_point_idxs]
        pcd.normals = o3d.utility.Vector3dVector((pcd_normals * -np.sign(normal_ray_alignment)).T)
        assert np.allclose(np.linalg.norm(np.array(pcd.normals), axis=1), 1)

        triangulated_point_normals = np.full((3, model_triangulated_points.shape[1]), fill_value=np.nan, dtype=np.float32)
        triangulated_point_normals[:, frame_triangulated_point_idxs] = np.array(pcd.normals).T
        frame_triangulated_point_normals.append(triangulated_point_normals)

    # %%

    plt.figure(figsize=(16, 10))
    ax3 = plt.subplot(1, 1, 1, projection='3d')
    ax3.scatter(*model_triangulated_points[:3, :], c='b', s=2)
    ax3.set_xlim((-20, 20))
    ax3.set_ylim((-20, 20))
    ax3.set_zlim((0, 40))
    ax3.set_aspect('equal', adjustable='datalim')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.view_init(elev=-135, azim=-90, roll=0)
    plt.tight_layout()

    # %%

    plt.figure(figsize=(16, 10))
    ax3 = plt.subplot(1, 1, 1, projection='3d')
    ax3.scatter(*model_triangulated_points[:3, np.all(np.isfinite(frame_triangulated_point_normals[0]), axis=0)], c='b', s=2)
    ax3.scatter(*model_triangulated_points[:3, np.all(np.isfinite(frame_triangulated_point_normals[28]), axis=0)], c='r', s=2)
    ax3.scatter(*model_triangulated_points[:3, np.all(np.isfinite(frame_triangulated_point_normals[0] * frame_triangulated_point_normals[28]), axis=0)], c='k')
    ax3.set_xlim((-20, 20))
    ax3.set_ylim((-20, 20))
    ax3.set_zlim((0, 40))
    ax3.set_aspect('equal', adjustable='datalim')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.view_init(elev=-135, azim=-90, roll=0)
    plt.tight_layout()

    # %%

    interframe_incoherence = np.full((len(key_frame_indices),) * 2, fill_value=np.nan, dtype=np.float32)
    for ref_frame_idx in range(len(key_frame_indices)):
        for frame_idx in range(len(key_frame_indices)):
            normal_misalignment = 1 - np.sum(frame_triangulated_point_normals[frame_idx] * frame_triangulated_point_normals[ref_frame_idx], axis=0)
            if np.any(np.isfinite(normal_misalignment)):
                interframe_incoherence[ref_frame_idx, frame_idx] = np.sqrt(np.nanmean(np.power(normal_misalignment, 2)))

    plt.figure(figsize=(16, 10))
    plt.imshow(interframe_incoherence, vmin=0, vmax=1, cmap='jet', interpolation='none')
    '''

    # %%

    """
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

    for frame_idx, camera_extrinsic in enumerate(camera_extrinsics):
        # The extrinsic matrix transforms from world coordinates to camera coordinates
        camera_lines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=image_size[0], view_height_px=image_size[1],
                                                                        intrinsic=camera_intrinsic,
                                                                        extrinsic=camera_extrinsic)
        camera_lines.paint_uniform_color([0, 0.5, 1])
        vis.add_geometry(camera_lines, reset_bounding_box=False)

        for prev_frame_idx, prev_camera_extrinsic in enumerate(camera_extrinsics[:frame_idx]):
            if interframe_confidences[prev_frame_idx, frame_idx] > 0:
                line_points = np.vstack([np.linalg.inv(prev_camera_extrinsic)[:3, 3], np.linalg.inv(camera_extrinsic)[:3, 3]])

                line_length = np.linalg.norm(np.diff(line_points, axis=0))
                line_unit_vector = np.diff(line_points, axis=0) / line_length

                rot_cross_vec = np.cross(np.array([0, 0, 1]), line_unit_vector)
                rot_angle = np.arccos(np.clip(np.array([0, 0, 1]) @ line_unit_vector.T, -1, 1))
                rot_vec = rot_angle * rot_cross_vec / np.linalg.norm(rot_cross_vec)

                radius = interframe_confidences[prev_frame_idx, frame_idx] * 0.1
                line_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=line_length)
                line_mesh.paint_uniform_color([1, 0, 0.5])

                R, _ = cv2.Rodrigues(rot_vec)
                line_mesh.rotate(R)
                line_mesh.translate(np.mean(line_points, axis=0))

                vis.add_geometry(line_mesh, reset_bounding_box=False)

    geometry = integrated_merged_rgbd_images_mesh.crop(o3d.geometry.AxisAlignedBoundingBox([-100, -100, -100], [100, 100, 100]))
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
    """

    # %%

    # TODO: only map the set of coherent / low cost secondary frames for each primary frame

    def get_frame_idxs_cmap(frame_idxs, N):
        assert np.all(frame_idxs) >= 0
        padded_frame_idxs = np.pad(frame_idxs, pad_width=((1, 1), (1, 1)), mode='constant', constant_values=-1)
        neighbouring_frame_idxs = np.stack([padded_frame_idxs[1:-1, 1:-1],
                                            padded_frame_idxs[:-2, :-2],
                                            padded_frame_idxs[1:-1, :-2],
                                            padded_frame_idxs[2:, :-2],
                                            padded_frame_idxs[:-2, 1:-1],
                                            padded_frame_idxs[2:, 1:-1],
                                            padded_frame_idxs[:-2, 2:],
                                            padded_frame_idxs[1:-1, 2:],
                                            padded_frame_idxs[2:, 2:]],
                                           axis=-1)
        neighbour_map = collections.defaultdict(set)
        for idxs in neighbouring_frame_idxs.reshape((-1, neighbouring_frame_idxs.shape[-1])):
            neighbour_map[idxs[0]].update(idxs[1:])
        G = networkx.Graph()
        for idx, neighbour_idxs in neighbour_map.items():
            for neighbour_idx in neighbour_idxs - set([idx, -1]):
                G.add_edge(idx, neighbour_idx)
        graph_colouring = networkx.coloring.greedy_color(G)
        graph_colouring_counts = collections.Counter(graph_colouring.values())
        assert sorted(graph_colouring_counts) == list(range(len(graph_colouring_counts)))
        graph_colouring_cmap_boundaries = np.cumsum([0] + [graph_colouring_counts[graph_colour] for graph_colour in sorted(graph_colouring_counts)]) / len(graph_colouring)
        graph_colour_cmap_boundaries = np.vstack([graph_colouring_cmap_boundaries[:-1], graph_colouring_cmap_boundaries[1:]]).T
        graph_colour_cmap_boundary_contraction = 0.2 / max(len(graph_colouring_counts) - 1, 1)
        graph_colour_cmap_boundaries[1:, 0] += graph_colour_cmap_boundary_contraction
        graph_colour_cmap_boundaries[:-1, 1] -= graph_colour_cmap_boundary_contraction
        cmap_colours = np.zeros((N, 4))
        for graph_colour in sorted(graph_colouring_counts):
            cmap_seq = list(matplotlib.colormaps['nipy_spectral'](np.linspace(graph_colour_cmap_boundaries[graph_colour, 0],
                                                                              graph_colour_cmap_boundaries[graph_colour, 1],
                                                                              graph_colouring_counts[graph_colour])))
            for frame_idx, colour_idx in graph_colouring.items():
                if colour_idx == graph_colour:
                    cmap_colours[frame_idx] = cmap_seq.pop(0)
        cmap = matplotlib.colors.ListedColormap(cmap_colours, N=mapping_scores.shape[0])
        return cmap

    plt.close('Synthetic frame scores')
    plt.close('Synthetic view')

    primary_frame_synthesis_elements = {}
    for primary_frame_idx in target_frame_idxs:

        print('synthesising primary_frame_idx', primary_frame_idx)

        synthetic_camera_extrinsic = np.block([[np.identity(3), np.array([0, 0, camera_pull_back_z])[:, None]], [0, 0, 0, 1]]) @ camera_extrinsics[primary_frame_idx]

        img = frame_images[primary_frame_idx]

        h, w = img.shape[:2]
        hj, wj = h * 3, w * 3

        grid_step = 5
        camera_intrinsic_stepped = np.block([[camera_intrinsic_synthetic[:2, :2] / grid_step, (camera_intrinsic_synthetic[:2, 2:] + 0.5) / grid_step - 0.5], [0, 0, 1]])

        assert wj % grid_step == 0 and hj % grid_step == 0
        ws, hs = wj // grid_step, hj // grid_step
        primary_mesh_xyzsg, xy1sg, _, _, _, _ = ray_cast_grid_points(primary_frame_idx, synthetic_camera_extrinsic, camera_intrinsic_stepped, ws, hs)

        projected_points = camera_intrinsic_synthetic @ xy1sg
        projected_points = projected_points[:2, :] / projected_points[2, :]
        xys_extent = (np.amin(projected_points[0, :]) - grid_step / 2, np.amax(projected_points[0, :]) + grid_step / 2,
                      np.amax(projected_points[1, :]) + grid_step / 2, np.amin(projected_points[1, :]) - grid_step / 2)
        assert np.allclose(xys_extent, (-0.5, wj - 0.5, hj - 0.5, -0.5))

        mapping_scores = np.full((len(key_frame_indices), hs, ws), fill_value=np.nan, dtype=np.float32)
        for secondary_frame_idx in range(len(key_frame_indices)):
            secondary_mesh_xyzsg, _, _, _, _, _ = ray_cast_grid_points(secondary_frame_idx, synthetic_camera_extrinsic, camera_intrinsic_stepped, ws, hs)

            secondary_camera_extrinsic = camera_extrinsics[secondary_frame_idx]
            camera_transform = secondary_camera_extrinsic @ np.linalg.inv(synthetic_camera_extrinsic)

            primary_mesh_projected_points = camera_intrinsic @ (camera_transform[:3, :3] @ primary_mesh_xyzsg + camera_transform[:3, 3:])
            primary_mesh_projected_points = primary_mesh_projected_points[:2, :] / primary_mesh_projected_points[2, :]

            secondary_mesh_projected_points = camera_intrinsic @ (camera_transform[:3, :3] @ secondary_mesh_xyzsg + camera_transform[:3, 3:])
            secondary_mesh_projected_points = secondary_mesh_projected_points[:2, :] / secondary_mesh_projected_points[2, :]

            primary_mesh_cross_flow = primary_mesh_projected_points.T.reshape((hs, ws, 2))
            secondary_mesh_cross_flow = secondary_mesh_projected_points.T.reshape((hs, ws, 2))

            cross_flow_inlier = ((primary_mesh_cross_flow[:, :, 0] > -0.5) & (primary_mesh_cross_flow[:, :, 0] < w - 0.5)
                                 & (primary_mesh_cross_flow[:, :, 1] > -0.5) & (primary_mesh_cross_flow[:, :, 1] < h - 0.5))
            inlier_scores = nan_gaussian_filter(cross_flow_inlier)
            inlier_scores[~cross_flow_inlier] = 0

            flow_discrepancy_score = cauchy(np.linalg.norm(secondary_mesh_cross_flow - primary_mesh_cross_flow, axis=-1), 10)

            flow_gradient_x = np.gradient(primary_mesh_cross_flow, grid_step, edge_order=2, axis=1)
            flow_gradient_y = np.gradient(primary_mesh_cross_flow, grid_step, edge_order=2, axis=0)

            flow_grid_scaling_aspect = np.linalg.norm(flow_gradient_x, axis=-1) / np.linalg.norm(flow_gradient_y, axis=-1)
            flow_grid_scaling_aspect_score = cauchy(np.clip(np.max(np.stack([flow_grid_scaling_aspect, 1 / flow_grid_scaling_aspect]) - 1, axis=0), 0, np.inf), 2.0)

            flow_grid_gradient_cross = np.cross(flow_gradient_x, flow_gradient_y, axis=-1)
            flow_grid_gradient_cross[flow_grid_gradient_cross <= 0] = np.nan
            # Note that the acute/obtuse angle ambiguity between flow_gradient_x and flow_gradient_y is not
            # relevant providing we only consider the magnitude of the offset from np.pi / 2
            assert ~np.any(np.abs(flow_grid_gradient_cross) > (1 + 1e-6) * np.linalg.norm(flow_gradient_x, axis=-1) * np.linalg.norm(flow_gradient_y, axis=-1))
            flow_grid_orthogonality_score = cauchy(np.pi / 2 - np.arcsin(np.clip(flow_grid_gradient_cross / np.linalg.norm(flow_gradient_x, axis=-1) / np.linalg.norm(flow_gradient_y, axis=-1), -1, 1)),
                                                   #np.pi / 2 * cauchy(interframe_angles[primary_frame_idx, secondary_frame_idx], np.pi / 4))
                                                   np.pi / 4)

            flow_laplacian_x = np.gradient(flow_gradient_x, grid_step, edge_order=2, axis=1)
            flow_laplacian_x_transverse_normed = np.abs(np.sum(np.stack([-flow_laplacian_x[:, :, 1], flow_laplacian_x[:, :, 0]], axis=-1) * flow_gradient_x, axis=-1)) / np.sum(np.power(flow_gradient_x, 2), axis=-1)
            flow_laplacian_y = np.gradient(flow_gradient_y, grid_step, edge_order=2, axis=0)
            flow_laplacian_y_transverse_normed = np.abs(np.sum(np.stack([-flow_laplacian_y[:, :, 0], flow_laplacian_y[:, :, 1]], axis=-1) * flow_gradient_y, axis=-1)) / np.sum(np.power(flow_gradient_y, 2), axis=-1)
            flow_laplacian_score = cauchy(np.linalg.norm(np.stack([flow_laplacian_x_transverse_normed, flow_laplacian_y_transverse_normed]), axis=0), 0.1)

            if True:
                mapping_score = (inlier_scores
                                 * nan_gaussian_filter(flow_discrepancy_score)
                                 * nan_gaussian_filter(flow_grid_scaling_aspect_score)
                                 * nan_gaussian_filter(flow_grid_orthogonality_score)
                                 * nan_gaussian_filter(flow_laplacian_score)
                                 * cauchy(key_frame_motion_blurs[secondary_frame_idx], 8.0))
            else:
                mapping_score = np.power(6 / (1 / np.maximum(inlier_scores, 1e-8)
                                              + 1 / np.maximum(nan_gaussian_filter(flow_discrepancy_score), 1e-8)
                                              + 1 / np.maximum(nan_gaussian_filter(flow_grid_scaling_aspect_score), 1e-8)
                                              + 1 / np.maximum(nan_gaussian_filter(flow_grid_orthogonality_score), 1e-8)
                                              + 1 / np.maximum(nan_gaussian_filter(flow_laplacian_score), 1e-8)
                                              + 1 / np.maximum(cauchy(key_frame_motion_blurs[secondary_frame_idx], 8.0), 1e-8)), 2)

            mapping_score[~cross_flow_inlier] = np.nan

            mapping_scores[secondary_frame_idx, :, :] = mapping_score

            sec_img = frame_images[secondary_frame_idx]

            plt.figure('Synthetic frame scores', figsize=(24, 12))
            setup_new_fig_page()
            plt.suptitle(f'primary, secondary frame idxs: {primary_frame_idx}, {secondary_frame_idx}')
            plt.subplot(3, 3, 1)
            plt.imshow(img)
            plt.subplot(3, 3, 2)
            plt.imshow(sec_img)
            inlier_mask = ((primary_mesh_projected_points[0, :] > -0.5) & (primary_mesh_projected_points[0, :] < sec_img.shape[1] - 0.5)
                           & (primary_mesh_projected_points[1, :] > -0.5) & (primary_mesh_projected_points[1, :] < sec_img.shape[0] - 0.5))
            plt.scatter(*primary_mesh_projected_points[:, inlier_mask], s=1, c='b', alpha=0.9)
            ax = plt.subplot(3, 3, 3)
            plt.imshow(mapping_score, vmin=0, vmax=1, extent=xys_extent)
            plt.title('mapping_score')
            plt.subplot(3, 3, 4, sharex=ax, sharey=ax)
            plt.imshow(flow_discrepancy_score, vmin=0, vmax=1, extent=xys_extent)
            plt.title('flow_discrepancy_score')
            plt.subplot(3, 3, 5, sharex=ax, sharey=ax)
            plt.imshow(flow_grid_scaling_aspect_score, vmin=0, vmax=1, extent=xys_extent)
            plt.title('flow_grid_scaling_aspect_score')
            plt.subplot(3, 3, 6, sharex=ax, sharey=ax)
            plt.imshow(flow_grid_orthogonality_score, vmin=0, vmax=1, extent=xys_extent)
            plt.title('flow_grid_orthogonality_score')
            plt.subplot(3, 3, 7, sharex=ax, sharey=ax)
            plt.imshow(flow_laplacian_score, vmin=0, vmax=1, extent=xys_extent)
            plt.title('flow_laplacian_score')
            plt.tight_layout()
            stash_fig_page()


        # Generate the synthetic view using the frame mapping scores
        masked_mapping_scores = np.array(mapping_scores)
        masked_mapping_scores[primary_frame_idx, ~np.isfinite(masked_mapping_scores[primary_frame_idx, :, :])] = 0
        secondary_frame_idxs = np.nanargmax(masked_mapping_scores, axis=0)

        synthetic_frame_idxs = cv2.resize(secondary_frame_idxs, dsize=(wj, hj), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)

        xyzfg, _, xfg, yfg, _, _ = ray_cast_grid_points(primary_frame_idx, synthetic_camera_extrinsic, camera_intrinsic_synthetic, wj, hj)

        synthetic_frame_img = np.full((hj, wj, 3), fill_value=np.nan, dtype=np.float32)

        for secondary_frame_idx in np.unique(synthetic_frame_idxs):
            sec_img = frame_images[secondary_frame_idx]

            secondary_camera_extrinsic = camera_extrinsics[secondary_frame_idx]
            camera_transform = secondary_camera_extrinsic @ np.linalg.inv(synthetic_camera_extrinsic)

            secondary_projected_points = camera_intrinsic @ (camera_transform[:3, :3] @ xyzfg + camera_transform[:3, 3:])
            secondary_projected_points = secondary_projected_points[:2, :] / secondary_projected_points[2, :]

            cross_flow = np.full((hj, wj, 2), fill_value=np.nan, dtype=np.float32)
            cross_flow[yfg, xfg] = secondary_projected_points.T

            if False:
                sec_img_warp = cv2.remap(cv2.copyMakeBorder(sec_img.astype(np.float32), top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_REPLICATE),
                                         cross_flow + 1, None, cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(np.nan,)*sec_img.shape[-1])
            else:
                sec_img_warp = cv2.remap(np.array(sec_img, dtype=np.float32), cross_flow, None, cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(np.nan,)*sec_img.shape[-1])

            render_mask = synthetic_frame_idxs == secondary_frame_idx
            synthetic_frame_img[render_mask] = sec_img_warp[render_mask]

        max_mapping_scores = np.nanmax(masked_mapping_scores, axis=0)
        max_mapping_scores[np.all(~np.isfinite(mapping_scores), axis=0)] = np.nan
        up_max_mapping_scores = cv2.resize(max_mapping_scores, dsize=(wj, hj), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        assert np.nanmin(up_max_mapping_scores) >= 0 and np.nanmax(up_max_mapping_scores) <= 1
        filtered_up_max_mapping_scores = nan_gaussian_filter(up_max_mapping_scores, ksize=(15, 15), unfiltered_point_value=np.nan)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(5, 5))
        kernel = kernel | kernel.T
        pad_width = 8
        padded_patch_img = cv2.copyMakeBorder(~np.isfinite(up_max_mapping_scores).astype(np.uint8),
                                              top=pad_width, bottom=pad_width, left=pad_width, right=pad_width,
                                              borderType=cv2.BORDER_CONSTANT, value=0)
        patch_mask = cv2.morphologyEx(padded_patch_img, op=cv2.MORPH_TOPHAT, kernel=kernel,
                                      iterations=2, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        patch_mask = patch_mask[pad_width:-pad_width, pad_width:-pad_width].astype(bool)
        filter_mask = np.isfinite(up_max_mapping_scores) | patch_mask
        filtered_up_max_mapping_scores[~filter_mask] = np.nan

        #ksizes = 7 * np.power(3, np.arange(4))
        ksizes = np.power(2, np.arange(6) + 2) - 1
        filtered_synthetic_frame_imgs = [synthetic_frame_img] + [nan_gaussian_filter(synthetic_frame_img, ksize=(ksize, ksize), unfiltered_point_value=np.nan)
                                                                 for ksize in ksizes]
        score_thresholds = np.hstack([np.power(0.5, np.arange(len(filtered_synthetic_frame_imgs) - 1)), [0]])

        filtered_synthetic_frame_img = np.full((hj, wj, 3), fill_value=np.nan, dtype=np.float32)
        for thresh_low, thresh_high, img_low, img_high in zip(score_thresholds[1:], score_thresholds[:-1],
                                                              filtered_synthetic_frame_imgs[1:], filtered_synthetic_frame_imgs[:-1]):
            level_mask = (filtered_up_max_mapping_scores > thresh_low) & (filtered_up_max_mapping_scores <= thresh_high)
            interp_weight_high = (filtered_up_max_mapping_scores[:, :, None] - thresh_low) / (thresh_high - thresh_low)
            interp_weight_low = 1 - interp_weight_high
            filtered_synthetic_frame_img[level_mask, :] = (interp_weight_low * img_low + interp_weight_high * img_high)[level_mask, :]

        level_mask = filtered_up_max_mapping_scores <= score_thresholds[-1]
        filtered_synthetic_frame_img[level_mask, :] = filtered_synthetic_frame_imgs[-1][level_mask, :]

        primary_frame_synthesis_elements[primary_frame_idx] = (img, h, w, hj, wj, xys_extent,
                                                               synthetic_camera_extrinsic, camera_intrinsic_synthetic,
                                                               mapping_scores, xyzfg, xfg, yfg)

        plt.figure('Synthetic view', figsize=(24, 12))
        setup_new_fig_page()
        plt.suptitle(f'primary frame idx: {primary_frame_idx}')
        ax = plt.subplot(2, 2, 1)
        synthetic_frame_cmap = get_frame_idxs_cmap(synthetic_frame_idxs, N=mapping_scores.shape[0])
        plt.imshow(synthetic_frame_idxs, cmap=synthetic_frame_cmap, vmin=0, vmax=mapping_scores.shape[0], interpolation_stage='rgba')
        ax = plt.subplot(2, 2, 2, sharex=ax, sharey=ax)
        ax.set_facecolor('grey')
        plt.imshow(synthetic_frame_img / 255)
        plt.subplot(2, 2, 3, sharex=ax, sharey=ax)
        plt.imshow(up_max_mapping_scores, vmin=0, vmax=1)
        ax = plt.subplot(2, 2, 4, sharex=ax, sharey=ax)
        ax.set_facecolor('grey')
        plt.imshow(filtered_synthetic_frame_img / 255)
        plt.tight_layout()
        stash_fig_page()

    # %%

    # TODO: only map & optimise the set of coherent secondary frames

    plt.close('Loss components')
    plt.close('Grayscale intensity optimisation')
    plt.close('Optimised mapping')
    plt.close('Optimised synthetic view')

    for primary_frame_idx in primary_frame_synthesis_elements:

        print('optimising primary_frame_idx', primary_frame_idx)

        (img, h, w, hj, wj, xys_extent,
         synthetic_camera_extrinsic, camera_intrinsic_synthetic,
         mapping_scores, xyzfg, xfg, yfg) = primary_frame_synthesis_elements[primary_frame_idx]

        # Optimise the frame mapping with a regularisation loss function
        # mapping_score_mask_threshold is the value representing the order of magnitude of the mapping score
        # around which masking takes effect
        mapping_score_mask_threshold = 0.01
        log_clamp_min = 1e-4 * mapping_score_mask_threshold
        down_mapping_scores = []
        for mapping_score in mapping_scores:
            filtered_mapping_score = np.array(mapping_score)
            filtered_mapping_score[~np.isfinite(mapping_score)] = 0
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(5, 5))
            kernel = kernel | kernel.T
            filtered_mapping_score = cv2.dilate(filtered_mapping_score, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(9, 9))
            kernel = kernel | kernel.T
            filtered_mapping_score = cv2.erode(filtered_mapping_score, kernel, iterations=1, borderType=cv2.BORDER_REPLICATE)
            filtered_mapping_score[~np.isfinite(mapping_score)] = 0
            down_mapping_scores.append(cv2.resize(filtered_mapping_score, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR))
        down_mapping_scores = np.stack(down_mapping_scores)

        model_mapping_scores = torch.tensor(down_mapping_scores, dtype=torch.float32)
        model_max_mapping_scores = torch.max(model_mapping_scores, dim=0).values
        model_mapping_scores_mask = torch.clamp(model_max_mapping_scores / mapping_score_mask_threshold, min=log_clamp_min, max=1.0)
        model_frames = torch.tensor(np.identity(down_mapping_scores.shape[0])[primary_frame_idx, :][:, None, None] * np.ones(down_mapping_scores.shape[1:]),
                                    dtype=torch.float32, requires_grad=True)
        # TODO: only map & optimise the set of coherent secondary frames, which would make the application of np.nan_to_num() redundant
        model_interframe_costs = torch.tensor(np.nan_to_num(interframe_costs, nan=1.0), dtype=torch.float32)

        def loss_fn():
            prob_model_frames = torch.nn.functional.softmax(model_frames, dim=0)
            loss_components = {}

            #loss_components['mapping cost'] = 10 * (model_max_mapping_scores - torch.sum(prob_model_frames * model_mapping_scores, dim=0))
            #loss_components['mapping cost'] = 1 - torch.sum(prob_model_frames * model_mapping_scores, dim=0) / torch.clamp(model_max_mapping_scores, min=log_clamp_min)
            #loss_components['mapping cost'] = model_max_mapping_scores / torch.clamp(torch.sum(prob_model_frames * model_mapping_scores, dim=0), min=log_clamp_min)
            #loss_components['mapping cost'] = torch.pow(torch.log(torch.clamp(torch.sum(prob_model_frames * model_mapping_scores, dim=0), min=log_clamp_min)) - torch.log(torch.clamp(model_max_mapping_scores, min=log_clamp_min)), 2)
            loss_components['mapping cost'] = torch.log(torch.clamp(model_max_mapping_scores, min=log_clamp_min)) - torch.log(torch.clamp(torch.sum(prob_model_frames * model_mapping_scores, dim=0), min=log_clamp_min))

            prob_interframe_costs = torch.einsum('ij,jab->iab', model_interframe_costs, prob_model_frames)
            dist_ud = (torch.sum(prob_model_frames[:, :-1, :] * prob_interframe_costs[:, 1:, :], dim=0)
                       * (model_mapping_scores_mask[:-1, :] + model_mapping_scores_mask[1:, :]))
            dist_lr = (torch.sum(prob_model_frames[:, :, :-1] * prob_interframe_costs[:, :, 1:], dim=0)
                       * (model_mapping_scores_mask[:, :-1] + model_mapping_scores_mask[:, 1:]))
            dist_uldr = (torch.sum(prob_model_frames[:, :-1, :-1] * prob_interframe_costs[:, 1:, 1:], dim=0)
                         * (model_mapping_scores_mask[:-1, :-1] + model_mapping_scores_mask[1:, 1:]))
            dist_urdl = (torch.sum(prob_model_frames[:, 1:, :-1] * prob_interframe_costs[:, :-1, 1:], dim=0)
                         * (model_mapping_scores_mask[1:, :-1] + model_mapping_scores_mask[:-1, 1:]))
            # torch.nn.functional.pad() pad parameter is specified as left, right pairs in from the last dimension to the first dimension
            dist_ud = torch.nn.functional.pad(dist_ud, (0, 0, 0, 1)) + torch.nn.functional.pad(dist_ud, (0, 0, 1, 0))
            dist_lr = torch.nn.functional.pad(dist_lr, (0, 1, 0, 0)) + torch.nn.functional.pad(dist_lr, (1, 0, 0, 0))
            dist_uldr = torch.nn.functional.pad(dist_uldr, (0, 1, 0, 1)) + torch.nn.functional.pad(dist_uldr, (1, 0, 1, 0))
            dist_urdl = torch.nn.functional.pad(dist_urdl, (1, 0, 0, 1)) + torch.nn.functional.pad(dist_urdl, (0, 1, 1, 0))
            loss_components['interframe cost'] = 0.1 * (dist_ud + dist_lr + dist_uldr + dist_urdl)

            """
            # The diagonal terms help to balance the loss for all border directions between frame mapping transitions.
            # Without the diagonal terms, the loss for horizontal and vertical borders is less than the loss for diagonals,
            # so borders tend to align horizontally or vertically.
            if False:
                # The L2 norm loss may be too much at odds with transitioning quickly between one-hot frame map vectors.
                # It appears to result in thin strips stacked with the thinner dimension in the direction of camera motion
                # for some regions of the map, particularly around the centre.
                grad_ud = (torch.sum(torch.pow(prob_model_frames[:, :-1, :] - prob_model_frames[:, 1:, :], 2), dim=0)
                           * (model_mapping_scores_mask[:-1, :] + model_mapping_scores_mask[1:, :]))
                grad_lr = (torch.sum(torch.pow(prob_model_frames[:, :, :-1] - prob_model_frames[:, :, 1:], 2), dim=0)
                           * (model_mapping_scores_mask[:, :-1] + model_mapping_scores_mask[:, 1:]))
                grad_uldr = (torch.sum(torch.pow(prob_model_frames[:, :-1, :-1] - prob_model_frames[:, 1:, 1:], 2), dim=0)
                             * (model_mapping_scores_mask[:-1, :-1] + model_mapping_scores_mask[1:, 1:]))
                grad_urdl = (torch.sum(torch.pow(prob_model_frames[:, 1:, :-1] - prob_model_frames[:, :-1, 1:], 2), dim=0)
                             * (model_mapping_scores_mask[1:, :-1] + model_mapping_scores_mask[:-1, 1:]))
            else:
                grad_ud = (torch.sum(torch.abs(prob_model_frames[:, :-1, :] - prob_model_frames[:, 1:, :]), dim=0)
                           * (model_mapping_scores_mask[:-1, :] + model_mapping_scores_mask[1:, :]))
                grad_lr = (torch.sum(torch.abs(prob_model_frames[:, :, :-1] - prob_model_frames[:, :, 1:]), dim=0)
                           * (model_mapping_scores_mask[:, :-1] + model_mapping_scores_mask[:, 1:]))
                grad_uldr = (torch.sum(torch.abs(prob_model_frames[:, :-1, :-1] - prob_model_frames[:, 1:, 1:]), dim=0)
                             * (model_mapping_scores_mask[:-1, :-1] + model_mapping_scores_mask[1:, 1:]))
                grad_urdl = (torch.sum(torch.abs(prob_model_frames[:, 1:, :-1] - prob_model_frames[:, :-1, 1:]), dim=0)
                             * (model_mapping_scores_mask[1:, :-1] + model_mapping_scores_mask[:-1, 1:]))
            # torch.nn.functional.pad() pad parameter is specified as left, right pairs in from the last dimension to the first dimension
            grad_ud = torch.nn.functional.pad(grad_ud, (0, 0, 0, 1)) + torch.nn.functional.pad(grad_ud, (0, 0, 1, 0))
            grad_lr = torch.nn.functional.pad(grad_lr, (0, 1, 0, 0)) + torch.nn.functional.pad(grad_lr, (1, 0, 0, 0))
            grad_uldr = torch.nn.functional.pad(grad_uldr, (0, 1, 0, 1)) + torch.nn.functional.pad(grad_uldr, (1, 0, 1, 0))
            grad_urdl = torch.nn.functional.pad(grad_urdl, (1, 0, 0, 1)) + torch.nn.functional.pad(grad_urdl, (0, 1, 1, 0))
            loss_components['gradient'] = 0.05 * (grad_ud + grad_lr + grad_uldr + grad_urdl)
            """

            #prob_model_frames_neighbourhood_mean = torch.nn.functional.conv2d(prob_model_frames[:, None, :, :],
            #                                                                  torch.ones((1, 1, 2, 2), dtype=torch.float32),
            #                                                                  padding='valid')[:, 0, :, :] / 4
            prob_model_frames_neighbourhood_mean = (prob_model_frames[:, :-1, :-1] + prob_model_frames[:, :-1, 1:]
                                                    + prob_model_frames[:, 1:, :-1] + prob_model_frames[:, 1:, 1:]) / 4
            prob_model_frames_neighbourhood_norm_var = (torch.sum(torch.pow(prob_model_frames[:, :-1, :-1] - prob_model_frames_neighbourhood_mean, 2), dim=0) * model_mapping_scores_mask[:-1, :-1]
                                                        + torch.sum(torch.pow(prob_model_frames[:, :-1, 1:] - prob_model_frames_neighbourhood_mean, 2), dim=0) * model_mapping_scores_mask[:-1, 1:]
                                                        + torch.sum(torch.pow(prob_model_frames[:, 1:, :-1] - prob_model_frames_neighbourhood_mean, 2), dim=0) * model_mapping_scores_mask[1:, :-1]
                                                        + torch.sum(torch.pow(prob_model_frames[:, 1:, 1:] - prob_model_frames_neighbourhood_mean, 2), dim=0) * model_mapping_scores_mask[1:, 1:]) / 4
            loss_components['variance'] = 0.2 * prob_model_frames_neighbourhood_norm_var

            #loss_components['entropy'] = 0.1 * -torch.log(torch.clamp(torch.var(prob_model_frames, dim=0) * prob_model_frames.shape[0] * model_mapping_scores_mask, min=log_clamp_min))
            # Divide by np.log(prob_model_frames.shape[0]) to calculate the entropy relative to that of a uniform distribution
            #loss_components['entropy'] = 1.0 * -torch.sum(torch.log(torch.clamp(prob_model_frames, min=log_clamp_min)) * prob_model_frames, dim=0) / np.log(prob_model_frames.shape[0]) * model_mapping_scores_mask
            loss_components['entropy'] = 1.0 * torch.pow(torch.sum(torch.log(torch.clamp(prob_model_frames, min=log_clamp_min)) * prob_model_frames, dim=0) / np.log(prob_model_frames.shape[0]), 2) * model_mapping_scores_mask

            #loss_components['information'] = 1e-3 * torch.log(1 + 10 * torch.sum(prob_model_frames, dim=(1, 2)))

            return sum(torch.mean(loss_component) for loss_component in loss_components.values()), loss_components

        lr = 0.005
        num_steps = 20000
        convergence_criterion = {'atol': 1e-5, 'window_size': 200, 'min_num_steps': 2000}
        optimiser = torch.optim.Adam([model_frames], lr=lr)
        lr_lambda = lambda epoch: (np.sin(min((epoch + 1) / convergence_criterion['min_num_steps'], 0.5) * np.pi)
                                   * np.power(0.1, max(epoch - 0.5 * convergence_criterion['min_num_steps'], 0) / num_steps))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lr_lambda)
        losses = []
        loss_components = {}
        for optim_step in range(num_steps):
            optimiser.zero_grad()
            loss, loss_components = loss_fn()
            loss.backward()
            optimiser.step()
            scheduler.step()
            losses.append(loss.numpy(force=True))
            loss_window_std = np.std(losses[-convergence_criterion['window_size']:])
            if optim_step % 100 == 0:
                print('optim_step', optim_step, 'loss', losses[-1], 'loss_window_std', loss_window_std, 'learning rate', scheduler.get_last_lr()[0])
            if optim_step >= convergence_criterion['min_num_steps'] and loss_window_std < convergence_criterion['atol']:
                break

        print('len(losses)', len(losses))
        print('initial, first, final loss', losses[0], losses[convergence_criterion['min_num_steps']], losses[-1])
        print('highest, lowest loss', np.max(losses[convergence_criterion['min_num_steps']:]), np.min(losses[convergence_criterion['min_num_steps']:]))

        prob_model_frames = torch.nn.functional.softmax(model_frames, dim=0).numpy(force=True)
        model_frames_idxs = np.argmax(prob_model_frames, axis=0)
        model_core_frame_idxs = np.sort(np.unique(model_frames_idxs))


        # Optimise the grayscale intensity transforms for each of the core frames
        gray_step = 2
        assert wj % gray_step == 0 and hj % gray_step == 0
        wg, hg = wj // gray_step, hj // gray_step
        camera_intrinsic_gray = np.block([[camera_intrinsic_synthetic[:2, :2] / gray_step, (camera_intrinsic_synthetic[:2, 2:] + 0.5) / gray_step - 0.5], [0, 0, 1]])

        xyzsg, _, xsg, ysg, _, _ = ray_cast_grid_points(primary_frame_idx, synthetic_camera_extrinsic, camera_intrinsic_gray, wg, hg)

        gray_prob_model_frames = np.stack([cv2.resize(prob_model_frames[idx, :, :], dsize=(wg, hg), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
                                           for idx in model_core_frame_idxs])
        gray_model_frames_idxs = model_core_frame_idxs[np.argmax(gray_prob_model_frames, axis=0)]

        core_frame_imgs = []
        core_frame_pixels_and_weights = []
        pixel_frame_bins = collections.defaultdict(dict)
        for core_idx, secondary_frame_idx in enumerate(model_core_frame_idxs):
            sec_gray = rgb_to_gray(filtered_frame_images[secondary_frame_idx], dtype=np.float32)
            filtered_sec_gray = nan_gaussian_filter(sec_gray, ksize=(31, 31))
            filtered_sec_gray[frame_img_masks[secondary_frame_idx]] = np.nan

            _, _, _, _, depth_img, normal_coincidence_img = ray_cast_grid_points(secondary_frame_idx, camera_extrinsics[secondary_frame_idx], camera_intrinsic, w, h)
            filtered_depth_img = nan_gaussian_filter(depth_img, ksize=(101, 101), unfiltered_point_value=np.nan)
            filtered_normal_coincidence_img = nan_gaussian_filter(normal_coincidence_img, ksize=(101, 101), unfiltered_point_value=np.nan)
            core_frame_imgs.append((filtered_depth_img, filtered_normal_coincidence_img))

            sec_frame_img = np.stack([filtered_sec_gray, filtered_depth_img, filtered_normal_coincidence_img], axis=-1, dtype=np.float32)

            secondary_camera_extrinsic = camera_extrinsics[secondary_frame_idx]
            camera_transform = secondary_camera_extrinsic @ np.linalg.inv(synthetic_camera_extrinsic)

            secondary_object_points = camera_transform[:3, :3] @ xyzsg + camera_transform[:3, 3:]
            secondary_projected_points = camera_intrinsic @ secondary_object_points
            secondary_projected_points = secondary_projected_points[:2, :] / secondary_projected_points[2, :]

            cross_flow = np.full((hg, wg, 2), fill_value=np.nan, dtype=np.float32)
            cross_flow[ysg, xsg] = secondary_projected_points.T

            if False:
                sec_frame_img_warp = cv2.remap(cv2.copyMakeBorder(sec_frame_img, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_REPLICATE),
                                               cross_flow + 1, None, cv2.INTER_LINEAR,
                                               borderMode=cv2.BORDER_CONSTANT, borderValue=(np.nan,)*sec_frame_img.shape[-1])
            else:
                sec_frame_img_warp = cv2.remap(sec_frame_img, cross_flow, None, cv2.INTER_LINEAR,
                                               borderMode=cv2.BORDER_CONSTANT, borderValue=(np.nan,)*sec_frame_img.shape[-1])

            xyr_img = np.full((hg, wg, 3), fill_value=np.nan, dtype=np.float32)
            xy = secondary_object_points[:2, :] / secondary_object_points[2, :]
            xyr_img[ysg, xsg] = np.vstack([xy, np.linalg.norm(xy, axis=0)]).T

            render_mask = gray_model_frames_idxs == secondary_frame_idx
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(25, 25))
            kernel = kernel | kernel.T
            render_mask = cv2.dilate(render_mask.astype(np.uint8), kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)

            mapping_score = mapping_scores[secondary_frame_idx, :, :]
            gray_mapping_score = cv2.resize(mapping_score, dsize=(wg, hg), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)

            pixel_frame_bin_mask = (render_mask
                                    & np.all(np.isfinite(sec_frame_img_warp), axis=-1)
                                    & (gray_mapping_score >= mapping_score_mask_threshold))

            core_frame_pixels_and_weights.append(np.hstack([sec_frame_img_warp[pixel_frame_bin_mask, :],
                                                            xyr_img[pixel_frame_bin_mask, :],
                                                            gray_mapping_score[pixel_frame_bin_mask][:, None]]))

            for pixel_idx, (y, x) in enumerate(np.stack(np.where(pixel_frame_bin_mask)).T):
                pixel_frame_bins[(x, y)][core_idx] = pixel_idx

        pixel_frame_bins = {xy: pixel_frame_bin for xy, pixel_frame_bin in pixel_frame_bins.items() if len(pixel_frame_bin) >= 2}

        intersecting_frame_pixels = []
        intersecting_frame_pixel_weights = []
        for xy, pixel_frame_bin in pixel_frame_bins.items():
            frame_pixels = []
            frame_pixel_weights = []
            for core_idx in range(len(model_core_frame_idxs)):
                if core_idx in pixel_frame_bin:
                    pixel_idx = pixel_frame_bin[core_idx]
                    gray, depth, normal_coincidence, x, y, r, weight = core_frame_pixels_and_weights[core_idx][pixel_idx, :]
                    frame_pixels.append((16,
                                         gray,
                                         gray * depth / 10,
                                         gray * np.power(depth / 10, 2),
                                         gray * (1 - np.clip(normal_coincidence, 0, np.inf)),
                                         gray * np.power(1 - np.clip(normal_coincidence, 0, np.inf), 2),
                                         gray * x,
                                         gray * np.power(x, 2),
                                         gray * y,
                                         gray * np.power(y, 2),
                                         gray * r,
                                         gray * np.power(r, 2)))
                    frame_pixel_weights.append(np.clip(weight, 1 / len(pixel_frame_bin), np.inf))
                else:
                    frame_pixels.append((0,) * 12)
                    frame_pixel_weights.append(0)
            intersecting_frame_pixels.append(frame_pixels)
            intersecting_frame_pixel_weights.append(frame_pixel_weights)

        model_frame_intensity_transform_coeffs = torch.tensor(np.zeros((len(model_core_frame_idxs), 12)), dtype=torch.float32, requires_grad=True)
        model_intersecting_frame_pixels = torch.tensor(np.array(intersecting_frame_pixels), dtype=torch.float32)
        model_intersecting_frame_pixel_weights = np.array(intersecting_frame_pixel_weights)
        model_intersecting_frame_pixel_counts = torch.tensor(np.sum(model_intersecting_frame_pixel_weights > 0, axis=-1), dtype=torch.float32)
        model_intersecting_frame_pixel_normed_weights = torch.tensor(model_intersecting_frame_pixel_weights / np.sum(model_intersecting_frame_pixel_weights, axis=-1, keepdims=True), dtype=torch.float32)

        untransformed_pixels = model_intersecting_frame_pixels[:, :, 1] / 255
        weighted_untransformed_pixels_mean_std = torch.std(torch.sum(model_intersecting_frame_pixel_normed_weights * untransformed_pixels, axis=-1))

        def loss_fn():
            transformed_pixel_deltas = torch.sum(model_intersecting_frame_pixels * model_frame_intensity_transform_coeffs, axis=-1) / 255
            transformed_pixels = untransformed_pixels + transformed_pixel_deltas

            weighted_pixel_deltas = torch.sum(model_intersecting_frame_pixel_normed_weights * transformed_pixel_deltas, axis=-1)
            weighted_pixels_mean = torch.sum(model_intersecting_frame_pixel_normed_weights * transformed_pixels, axis=-1)
            weighted_pixels_var = torch.sum(model_intersecting_frame_pixel_normed_weights * torch.pow(transformed_pixels, 2), axis=-1) - torch.pow(weighted_pixels_mean, 2)

            # Heavy tailed bell shaped function based on Student's t-distribution with v = 1 (i.e. Cauchy distribution)
            # y=1/(1+x**2) has a knee point at x ~ 3.0 beyond which it slowly converges to 0, e.g. [1.0, 0.5], [3.0, 0.1], [9.9, 0.01]
            threshold = 64.0 / 255
            pixel_var_losses = 1 - 1 / (1 + weighted_pixels_var / np.power(threshold / 3.0, 2))
            pixel_var_losses *= model_intersecting_frame_pixel_counts

            pixel_loss_components = {}
            pixel_loss_components['pixel var'] = torch.mean(pixel_var_losses)
            pixel_loss_components['pixel mean std delta'] = 1e5 * torch.pow(torch.std(weighted_pixels_mean) - weighted_untransformed_pixels_mean_std, 2)
            pixel_loss_components['pixel mean saturation'] = 10.0 * (torch.mean(torch.pow(torch.clamp(weighted_pixels_mean - 1, min=0), 2))
                                                                     + torch.mean(torch.pow(torch.clamp(weighted_pixels_mean, max=0), 2)))
            # TODO: if the original source images are not well balanced in the first place,
            #       then these distribution regularisation constraints on weighted_pixel_deltas may not be valid
            pixel_loss_components['weighted_pixel_deltas mean'] = 100.0 * torch.pow(torch.mean(weighted_pixel_deltas), 2)
            #pixel_loss_components['weighted_pixel_deltas mean'] = 1.0 * torch.abs(torch.mean(weighted_pixel_deltas))
            pixel_loss_components['weighted_pixel_deltas var'] = 10.0 * torch.var(weighted_pixel_deltas)
            # Note that dividing the 3rd moment by the standard deviation to calculate the normalised / standardised skewness
            # results in poorer convergence (i.e. oscillations)
            pixel_loss_components['weighted_pixel_deltas skew'] = 50.0 * torch.abs(torch.mean(torch.pow(weighted_pixel_deltas - torch.mean(weighted_pixel_deltas), 3)))

            return sum(pixel_loss_components.values()), pixel_loss_components, weighted_pixel_deltas, weighted_pixels_mean, weighted_pixels_var

        lr = 0.005
        num_steps = 10000
        convergence_criterion = {'atol': 1e-4, 'window_size': 100, 'min_num_steps': 1000}
        optimiser = torch.optim.Adam([model_frame_intensity_transform_coeffs], lr=lr)
        lr_lambda = lambda epoch: (np.sin(min((epoch + 1) / convergence_criterion['min_num_steps'], 0.5) * np.pi)
                                   * np.power(0.1, max(epoch - 0.5 * convergence_criterion['min_num_steps'], 0) / num_steps))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lr_lambda)
        losses = []
        pixel_loss_components = {}
        weighted_pixel_deltas, weighted_pixels_mean, weighted_pixels_var = None, None, None
        for optim_step in range(num_steps):
            optimiser.zero_grad()
            loss, pixel_loss_components, weighted_pixel_deltas, weighted_pixels_mean, weighted_pixels_var = loss_fn()
            loss.backward()
            optimiser.step()
            scheduler.step()
            losses.append(loss.numpy(force=True))
            loss_window_std = np.std(losses[-convergence_criterion['window_size']:])
            if optim_step % 100 == 0:
                print('optim_step', optim_step, 'loss', losses[-1], 'loss_window_std', loss_window_std, 'learning rate', scheduler.get_last_lr()[0])
            if optim_step >= convergence_criterion['min_num_steps'] and loss_window_std < convergence_criterion['atol']:
                break

        print('len(losses)', len(losses))
        print('initial, first, final loss', losses[0], losses[convergence_criterion['min_num_steps']], losses[-1])
        print('highest, lowest loss', np.max(losses[convergence_criterion['min_num_steps']:]), np.min(losses[convergence_criterion['min_num_steps']:]))

        for title, loss_component in pixel_loss_components.items():
            print(title, loss_component.numpy(force=True))

        frame_intensity_transform_coeffs = model_frame_intensity_transform_coeffs.numpy(force=True)
        #print(frame_intensity_transform_coeffs)


        # Generate the synthetic view using the optimised frame and intensity transforms
        up_prob_model_frames = np.stack([cv2.resize(prob_model_frames[idx, :, :], dsize=(wj, hj), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
                                         for idx in model_core_frame_idxs])
        up_model_frames_idxs = model_core_frame_idxs[np.argmax(up_prob_model_frames, axis=0)]

        up_model_synthetic_frame_img = np.full((hj, wj, 3), fill_value=np.nan, dtype=np.float32)
        up_max_model_mapping_scores = np.full((hj, wj), fill_value=np.nan, dtype=np.float32)

        h, w = img.shape[:2]
        yg, xg = np.mgrid[0:h, 0:w].reshape((2, -1)).astype(int)
        uvg = np.vstack([xg, yg]) - camera_intrinsic[:2, 2:]
        xyg = np.linalg.inv(camera_intrinsic[:2, :2]) @ uvg

        xyr_img = np.full((h, w, 3), fill_value=np.nan, dtype=np.float32)
        xyr_img[yg, xg] = np.vstack([xyg, np.linalg.norm(xyg, axis=0)]).T

        for core_idx, secondary_frame_idx in enumerate(model_core_frame_idxs):
            sec_img = frame_images[secondary_frame_idx]
            filtered_depth_img, filtered_normal_coincidence_img = core_frame_imgs[core_idx]
            transformed_sec_img = (16 * frame_intensity_transform_coeffs[core_idx, 0]
                                   + sec_img.astype(np.float32) * (1 + frame_intensity_transform_coeffs[core_idx, 1]
                                                                   + filtered_depth_img / 10 * frame_intensity_transform_coeffs[core_idx, 2]
                                                                   + np.power(filtered_depth_img / 10, 2) * frame_intensity_transform_coeffs[core_idx, 3]
                                                                   + (1 - np.clip(filtered_normal_coincidence_img, 0, np.inf)) * frame_intensity_transform_coeffs[core_idx, 4]
                                                                   + np.power((1 - np.clip(filtered_normal_coincidence_img, 0, np.inf)), 2) * frame_intensity_transform_coeffs[core_idx, 5]
                                                                   + np.sum(xyr_img * frame_intensity_transform_coeffs[core_idx, [6, 8, 10]], axis=-1)
                                                                   + np.sum(np.power(xyr_img, 2) * frame_intensity_transform_coeffs[core_idx, [7, 9, 11]], axis=-1)
                                                                   )[:, :, None])

            secondary_camera_extrinsic = camera_extrinsics[secondary_frame_idx]
            camera_transform = secondary_camera_extrinsic @ np.linalg.inv(synthetic_camera_extrinsic)

            secondary_projected_points = camera_intrinsic @ (camera_transform[:3, :3] @ xyzfg + camera_transform[:3, 3:])
            secondary_projected_points = secondary_projected_points[:2, :] / secondary_projected_points[2, :]

            cross_flow = np.full((hj, wj, 2), fill_value=np.nan, dtype=np.float32)
            cross_flow[yfg, xfg] = secondary_projected_points.T

            if False:
                sec_img_warp = cv2.remap(cv2.copyMakeBorder(transformed_sec_img, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_REPLICATE),
                                         cross_flow + 1, None, cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(np.nan,)*transformed_sec_img.shape[-1])
            else:
                sec_img_warp = cv2.remap(transformed_sec_img, cross_flow, None, cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(np.nan,)*transformed_sec_img.shape[-1])

            render_mask = up_model_frames_idxs == secondary_frame_idx
            up_model_synthetic_frame_img[render_mask] = sec_img_warp[render_mask]
            up_max_model_mapping_scores[render_mask] = cv2.resize(mapping_scores[secondary_frame_idx, :, :], dsize=(wj, hj), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)[render_mask]

        assert np.nanmin(up_max_model_mapping_scores) >= 0 and np.nanmax(up_max_model_mapping_scores) <= 1
        filtered_up_max_model_mapping_scores = nan_gaussian_filter(up_max_model_mapping_scores, ksize=(15, 15), unfiltered_point_value=np.nan)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(5, 5))
        kernel = kernel | kernel.T
        pad_width = 8
        padded_patch_img = cv2.copyMakeBorder(~np.isfinite(up_max_model_mapping_scores).astype(np.uint8),
                                              top=pad_width, bottom=pad_width, left=pad_width, right=pad_width,
                                              borderType=cv2.BORDER_CONSTANT, value=0)
        patch_mask = cv2.morphologyEx(padded_patch_img, op=cv2.MORPH_TOPHAT, kernel=kernel,
                                      iterations=2, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        patch_mask = patch_mask[pad_width:-pad_width, pad_width:-pad_width].astype(bool)
        filter_mask = np.isfinite(up_max_model_mapping_scores) | patch_mask
        filtered_up_max_model_mapping_scores[~filter_mask] = np.nan

        #ksizes = 7 * np.power(3, np.arange(4))
        ksizes = np.power(2, np.arange(6) + 2) - 1
        filtered_up_model_synthetic_frame_imgs = [up_model_synthetic_frame_img] + [nan_gaussian_filter(up_model_synthetic_frame_img, ksize=(ksize, ksize), unfiltered_point_value=np.nan)
                                                                                   for ksize in ksizes]
        score_thresholds = np.hstack([np.power(0.5, np.arange(len(filtered_up_model_synthetic_frame_imgs) - 1)), [0]])

        filtered_up_model_synthetic_frame_img = np.full((hj, wj, 3), fill_value=np.nan, dtype=np.float32)
        for thresh_low, thresh_high, img_low, img_high in zip(score_thresholds[1:], score_thresholds[:-1],
                                                              filtered_up_model_synthetic_frame_imgs[1:], filtered_up_model_synthetic_frame_imgs[:-1]):
            level_mask = (filtered_up_max_model_mapping_scores > thresh_low) & (filtered_up_max_model_mapping_scores <= thresh_high)
            interp_weight_high = (filtered_up_max_model_mapping_scores[:, :, None] - thresh_low) / (thresh_high - thresh_low)
            interp_weight_low = 1 - interp_weight_high
            filtered_up_model_synthetic_frame_img[level_mask, :] = (interp_weight_low * img_low + interp_weight_high * img_high)[level_mask, :]

        level_mask = filtered_up_max_model_mapping_scores <= score_thresholds[-1]
        filtered_up_model_synthetic_frame_img[level_mask, :] = filtered_up_model_synthetic_frame_imgs[-1][level_mask, :]

        filename_stem = f'{primary_frame_idx}.{datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")}'

        output_img = np.array(filtered_up_model_synthetic_frame_img)
        output_img[~np.isfinite(output_img)] = 127
        output_img = np.clip(output_img, 0, 255).astype(np.uint8)
        output_path = output_dirpath / (filename_stem + '.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR), params=[cv2.IMWRITE_PNG_COMPRESSION, 1])

        canvas_mesh = integrated_weighted_canvas_meshes[primary_frame_idx]
        output_path = output_dirpath / (filename_stem + '.pickle')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as pickle_file:
            pickle.dump({'synthetic_camera_extrinsic': synthetic_camera_extrinsic,
                         'camera_intrinsic_synthetic': camera_intrinsic_synthetic,
                         'filtered_up_model_synthetic_frame_img': filtered_up_model_synthetic_frame_img,
                         'vertices': np.array(canvas_mesh.vertices, dtype=np.float32),
                         'triangles': np.array(canvas_mesh.triangles, dtype=np.int32),
                         'vertex_colors': np.array(canvas_mesh.vertex_colors, dtype=np.float32)},
                        pickle_file)

        model_mapping_scores_frame_idxs = np.argmax(model_mapping_scores.numpy(force=True), axis=0)
        model_mapping_scores_cmap = get_frame_idxs_cmap(model_mapping_scores_frame_idxs, N=model_mapping_scores.shape[0])
        up_model_cmap = get_frame_idxs_cmap(up_model_frames_idxs, N=model_mapping_scores.shape[0])


        plt.figure('Loss components', figsize=(24, 12))
        setup_new_fig_page()
        loss = sum(np.mean(loss_component.numpy(force=True)) for loss_component in loss_components.values())
        plt.suptitle(f'primary frame idx: {primary_frame_idx} {loss:.5f}')
        ncols = 3
        nrows = int(np.ceil((2 + len(loss_components)) / ncols))
        ax = plt.subplot(nrows, ncols, 1)
        plt.imshow(model_frames_idxs, cmap=up_model_cmap, vmin=0, vmax=model_mapping_scores.shape[0],
                   extent=xys_extent, interpolation_stage='rgba')
        ax = plt.subplot(nrows, ncols, 2, sharex=ax, sharey=ax)
        ax.set_facecolor('grey')
        plt.imshow(np.clip(filtered_up_model_synthetic_frame_img / 255, 0, 1))
        for idx, (title, loss_component) in enumerate(loss_components.items()):
            loss_component = loss_component.numpy(force=True)
            if len(loss_component.shape) < 2:
                plt.subplot(nrows, ncols, 3 + idx)
                plt.plot(loss_component)
            else:
                plt.subplot(nrows, ncols, 3 + idx, sharex=ax, sharey=ax)
                wp = (xys_extent[1] - xys_extent[0]) / model_frames_idxs.shape[1]
                hp = (xys_extent[2] - xys_extent[3]) / model_frames_idxs.shape[0]
                shape_diff = np.array(model_frames_idxs.shape[:2]) - np.array(loss_component.shape[:2])
                sub_xys_extent = (xys_extent[0] + wp * shape_diff[1] / 2, xys_extent[1] - wp * shape_diff[1] / 2,
                                  xys_extent[2] - hp * shape_diff[0] / 2, xys_extent[3] + hp * shape_diff[0] / 2)
                plt.imshow(loss_component, extent=sub_xys_extent)
            plt.title(f'{title} {np.mean(loss_component):.5f}')
        plt.tight_layout()
        stash_fig_page()

        plt.figure('Grayscale intensity optimisation', figsize=(24, 12))
        setup_new_fig_page()
        plt.suptitle(f'primary frame idx: {primary_frame_idx}')
        ax = plt.subplot(3, 3, 1)
        plt.imshow(up_model_frames_idxs, cmap=up_model_cmap, vmin=0, vmax=model_mapping_scores.shape[0], interpolation_stage='rgba')
        plt.title('up_model_frames_idxs')
        ax = plt.subplot(3, 3, 2, sharex=ax, sharey=ax)
        ax.set_facecolor('grey')
        plt.imshow(np.clip(up_model_synthetic_frame_img / 255, 0, 1))
        plt.title('up_model_synthetic_frame_img')
        plt.subplot(3, 3, 3, sharex=ax, sharey=ax)
        plt.imshow(up_max_model_mapping_scores, vmin=0, vmax=1)
        plt.title('up_max_model_mapping_scores')
        ax = plt.subplot(3, 3, 4, sharex=ax, sharey=ax)
        ax.set_facecolor('grey')
        pixel_frame_bin_counts = np.full((hg, wg), fill_value=np.nan, dtype=np.float32)
        for xy, pixel_frame_bin in pixel_frame_bins.items():
            pixel_frame_bin_counts[xy[1], xy[0]] = len(pixel_frame_bin)
        plt.imshow(pixel_frame_bin_counts, extent=xys_extent)
        plt.title('pixel_frame_bin_counts')
        if weighted_pixel_deltas is not None:
            ax = plt.subplot(3, 3, 5, sharex=ax, sharey=ax)
            ax.set_facecolor('grey')
            weighted_pixel_delta_img = np.full((hg, wg), fill_value=np.nan, dtype=np.float32)
            for xy, value in zip(pixel_frame_bins, weighted_pixel_deltas.numpy(force=True)):
                weighted_pixel_delta_img[xy[1], xy[0]] = value
            absvmax = np.nanmax(np.abs(weighted_pixel_delta_img))
            plt.imshow(weighted_pixel_delta_img, cmap='seismic', vmin=-absvmax, vmax=absvmax, extent=xys_extent)
            plt.title('weighted_pixel_delta_img')
            plt.subplot(3, 3, 6)
            plt.hist(weighted_pixel_deltas.numpy(force=True), bins=100)
            plt.title('weighted_pixel_deltas')
        if weighted_pixels_mean is not None:
            ax = plt.subplot(3, 3, 7, sharex=ax, sharey=ax)
            ax.set_facecolor('darkslateblue')
            weighted_pixels_mean_img = np.full((hg, wg), fill_value=np.nan, dtype=np.float32)
            for xy, value in zip(pixel_frame_bins, weighted_pixels_mean.numpy(force=True)):
                weighted_pixels_mean_img[xy[1], xy[0]] = value
            plt.imshow(weighted_pixels_mean_img, cmap='gray', vmin=0, vmax=1, extent=xys_extent)
            plt.title('weighted_pixels_mean_img')
        if weighted_pixels_var is not None:
            ax = plt.subplot(3, 3, 8, sharex=ax, sharey=ax)
            ax.set_facecolor('grey')
            weighted_pixels_var_img = np.full((hg, wg), fill_value=np.nan, dtype=np.float32)
            for xy, value in zip(pixel_frame_bins, weighted_pixels_var.numpy(force=True)):
                weighted_pixels_var_img[xy[1], xy[0]] = value
            plt.imshow(weighted_pixels_var_img, extent=xys_extent)
            plt.title('weighted_pixels_var_img')
            plt.subplot(3, 3, 9)
            plt.hist(weighted_pixels_var.numpy(force=True), bins=100, log=True)
            plt.title('weighted_pixels_var')
        plt.tight_layout()
        stash_fig_page()

        plt.figure('Optimised mapping', figsize=(24, 12))
        setup_new_fig_page()
        plt.suptitle(f'primary frame idx: {primary_frame_idx}')
        ax = plt.subplot(2, 3, 1)
        plt.imshow(model_mapping_scores_frame_idxs, cmap=model_mapping_scores_cmap, vmin=0, vmax=model_mapping_scores.shape[0],
                   extent=xys_extent, interpolation_stage='rgba')
        plt.subplot(2, 3, 2, sharex=ax, sharey=ax)
        plt.imshow(model_frames_idxs, cmap=up_model_cmap, vmin=0, vmax=model_mapping_scores.shape[0],
                   extent=xys_extent, interpolation_stage='rgba')
        plt.subplot(2, 3, 3, sharex=ax, sharey=ax)
        plt.imshow(up_model_frames_idxs, cmap=up_model_cmap, vmin=0, vmax=model_mapping_scores.shape[0], interpolation_stage='rgba')
        ax = plt.subplot(2, 3, 4, sharex=ax, sharey=ax)
        ax.set_facecolor('grey')
        plt.imshow(np.clip(up_model_synthetic_frame_img / 255, 0, 1))
        plt.subplot(2, 3, 5, sharex=ax, sharey=ax)
        plt.imshow(up_max_model_mapping_scores, vmin=0, vmax=1)
        ax = plt.subplot(2, 3, 6, sharex=ax, sharey=ax)
        ax.set_facecolor('grey')
        plt.imshow(np.clip(filtered_up_model_synthetic_frame_img / 255, 0, 1))
        plt.tight_layout()
        stash_fig_page()

        plt.figure('Optimised synthetic view', figsize=(24, 12))
        setup_new_fig_page()
        plt.suptitle(f'primary frame idx: {primary_frame_idx}')
        ax = plt.subplot(1, 1, 1)
        ax.set_facecolor('grey')
        plt.imshow(np.clip(filtered_up_model_synthetic_frame_img / 255, 0, 1))
        plt.tight_layout()
        stash_fig_page()

    # %%

    post_to_pipeline_server((f'{working_subdir} / {output_dirpath.name}', None))
