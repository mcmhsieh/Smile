"""
Compute stereo disparity between key and auxiliary frames to generate depth images.

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
import cv2
import open3d as o3d
import shapely
import trimesh

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

from image_filtering import normalise_img_intensities
from fig_paging import setup_new_fig_page, stash_fig_page

from pipeline_server import start_pipeline_server, post_to_pipeline_server, get_queue_from_pipeline_server


if __name__ == '__main__':

    working_subdir_config_path = pathlib.Path(r'../pipeline-workspace/working_subdir.txt')
    with open(working_subdir_config_path, 'r') as config_file:
        working_subdir = config_file.read().rstrip('\n')

    workspace_dirpath = pathlib.Path(r'../pipeline-workspace') / working_subdir
    image_source_dirpath = workspace_dirpath / 'calc_sequential_flow_and_blur'
    input_source_dirpath = workspace_dirpath / 'select_and_position_inter_key_aux_frames'
    key_frames_dirpath = workspace_dirpath / 'select_key_frames'
    output_dirpath = workspace_dirpath / 'compute_depth_images'

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

    input_path = input_source_dirpath / 'positioned_inter_key_aux_frames.pickle'
    with open(input_path, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
        inter_key_aux_frames = data['inter_key_aux_frames']
        aux_frame_indices = data['aux_frame_indices']
        aux_frame_image_triangulated_point_idxs = data['aux_frame_image_triangulated_point_idxs']
        aux_frame_camera_extrinsics = data['aux_frame_camera_extrinsics']

    # %%

    aux_frames_filepaths = sorted(key_frames_dirpath.glob('*.flow.aux.pickle'))

    key_frame_aux_map = collections.defaultdict(list)
    for input_filepath in sorted(aux_frames_filepaths):
        frame_time, frame_index, _, frame_type, filename_ext = input_filepath.name.split('.')
        frame_time = datetime.datetime.strptime(frame_time, '%Y%m%d-%H%M%S%f')
        frame_index = int(frame_index)

        with open(input_filepath, 'rb') as pickle_file:
            frame_data = pickle.load(pickle_file)

        ref_img_indices, img_indices, camera_matrix, flow_displacement, ref_motion_blur, motion_blur, (xfp, yfp, xfn, yfn) = [
            frame_data[name]
            for name in ['ref_img_indices', 'img_indices', 'camera_matrix', 'flow_displacement', 'ref_motion_blur', 'motion_blur', 'flow_vectors']]

        assert ref_img_indices in key_frame_indices

        key_frame_aux_map[ref_img_indices].append(img_indices)

    # %%

    assert set(key_frame_indices[:-1]) == set(key_frame_aux_map)

    frame_fractional_idxs = {}
    for frame_idx, ref_img_indices in enumerate(key_frame_indices):
        frame_fractional_idxs[ref_img_indices] = frame_idx
        if ref_img_indices in key_frame_aux_map:
            for frame_sub_idx, img_indices in enumerate(key_frame_aux_map[ref_img_indices]):
                if img_indices in aux_frame_indices:
                    frame_fractional_idxs[img_indices] = frame_idx + fractions.Fraction(frame_sub_idx + 1, len(key_frame_aux_map[ref_img_indices]) + 1)

    # %%

    frame_images = []
    for frame_index, frame_time in key_frame_indices:
        filename = f'{frame_time.strftime("%Y%m%d-%H%M%S%f")}.{frame_index:03d}.resized.png'
        frame_images.append(cv2.cvtColor(cv2.imread(image_source_dirpath / filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB))

    filtered_frame_images = {}
    frame_img_masks = {}
    for frame_index, frame_time in frame_fractional_idxs:
        fractional_idx = frame_fractional_idxs[(frame_index, frame_time)]

        filename = f'{frame_time.strftime("%Y%m%d-%H%M%S%f")}.{frame_index:03d}.filtered.png'
        filtered_frame_images[fractional_idx] = cv2.cvtColor(cv2.imread(image_source_dirpath / filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB)

        filename = f'{frame_time.strftime("%Y%m%d-%H%M%S%f")}.{frame_index:03d}.mask.png'
        frame_img_masks[fractional_idx] = cv2.imread(image_source_dirpath / filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(bool)

    frame_camera_extrinsics = dict(list(enumerate(camera_extrinsics))
                                   + list(zip([frame_fractional_idxs[frame_indices] for frame_indices in aux_frame_indices],
                                              aux_frame_camera_extrinsics)))

    image_sizes = set([img.shape[1::-1] for img in frame_images])
    assert len(image_sizes) == 1
    image_size = image_sizes.pop()

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
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(model_triangulated_points[:3, post_optimise_triangulated_idxs_mask].T))
    pcd.colors = o3d.utility.Vector3dVector(model_triangulated_points[3:, post_optimise_triangulated_idxs_mask].T / 255)
    visualise_geometries([pcd])

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(model_triangulated_points[:3, :].T))
    pcd.colors = o3d.utility.Vector3dVector(model_triangulated_points[3:, :].T / 255)
    visualise_geometries([pcd])

    # %%

    valid_mask = np.all(np.isfinite(model_triangulated_points[:3, :]), axis=0)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(model_triangulated_points[:3, valid_mask].T))
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
    normal_ray_alignment = np.zeros((np.sum(valid_mask),), dtype=np.float32)
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
        normal_ray_alignment += np.sum(object_point_ray * object_normals, axis=0) * image_points_weights[valid_mask]
    pcd.normals = o3d.utility.Vector3dVector((pcd_normals * -np.sign(normal_ray_alignment)).T)
    assert np.allclose(np.linalg.norm(np.array(pcd.normals), axis=1), 1)

    model_triangulated_normals = np.full((3, model_triangulated_points.shape[1]), fill_value=np.nan, dtype=np.float32)
    model_triangulated_normals[:, valid_mask] = np.array(pcd.normals).T

    # %%

    zoom_scale = 0.125
    camera_intrinsic_zoomed = np.block([[camera_intrinsic[:2, :2] * zoom_scale, (camera_intrinsic[:2, 2:] + 0.5) * zoom_scale - 0.5], [0, 0, 1]])

    voxel_length = 2.0 * 8 / np.sqrt(np.linalg.det(camera_intrinsic_zoomed[:2, :2]))

    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_length, sdf_trunc=1.5,
                                                          color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for rgbd_frame_idx in range(len(key_frame_indices)):

        rgbd_img = cv2.resize(frame_images[rgbd_frame_idx], dsize=None, fx=zoom_scale, fy=zoom_scale, interpolation=cv2.INTER_AREA)

        image_to_triangulated_point_idxs = key_frame_image_triangulated_point_idxs[rgbd_frame_idx]
        triangulated_image_idxs = np.where(image_to_triangulated_point_idxs >= 0)[0]
        triangulated_point_idxs = image_to_triangulated_point_idxs[triangulated_image_idxs]
        valid_mask = np.all(np.isfinite(model_triangulated_points[:3, triangulated_point_idxs]), axis=0)
        triangulated_image_idxs = triangulated_image_idxs[valid_mask]
        triangulated_point_idxs = triangulated_point_idxs[valid_mask]

        for opt_mode in [False] * 1 + [True] * 1:

            if opt_mode:
                post_optimise_triangulated_points_idxs = triangulated_point_idxs[post_optimise_triangulated_idxs_mask[triangulated_point_idxs]]
                triangulated_points = model_triangulated_points[:3, post_optimise_triangulated_points_idxs].T
                triangulated_normals = model_triangulated_normals[post_optimise_triangulated_points_idxs, :]
            else:
                triangulated_points = model_triangulated_points[:3, triangulated_point_idxs].T
                triangulated_normals = model_triangulated_normals[triangulated_point_idxs, :]

            object_points = camera_extrinsics[rgbd_frame_idx] @ np.vstack([triangulated_points.T, np.ones((triangulated_points.shape[0],))])
            object_points = object_points[:, np.argsort(object_points[2, :])[::-1]]
            object_normals = camera_extrinsics[rgbd_frame_idx][:3, :3] @ triangulated_normals.T
            projected_points = camera_intrinsic_zoomed @ object_points[:3, :]
            projected_points = np.round(projected_points[:2, :] / projected_points[2, :]).astype(int)
            mask = ((projected_points[0, :] > -0.5) & (projected_points[0, :] < rgbd_img.shape[1] - 0.5)
                    & (projected_points[1, :] > -0.5) & (projected_points[1, :] < rgbd_img.shape[0] - 0.5))
            if False:
                mask &= object_normals[2, :] < -np.cos(np.pi / 4)
            object_points = object_points[:, mask]
            projected_points = projected_points[:, mask]

            print(rgbd_frame_idx, projected_points.shape[1], np.unique(projected_points, axis=1).shape[1])

            depth_img = np.full(rgbd_img.shape[:2], fill_value=np.nan, dtype=np.float32)
            depth_img[projected_points[1, :], projected_points[0, :]] = object_points[2, :]
            depth_img[depth_img >= 30] = np.nan

            # Depth values larger than depth_trunc are truncated to 0
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgbd_img),
                                                                            o3d.geometry.Image(depth_img),
                                                                            depth_scale=1.0, depth_trunc=np.inf,
                                                                            convert_rgb_to_intensity=False)

            volume.integrate(rgbd_image,
                             o3d.camera.PinholeCameraIntrinsic(np.array(rgbd_image.depth).shape[1],
                                                               np.array(rgbd_image.depth).shape[0],
                                                               camera_intrinsic_zoomed),
                             camera_extrinsics[rgbd_frame_idx])

    visualise_geometries([volume.extract_triangle_mesh()])
    """

    # %%

    plt.close('RGBD disparity')
    plt.close('RGBD dense triangulated points')

    def fraction_idx_to_str(value):
        assert value >= 0
        quotient, remainder_fraction = divmod(value, 1)
        if remainder_fraction > 0:
            return f'{quotient}+{remainder_fraction}'
        return f'{quotient}'

    output_path = output_dirpath / 'depth_images'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for rgbd_frame_idx in range(len(key_frame_indices)):
        next_frames = inter_key_aux_frames[key_frame_indices[rgbd_frame_idx]]
        if rgbd_frame_idx > 0:
            next_frames = [key_frame_indices[rgbd_frame_idx - 1]] + next_frames
        if rgbd_frame_idx < len(key_frame_indices) - 1:
            next_frames = next_frames + [key_frame_indices[rgbd_frame_idx + 1]]
        next_frame_idxs = [frame_fractional_idxs[frame_indices] for frame_indices in next_frames]
        assert np.all(np.diff(next_frame_idxs) > 0)

        frame_depth_images = {}
        for next_frame_idx, next_frame_indices in zip(next_frame_idxs, next_frames):

            print(f'computing dense disparity between frames {rgbd_frame_idx}, {fraction_idx_to_str(next_frame_idx)}')

            ref_img = filtered_frame_images[rgbd_frame_idx]
            img = filtered_frame_images[next_frame_idx]

            ref_img_mask = frame_img_masks[rgbd_frame_idx]
            img_mask = frame_img_masks[next_frame_idx]

            prev_camera_extrinsic = frame_camera_extrinsics[rgbd_frame_idx]
            camera_extrinsic = frame_camera_extrinsics[next_frame_idx]

            camera_transform = camera_extrinsic @ np.linalg.inv(prev_camera_extrinsic)
            R = camera_transform[:3, :3]
            t = camera_transform[:3, 3:]

            # Project the next frame's camera view cone onto the reference frame's camera image and calculate
            # the intersection as a measure of the disparity that can potentially be derived
            uvs = np.array([[0, 0], [img.shape[1] - 1, 0], [img.shape[1] - 1, img.shape[0] - 1], [0, img.shape[0] - 1]]).T
            uvcs = uvs - camera_intrinsic[:2, 2:]
            xys = np.linalg.inv(camera_intrinsic[:2, :2]) @ uvcs
            xyzs = np.vstack([xys, np.ones(xys.shape[1],)])
            xyzs = np.hstack([xyzs * 5, xyzs * 30])
            xyzs = R.T @ xyzs - R.T @ t
            projected_points = camera_intrinsic @ xyzs
            projected_points = projected_points[:2, :] / projected_points[2, :]
            projected_points_geometry = shapely.MultiPoint(projected_points.T).convex_hull
            rgbd_image_geometry = shapely.MultiPoint(uvs.T).convex_hull
            projected_cone_coverage = rgbd_image_geometry.intersection(projected_points_geometry).area / rgbd_image_geometry.area
            if projected_cone_coverage < 0.5:
                print('projected_cone_coverage', projected_cone_coverage)
                continue

            """
            @param cameraMatrix1 First camera intrinsic matrix.
            @param distCoeffs1 First camera distortion parameters.
            @param cameraMatrix2 Second camera intrinsic matrix.
            @param distCoeffs2 Second camera distortion parameters.
            @param imageSize Size of the image used for stereo calibration.
            @param R Rotation matrix from the coordinate system of the first camera to the second camera,
                    see @ref stereoCalibrate.
            @param T Translation vector from the coordinate system of the first camera to the second camera,
                    see @ref stereoCalibrate.
            @param R1 Output 3x3 rectification transform (rotation matrix) for the first camera. This matrix
                    brings points given in the unrectified first camera's coordinate system to points in the rectified
                    first camera's coordinate system. In more technical terms, it performs a change of basis from the
                    unrectified first camera's coordinate system to the rectified first camera's coordinate system.
            @param R2 Output 3x3 rectification transform (rotation matrix) for the second camera. This matrix
                    brings points given in the unrectified second camera's coordinate system to points in the rectified
                    second camera's coordinate system. In more technical terms, it performs a change of basis from the
                    unrectified second camera's coordinate system to the rectified second camera's coordinate system.
            @param P1 Output 3x4 projection matrix in the new (rectified) coordinate systems for the first
                    camera, i.e. it projects points given in the rectified first camera coordinate system into the
                    rectified first camera's image.
            @param P2 Output 3x4 projection matrix in the new (rectified) coordinate systems for the second
                    camera, i.e. it projects points given in the rectified first camera coordinate system into the
                    rectified second camera's image.
            @param Q Output \f$4 \times 4\f$ disparity-to-depth mapping matrix (see @ref reprojectImageTo3D).
            @param flags Operation flags that may be zero or @ref CALIB_ZERO_DISPARITY . If the flag is set,
                    the function makes the principal points of each camera have the same pixel coordinates in the
                    rectified views. And if the flag is not set, the function may still shift the images in the
                    horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the
                    useful image area.
            @param alpha Free scaling parameter. If it is -1 or absent, the function performs the default
                    scaling. Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified
                    images are zoomed and shifted so that only valid pixels are visible (no black areas after
                    rectification). alpha=1 means that the rectified image is decimated and shifted so that all the
                    pixels from the original images from the cameras are retained in the rectified images (no source
                    image pixels are lost). Any intermediate value yields an intermediate result between
                    those two extreme cases.
            @param newImageSize New image resolution after rectification. The same size should be passed to
                    #initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0)
                    is passed (default), it is set to the original imageSize . Setting it to a larger value can help you
                    preserve details in the original image, especially when there is a big radial distortion.
            @param validPixROI1 Optional output rectangles inside the rectified images where all the pixels
                    are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
                    (see the picture below).
            @param validPixROI2 Optional output rectangles inside the rectified images where all the pixels
                    are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
                    (see the picture below).
            """

            imageSize = img.shape[1::-1]
            R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(camera_intrinsic, None,
                                                                              camera_intrinsic, None,
                                                                              imageSize,
                                                                              R.astype(float), t.astype(float),
                                                                              flags=cv2.CALIB_ZERO_DISPARITY,
                                                                              alpha=-1,
                                                                              newImageSize=imageSize)

            assert np.allclose(P1[:, :3], P2[:, :3])
            assert np.allclose(np.std(np.diag(P1[:2, :2])), 0)

            # This camera has a wider angle of view than a normal camera, the object is closer, and the
            # translation between the stereo viewpoints is relatively small, and can include an unconventionally
            # significant Z component.
            # So the P1 and P2 camera matrices returned by cv2.stereoRectify() can be unusable, irrespective of
            # the value of alpha set as -1, 0 or 1.
            #rect_map_prev, _ = cv2.initUndistortRectifyMap(camera_intrinsic, None, R1, P1[:3, :3], newImageSize, cv2.CV_32FC2)
            #rect_map_next, _ = cv2.initUndistortRectifyMap(camera_intrinsic, None, R2, P2[:3, :3], newImageSize, cv2.CV_32FC2)

            if P2[1, 3] != 0:
                # cv2.stereoRectify() return values are targeted for vertical stereo
                # (the epipolar lines in the rectified images are vertical and have the same x-coordinate).
                # Rotate the space in order to use horizontal stereo.
                Ry2x, jacobian = cv2.Rodrigues(np.array([0, 0, -np.pi / 2]))
                R1 = Ry2x @ R1
                R2 = Ry2x @ R2

            # Project the visible volume in the Z range 5 to 30 in the previous and next cameras' frames of reference
            # onto the rectified stereo cameras
            for img_size_trim in range(0, max(img.shape[:2]) // 2, 5):
                uvs = np.array([[0, 0], [img.shape[1] - 1, 0], [img.shape[1] - 1, img.shape[0] - 1], [0, img.shape[0] - 1]]).T
                uvcs = uvs - camera_intrinsic[:2, 2:]
                uvcs_clip = (max(img.shape[:2]) - 1) / 2 - img_size_trim
                uvcs = np.clip(uvcs, -uvcs_clip, uvcs_clip)
                xys = np.linalg.inv(camera_intrinsic[:2, :2]) @ uvcs
                xyzs = np.vstack([xys, np.ones(xys.shape[1],)])
                # Replicate 4 corners at depths 5 and 30 respectively
                xyzs = np.hstack([xyzs * 5, xyzs * 30])
                view_frustum_mesh, _ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyzs.T)).compute_convex_hull()
                view_frustum_vertices = np.array(view_frustum_mesh.vertices)
                view_frustum_triangles = np.array(view_frustum_mesh.triangles)
                slicing_mesh = o3d.geometry.TriangleMesh(view_frustum_mesh)
                slicing_mesh.transform(np.linalg.inv(camera_transform))
                slicing_mesh.compute_triangle_normals()
                overtrimmed_common_view_frustum = False
                for triangle, normal in zip(np.array(slicing_mesh.triangles), np.array(slicing_mesh.triangle_normals)):
                    sliced_vertices, _, _ = trimesh.intersections.slice_faces_plane(vertices=view_frustum_vertices,
                                                                                    faces=view_frustum_triangles,
                                                                                    plane_normal=-normal,
                                                                                    plane_origin=np.array(slicing_mesh.vertices)[triangle[0], :])
                    if sliced_vertices.shape[0] < 4:
                        overtrimmed_common_view_frustum = True
                        break
                    sliced_vertices_pcd = trimesh.PointCloud(vertices=sliced_vertices)
                    sliced_vertices_pcd.merge_vertices()
                    if sliced_vertices_pcd.vertices.shape[0] < 4:
                        overtrimmed_common_view_frustum = True
                        break
                    sliced_mesh = sliced_vertices_pcd.convex_hull
                    if sliced_mesh.volume < 1000:
                        overtrimmed_common_view_frustum = True
                        break
                    view_frustum_vertices = np.array(sliced_mesh.vertices)
                    view_frustum_triangles = np.array(sliced_mesh.faces)

                if overtrimmed_common_view_frustum:
                    break

                common_view_frustum_xyzs = view_frustum_vertices.T

                common_view_frustum_xyzs1 = R1 @ common_view_frustum_xyzs
                common_view_frustum_xyzs2 = R2 @ (R @ common_view_frustum_xyzs + t)

                rect_proximity = np.min(np.hstack([common_view_frustum_xyzs1[2, :], common_view_frustum_xyzs2[2, :]]))
                if np.all(rect_proximity >= 1.0):
                    break
            else:
                print('img_size_trim', img_size_trim)
                continue

            if overtrimmed_common_view_frustum:
                print('overtrimmed_common_view_frustum, img_size_trim', overtrimmed_common_view_frustum, img_size_trim)
                continue

            # common_view_frustum_xyzs are the vertices of the convex hull which can
            # be used to calculate the extremities of the geometry.
            # common_view_frustum_sample_points represents a distribution of points which can
            # be used to calculate statistical properties of the geometry.
            # Since the common view frustum is convex, it is much quicker to slice the point cloud
            # than to use trimesh.Trimesh.contains().
            hs, ws = np.round(np.array(img.shape[:2]) / 10).astype(int) + 1
            uvs = np.vstack([uv.flatten() for uv in np.mgrid[0:img.shape[0]-1:hs*1j, 0:img.shape[1]-1:ws*1j][::-1]])
            uvcs = uvs - camera_intrinsic[:2, 2:]
            xys = np.linalg.inv(camera_intrinsic[:2, :2]) @ uvcs
            xyzs = np.vstack([xys, np.ones(xys.shape[1],)])
            common_view_frustum_sample_points = np.hstack([xyzs * z for z in 1 / np.linspace(1 / 29.999, 1 / 5.001, 129)])
            common_view_frustum_sample_points_weights = np.power(common_view_frustum_sample_points[2, :], -1.25)

            common_view_frustum_inlier_mask = np.full((common_view_frustum_sample_points.shape[1],), fill_value=True, dtype=bool)
            slicing_mesh, _ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(common_view_frustum_xyzs.T)).compute_convex_hull()
            slicing_mesh.compute_triangle_normals()
            for triangle, normal in zip(np.array(slicing_mesh.triangles), np.array(slicing_mesh.triangle_normals)):
                point_to_plane_distances = trimesh.points.point_plane_distance(common_view_frustum_sample_points.T,
                                                                               plane_normal=-normal,
                                                                               plane_origin=np.array(slicing_mesh.vertices)[triangle[0], :])
                common_view_frustum_inlier_mask &= point_to_plane_distances >= 0

            common_view_frustum_weighted_inlier_ratio = (np.sum(common_view_frustum_sample_points_weights[common_view_frustum_inlier_mask])
                                                         / np.sum(common_view_frustum_sample_points_weights))
            if common_view_frustum_weighted_inlier_ratio < 0.2:
                print('common_view_frustum_weighted_inlier_ratio', common_view_frustum_weighted_inlier_ratio)
                continue

            common_view_frustum_sample_points1 = R1 @ common_view_frustum_sample_points
            common_view_frustum_sample_points2 = R2 @ (R @ common_view_frustum_sample_points + t)

            disparity_map_zoom = 0.25
            fxy = (disparity_map_zoom * np.diag(camera_intrinsic[:2, :2])
                   * np.mean(np.hstack([common_view_frustum_sample_points1[2, :], common_view_frustum_sample_points2[2, :]]))
                   / np.mean(common_view_frustum_sample_points[2, :]))

            uvs1 = common_view_frustum_xyzs1[:2, :] / common_view_frustum_xyzs1[2, :] * fxy[:, None]
            uvs2 = common_view_frustum_xyzs2[:2, :] / common_view_frustum_xyzs2[2, :] * fxy[:, None]

            # Enclose both camera projections of the common view frustum in the vertical image axes
            tb1 = (np.min(uvs1[1, :]), np.max(uvs1[1, :]))
            tb2 = (np.min(uvs2[1, :]), np.max(uvs2[1, :]))

            img_height = int(np.ceil(max(tb1[1], tb2[1]) - min(tb1[0], tb2[0])))
            cy = 0.5 * (img_height - 1) - np.mean(tb2 + tb2)

            # Enclose each camera projection of the common view frustum in each horizontal image axis
            lr1 = (np.min(uvs1[0, :]), np.max(uvs1[0, :]))
            lr2 = (np.min(uvs2[0, :]), np.max(uvs2[0, :]))

            img_width = int(np.ceil(max(lr1[1] - lr1[0], lr2[1] - lr2[0])))
            cx1 = 0.5 * (img_width - 1) - np.mean(lr1)
            cx2 = 0.5 * (img_width - 1) - np.mean(lr2)

            P1 = np.array([[fxy[0], 0, cx1], [0, fxy[1], cy], [0, 0, 1]])
            P2 = np.array([[fxy[0], 0, cx2], [0, fxy[1], cy], [0, 0, 1]])

            # Disparity is defined as left_x - right_x, i.e. in the negative direction of the x-axis relative to the reference image,
            # perhaps following the concept that the camera on the "right" has an extrinsic negative horizontal translation.
            uvs1 = common_view_frustum_sample_points1[:2, :] / common_view_frustum_sample_points1[2, :] * fxy[:, None]
            uvs2 = common_view_frustum_sample_points2[:2, :] / common_view_frustum_sample_points2[2, :] * fxy[:, None]
            dx12s = (uvs1[0, :] + cx1) - (uvs2[0, :] + cx2)

            newImageSize = (img_width, img_height)

            # Measure lower and upper percentiles and the spread of the disparity distribution
            disparity_lower, disparity_upper = np.round(np.percentile(dx12s[common_view_frustum_inlier_mask], [10, 90])).astype(int)
            disparity_spread = disparity_upper + 1 - disparity_lower

            """
            plt.figure('dx12s', figsize=(16, 10))
            setup_new_fig_page()
            plt.suptitle(f'rgbd, next frame idxs: {rgbd_frame_idx}, {next_frame_idx}\ndisparity_spread {disparity_spread}')
            hist_bins = np.arange(np.floor(np.min(dx12s[common_view_frustum_inlier_mask])),
                                  np.floor(np.max(dx12s[common_view_frustum_inlier_mask])) + 1)
            plt.hist(dx12s[common_view_frustum_inlier_mask], bins=hist_bins,
                     weights=common_view_frustum_sample_points_weights[common_view_frustum_inlier_mask])
            plt.tight_layout()
            stash_fig_page()
            """

            if img_size_trim > 0 or rect_proximity < 2.0 or disparity_spread < 16 * disparity_map_zoom or disparity_spread > 192 * disparity_map_zoom:
                print('img_size_trim, rect_proximity, disparity_spread', img_size_trim, rect_proximity, disparity_spread)
                continue

            # Noting that positive disparity is in the negative x-axis direction relative to the reference (left) image,
            # for each left image pixel x location, the disparity search range within the right image extends:
            #  from: x - disparity_upper
            #    to: x - disparity_lower
            # Skip disparity computation if the search range falls entirely outside the right image for over half of the left image pixels.
            if 0.5 * newImageSize[0] - disparity_upper > newImageSize[0] or 0.5 * newImageSize[0] - disparity_lower < 0:
                print('disparity_lower, disparity_upper', disparity_lower, disparity_upper)
                continue
            # Skip disparity computation if the search range exceeds the width of the image.
            # There is potentially much more occlusion, dissimilar lighting between the images,
            # and a smaller resulting area of useful disparity.
            # For a baseline b = 3mm, fx * disparity_map_zoom = 360 * 0.25 = 90, and a depth z range of [5mm, 30mm],
            # the disparty b * fx * disparity_map_zoom / z range is [3 * 90 / 30, 3 * 90 / 5] = [9, 54]
            if disparity_spread > newImageSize[0]:
                print('disparity_spread', disparity_spread)
                continue

            # StereoSGBM requires num_disparities to be a multiple of 16.
            if True:
                num_disparities = int(max(np.round(disparity_spread / 16), 1)) * 16
            else:
                num_disparities = int(max(np.ceil(disparity_spread / 16), 1)) * 16

            # dx12s values are associated with the distribution of view frustum sample points ranging from depths 5 to 30.
            # If num_disparities is lower than the disparity range, then find the range that covers the highest weight of sample points.
            # Otherwise, set the disparity range boundary to align with the far field depth
            # and let the excess range extend into the near field depth.
            dx12s_quantised = np.round(dx12s[common_view_frustum_inlier_mask]).astype(int)
            dx12s_quantised_min = np.min(dx12s_quantised)
            dx12s_quantised_max = np.max(dx12s_quantised)
            if num_disparities < dx12s_quantised_max + 1 - dx12s_quantised_min:
                dx12s_quantised_bincount = np.bincount(dx12s_quantised - dx12s_quantised_min,
                                                       weights=common_view_frustum_sample_points_weights[common_view_frustum_inlier_mask])
                dx12s_quantised_bincount_cumsum = np.cumsum(dx12s_quantised_bincount)
                dx12s_quantised_bincount_sum_interval = dx12s_quantised_bincount_cumsum[num_disparities:] - dx12s_quantised_bincount_cumsum[:-num_disparities]
                min_disparity = np.argmax(dx12s_quantised_bincount_sum_interval) + dx12s_quantised_min
                max_disparity = min_disparity + (num_disparities - 1)
            else:
                disparity_slope = scipy.stats.linregress(common_view_frustum_sample_points[2, common_view_frustum_inlier_mask],
                                                         dx12s[common_view_frustum_inlier_mask]).slope
                assert disparity_slope != 0
                if disparity_slope > 0:
                    # max_disparity represents far field depth
                    max_disparity = dx12s_quantised_max
                    min_disparity = max_disparity - (num_disparities - 1)
                else:
                    # min_disparity represents far field depth
                    min_disparity = dx12s_quantised_min
                    max_disparity = min_disparity + (num_disparities - 1)

            # StereoSGBM returns left and right borders of invalid disparity for columns where the full search
            # width spanning min_disparity to max_disparity cannot be accommodated.
            #  - the left border is of width max(max_disparity, 0)
            #  - the right border is of width max(-min_disparity, 0)
            # Left and right pad the rectified images to accommodate the borders with invalid disparity values
            # and position the invalid columns outside the original image region.
            if min_disparity > 0:
                # The entire search range is in the negative x-axis direction and the leftmost min_disparity columns cannot be matched anyway.
                # With a total left border width of max_disparity, then max_disparity - min_disparity remaining padding is required.
                left_pad_width = max_disparity - min_disparity
            else:
                # min_disparity <= 0, so the search range is partially or entirely in the positive x-axis direction
                # and therefore the total left border width of max(max_disparity, 0) is required for left padding
                left_pad_width = max(max_disparity, 0)
            if max_disparity < 0:
                # The entire search range is in the positive x-axis direction and the rightmost -max_disparity columns
                # cannot be matched anyway.
                # With a total right border width of -min_disparity, then max_disparity - min_disparity remaining padding is required.
                right_pad_width = max_disparity - min_disparity
            else:
                # max_disparity >= 0, so the search range is partially or entirely in the negative x-axis direction
                # and therefore the total right border width of max(-min_disparity, 0) is required for right padding
                right_pad_width = max(-min_disparity, 0)

            # The right to left disparity matcher has a complementary search range
            right_matcher_min_disparity = -max_disparity
            if right_matcher_min_disparity > 0:
                left_pad_width = max(left_pad_width, max_disparity - min_disparity)
            else:
                left_pad_width = max(left_pad_width, right_matcher_min_disparity + max_disparity - min_disparity)
            if right_matcher_min_disparity + max_disparity - min_disparity < 0:
                right_pad_width = max(right_pad_width, max_disparity - min_disparity)
            else:
                right_pad_width = max(right_pad_width, -right_matcher_min_disparity)

            left_right_slice = slice(left_pad_width, left_pad_width + newImageSize[0])

            rect_map_prev, _ = cv2.initUndistortRectifyMap(camera_intrinsic, None, R1, P1, newImageSize, cv2.CV_32FC2)
            rect_map_next, _ = cv2.initUndistortRectifyMap(camera_intrinsic, None, R2, P2, newImageSize, cv2.CV_32FC2)

            ref_img_resized = cv2.resize(ref_img, dsize=None, fx=disparity_map_zoom, fy=disparity_map_zoom, interpolation=cv2.INTER_AREA)
            img_resized = cv2.resize(img, dsize=None, fx=disparity_map_zoom, fy=disparity_map_zoom, interpolation=cv2.INTER_AREA)

            # Reduce image interpolation artefacts and blurring by mapping rectified images from higher resolution normalised images
            # Note that cv2.remap() does not support cv2.INTER_AREA
            ref_img_normed_rect = cv2.remap(cv2.resize(normalise_img_intensities(ref_img_resized, disparity_map_zoom), dsize=None, fx=1/disparity_map_zoom, fy=1/disparity_map_zoom, interpolation=cv2.INTER_LINEAR),
                                            rect_map_prev, None, cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 0, 0))
            img_normed_rect = cv2.remap(cv2.resize(normalise_img_intensities(img_resized, disparity_map_zoom), dsize=None, fx=1/disparity_map_zoom, fy=1/disparity_map_zoom, interpolation=cv2.INTER_LINEAR),
                                        rect_map_next, None, cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 127))

            # Paint specular reflection and margin regions in diametrical colours
            ref_img_mask_rect = cv2.remap(ref_img_mask.astype(np.uint8), rect_map_prev, None, cv2.INTER_NEAREST,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
            img_mask_rect = cv2.remap(img_mask.astype(np.uint8), rect_map_next, None, cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)

            block_size = {0.0625: 5, 0.125: 7, 0.25: 9}[disparity_map_zoom]

            """
            mask_margin = block_size / disparity_map_zoom
            ref_img_mask_rect |= ((rect_map_prev[:, :, 0] < -mask_margin) | (rect_map_prev[:, :, 0] > img.shape[1] - 1 + mask_margin)
                                  | (rect_map_prev[:, :, 1] < -mask_margin) | (rect_map_prev[:, :, 1] > img.shape[0] - 1 + mask_margin))
            img_mask_rect |= ((rect_map_next[:, :, 0] < -mask_margin) | (rect_map_next[:, :, 0] > img.shape[1] - 1 + mask_margin)
                              | (rect_map_next[:, :, 1] < -mask_margin) | (rect_map_next[:, :, 1] > img.shape[0] - 1 + mask_margin))
            """
            ref_img_normed_rect[ref_img_mask_rect, :] = (127, 0, 0)
            img_normed_rect[img_mask_rect, :] = (0, 0, 127)

            ref_img_normed_rect_padded = cv2.copyMakeBorder(ref_img_normed_rect, top=0, bottom=0, left=left_pad_width, right=right_pad_width,
                                                            borderType=cv2.BORDER_CONSTANT, value=(127, 0, 0))
            img_normed_rect_padded = cv2.copyMakeBorder(img_normed_rect, top=0, bottom=0, left=left_pad_width, right=right_pad_width,
                                                        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 127))

            """
            @param minDisparity Minimum possible disparity value. Normally, it is zero but sometimes
                    rectification algorithms can shift images, so this parameter needs to be adjusted accordingly.
            @param numDisparities Maximum disparity minus minimum disparity. The value is always greater than
                    zero. In the current implementation, this parameter must be divisible by 16.
            @param blockSize Matched block size. It must be an odd number \>=1 . Normally, it should be
                    somewhere in the 3..11 range.
            @param P1 The first parameter controlling the disparity smoothness. See below.
            @param P2 The second parameter controlling the disparity smoothness. The larger the values are,
                    the smoother the disparity is. P1 is the penalty on the disparity change by plus or minus 1
                    between neighbor pixels. P2 is the penalty on the disparity change by more than 1 between neighbor
                    pixels. The algorithm requires P2 \> P1 . See stereo_match.cpp sample where some reasonably good
                    P1 and P2 values are shown (like 8\*number_of_image_channels\*blockSize\*blockSize and
                    32\*number_of_image_channels\*blockSize\*blockSize , respectively).
            @param disp12MaxDiff Maximum allowed difference (in integer pixel units) in the left-right
                    disparity check. Set it to a non-positive value to disable the check.
            @param preFilterCap Truncation value for the prefiltered image pixels. The algorithm first
                    computes x-derivative at each pixel and clips its value by [-preFilterCap, preFilterCap] interval.
                    The result values are passed to the Birchfield-Tomasi pixel cost function.
            @param uniquenessRatio Margin in percentage by which the best (minimum) computed cost function
                    value should "win" the second best value to consider the found match correct. Normally, a value
                    within the 5-15 range is good enough.
            @param speckleWindowSize Maximum size of smooth disparity regions to consider their noise speckles
                    and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the
                    50-200 range.
            @param speckleRange Maximum disparity variation within each connected component. If you do speckle
                    filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
                    Normally, 1 or 2 is good enough.
            @param mode Set it to StereoSGBM::MODE_HH to run the full-scale two-pass dynamic programming
                    algorithm. It will consume O(W\*H\*numDisparities) bytes, which is large for 640x480 stereo and
                    huge for HD-size pictures. By default, it is set to false .

            The first constructor initializes StereoSGBM with all the default parameters. So, you only have to
            set StereoSGBM::numDisparities at minimum. The second constructor enables you to set each parameter
            to a custom value.
            """

            if False:
                """
                https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/ximgproc/src/disparity_filters.cpp
                Ptr<StereoSGBM> right_sgbm = StereoSGBM::create(-(min_disp+num_disp)+1,num_disp,wsize);
                right_sgbm->setUniquenessRatio(0);
                right_sgbm->setP1(sgbm->getP1());
                right_sgbm->setP2(sgbm->getP2());
                right_sgbm->setMode(sgbm->getMode());
                right_sgbm->setPreFilterCap(sgbm->getPreFilterCap());
                right_sgbm->setDisp12MaxDiff(1000000);
                right_sgbm->setSpeckleWindowSize(0);
                """
                matcher_params = {'numDisparities': num_disparities,
                                  'blockSize': block_size,
                                  'P1': 8*3*block_size**2,
                                  'P2': 32*3*block_size**2,
                                  'disp12MaxDiff': 1000000,
                                  'preFilterCap': 0,
                                  'uniquenessRatio': 0,
                                  'speckleWindowSize': 0,
                                  'speckleRange': None,
                                  'mode': None}
            else:
                # Applying these stricter parameters (uniqueness, and maybe also speckle) appears to
                # avoid producing resulting regions with incorrect disparity (even after the WLS filter's
                # left-right-consistency check).
                # Note that the WLS filter helps to fill regions with unknown disparity (output by the left
                # and right matchers).
                matcher_params = {'numDisparities': num_disparities,
                                  'blockSize': block_size,
                                  'P1': 8*3*block_size**2,
                                  'P2': 32*3*block_size**2,
                                  'disp12MaxDiff': 0,
                                  'preFilterCap': 0,
                                  'uniquenessRatio': 15,
                                  'speckleWindowSize': 200,
                                  'speckleRange': 1,
                                  'mode': None}
            left_matcher = cv2.StereoSGBM_create(minDisparity=min_disparity, **matcher_params)
            if False:
                right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
                assert right_matcher.getMinDisparity() == right_matcher_min_disparity
            else:
                right_matcher = cv2.StereoSGBM_create(minDisparity=right_matcher_min_disparity, **matcher_params)

            left_disparity = left_matcher.compute(ref_img_normed_rect_padded, img_normed_rect_padded)
            right_disparity = right_matcher.compute(img_normed_rect_padded, ref_img_normed_rect_padded)

            left_mask = ((rect_map_prev[:, :, 0] < 0) | (rect_map_prev[:, :, 0] > img.shape[1] - 1)
                         | (rect_map_prev[:, :, 1] < 0) | (rect_map_prev[:, :, 1] > img.shape[0] - 1))
            left_disparity[:, left_right_slice][left_mask] = (min_disparity - 1) * 16
            left_disparity[:, :left_right_slice.start] = (min_disparity - 1) * 16
            left_disparity[:, left_right_slice.stop:] = (min_disparity - 1) * 16

            right_mask = ((rect_map_next[:, :, 0] < 0) | (rect_map_next[:, :, 0] > img.shape[1] - 1)
                          | (rect_map_next[:, :, 1] < 0) | (rect_map_next[:, :, 1] > img.shape[0] - 1))
            right_disparity[:, left_right_slice][right_mask] = (right_matcher_min_disparity - 1) * 16
            right_disparity[:, :left_right_slice.start] = (right_matcher_min_disparity - 1) * 16
            right_disparity[:, left_right_slice.stop:] = (right_matcher_min_disparity - 1) * 16

            # The WLS filter does not take invalid disparity values (i.e. < min_disparity * 16) into consideration.
            # Apply inpainting to work around pixel disparity values < min_disparity * 16 being treated as actual disparity.
            left_disparity_floor = (min_disparity - 1) * 16
            assert np.all(left_disparity >= left_disparity_floor)
            mask_img = (left_disparity < min_disparity * 16).astype(np.uint8)
            left_disparity_wls = cv2.inpaint((left_disparity - left_disparity_floor).astype(np.uint16),
                                             inpaintMask=mask_img, inpaintRadius=3, flags=cv2.INPAINT_NS).astype(np.int16) + left_disparity_floor

            # Replace the invalid disparity values of the right matcher to avoid inadvertent left-right-consistency
            # confidence matches
            right_disparity_wls = np.array(right_disparity)
            right_disparity_wls[right_disparity_wls < right_matcher_min_disparity * 16] = np.iinfo(right_disparity_wls.dtype).min

            ref_img_rect = cv2.remap(cv2.resize(ref_img_resized // 2 + 128, dsize=None, fx=1/disparity_map_zoom, fy=1/disparity_map_zoom, interpolation=cv2.INTER_LINEAR),
                                     rect_map_prev, None, cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            ref_img_rect_padded = cv2.copyMakeBorder(ref_img_rect, top=0, bottom=0, left=left_pad_width, right=right_pad_width,
                                                     borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

            """
            Edge-Preserving Decompositions for Multi-Scale Tone and Detail Manipulation
            https://www.cs.huji.ac.il/~danix/epd/epd.pdf
            WLS filter seeks to minimise the cost comprising:
             - the distance term between the output and input images
               - (input image - output image) ** 2
             - the regularisation term
               - λ * [(x-derivative of output image) ** 2 / (abs(x-gradient of log luminance of input image) ** α + ε)
                      + (y-derivative of output image) ** 2 / (abs(y-gradient of log luminance of input image) ** α + ε)]
               - typically ε = 0.0001

            Basically, larger gradients in the input image can suppress regularisation of gradients in the output image
            (i.e. gradients in the output image are more likely to be aligned with gradients in the input image)

            Assume for the disparity WLS filter that:
             - the distance term between the output and input disparity images
               - (input disparity image - output disparity image) ** 2
             - the regularisation term
               - λ * [(x-derivative of output disparity image) ** 2 / (abs(x-gradient of luminance of input image) ** α + ε)
                      + (y-derivative of output disparity image) ** 2 / (abs(y-gradient of luminance of input image) ** α + ε)]
               - typically ε = 0.0001
               - it appears that it is the luminance of input image (and not the log luminance of input image) that is
                 used because results for (input image // 2) are the same as for (input image // 2 + 128)
            """
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
            # Lambda is a parameter defining the amount of regularization during filtering.
            # Larger values force filtered disparity map edges to adhere more to source image edges.
            # Typical value is 8000.
            wls_filter.setLambda(8.0)
            # SigmaColor is a parameter defining how sensitive the filtering process is to source image edges.
            # Large values can lead to disparity leakage through low-contrast edges.
            # Small values can make the filter too sensitive to noise and textures in the source image.
            # Typical values range from 0.8 to 2.0.
            wls_filter.setSigmaColor(2.0)
            # LRCthresh is a threshold of disparity difference used in left-right-consistency check during
            # confidence map computation. The default value of 24 (1.5 pixels) is virtually always good enough.
            wls_filter.setLRCthresh(24)
            # DepthDiscontinuityRadius is a parameter used in confidence computation. It defines the size of
            # low-confidence regions around depth discontinuities.
            wls_filter.setDepthDiscontinuityRadius(0)
            # right_view does not appear to be used by the WLS filter (the results are the same either with or without)
            filtered_disparity = wls_filter.filter(disparity_map_left=left_disparity_wls, left_view=ref_img_rect_padded,
                                                   disparity_map_right=right_disparity_wls)
            confidence_map = wls_filter.getConfidenceMap()[:, left_right_slice] / 255

            left_disparity = left_disparity[:, left_right_slice].astype(np.float32) / 16.0
            left_disparity[left_disparity < min_disparity] = np.nan

            right_disparity = right_disparity[:, left_right_slice].astype(np.float32) / 16.0
            right_disparity[right_disparity < right_matcher_min_disparity] = np.nan

            left_disparity_wls = left_disparity_wls[:, left_right_slice].astype(np.float32) / 16.0
            left_disparity_wls[left_disparity_wls < min_disparity] = np.nan

            right_disparity_wls = right_disparity_wls[:, left_right_slice].astype(np.float32) / 16.0
            right_disparity_wls[right_disparity_wls < right_matcher_min_disparity] = np.nan

            filtered_disparity = filtered_disparity[:, left_right_slice].astype(np.float32) / 16.0
            filtered_disparity[filtered_disparity < min_disparity] = np.nan
            filtered_disparity[left_mask] = np.nan

            confidence_map[~np.isfinite(filtered_disparity)] = np.nan

            # Because the disparity search range is limited, output values may effectively be clamped to the
            # min and max disparity limits, i.e. clamped to max disparity if the true disparity > max disparity,
            # and clamped to min disparity if the true disparity < min disparity.
            confidence_map[(filtered_disparity == min_disparity) | (filtered_disparity == max_disparity)] *= 2 / 3

            # Derive optical flow from disparity map
            # Project the cross frame's grid of image coordinates into object space
            uvs = np.vstack([uv.flatten() for uv in np.mgrid[0:img.shape[0], 0:img.shape[1]][::-1]])
            uvcs = uvs - camera_intrinsic[:2, 2:]
            xys = np.linalg.inv(camera_intrinsic[:2, :2]) @ uvcs
            xyzs = np.vstack([xys, np.ones(xys.shape[1],)])

            # Apply the cross frame's stereo camera rectification transformation to its grid of image coordinates
            xyzs1 = R1 @ xyzs
            zs1_map = xyzs1[2, :].reshape(img.shape[:2]).astype(np.float32)

            # Apply the cross frame's stereo camera projection to its rectified grid of image coordinates
            xys1 = xyzs1[:2, :] / xyzs1[2, :]
            uvs1 = xys1 * fxy[:, None] + np.array([cx1, cy])[:, None]

            # Map the filtered disparity and confidence map onto the cross frame's grid of image coordinates
            uvs1_map = uvs1.T.reshape(img.shape[:2] + (2,)).astype(np.float32)

            uvs12_disparity = cv2.remap(np.array(filtered_disparity), uvs1_map, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
            uvs12_confidence_map = cv2.remap(np.array(confidence_map), uvs1_map, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)

            # Use disparity to compute the mapping of the cross frame's grid of image coordinates between
            # the cross and current frames in the stereo camera spaces
            uvs12 = np.vstack([uvs1[0, :] - uvs12_disparity.flatten(), uvs1[1, :]])

            # Project the disparity mapped points from the current frame's stereo camera into object space
            xys12 = (uvs12 - np.array([cx2, cy])[:, None]) / fxy[:, None]
            xyz12 = np.vstack([xys12, np.ones(xys12.shape[1],)])

            # Apply the inverse of the current frame's stereo camera rectification transformation to the disparity mapped points
            xyzs12 = np.linalg.inv(R2) @ xyz12

            # Project the disparity mapped points onto the current frame's image space
            xys12 = xyzs12[:2, :] / xyzs12[2, :]
            uvs12 = camera_intrinsic[:2, :2] @ xys12 + camera_intrinsic[:2, 2:]

            # Apply the current frame's stereo camera rectification transformation to the
            # cross frame's disparity mapped grid of image coordinates
            zs12_map = (R2 @ (xyzs12 / xyzs12[2, :]))[2, :].reshape(img.shape[:2])

            print('computing RGBD')

            if True:
                plt.figure('RGBD disparity', figsize=(24, 12))
                setup_new_fig_page()
                rvec1, _ = cv2.Rodrigues(R1)
                rvec2, _ = cv2.Rodrigues(R2)
                title = '\n'.join([f'rgbd, next frame idxs: {rgbd_frame_idx}, {fraction_idx_to_str(next_frame_idx)}',
                                   f'min, max disparity {min_disparity}, {max_disparity}',
                                   f'rvec1, rvec2 {np.round(rvec1.flatten(), 3)}, {np.round(rvec2.flatten(), 3)}',
                                   f'rect_proximity {rect_proximity}',
                                   f'img_size_trim {img_size_trim}'])
                plt.suptitle(title)
                ax = plt.subplot(3, 4, 1)
                plt.imshow(ref_img_normed_rect)
                plt.title('ref_img_normed_rect')
                ax = plt.subplot(3, 4, 2, sharex=ax, sharey=ax)
                plt.imshow(img_normed_rect)
                plt.title('img_normed_rect')
                plt.subplot(3, 4, 3)
                plt.imshow(zs1_map)
                plt.title('zs1_map')
                plt.subplot(3, 4, 4)
                plt.imshow(zs12_map)
                plt.title('zs12_map')
                ax = plt.subplot(3, 4, 5, sharex=ax, sharey=ax)
                plt.imshow(left_disparity)
                plt.title('left_disparity')
                ax = plt.subplot(3, 4, 6, sharex=ax, sharey=ax)
                plt.imshow(right_disparity)
                plt.title('right_disparity')
                ax = plt.subplot(3, 4, 7, sharex=ax, sharey=ax)
                plt.imshow(ref_img_rect)
                plt.title('ref_img_rect')
                ax = plt.subplot(3, 4, 9, sharex=ax, sharey=ax)
                plt.imshow(left_disparity_wls)
                plt.title('left_disparity_wls')
                ax = plt.subplot(3, 4, 10, sharex=ax, sharey=ax)
                plt.imshow(right_disparity_wls)
                plt.title('right_disparity_wls')
                ax = plt.subplot(3, 4, 11, sharex=ax, sharey=ax)
                plt.imshow(filtered_disparity)
                plt.title('filtered_disparity')
                ax = plt.subplot(3, 4, 12, sharex=ax, sharey=ax)
                plt.imshow(confidence_map)
                plt.title('confidence_map')
                plt.tight_layout()
                stash_fig_page()

            prev_proj_mat = camera_intrinsic @ np.block([np.identity(3), np.zeros((3, 1))])
            next_proj_mat = camera_intrinsic @ camera_transform[:3, :]
            triangulatedPoints = cv2.triangulatePoints(prev_proj_mat, next_proj_mat,
                                                       uvs.T.astype(np.float32).reshape((-1, 1, 2)),
                                                       uvs12.T.astype(np.float32).reshape((-1, 1, 2)))
            triangulated_points = cv2.convertPointsFromHomogeneous(triangulatedPoints.T).reshape((-1, 3))

            triangulated_z_bound = 100
            mask = triangulated_points[:, 2] > triangulated_z_bound
            triangulated_points[mask, :] *= triangulated_z_bound / triangulated_points[mask, 2:]

            triangulated_points[triangulated_points[:, 2] <= 0, :] = np.nan
            triangulated_points[triangulated_points[:, 2] >= 30, :] = np.nan

            triangulated_points = triangulated_points.reshape(img.shape[:2] + (3,))

            depth_img = np.array(triangulated_points[:, :, 2])

            # TODO: find speckle, by e.g. comparing sequential optical flow warped image to previous image
            # and detecting large impulsive differences

            # Attenuate confidence of points where the local stereo rectification transformation scaling makes disparity less reliable
            uvs12_attenuated_confidence_map = (uvs12_confidence_map
                                               * 0.5 * (1 + scipy.special.erf((np.min(np.stack([zs1_map, 1 / np.clip(zs1_map, 1e-5, np.inf)]), axis=0) - 0.6) / 0.2))
                                               * 0.5 * (1 + scipy.special.erf((np.min(np.stack([zs12_map, 1 / np.clip(zs12_map, 1e-5, np.inf)]), axis=0) - 0.6) / 0.2)))

            # Scale down confidence map values for higher depth gradients
            # Z = depth
            # u = X / Z * M + c
            # dZ/dX = dZ/du * du/dX
            #       = dZ/du * M / Z
            #depth_gradient = np.stack(np.gradient(depth_img)) * np.diag(camera_intrinsic)[:2][:, None, None] / depth_img
            #uvs12_attenuated_confidence_map = uvs12_confidence_map * np.exp(-0.5 * np.sum(np.power(depth_gradient / (15 / 3), 2), axis=0))
            #uvs12_attenuated_confidence_map = uvs12_confidence_map * np.exp(-0.5 * np.sum(np.power(depth_gradient / (3 / 3), 2), axis=0))
            #uvs12_attenuated_confidence_map[np.linalg.norm(depth_gradient, axis=0) > 5] = np.nan

            uvs12_attenuated_confidence_map[~np.isfinite(depth_img)] = np.nan

            # Mask out a margin around the edge of the depth image where the disparity is less reliable
            margin = int(np.ceil(block_size / 2 / disparity_map_zoom))
            uvs12_attenuated_confidence_map[:margin, :] = np.nan
            uvs12_attenuated_confidence_map[-margin:, :] = np.nan
            uvs12_attenuated_confidence_map[:, :margin] = np.nan
            uvs12_attenuated_confidence_map[:, -margin:] = np.nan

            filtered_depth_img = np.array(depth_img)
            filtered_depth_img[~(uvs12_attenuated_confidence_map >= 1e-2)] = np.nan

            """
            # Mask out low confidence points
            depth_img[~np.isfinite(uvs12_attenuated_confidence_map)] = np.nan
            depth_img[uvs12_attenuated_confidence_map < 0.95] = np.nan

            # Mask out points where the local stereo rectification transformation scaling makes disparity less reliable
            depth_img[zs1_map < 0.3] = np.nan
            depth_img[zs12_map < 0.3] = np.nan
            # Mask out a margin around the edge of the depth image where the disparity is less reliable
            margin = int(np.ceil(block_size / 2 / disparity_map_zoom))
            depth_img[:margin, :] = np.nan
            depth_img[-margin:, :] = np.nan
            depth_img[:, :margin] = np.nan
            depth_img[:, -margin:] = np.nan
            """

            # Depth values larger than depth_trunc are truncated to 0
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(ref_img),
                                                                            o3d.geometry.Image(filtered_depth_img),
                                                                            depth_scale=1.0, depth_trunc=np.inf,
                                                                            convert_rgb_to_intensity=False)

            rgbd_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                                      o3d.camera.PinholeCameraIntrinsic(np.array(rgbd_image.depth).shape[1],
                                                                                                        np.array(rgbd_image.depth).shape[0],
                                                                                                        camera_intrinsic),
                                                                      np.identity(4))
            rgbd_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
            rgbd_pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))

            normal_img = np.full(depth_img.shape + (3,), fill_value=np.nan, dtype=np.float32)
            normal_img[np.isfinite(filtered_depth_img), :] = np.array(rgbd_pcd.normals)

            camera_rays = triangulated_points / np.clip(np.linalg.norm(triangulated_points, axis=-1), 1e-6, np.inf)[:, :, None]
            normal_ray_alignment = np.sum(camera_rays * normal_img, axis=-1)
            camera_ray_to_object_plane = camera_rays / np.clip(-normal_ray_alignment, 1e-8, 1)[:, :, None]
            camera_ray_to_image_plane = camera_rays * -normal_img[:, :, 2:] / camera_rays[:, :, 2:]

            object_to_image_ratio = (np.linalg.norm(normal_img + camera_ray_to_object_plane, axis=-1)
                                     / np.clip(np.linalg.norm(normal_img + camera_ray_to_image_plane, axis=-1), 1e-8, np.inf))
            perspective_distortion = 1 - np.min(np.stack([object_to_image_ratio,
                                                          np.clip(1 / object_to_image_ratio, 1e-8, np.inf)]), axis=0)

            perspective_distortion_scores = 0.5 * (1 - scipy.special.erf((perspective_distortion - 0.4) / 0.15))
            perspective_distortion_scores[normal_ray_alignment >= 0] = 0

            uvs12_attenuated_confidence_map *= perspective_distortion_scores

            frame_depth_images[next_frame_idx] = (np.array(rgbd_image.depth, dtype=np.float32), normal_img, uvs12_attenuated_confidence_map.astype(np.float32))

            rgbd_pcd.transform(np.linalg.inv(prev_camera_extrinsic))

            if True:
                plt.figure('RGBD dense triangulated points', figsize=(24, 12))
                setup_new_fig_page()
                plt.suptitle(f'rgbd, next frame idxs: {rgbd_frame_idx}, {fraction_idx_to_str(next_frame_idx)}')
                ax = plt.subplot(2, 3, 1)
                plt.imshow(np.require(ref_img, dtype=np.uint8))
                plt.title('ref_img')
                ax = plt.subplot(2, 3, 2, sharex=ax, sharey=ax)
                plt.imshow(np.require(img, dtype=np.uint8))
                plt.title('img')
                ax = plt.subplot(2, 3, 3, sharex=ax, sharey=ax)
                plt.imshow(depth_img, vmin=0, vmax=15)
                plt.title('depth_img')
                ax = plt.subplot(2, 3, 4, sharex=ax, sharey=ax)
                plt.imshow(perspective_distortion)
                plt.title('perspective_distortion')
                ax = plt.subplot(2, 3, 5, sharex=ax, sharey=ax)
                plt.imshow(uvs12_attenuated_confidence_map, vmin=0, vmax=1)
                plt.title('uvs12_attenuated_confidence_map')
                rgbd_pcd_down = rgbd_pcd.voxel_down_sample(voxel_size=0.2)
                ax3 = plt.subplot(2, 3, 6, projection='3d')
                ax3.scatter(*np.array(rgbd_pcd_down.points).T, s=2, c=np.array(rgbd_pcd_down.colors))
                ax3.set_xlim((-20, 20))
                ax3.set_ylim((-20, 20))
                ax3.set_zlim((0, 40))
                ax3.set_aspect('equal', adjustable='datalim')
                ax3.set_xlabel('X')
                ax3.set_ylabel('Y')
                ax3.set_zlabel('Z')
                plt.title('rgbd_pcd_down')
                ax3.view_init(elev=-135, azim=-90, roll=0)
                plt.tight_layout()
                stash_fig_page()

        if len(frame_depth_images) > 0:
            with shelve.open(output_path) as depth_images:
                depth_images[str(rgbd_frame_idx)] = frame_depth_images

    # %%

    with shelve.open(output_path) as depth_images:
        depth_image_frame_idxs = set(map(int, depth_images.keys()))

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

            pcds = []
            if rgbd_frame_idx in depth_image_frame_idxs:
                with shelve.open(output_path) as depth_images:
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
                    #rgbd_pcd = rgbd_pcd.voxel_down_sample(voxel_size=0.1)
                    rgbd_pcd = rgbd_pcd.crop(o3d.geometry.AxisAlignedBoundingBox([-100, -100, -100], [100, 100, 100]))
                    pcds.append(rgbd_pcd)
            else:
                pcds.append(None)

            for pcd in pcds:
                if pcd is not None:
                    vis.add_geometry(pcd, reset_bounding_box=False)

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

        if close_vis:
            break

        for geometry in geometries:
            vis.remove_geometry(geometry, reset_bounding_box=False)

    vis.destroy_window()

    # %%

    post_to_pipeline_server((f'{working_subdir} / {output_dirpath.name}', None))
