"""
Select and position auxiliary frames located between key frames
using a cost function based on estimated flow displacement and motion blur
to compare the suitability of each frame.

This is one stage of the processing pipeline for https://github.com/mcmhsieh/Smile

SPDX-FileCopyrightText: 2026 Mark Hsieh
SPDX-License-Identifier: MIT
"""

import sys
import warnings
import time
import datetime
import pathlib
import shutil
import pickle
import collections

import numpy as np
import scipy
import sklearn.cluster
import skimage
import cv2
import open3d as o3d
import torch
import pomegranate.gmm
import pomegranate.distributions

# Workaround for the obstacles installing PyTorch3D described in README
sys.path.append('../external/pytorch3d')
import pytorch3d.transforms

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

from image_filtering import calc_image_point_weights
from weighting_functions import gamma_softplus
from fig_paging import setup_new_fig_page, stash_fig_page

from pipeline_server import start_pipeline_server, post_to_pipeline_server, get_queue_from_pipeline_server


if __name__ == '__main__':

    torch.set_default_device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    working_subdir_config_path = pathlib.Path(r'../pipeline-workspace/working_subdir.txt')
    with open(working_subdir_config_path, 'r') as config_file:
        working_subdir = config_file.read().rstrip('\n')

    workspace_dirpath = pathlib.Path(r'../pipeline-workspace') / working_subdir
    image_source_dirpath = workspace_dirpath / 'calc_sequential_flow_and_blur'
    input_source_dirpath = workspace_dirpath / 'stitch_key_frames'
    key_frames_dirpath = workspace_dirpath / 'select_key_frames'
    output_dirpath = workspace_dirpath / 'select_and_position_inter_key_aux_frames'

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

    input_path = input_source_dirpath / 'stitched_key_frames.pickle'
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

    key_frames_filepaths = sorted(key_frames_dirpath.glob('*.flow.key.pickle'))
    aux_frames_filepaths = sorted(key_frames_dirpath.glob('*.flow.aux.pickle'))

    assert len(key_frames_filepaths) == len(key_frame_indices) - 1

    # %%

    frame_images = []
    for frame_index, frame_time in key_frame_indices:
        filename = f'{frame_time.strftime("%Y%m%d-%H%M%S%f")}.{frame_index:03d}.resized.png'
        frame_images.append(cv2.cvtColor(cv2.imread(image_source_dirpath / filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB))

    image_sizes = set([img.shape[1::-1] for img in frame_images])
    assert len(image_sizes) == 1
    image_size = image_sizes.pop()

    # %%

    frame_flow_displacements = {}
    frame_motion_blurs = {}
    frame_data_map = collections.defaultdict(list)
    for input_filepath in sorted(key_frames_filepaths + aux_frames_filepaths):
        frame_time, frame_index, _, frame_type, filename_ext = input_filepath.name.split('.')
        frame_time = datetime.datetime.strptime(frame_time, '%Y%m%d-%H%M%S%f')
        frame_index = int(frame_index)

        with open(input_filepath, 'rb') as pickle_file:
            frame_data = pickle.load(pickle_file)

        ref_img_indices, img_indices, camera_matrix, flow_displacement, ref_motion_blur, motion_blur, (xfp, yfp, xfn, yfn) = [
            frame_data[name]
            for name in ['ref_img_indices', 'img_indices', 'camera_matrix', 'flow_displacement', 'ref_motion_blur', 'motion_blur', 'flow_vectors']]

        assert ref_img_indices in key_frame_indices

        assert img_indices not in frame_flow_displacements
        frame_flow_displacements[img_indices] = (ref_img_indices, flow_displacement)

        for frame_indices, frame_motion_blur in [(ref_img_indices, ref_motion_blur),
                                                 (img_indices, motion_blur)]:
            if frame_indices not in frame_motion_blurs:
                frame_motion_blurs[frame_indices] = frame_motion_blur
            else:
                assert frame_motion_blurs[frame_indices] == frame_motion_blur

        frame_data_map[ref_img_indices].append((img_indices, frame_type, flow_displacement, motion_blur))

    # %%

    plt.close('Flow costs')

    selected_inter_key_aux_frames = {ref_img_indices: [] for ref_img_indices in key_frame_indices}

    assert set(key_frame_indices[:-1]) == set(frame_data_map)

    for prev_key_frame_indices, next_key_frame_indices, prev_key_frame_motion_blur in zip(key_frame_indices[:-1], key_frame_indices[1:], key_frame_motion_blurs[:-1]):
        frame_types = [frame_type for _, frame_type, _, _ in frame_data_map[prev_key_frame_indices]]
        assert frame_types == ['aux'] * len(frame_types[:-1]) + ['key']

        frame_indices_seq = [prev_key_frame_indices] + [img_indices for img_indices, _, _, _ in frame_data_map[prev_key_frame_indices]]
        assert frame_indices_seq[-1] == next_key_frame_indices

        for direction in ['forward', 'reverse']:

            flow_displacements_and_motion_blurs = np.array([(0, prev_key_frame_motion_blur)] +
                                                           [(flow_displacement, motion_blur)
                                                            for _, _, flow_displacement, motion_blur in frame_data_map[prev_key_frame_indices]])
            frame_idxs = np.arange(flow_displacements_and_motion_blurs.shape[0])

            if direction == 'reverse':
                frame_indices_seq = frame_indices_seq[::-1]
                flow_displacements_and_motion_blurs = flow_displacements_and_motion_blurs[::-1, :]
                flow_displacements_and_motion_blurs[:, 0] = flow_displacements_and_motion_blurs[0, 0] - flow_displacements_and_motion_blurs[:, 0]
                frame_idxs = -frame_idxs

            # Select and position auxiliary frames by iteratively bisecting gaps between selected frames
            selected_frame_offsets = []
            selected_gaps_flow_displacements = set()
            targeted_flow_displacements = [0, flow_displacements_and_motion_blurs[-1, 0]]

            while True:
                targeted_flow_displacement_gaps = np.diff(targeted_flow_displacements)
                gap_idxs = [gap_idx for gap_idx in np.where(targeted_flow_displacement_gaps >= 48)[0]
                            if tuple(targeted_flow_displacements[gap_idx:gap_idx+2]) not in selected_gaps_flow_displacements]
                if len(gap_idxs) == 0:
                    break

                motion_blur_aux_frame_target = 0.5
                targeted_gap_flow_displacements = tuple(targeted_flow_displacements[gap_idxs[0]:gap_idxs[0]+2])
                selected_gaps_flow_displacements.add(targeted_gap_flow_displacements)
                target_flow_displacement = 0.5 * (targeted_gap_flow_displacements[1] - targeted_gap_flow_displacements[0])

                def calc_flow_costs(flow_displacements, motion_blurs):
                    flow_displacements = flow_displacements - targeted_gap_flow_displacements[0]
                    # The gradient of these mirror symmetric components wrt flow_displacements at flow_displacements = target_flow_displacement
                    #   = +/- 2 * (target_flow_displacement ** 0.5) / (target_flow_displacement ** 1.5)
                    #   = +/- 2 / target_flow_displacement
                    cost_magnitude = 1.1 / (motion_blurs / motion_blur_aux_frame_target + 0.1)
                    flow_costs = cost_magnitude * 4 * (np.sqrt(target_flow_displacement / np.maximum(2 * target_flow_displacement - flow_displacements, 1e-8)) - 1)
                    flow_costs += cost_magnitude * 4 * (np.sqrt(target_flow_displacement / np.maximum(flow_displacements, 1e-8)) - 1)

                    flow_costs += 0.2 * np.power(motion_blurs / motion_blur_aux_frame_target, 2)
                    return flow_costs

                flow_costs = calc_flow_costs(flow_displacements_and_motion_blurs[1:-1, 0], flow_displacements_and_motion_blurs[1:-1, 1])

                selected_frame_offset = np.argmin(flow_costs) + 1

                assert selected_frame_offset not in selected_frame_offsets
                selected_frame_offsets.append(selected_frame_offset)

                targeted_flow_displacements.append(flow_displacements_and_motion_blurs[selected_frame_offset, 0])
                targeted_flow_displacements.sort()

                selected_frame_idx = frame_idxs[selected_frame_offset]

                plt.figure('Flow costs', figsize=(16, 10))
                setup_new_fig_page()
                plt.suptitle('\n'.join(['Ref key frame: {} {}'.format(*frame_indices_seq[0]),
                                        f'{direction} {target_flow_displacement}']))
                ax = plt.subplot(3, 2, 1)
                plt.plot(frame_idxs, flow_displacements_and_motion_blurs[:, 0], 'o-')
                plt.axvline(selected_frame_idx, linestyle='--', color='black')
                #plt.ylim((0, plt.ylim()[1]))
                plt.title('flow_displacements')
                plt.subplot(3, 2, 3, sharex=ax)
                plt.plot(frame_idxs, flow_displacements_and_motion_blurs[:, 1], 'o-')
                plt.axvline(selected_frame_idx, linestyle='--', color='black')
                plt.ylim((0, plt.ylim()[1]))
                plt.title('motion_blurs')
                plt.subplot(3, 2, 5, sharex=ax)
                plt.plot(frame_idxs[1:-1], flow_costs, 'o-')
                plt.axvline(selected_frame_idx, linestyle='--', color='black')
                plt.ylim((-0.5, 6.5))
                plt.title('flow_costs')
                ax = plt.subplot(1, 2, 2, projection='3d')
                ax.plot(flow_displacements_and_motion_blurs[1:-1, 0], flow_displacements_and_motion_blurs[1:-1, 1], flow_costs, 'ko:')
                flow_displacements_grid, motion_blurs_grid = np.meshgrid(np.arange(0.2, 180, 0.2), np.arange(0.01, 9, 0.01))
                flow_costs_grid = calc_flow_costs(flow_displacements_grid, motion_blurs_grid)
                ax.contour(flow_displacements_grid, motion_blurs_grid, flow_costs_grid, levels=np.arange(-0.1, 6.1, 0.1))
                ax.set_zlim((-0.5, 6.5))
                ax.set_xlabel('flow_displacements')
                ax.set_ylabel('motion_blurs')
                ax.set_zlabel('flow_costs')
                ax.view_init(elev=80, azim=-90, roll=0)
                plt.tight_layout()
                stash_fig_page()

            for selected_frame_offset in sorted(selected_frame_offsets, reverse=(direction == 'reverse')):
                selected_inter_key_aux_frames[frame_indices_seq[0]].append(frame_indices_seq[selected_frame_offset])

    # %%

    plt.close('Selected inter key aux frames')

    def read_img(frame_indices):
        frame_index, frame_time = frame_indices
        filename = f'{frame_time.strftime("%Y%m%d-%H%M%S%f")}.{frame_index:03d}.resized.png'
        return cv2.cvtColor(cv2.imread(image_source_dirpath / filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB)

    for ref_frame_indices, aux_frames in selected_inter_key_aux_frames.items():
        plt.figure('Selected inter key aux frames', figsize=(16, 10))
        setup_new_fig_page()
        plt.suptitle('Ref key frame: {} {}'.format(*ref_frame_indices))
        n_subplots = len(aux_frames) + 1
        n_rows = int(np.ceil(np.sqrt(n_subplots / 1.5)))
        n_cols = int(np.ceil(n_subplots / n_rows))
        ax = plt.subplot(n_rows, n_cols, 1)
        plt.imshow(read_img(ref_frame_indices))
        plt.title(f'ref key frame, motion blur {frame_motion_blurs[ref_frame_indices]:.3f}')
        for plot_idx, aux_frame_indices in enumerate(aux_frames, start=2):
            plt.subplot(n_rows, n_cols, plot_idx, sharex=ax, sharey=ax)
            plt.imshow(read_img(aux_frame_indices))
            frame_flow_displacements_ref_indices, flow_displacement = frame_flow_displacements[aux_frame_indices]
            if frame_flow_displacements_ref_indices != ref_frame_indices:
                _, ref_frame_flow_displacement = frame_flow_displacements[ref_frame_indices]
                flow_displacement -= ref_frame_flow_displacement
            plt.title(f'aux frame, displacement {flow_displacement:.3f}, motion blur {frame_motion_blurs[aux_frame_indices]:.3f}')
        plt.tight_layout()
        stash_fig_page()

    # %%

    assert set(key_frame_indices[:-1]) == set(frame_data_map)

    frame_indices_seq = []
    cumulative_flow_displacements_and_motion_blurs = []
    for prev_key_frame_indices, next_key_frame_indices, prev_key_frame_motion_blur in zip(key_frame_indices[:-1], key_frame_indices[1:], key_frame_motion_blurs[:-1]):
        frame_types = [frame_type for _, frame_type, _, _ in frame_data_map[prev_key_frame_indices]]
        assert frame_types == ['aux'] * len(frame_types[:-1]) + ['key']

        if len(frame_indices_seq) == 0:
            frame_indices_seq.append(prev_key_frame_indices)
            cumulative_flow_displacements_and_motion_blurs.append((0, prev_key_frame_motion_blur))

        frame_indices_seq.extend([img_indices for img_indices, _, _, _ in frame_data_map[prev_key_frame_indices]])
        assert frame_indices_seq[-1] == next_key_frame_indices
        prev_flow_displacement, _ = cumulative_flow_displacements_and_motion_blurs[-1]
        cumulative_flow_displacements_and_motion_blurs.extend([(flow_displacement + prev_flow_displacement, motion_blur)
                                                               for _, _, flow_displacement, motion_blur in frame_data_map[prev_key_frame_indices]])

    key_frame_idxs = [frame_indices_seq.index(frame_indices) for frame_indices in key_frame_indices]
    selected_frame_idxs = sorted(set([frame_indices_seq.index(frame_indices)
                                      for aux_frames in selected_inter_key_aux_frames.values()
                                      for frame_indices in aux_frames]))
    cumulative_flow_displacements_and_motion_blurs = np.array(cumulative_flow_displacements_and_motion_blurs)

    plt.figure('Cumulative flow displacements and motion blurs', figsize=(16, 10))
    plt.clf()
    ax = plt.subplot(2, 1, 1)
    plt.plot(cumulative_flow_displacements_and_motion_blurs[:, 0])
    plt.scatter(key_frame_idxs, cumulative_flow_displacements_and_motion_blurs[key_frame_idxs, 0],
                s=10+30*cumulative_flow_displacements_and_motion_blurs[key_frame_idxs, 1],
                c=cumulative_flow_displacements_and_motion_blurs[key_frame_idxs, 1],
                marker='o', label='key')
    plt.scatter(selected_frame_idxs, cumulative_flow_displacements_and_motion_blurs[selected_frame_idxs, 0],
                s=10+30*np.power(cumulative_flow_displacements_and_motion_blurs[selected_frame_idxs, 1], 2),
                c=cumulative_flow_displacements_and_motion_blurs[selected_frame_idxs, 1],
                marker='P', label='aux')
    for idx in key_frame_idxs:
        plt.axvline(idx, alpha=0.2)
    plt.ylabel('flow displacement')
    plt.legend()
    ax1 = plt.subplot(2, 1, 2, sharex=ax)
    ax1.plot(cumulative_flow_displacements_and_motion_blurs[:, 1])
    ax1.scatter(key_frame_idxs, cumulative_flow_displacements_and_motion_blurs[key_frame_idxs, 1],
                s=10+30*cumulative_flow_displacements_and_motion_blurs[key_frame_idxs, 1],
                c=cumulative_flow_displacements_and_motion_blurs[key_frame_idxs, 1],
                marker='o')
    ax1.scatter(selected_frame_idxs, cumulative_flow_displacements_and_motion_blurs[selected_frame_idxs, 1],
                s=10+30*np.power(cumulative_flow_displacements_and_motion_blurs[selected_frame_idxs, 1], 2),
                c=cumulative_flow_displacements_and_motion_blurs[selected_frame_idxs, 1],
                marker='P')
    ax1.set_ylabel('motion blur')
    ax2 = ax1.twinx()
    ax2.plot(np.gradient(cumulative_flow_displacements_and_motion_blurs[:, 0]), '--', alpha=0.2)
    for idx in key_frame_idxs:
        ax2.axvline(idx, alpha=0.2)
    ax2.set_ylabel('flow displacement gradient')
    plt.tight_layout()

    # %%

    aux_frame_indices = []
    aux_frame_image_triangulated_point_idxs = []
    aux_frame_camera_extrinsics = []

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

    motion_blur_threshold = 1.5
    aux_frame_set = set([frame_indices for aux_frames in selected_inter_key_aux_frames.values()
                         for frame_indices in aux_frames
                         if frame_motion_blurs[frame_indices] < motion_blur_threshold])

    for input_filepath in sorted(aux_frames_filepaths):

        frame_time, frame_index, _, frame_type, filename_ext = input_filepath.name.split('.')
        frame_time = datetime.datetime.strptime(frame_time, '%Y%m%d-%H%M%S%f')
        frame_index = int(frame_index)

        if (frame_index, frame_time) not in aux_frame_set:
            continue

        print('-' * 80)
        print('frame', frame_index, frame_time, frame_type)

        with open(input_filepath, 'rb') as pickle_file:
            frame_data = pickle.load(pickle_file)

        ref_img_indices, img_indices, camera_matrix, flow_displacement, ref_motion_blur, motion_blur, (xfp, yfp, xfn, yfn) = [
            frame_data[name]
            for name in ['ref_img_indices', 'img_indices', 'camera_matrix', 'flow_displacement', 'ref_motion_blur', 'motion_blur', 'flow_vectors']]

        images = []
        for frame_index, frame_time in [ref_img_indices, img_indices]:
            filename = f'{frame_time.strftime("%Y%m%d-%H%M%S%f")}.{frame_index:03d}.filtered.png'
            images.append(cv2.cvtColor(cv2.imread(image_source_dirpath / filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB))
        ref_img, img = images

        image_masks = []
        for frame_index, frame_time in [ref_img_indices, img_indices]:
            filename = f'{frame_time.strftime("%Y%m%d-%H%M%S%f")}.{frame_index:03d}.mask.png'
            image_masks.append(cv2.imread(image_source_dirpath / filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(bool))
        ref_img_mask, img_mask = image_masks

        (_, _, primary_eig_weights, secondary_eig_weights, primary_eig_vecs, _) = calc_image_point_weights(img)
        frame_image_point_weights = (primary_eig_weights, secondary_eig_weights, primary_eig_vecs)

        current_key_frame_idx = key_frame_indices.index(ref_img_indices)

        xyfn_flow = np.full(img.shape[:2] + (2,), fill_value=np.nan, dtype=np.float32)
        xyfn_flow[yfp, xfp] = np.vstack([xfn, yfn]).T

        # Mask out optical flow in specular reflection regions
        xyfn_flow_masked = np.array(xyfn_flow)
        xyfn_flow_masked[ref_img_mask, :] = np.nan

        img_mask_warp = cv2.remap(img_mask.astype(np.uint8), xyfn_flow, None, cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=1).astype(bool)
        xyfn_flow_masked[img_mask_warp, :] = np.nan

        # Subsampled point correspondences between the previous image the current / next image,
        # for feeding (after residual optical flow adjustment) to cv2.findEssentialMat() and cv2.recoverPose()
        # to estimate camera motion.
        interp = scipy.interpolate.RegularGridInterpolator((np.arange(xyfn_flow_masked.shape[0]), np.arange(xyfn_flow_masked.shape[1])),
                                                           xyfn_flow_masked[:, :, 0] + xyfn_flow_masked[:, :, 1] * 1j,
                                                           method='linear', bounds_error=False, fill_value=np.nan)

        xsp, ysp = key_frame_image_sample_points[current_key_frame_idx]
        xysn = interp((ysp, xsp))
        xsn, ysn = xysn.real, xysn.imag
        # Mask out a margin around the edge of the image where the flow estimation is less reliable
        # TODO: check for correspondence between optical flow estimates and disparity based flow estimates, so that
        # points in the margin can still be triangulated
        margin = 25
        mask = (np.isfinite(xsp) & np.isfinite(ysp) & np.isfinite(xsn) & np.isfinite(ysn)
                & (xsp >= margin) & (xsp < img.shape[1] - margin) & (ysp >= margin) & (ysp < img.shape[0] - margin)
                & (xsn >= margin) & (xsn < img.shape[1] - margin) & (ysn >= margin) & (ysn < img.shape[0] - margin))
        corres_idxs = np.where(mask)[0]

        print('num corresponding flow points', len(corres_idxs))

        prev_image_to_triangulated_point_idxs = key_frame_image_triangulated_point_idxs[current_key_frame_idx]
        prev_triangulated_image_idxs = np.where(prev_image_to_triangulated_point_idxs >= 0)[0]
        prev_triangulated_idxs = prev_image_to_triangulated_point_idxs[prev_triangulated_image_idxs]
        valid_mask = np.all(np.isfinite(model_triangulated_points[:3, prev_triangulated_idxs]), axis=0)
        prev_triangulated_image_idxs = prev_triangulated_image_idxs[valid_mask]
        prev_triangulated_idxs = prev_triangulated_idxs[valid_mask]

        intersected_corres_idxs = np.array(list(set(list(prev_triangulated_image_idxs)) & set(list(corres_idxs))), dtype=int)
        next_image_points = np.vstack([xsn, ysn])[:, intersected_corres_idxs].T
        R = camera_extrinsics[current_key_frame_idx][:3, :3]
        t = camera_extrinsics[current_key_frame_idx][:3, 3:]
        prev_corres_triangulated_points = (R @ model_triangulated_points[:3, prev_image_to_triangulated_point_idxs[intersected_corres_idxs]] + t).T
        prev_corres_triangulated_idxs_weights = triangulated_idxs_weights[prev_image_to_triangulated_point_idxs[intersected_corres_idxs]]

        if True:
            new_triangulation_idxs = np.array(list(set(list(corres_idxs)) - set(list(intersected_corres_idxs))), dtype=int)
            prev_new_triangulation_points = np.vstack([xsp, ysp])[:, new_triangulation_idxs].T.astype(np.float32)
            next_new_triangulation_points = np.vstack([xsn, ysn])[:, new_triangulation_idxs].T.astype(np.float32)
        else:
            corres_pts_p = np.vstack([xsp, ysp])[:, corres_idxs].T.astype(np.float32)
            corres_pts_n = np.vstack([xsn, ysn])[:, corres_idxs].T.astype(np.float32)

        results = []
        for iter_idx in range(5000):
            np.random.seed(iter_idx)
            sample_fraction = np.random.uniform(1 / 3, 3 / 4)
            sample_size = np.clip(int(np.ceil(next_image_points.shape[0] * sample_fraction)), 50, next_image_points.shape[0])
            pts_idxs = np.random.choice(np.arange(next_image_points.shape[0]), size=(sample_size,), replace=False)

            # Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
            #threshold = min(0.5 * (1 + motion_blur), 2.0)
            threshold = 3.0
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints=prev_corres_triangulated_points[pts_idxs, :],
                                                             imagePoints=next_image_points[pts_idxs, :],
                                                             cameraMatrix=camera_intrinsic, distCoeffs=np.zeros((4,)),
                                                             rvec=None, tvec=None, useExtrinsicGuess=False,
                                                             iterationsCount=10, reprojectionError=threshold, confidence=0.999, inliers=None,
                                                             flags=cv2.SOLVEPNP_ITERATIVE)
            #assert retval == True
            if retval:
                R, jacobian = cv2.Rodrigues(rvec)
                t = tvec

                camera_transform = np.block([[R, t], [0, 0, 0, 1]])

                object_points = camera_transform @ np.vstack([prev_corres_triangulated_points.T, np.ones((prev_corres_triangulated_points.shape[0],))])
                projected_points = camera_intrinsic @ object_points[:3, :]
                projected_points = projected_points[:2, :] / projected_points[2, :]

                #threshold = min(0.5 * (1 + motion_blur), 2.0)
                threshold = 6.0
                triangulated_points_inliers = np.where(np.linalg.norm(next_image_points - projected_points.T, axis=1) < threshold)[0]

                E = R @ np.cross(np.identity(3), t.T)

                # F_xysp are the epipolar lines upon which the corresponding xysn points should lie
                if True:
                    F_xysp = np.linalg.inv(camera_intrinsic).T @ E @ np.linalg.inv(camera_intrinsic) @ np.vstack([prev_new_triangulation_points.T, np.ones((prev_new_triangulation_points.shape[0],))])
                    xysn = np.vstack([next_new_triangulation_points.T, np.ones((next_new_triangulation_points.shape[0],))])
                else:
                    F_xysp = np.linalg.inv(camera_intrinsic).T @ E @ np.linalg.inv(camera_intrinsic) @ np.vstack([corres_pts_p.T, np.ones((corres_pts_p.shape[0],))])
                    xysn = np.vstack([corres_pts_n.T, np.ones((corres_pts_n.shape[0],))])
                # Normalise the lines
                # https://math.stackexchange.com/questions/4585131/understanding-distance-from-point-to-line-in-homogeneous-coordinates
                # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/BEARDSLEY/node2.html
                F_xysp = F_xysp[:3, :] / np.clip(np.linalg.norm(F_xysp[:2, :], axis=0), 1e-8, np.inf)

                #threshold = min(0.5 * (1 + motion_blur), 2.0)
                threshold = 6.0
                new_triangulation_inliers = np.where(np.abs(np.sum(xysn * F_xysp, axis=0)) < threshold)[0]

                # TODO: include in score the entropy of the distribution of errors in space, in terms of whether
                # errors are smaller in one region than another, or whether errors are more negative in one region
                # than another

                if False:
                    score = (np.sum(prev_corres_triangulated_idxs_weights[triangulated_points_inliers])
                             + len(triangulated_points_inliers) + len(new_triangulation_inliers))
                    score = score / (np.sum(prev_corres_triangulated_idxs_weights + 1) + xysn.shape[1])
                elif False:
                    threshold_prev = 15.0
                    sigma_new = 3.0
                    projection_err2 = np.sum(np.power(next_image_points - projected_points.T, 2), axis=1)
                    # Heavy tailed bell shaped function based on Student's t-distribution with v = 1 (i.e. Cauchy distribution)
                    # y=1/(1+x**2) has a knee point at x ~ 3.0 beyond which it slowly converges to 0, e.g. [1.0, 0.5], [3.0, 0.1], [9.9, 0.01]
                    score = (np.sum((prev_corres_triangulated_idxs_weights + 1)
                                    * 1 / (1 + projection_err2 * np.power(3.0 / threshold_prev, 2)))
                             + np.sum(np.exp(-0.5 * np.power(np.sum(xysn * F_xysp, axis=0) / sigma_new, 2))))
                    score = score / (np.sum(prev_corres_triangulated_idxs_weights + 1) + xysn.shape[1])
                else:
                    sigma_prev = 3.0
                    sigma_new = 1.5
                    score = (0.2 * np.sum((prev_corres_triangulated_idxs_weights + 1)
                                          * np.exp(-0.5 * np.power(np.linalg.norm(next_image_points - projected_points.T, axis=1) / sigma_prev, 2)))
                             + 0.8 * np.sum(np.exp(-0.5 * np.power(np.sum(xysn * F_xysp, axis=0) / sigma_new, 2))))
                    score = score / (0.2 * np.sum(prev_corres_triangulated_idxs_weights + 1) + 0.8 * xysn.shape[1])

                results.append((rvec, tvec, triangulated_points_inliers, new_triangulation_inliers, score))

            if sum([score for _, _, _, _, score in results]) >= 100:
                break

        results.sort(key=lambda item: item[4], reverse=True)
        print('cv2.solvePnPRansac len(results)', len(results))

        transform_delta_ref_vec = np.array([(1, 0, 8), (0, 0, 8), (0, 0, 9)]).T
        transform_delta_vecs = []
        result_scores = []
        for rvec, tvec, _, _, score in results:
            R, jacobian = cv2.Rodrigues(rvec)
            t = tvec
            transform_delta_vecs.append((R @ transform_delta_ref_vec + t - transform_delta_ref_vec).flatten())
            result_scores.append(score)
        transform_delta_vecs = np.array(transform_delta_vecs)
        result_scores = np.array(result_scores)[:, None]

        """
        dbscan = sklearn.cluster.DBSCAN(eps=0.1, min_samples=5)
        dbscan.fit(transform_delta_vecs)
        dbscan_labels = sorted(set(dbscan.labels_) - set([-1]))
        for label in dbscan_labels:
            core_idxs = list(set(np.where(dbscan.labels_ == label)[0]) & set(dbscan.core_sample_indices_))
            if len(core_idxs) > 0:
                centroid = np.sum((transform_delta_vecs * result_scores)[core_idxs, :], axis=0) / np.sum(result_scores[core_idxs])
                std_dev = np.sqrt(np.sum((np.power(transform_delta_vecs - centroid, 2) * result_scores)[core_idxs, :], axis=0)
                                  / np.sum(result_scores[core_idxs]))
                print('transform delta vector dbscan cluster centroid, std dev', label, np.round(centroid, 3), np.round(std_dev, 3))
                #for core_idx in sorted(core_idxs):
                #    print('transform delta vector', np.round(transform_delta_vecs[core_idx], 3), 'score', result_scores[core_idx])
                print('score 0, 50, 100 percentiles', np.percentile(result_scores[core_idxs], [0, 50, 100]))

        kmeans = sklearn.cluster.KMeans(n_clusters=2, random_state=1)
        kmeans.fit(transform_delta_vecs, sample_weight=result_scores.flatten())
        assert set(kmeans.labels_) == {0, 1}
        for label in range(2):
            centroid = kmeans.cluster_centers_[label, :]
            std_dev = np.sqrt(np.sum((np.power(transform_delta_vecs - centroid, 2) * result_scores)[kmeans.labels_ == label, :], axis=0)
                              / np.sum(result_scores[kmeans.labels_ == label]))
            print('transform delta vector kmeans cluster centroid, std dev', label, np.round(centroid, 3), np.round(std_dev, 3))
            print('score 0, 50, 100 percentiles', np.percentile(result_scores[kmeans.labels_ == label, :], [0, 50, 100]))
        """

        torch.set_default_device(torch.device('cpu'))
        for trial_countdown in np.arange(10)[::-1]:
            gmm = pomegranate.gmm.GeneralMixtureModel([pomegranate.distributions.Normal(covariance_type='diag') for idx in range(2)],
                                                      random_state=trial_countdown)
            try:
                gmm.fit(transform_delta_vecs, sample_weight=result_scores)
                break
            except ValueError as e:
                # TODO: debug ValueError: Variances must be positive.
                if trial_countdown == 0:
                    raise
                print('gmm.fit raised exception', e)
        unsorted_labels = gmm.predict(transform_delta_vecs).numpy()
        sort_idxs = np.array([unsorted_labels[0], 1 - unsorted_labels[0]])
        label_map = {sort_idx: idx for idx, sort_idx in enumerate(sort_idxs)}
        labels = np.array([label_map[label] for label in unsorted_labels])
        gmm_priors = gmm.priors.numpy()[sort_idxs]
        gmm_distributions = [gmm.distributions[idx] for idx in sort_idxs]

        centroid_log_probs = np.array([[gmm_distribution.log_probability(gmm_sample_distribution.means.numpy()[None, :]).numpy()[0]
                                        for gmm_sample_distribution in gmm_distributions]
                                       for gmm_distribution in gmm_distributions])
        for idx in range(2):
            print(f'gmm[{idx}] prior', np.round(gmm_priors[idx], 3))
            print(f'gmm[{idx}] centroid', np.round(gmm_distributions[idx].means.numpy(), 3))
            print(f'gmm[{idx}] std devs', np.round(np.sqrt(gmm_distributions[idx].covs.numpy()), 3))
            print(f'gmm[{idx}] centroid_log_probs', centroid_log_probs[idx, :])

        log_prob_transform_delta_vecs = gmm_distributions[0].log_probability(transform_delta_vecs).numpy()
        sample_vec_idxs = (labels == 1) & (log_prob_transform_delta_vecs < centroid_log_probs[0, 0] - 3.0)
        sample_vec_idxs[0] = True
        sample_vec_idxs = np.where(sample_vec_idxs)[0][:10]
        sample_vec_labels = labels[sample_vec_idxs]
        sample_vecs = transform_delta_vecs[sample_vec_idxs, :]
        for idx, (label, sample_vec) in enumerate(zip(sample_vec_labels, sample_vecs)):
            print(f'sample_vecs[{idx}]', f'gmm[{label}]', np.round(sample_vec, 3))
        print('sample vec scores', np.round(result_scores.flatten()[sample_vec_idxs], 3))
        for idx in range(2):
            print(f'gmm[{idx}] log_prob(sample_vecs)', np.round(gmm_distributions[idx].log_probability(sample_vecs).numpy(), 3))
        torch.set_default_device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        """
        transform_delta_centroid = gmm.distributions[sort_idxs[0]].means.numpy().reshape(transform_delta_ref_vec.shape)
        source_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(transform_delta_ref_vec.T))
        target_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector((transform_delta_ref_vec + transform_delta_centroid).T))
        transform_estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        camera_transform = transform_estimation.compute_transformation(source=source_pcd, target=target_pcd,
                                                                       corres=o3d.utility.Vector2iVector([(idx, idx) for idx in range(3)]))
        rvec, jacobian = cv2.Rodrigues(camera_transform[:3, :3])
        tvec = camera_transform[:3, 3:]
        """

        rvec, tvec, triangulated_points_inliers, new_triangulation_inliers, score = results[0]

        print('cv2.solvePnPRansac extrinsic rotation', rvec.flatten())
        print('cv2.solvePnPRansac extrinsic translation', tvec.flatten())
        print('cv2.solvePnPRansac len(triangulated_points_inliers)', len(triangulated_points_inliers))
        print('cv2.solvePnPRansac len(new_triangulation_inliers)', len(new_triangulation_inliers))
        print('cv2.solvePnPRansac inliers score', score)

        R, jacobian = cv2.Rodrigues(rvec)
        t = tvec
        E = R @ np.cross(np.identity(3), t.T)
        camera_transform = np.block([[R, t], [0, 0, 0, 1]])

        prev_points = np.vstack([xsp, ysp])[:, intersected_corres_idxs].T.astype(np.float32)
        next_points = np.vstack([xsn, ysn])[:, intersected_corres_idxs].T.astype(np.float32)

        # F_xysp are the epipolar lines upon which the corresponding xysn points should lie
        F_xysp = np.linalg.inv(camera_intrinsic).T @ E @ np.linalg.inv(camera_intrinsic) @ np.vstack([prev_points.T, np.ones((prev_points.shape[0],))])
        xysn = np.vstack([next_points.T, np.ones((next_points.shape[0],))])
        # Normalise the lines
        # https://math.stackexchange.com/questions/4585131/understanding-distance-from-point-to-line-in-homogeneous-coordinates
        # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/BEARDSLEY/node2.html
        F_xysp = F_xysp[:3, :] / np.clip(np.linalg.norm(F_xysp[:2, :], axis=0), 1e-8, np.inf)

        #threshold = min(0.5 * (1 + motion_blur), 2.0)
        threshold = 6.0
        inlier_idxs = np.where(np.abs(np.sum(xysn * F_xysp, axis=0)) < threshold)[0]
        print('num corresponding previous triangulation points', len(intersected_corres_idxs))
        print('num corresponding previous triangulation point inliers', len(inlier_idxs))


        camera_fxy = torch.tensor(np.diag(camera_intrinsic[:2, :2])[:, None], dtype=torch.float32)
        camera_cxy = torch.tensor(camera_intrinsic[:2, 2:], dtype=torch.float32)

        # Although including all the camera extrinsics introduces redundant degrees of freedom,
        # this appears to help make the optimisation unbiased to each camera extrinsic.
        # Previously when the first frame was fixed to the origin/axes, it appeared to be susceptible
        # to higher projection errors especially for larger convergence criteria thresholds,
        # perhaps because the gradient graph for the first frame is far more convoluted.
        rvecs = []
        tvecs = []
        for camera_extrinsic in [camera_transform @ camera_extrinsics[current_key_frame_idx]]:
            rvec, jacobian = cv2.Rodrigues(camera_extrinsic[:3, :3])
            tvec = camera_extrinsic[:3, 3:]
            rvecs.append(rvec.flatten())
            tvecs.append(tvec)
        model_rvecs = torch.tensor(np.array(rvecs), dtype=torch.float32, requires_grad=True)
        model_tvecs = torch.tensor(np.array(tvecs), dtype=torch.float32, requires_grad=True)

        # TODO: try higher confidence map threshold

        current_image_to_triangulated_point_idxs = np.full((len(xsn),), fill_value=-1, dtype=int)
        current_image_to_triangulated_point_idxs[intersected_corres_idxs] = prev_image_to_triangulated_point_idxs[intersected_corres_idxs]

        triangulated_image_points = []
        triangulated_image_points_weights = []
        for (xs, ys), image_to_triangulated_point_idxs, image_point_weights, camera_extrinsic in zip([(xsn, ysn)],
                                                                                                     [current_image_to_triangulated_point_idxs],
                                                                                                     [frame_image_point_weights],
                                                                                                     [camera_transform @ camera_extrinsics[current_key_frame_idx]]):
            triangulated_image_idxs = np.where(image_to_triangulated_point_idxs >= 0)[0]
            triangulated_point_idxs = image_to_triangulated_point_idxs[triangulated_image_idxs]

            image_points = np.full((2, model_triangulated_points.shape[1]), fill_value=np.nan, dtype=np.float32)
            image_points[:, triangulated_point_idxs] = np.vstack([xs, ys])[:, triangulated_image_idxs]
            triangulated_image_points.append(image_points)

            image_points_weights = np.zeros((6, model_triangulated_points.shape[1]))

            primary_eig_weights, secondary_eig_weights, primary_eig_vecs = image_point_weights
            eig_weights = primary_eig_weights + secondary_eig_weights * 1j
            interp = scipy.interpolate.RegularGridInterpolator((np.arange(eig_weights.shape[0]), np.arange(eig_weights.shape[1])),
                                                               eig_weights,
                                                               method='linear', bounds_error=True)
            eig_weights = interp((image_points[1, triangulated_point_idxs], image_points[0, triangulated_point_idxs]))
            eig_vecs = primary_eig_vecs[:, :, 0] + primary_eig_vecs[:, :, 1] * 1j
            interp = scipy.interpolate.RegularGridInterpolator((np.arange(eig_vecs.shape[0]), np.arange(eig_vecs.shape[1])),
                                                               eig_vecs,
                                                               method='linear', bounds_error=True)
            eig_vecs = interp((image_points[1, triangulated_point_idxs], image_points[0, triangulated_point_idxs]))
            eig_weights = np.vstack([eig_weights.real, eig_weights.imag,
                                     eig_vecs.real, eig_vecs.imag,
                                     eig_vecs.imag, -eig_vecs.real])
            image_points_weights[:, triangulated_point_idxs] = eig_weights

            triangulated_points = camera_extrinsic[:3, :3] @ model_triangulated_points[:3, triangulated_point_idxs] + camera_extrinsic[:3, 3:]
            triangulated_normals = camera_extrinsic[:3, :3] @ model_triangulated_normals[:, triangulated_point_idxs]
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
            image_points_weights[:2, triangulated_point_idxs] *= triangulated_normal_weights

            triangulated_image_points_weights.append(image_points_weights)
        triangulated_image_points = np.array(triangulated_image_points)
        triangulated_image_points_weights = np.array(triangulated_image_points_weights)

        triangulated_point_mapping_counts = np.sum(np.all(np.isfinite(triangulated_image_points), axis=1), axis=0)
        # As the optimisation is only performed over one frame, the mapping counts are either 0 or 1
        bundle_triangulated_point_idxs = np.where(triangulated_point_mapping_counts > 0)[0]
        assert np.all(np.isfinite(model_triangulated_points[:3, bundle_triangulated_point_idxs]))

        print('len(bundle_triangulated_point_idxs)', len(bundle_triangulated_point_idxs),
              'of len(triangulated_point_mapping_counts)', len(triangulated_point_mapping_counts))

        model_triangulated_points_tensor = torch.tensor(model_triangulated_points[:3, bundle_triangulated_point_idxs], dtype=torch.float32)
        bundle_image_points = torch.tensor(triangulated_image_points[:, :, bundle_triangulated_point_idxs], dtype=torch.float32)
        bundle_image_points[~torch.isfinite(bundle_image_points)] = 0
        bundle_image_points_weights = torch.tensor(triangulated_image_points_weights[:, :, bundle_triangulated_point_idxs], dtype=torch.float32)
        #threshold_target = torch.tensor(3.0 + np.log2(triangulated_point_mapping_counts[bundle_triangulated_point_idxs]), dtype=torch.float32)
        threshold_target = torch.tensor(10.0, dtype=torch.float32)
        #gamma_softplus_alpha = torch.tensor(2.0 + np.log2(triangulated_point_mapping_counts[bundle_triangulated_point_idxs]), dtype=torch.float32)
        gamma_softplus_alpha = torch.tensor(6, dtype=torch.float32)

        def calc_projected_image_points():
            model_Rs = pytorch3d.transforms.axis_angle_to_matrix(model_rvecs)
            object_points = model_Rs @ model_triangulated_points_tensor + model_tvecs

            xd = object_points[:, 0:1, :] / torch.clamp(object_points[:, 2:3, :], 1e-3, np.inf)
            yd = object_points[:, 1:2, :] / torch.clamp(object_points[:, 2:3, :], 1e-3, np.inf)

            return torch.cat([xd, yd], dim=1) * camera_fxy + camera_cxy

        def loss_fn(temperature):
            bundle_projected_points = calc_projected_image_points()
            projection_offset = bundle_projected_points - bundle_image_points
            projection_err = torch.stack([torch.sum(projection_offset * bundle_image_points_weights[:, 2:4, :], dim=1),
                                          torch.sum(projection_offset * bundle_image_points_weights[:, 4:, :], dim=1)], dim=1)
            if False:
                projection_err_losses = projection_err2
            elif False:
                # softplus curved region spans from approximately -4.5 to 0.5
                # starts from around -4.5 with a span of 5.0
                projection_err_losses = torch.nn.functional.softplus(torch.sqrt(projection_err2) / 3.0 * 5.0 - 4.5)
            else:
                # Heavy tailed bell shaped function based on Student's t-distribution with v = 0
                # y=1/sqrt(1+x**2) has a knee point at x ~ 3.0 beyond which it slowly converges to 0, e.g. [1.0, 0.71], [3.0, 0.32], [9.9, 0.1]
                # d(1-y)/dx (the loss function's virtual spring tension) rises rapidly from zero, peaks at x ~ 0.71 and then tends to -1/x**2,
                # e.g. [0, 0], [0.71, 0.38], [3, 0.1], [9.9, 0.01], so the virtual spring is elastic x < 0.71 beyond which it becomes plastic.
                # Note that when x is scaled (e.g. by the threshold value) then the gradients d(1-y)/dx are naturally inversely scaled.
                # The gradients could be made invariant to x-scaling by using y=s/sqrt(1+(x/s)**2), but since that just scales the loss magnitude
                # that would not make any difference to gradient descent based optimisation. We also have to be aware that Adam's RMSProp and Momentum
                # states can be adversely disrupted by abrupt changes to the loss function.
                threshold = threshold_target * (1 - temperature) + 100.0 * temperature
                #projection_err_losses = 1 - 1 / torch.sqrt(1 + torch.pow(projection_err * 3.0 / threshold, 2))
                projection_err_losses = gamma_softplus(projection_err, threshold=threshold, alpha=gamma_softplus_alpha, relative_outer_gradient=0.01)

            return torch.sum(projection_err_losses * bundle_image_points_weights[:, :2, :])

        lr = 0.005
        num_steps = 1000
        convergence_criterion = {'rtol': 1e-4, 'window_size': 100, 'min_num_steps': 200}
        optimiser = torch.optim.Adam([model_rvecs, model_tvecs], lr=lr)
        lr_lambda = lambda epoch: (np.sin(min((epoch + 1) / convergence_criterion['min_num_steps'], 0.5) * np.pi)
                                   * np.power(0.1, max(epoch - 0.5 * convergence_criterion['min_num_steps'], 0) / num_steps))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lr_lambda)
        losses = []
        for optim_step in range(num_steps):
            optimiser.zero_grad()
            temperature = (1 + np.cos(np.pi * min(optim_step / convergence_criterion['min_num_steps'], 1))) / 2
            loss = loss_fn(temperature)
            loss.backward()
            optimiser.step()
            scheduler.step()
            losses.append(loss.numpy(force=True))
            loss_window_std = np.std(losses[-convergence_criterion['window_size']:])
            if optim_step % 100 == 0:
                print('optim_step', optim_step, 'loss', losses[-1], 'loss_window_std', loss_window_std, 'learning rate', scheduler.get_last_lr()[0])
            if optim_step >= convergence_criterion['min_num_steps'] and loss_window_std < convergence_criterion['rtol'] * losses[-1]:
                break

        print('len(losses)', len(losses))
        print('initial, first, final loss', losses[0], losses[convergence_criterion['min_num_steps']], losses[-1])
        print('highest, lowest loss', np.max(losses[convergence_criterion['min_num_steps']:]), np.min(losses[convergence_criterion['min_num_steps']:]))

        aux_frame_indices.append(img_indices)

        # Add the index array mapping current image points to triangulated points
        image_to_triangulated_point_idxs = np.full((len(xsn),), fill_value=-1, dtype=int)
        image_to_triangulated_point_idxs[intersected_corres_idxs] = prev_image_to_triangulated_point_idxs[intersected_corres_idxs]
        aux_frame_image_triangulated_point_idxs.append(image_to_triangulated_point_idxs)

        for rvec, tvec in zip(model_rvecs.numpy(force=True), model_tvecs.numpy(force=True)):
            R, jacobian = cv2.Rodrigues(rvec)
            t = tvec
            aux_frame_camera_extrinsics.append(np.block([[R, t], [0, 0, 0, 1]]))

    # %%

    inter_key_aux_frames = {ref_frame: [aux_frame for aux_frame in aux_frames if aux_frame in aux_frame_indices]
                            for ref_frame, aux_frames in selected_inter_key_aux_frames.items()}

    output_path = output_dirpath / 'positioned_inter_key_aux_frames.pickle'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as pickle_file:
        pickle.dump({'inter_key_aux_frames': inter_key_aux_frames,
                     'aux_frame_indices': aux_frame_indices,
                     'aux_frame_image_triangulated_point_idxs': aux_frame_image_triangulated_point_idxs,
                     'aux_frame_camera_extrinsics': aux_frame_camera_extrinsics},
                    pickle_file)

    # %%

    frame_ordinals = {}
    for ref_img_indices in key_frame_indices:
        frame_ordinals[ref_img_indices] = len(frame_ordinals)
        if ref_img_indices in frame_data_map:
            for img_indices, _, _, _ in frame_data_map[ref_img_indices]:
                if img_indices in aux_frame_indices:
                    frame_ordinals[img_indices] = len(frame_ordinals)

    camera_extrinsics_map = dict(list(zip(key_frame_indices, camera_extrinsics))
                                 + list(zip(aux_frame_indices, aux_frame_camera_extrinsics)))
    frame_camera_extrinsics = [camera_extrinsics_map[frame_indices] for frame_indices in frame_ordinals]

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

    for prev_camera_extrinsic, camera_extrinsic, frame_indices in zip([None] + frame_camera_extrinsics[:-1], frame_camera_extrinsics, frame_ordinals):
        # The extrinsic matrix transforms from world coordinates to camera coordinates
        camera_lines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=image_size[0], view_height_px=image_size[1],
                                                                        intrinsic=camera_matrix,
                                                                        extrinsic=camera_extrinsic)
        if frame_indices in key_frame_indices:
            camera_lines.paint_uniform_color([0, 0.5, 1])
        else:
            camera_lines.paint_uniform_color([0.5, 0.75, 0])
        if prev_camera_extrinsic is not None:
            camera_lines.points = o3d.utility.Vector3dVector(np.vstack([camera_lines.points,
                                                                        np.linalg.inv(prev_camera_extrinsic)[:3, 3],
                                                                        np.linalg.inv(camera_extrinsic)[:3, 3]]))
            camera_lines.lines = o3d.utility.Vector2iVector(np.vstack([camera_lines.lines,
                                                                       [len(camera_lines.points) - 2, len(camera_lines.points) - 1]]))
            camera_lines.colors = o3d.utility.Vector3dVector(np.vstack([camera_lines.colors,
                                                                        [1, 0, 0.5]]))
        vis.add_geometry(camera_lines, reset_bounding_box=False)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(model_triangulated_points[:3, post_optimise_triangulated_idxs_mask].T))
    pcd.colors = o3d.utility.Vector3dVector(model_triangulated_points[3:, post_optimise_triangulated_idxs_mask].T / 255)
    vis.add_geometry(pcd.crop(o3d.geometry.AxisAlignedBoundingBox([-100, -100, -100], [100, 100, 100])), reset_bounding_box=False)

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

    post_to_pipeline_server((f'{working_subdir} / {output_dirpath.name}', None))
