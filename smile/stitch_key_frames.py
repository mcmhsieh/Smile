"""
Stitch key frames by estimating and optimising their camera extrinsics
jointly with triangulated points in a sequence of iterative bundle adjustment
steps, incorporating geometric transformation image alignment for loop closure.

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
import collections

import numpy as np
import scipy
import sklearn.cluster
import skimage
import cv2
import open3d as o3d
import shapely
import trimesh
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

from image_filtering import img_to_normed_gray, nan_gaussian_filter, normalise_img_intensities, calc_image_point_weights
from weighting_functions import cauchy, gamma_softplus
from fig_paging import setup_new_fig_page, stash_fig_page

from pipeline_server import start_pipeline_server, post_to_pipeline_server, get_queue_from_pipeline_server


"""
def print(*args, **kwargs):
    builtins.print(*args, **kwargs)
    if 'file' not in kwargs:
        with open('log.txt', 'a+') as f:
            builtins.print(*args, file=f, **kwargs)
"""


if __name__ == '__main__':

    torch.set_default_device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    working_subdir_config_path = pathlib.Path(r'../pipeline-workspace/working_subdir.txt')
    with open(working_subdir_config_path, 'r') as config_file:
        working_subdir = config_file.read().rstrip('\n')

    workspace_dirpath = pathlib.Path(r'../pipeline-workspace') / working_subdir
    image_source_dirpath = workspace_dirpath / 'calc_sequential_flow_and_blur'
    input_source_dirpath = workspace_dirpath / 'select_key_frames'
    output_dirpath = workspace_dirpath / 'stitch_key_frames'

    exec_mode = ['all_key_frames', 'next_key_frame', 'debug_key_frame'][0]
    debug_key_frame_idx = 34

    if exec_mode == 'all_key_frames' and output_dirpath.exists():
        shutil.rmtree(output_dirpath)

    if exec_mode in ['all_key_frames', 'next_key_frame']:
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

    key_frames_filepaths = sorted(input_source_dirpath.glob('*.flow.key.pickle'))

    key_frame_indices = []
    key_frame_images = []
    key_frame_masks = []
    key_frame_motion_blurs = []
    key_frame_flow_vectors = {}

    for current_frame_idx, input_filepath in enumerate(key_frames_filepaths, start=1):

        with open(input_filepath, 'rb') as pickle_file:
            key_frame_data = pickle.load(pickle_file)

        ref_img_indices, img_indices, camera_matrix, flow_displacement, ref_motion_blur, motion_blur, (xfp, yfp, xfn, yfn) = [
            key_frame_data[name]
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

        if len(key_frame_indices) == 0:
            key_frame_indices.append(ref_img_indices)
            key_frame_images.append((ref_img, img_to_normed_gray(ref_img)))
            key_frame_masks.append(ref_img_mask)
            key_frame_motion_blurs.append(ref_motion_blur)

        key_frame_indices.append(img_indices)
        key_frame_images.append((img, img_to_normed_gray(img)))
        key_frame_masks.append(img_mask)
        key_frame_motion_blurs.append(motion_blur)
        key_frame_flow_vectors[current_frame_idx] = (xfp, yfp, xfn, yfn)

    image_sizes = set([img.shape[1::-1] for img, gray in key_frame_images])
    assert len(image_sizes) == 1
    image_size = image_sizes.pop()

    # %%

    key_frame_image_point_weights = []

    for frame_idx, (img, gray) in enumerate(key_frame_images):
        (primary_eig_vals, secondary_eig_vals,
         primary_eig_weights, secondary_eig_weights,
         primary_eig_vecs, secondary_eig_vecs) = calc_image_point_weights(img)

        key_frame_image_point_weights.append((primary_eig_weights, secondary_eig_weights, primary_eig_vecs))

        if False:
            h, w = img.shape[:2]
            xys_extent = (-0.5, w - 0.5, h - 0.5, -0.5)
            yxs = np.mgrid[0:h, 0:w]

            plt.figure('eigen val & vec image point weights', figsize=(24, 12))
            setup_new_fig_page()
            plt.suptitle(f'frame idx: {frame_idx}')
            ax = plt.subplot(3, 3, 2)
            plt.imshow(img)
            plt.title('filtered image')
            ax = plt.subplot(3, 3, 3, sharex=ax, sharey=ax)
            plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
            plt.title('normalised filtered grayscale image')
            ax = plt.subplot(3, 3, 4, sharex=ax, sharey=ax)
            plt.imshow(primary_eig_vals, extent=xys_extent)
            plt.title('primary_eig_vals')
            ax = plt.subplot(3, 3, 5, sharex=ax, sharey=ax)
            plt.imshow(primary_eig_weights, vmin=0, vmax=1, extent=xys_extent)
            plt.title('primary_eig_weights')
            ax = plt.subplot(3, 3, 6, sharex=ax, sharey=ax)
            plt.quiver(yxs[1, 4::8, 4::8], yxs[0, 4::8, 4::8],
                       primary_eig_vecs[4::8, 4::8, 0] * primary_eig_weights[4::8, 4::8], primary_eig_vecs[4::8, 4::8, 1] * primary_eig_weights[4::8, 4::8],
                       angles='xy', scale_units='xy', scale=1/8, pivot='middle',
                       headwidth=0, headlength=0, headaxislength=0)
            ax.set_aspect(aspect='equal')
            plt.title('weighted primary_eig_vecs')
            ax = plt.subplot(3, 3, 7, sharex=ax, sharey=ax)
            plt.imshow(secondary_eig_vals, extent=xys_extent)
            plt.title('secondary_eig_vals')
            ax = plt.subplot(3, 3, 8, sharex=ax, sharey=ax)
            plt.imshow(secondary_eig_weights, vmin=0, vmax=1, extent=xys_extent)
            plt.title('secondary_eig_weights')
            ax = plt.subplot(3, 3, 9, sharex=ax, sharey=ax)
            plt.quiver(yxs[1, 4::8, 4::8], yxs[0, 4::8, 4::8],
                       secondary_eig_vecs[4::8, 4::8, 0] * secondary_eig_weights[4::8, 4::8], secondary_eig_vecs[4::8, 4::8, 1] * secondary_eig_weights[4::8, 4::8],
                       angles='xy', scale_units='xy', scale=1/8, pivot='middle',
                       headwidth=0, headlength=0, headaxislength=0)
            ax.set_aspect(aspect='equal')
            plt.title('weighted secondary_eig_vecs')
            """
            ax = plt.subplot(3, 3, 7, sharex=ax, sharey=ax)
            plt.imshow(duniformity_dx, extent=xys_extent)
            plt.title('duniformity_dx')
            ax = plt.subplot(3, 3, 8, sharex=ax, sharey=ax)
            plt.imshow(duniformity_dy, extent=xys_extent)
            plt.title('duniformity_dy')
            ax = plt.subplot(3, 3, 9, sharex=ax, sharey=ax)
            plt.imshow(eig_vec_curvature, extent=xys_extent)
            plt.title('eig_vec_curvature')
            ax = plt.subplot(3, 3, 10, sharex=ax, sharey=ax)
            plt.imshow(eig_vals_magn * eig_vec_curvature, extent=xys_extent)
            plt.title('eig_vals_magn * eig_vec_curvature')
            ax = plt.subplot(3, 3, 11, sharex=ax, sharey=ax)
            plt.imshow(image_eigv_weights_down, extent=xys_extent)
            plt.title('image_eigv_weights_down')
            ax = plt.subplot(3, 3, 12, sharex=ax, sharey=ax)
            plt.imshow(image_eigv_weights)
            plt.title('image_eigv_weights')
            """
            plt.tight_layout()
            stash_fig_page()

    # %%

    # 3D triangulated points and rgb values, stacked from each stitched frame pair
    #  - in the absolute origin frame of reference when persisted
    #  - transformed into the current / next camera frame of reference when stitching
    # ndarray that is overwritten and mutates
    stitched_triangulated_points = None
    # Weights of each 3D triangulated point calculated from the projection errors for associated image points
    # ndarray that is overwritten
    triangulated_idxs_weights = None
    # 2D image sample points used with correspondences in previous or next frames for triangulating points
    # only append tuple of non-mutating ndarrays
    key_frame_image_sample_points = []
    # Linked indices between 3D triangulated points and 2D image sample points for each frame
    # append ndarray, non-mutating except for the last two (current and previous) frames
    key_frame_image_triangulated_point_idxs = []
    # Camera extrinsics for each stitched frame
    # mutating list of non-mutating ndarray
    key_frame_camera_extrinsics = []
    # Key frame cross stitch disparity confidence maps
    # only insert tuple of non-mutating ndarrays and immutable values
    cross_stitch_disparity_confidence_maps = {}

    processed_frame_idxs = []

    for current_frame_idx, input_filepath in enumerate(key_frames_filepaths, start=1):

        frame_time, frame_index, _, frame_type, filename_ext = input_filepath.name.split('.')
        frame_time = datetime.datetime.strptime(frame_time, '%Y%m%d-%H%M%S%f')
        frame_index = int(frame_index)

        filename = f'{current_frame_idx:04d} {frame_time.strftime("%Y%m%d-%H%M%S%f")}.{frame_index:03d}.stitch.pickle'
        stitched_key_frame_path = output_dirpath / filename
        if ((exec_mode == 'next_key_frame' and stitched_key_frame_path.exists())
            or (exec_mode == 'debug_key_frame' and current_frame_idx < debug_key_frame_idx)):
            with open(stitched_key_frame_path, 'rb') as pickle_file:
                data = pickle.load(pickle_file)
                stitched_triangulated_points = data['stitched_triangulated_points']
                triangulated_idxs_weights = data['triangulated_idxs_weights']
                key_frame_image_sample_points.extend(data['key_frame_image_sample_points'])
                del key_frame_image_triangulated_point_idxs[current_frame_idx-1:]
                key_frame_image_triangulated_point_idxs.extend(data['key_frame_image_triangulated_point_idxs'])
                key_frame_camera_extrinsics = data['key_frame_camera_extrinsics']
                cross_stitch_disparity_confidence_maps.update(data['cross_stitch_disparity_confidence_maps'])
            continue

        print('-' * 80)
        print('frame', current_frame_idx, frame_index, frame_time)

        ref_img, ref_gray = key_frame_images[current_frame_idx - 1]
        ref_img_mask = key_frame_masks[current_frame_idx - 1]
        img, gray = key_frame_images[current_frame_idx]
        img_mask = key_frame_masks[current_frame_idx]
        #motion_blur = np.max(key_frame_motion_blurs[current_frame_idx-1:current_frame_idx+1])
        (xfp, yfp, xfn, yfn) = key_frame_flow_vectors[current_frame_idx]

        xyfn_flow = np.full(img.shape[:2] + (2,), fill_value=np.nan, dtype=np.float32)
        xyfn_flow[yfp, xfp] = np.vstack([xfn, yfn]).T

        # Mask out optical flow in specular reflection regions
        xyfn_flow_masked = np.array(xyfn_flow)
        xyfn_flow_masked[ref_img_mask, :] = np.nan

        img_mask_warp = cv2.remap(img_mask.astype(np.uint8), xyfn_flow, None, cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=1).astype(bool)
        xyfn_flow_masked[img_mask_warp, :] = np.nan

        grid_step = 8
        h, w = img.shape[:2]
        ysg, xsg = np.mgrid[grid_step/2:h:grid_step, grid_step/2:w:grid_step].reshape((2, -1)).astype(np.int32)

        if len(key_frame_image_sample_points) == 0:
            key_frame_image_sample_points.append((xsg, ysg))

        # Subsampled point correspondences between the previous image the current / next image,
        # for feeding (after residual optical flow adjustment) to cv2.findEssentialMat() and cv2.recoverPose()
        # to estimate camera motion.
        interp = scipy.interpolate.RegularGridInterpolator((np.arange(xyfn_flow_masked.shape[0]), np.arange(xyfn_flow_masked.shape[1])),
                                                           xyfn_flow_masked[:, :, 0] + xyfn_flow_masked[:, :, 1] * 1j,
                                                           method='linear', bounds_error=False, fill_value=np.nan)

        xsp, ysp = key_frame_image_sample_points[current_frame_idx - 1]
        xysn = interp((ysp, xsp)).astype(np.complex64)
        xsn, ysn = xysn.real, xysn.imag
        # Mask out a margin around the edge of the image where the flow estimation is less reliable
        # TODO: check for correspondence between optical flow estimates and disparity based flow estimates, so that
        # points in the margin can still be triangulated
        margin = 5
        mask = (np.isfinite(xsp) & np.isfinite(ysp) & np.isfinite(xsn) & np.isfinite(ysn)
                & (xsp >= margin) & (xsp < img.shape[1] - margin) & (ysp >= margin) & (ysp < img.shape[0] - margin)
                & (xsn >= margin) & (xsn < img.shape[1] - margin) & (ysn >= margin) & (ysn < img.shape[0] - margin))
        corres_idxs = np.where(mask)[0]

        print('num corresponding flow points', len(corres_idxs))

        if len(key_frame_camera_extrinsics) == 0:

            corres_pts_p = np.vstack([xsp, ysp])[:, corres_idxs].T.astype(np.float32)
            corres_pts_n = np.vstack([xsn, ysn])[:, corres_idxs].T.astype(np.float32)

            results = []
            for iter_idx in range(1000):
                #F, mask = cv2.findFundamentalMat(points1=corres_pts_p, points2=corres_pts_n,
                #                                 method=cv2.FM_RANSAC, ransacReprojThreshold=3.0, confidence=0.99, maxIters=1000)

                # cv2 RNG seed doesn't appear to affect cv2.findEssentialMat() RANSAC
                #cv2.setRNGSeed(iter_idx)

                np.random.seed(iter_idx)
                sample_fraction = np.random.uniform(1 / 3, 3 / 4)
                sample_size = np.clip(int(np.ceil(corres_pts_p.shape[0] * sample_fraction)), 50, corres_pts_p.shape[0])
                pts_idxs = np.random.choice(np.arange(corres_pts_p.shape[0]), size=(sample_size,), replace=False)

                #threshold = min(0.5 * (1 + motion_blur), 2.0)
                threshold = 3.0
                E, mask = cv2.findEssentialMat(corres_pts_p[pts_idxs, :], corres_pts_n[pts_idxs, :], camera_matrix,
                                               method=cv2.RANSAC, prob=0.999, threshold=threshold, maxIters=10)
                assert np.all(np.isfinite(E))
                #print('num points from cv2.findEssentialMat()', np.sum(mask))
                #print(E)

                R1, R2, t = cv2.decomposeEssentialMat(E)
                assert np.all(np.isfinite(t))
                #print('t', t.flatten())

                results.append((E, t))

                if len(results) >= 100:
                    break

            """
            plt.figure(figsize=(16, 10))
            ax = plt.subplot(1, 1, 1, projection='3d')
            ax.scatter(*np.hstack([t for E, t in results]), s=2, c='b', alpha=0.2)
            ax.set_aspect('equal', adjustable='datalim')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.tight_layout()
            """

            dbscan = sklearn.cluster.DBSCAN(eps=0.05, min_samples=5)
            dbscan.fit(np.hstack([np.hstack([t, -t]) for E, t in results]).T)
            dbscan_labels = sorted(set(dbscan.labels_) - set([-1]))
            print('dbscan_labels', dbscan_labels)
            cluster_centroids = []
            for label in dbscan_labels:
                core_idxs = [idx // 2 for idx in set(np.where(dbscan.labels_ == label)[0]) & set(dbscan.core_sample_indices_)
                             if idx % 2 == 0]
                if len(core_idxs) > 0:
                    cluster_centroids.append(np.mean(np.hstack([t for E, t in results]).T[core_idxs, :], axis=0))
            else:
                cluster_centroids.append(np.mean(np.hstack([t for E, t in results]).T, axis=0))

            # TODO: sample from pairs from previous and next sets of cluster core samples and find the combination
            # resulting in the most reprojected triangulated point inliers

            # Select the Essential Matrix E which has a decomposed translation vector t closest
            # to the translation vector cluster centroid with the smallest Z component
            cluster_centroids = np.array(sorted(cluster_centroids, key=lambda t: np.abs(t[2])))
            print('cluster_centroids', cluster_centroids)
            results.sort(key=lambda item: np.abs(np.sum(cluster_centroids[0, :] * item[1].flatten())), reverse=True)
            E, t = results[0]
            print('selected unit length t', t.flatten())

            # F_xysp are the epipolar lines upon which the corresponding xysn points should lie
            F_xysp = np.linalg.inv(camera_matrix).T @ E @ np.linalg.inv(camera_matrix) @ np.vstack([corres_pts_p.T, np.ones((corres_pts_p.shape[0],))])
            xysn = np.vstack([corres_pts_n.T, np.ones((corres_pts_n.shape[0],))])
            # Normalise the lines
            # https://math.stackexchange.com/questions/4585131/understanding-distance-from-point-to-line-in-homogeneous-coordinates
            # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/BEARDSLEY/node2.html
            F_xysp = F_xysp[:3, :] / np.clip(np.linalg.norm(F_xysp[:2, :], axis=0), 1e-8, np.inf)

            #threshold = min(0.5 * (1 + motion_blur), 2.0)
            threshold = 6.0
            inlier_idxs = np.where(np.abs(np.sum(xysn * F_xysp, axis=0)) < threshold)[0]

            # Recovers the relative camera rotation and the translation from corresponding points in two images from two different cameras,
            # using cheirality check.
            #retval = cv2.recoverPose(corres_pts_p, corres_pts_n, camera_matrix, np.zeros((4,)), camera_matrix, np.zeros((4,)),
            #                         method=cv2.RANSAC, prob=0.999, threshold=2.0)
            #print(retval)
            print('num points to cv2.recoverPose()', len(inlier_idxs))
            retval, R, t, mask, triangulatedPoints = cv2.recoverPose(E, corres_pts_p[inlier_idxs, :], corres_pts_n[inlier_idxs, :],
                                                                     camera_matrix, distanceThresh=1e3, mask=None)
            print('num points from cv2.recoverPose()', retval)

            rvec, jacobian = cv2.Rodrigues(R)
            model_rvec = torch.tensor(rvec.T, dtype=torch.float32, requires_grad=True)
            model_tvec = torch.tensor(t, dtype=torch.float32, requires_grad=True)

            inv_camera_matrix = torch.tensor(np.linalg.inv(camera_matrix), dtype=torch.float32)
            xysp = torch.tensor(np.vstack([corres_pts_p.T, np.ones((corres_pts_p.shape[0],))]), dtype=torch.float32)
            xysn = torch.tensor(np.vstack([corres_pts_n.T, np.ones((corres_pts_n.shape[0],))]), dtype=torch.float32)

            primary_eig_weights, secondary_eig_weights, primary_eig_vecs = key_frame_image_point_weights[current_frame_idx]
            eig_weights = primary_eig_weights + secondary_eig_weights * 1j
            interp = scipy.interpolate.RegularGridInterpolator((np.arange(eig_weights.shape[0]), np.arange(eig_weights.shape[1])),
                                                               eig_weights,
                                                               method='linear', bounds_error=True)
            eig_weights = interp((corres_pts_n[:, 1], corres_pts_n[:, 0])).astype(np.complex64)
            eig_vecs = primary_eig_vecs[:, :, 0] + primary_eig_vecs[:, :, 1] * 1j
            interp = scipy.interpolate.RegularGridInterpolator((np.arange(eig_vecs.shape[0]), np.arange(eig_vecs.shape[1])),
                                                               eig_vecs,
                                                               method='linear', bounds_error=True)
            eig_vecs = interp((corres_pts_n[:, 1], corres_pts_n[:, 0])).astype(np.complex64)
            xysn_weights = torch.tensor(np.vstack([eig_weights.real, eig_weights.imag,
                                                   eig_vecs.real, eig_vecs.imag,
                                                   eig_vecs.imag, -eig_vecs.real]), dtype=torch.float32)

            threshold_target = torch.tensor(3.0, dtype=torch.float32)
            gamma_softplus_alpha = torch.tensor(2, dtype=torch.float32)

            def loss_fn(temperature):
                R = pytorch3d.transforms.axis_angle_to_matrix(model_rvec)[0, :, :]
                model_tvec_normed = model_tvec / torch.clamp(torch.norm(model_tvec), 1e-6, np.inf)
                E = R @ torch.linalg.cross(torch.eye(3, dtype=torch.float32), torch.cat([model_tvec_normed] * 3, dim=1).T)

                # F_xysp are the epipolar lines upon which the corresponding xysn points should lie
                F_xysp = inv_camera_matrix.T @ E @ inv_camera_matrix @ xysp
                # Normalise the lines
                # https://math.stackexchange.com/questions/4585131/understanding-distance-from-point-to-line-in-homogeneous-coordinates
                # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/BEARDSLEY/node2.html
                F_xysp = F_xysp[:3, :] / torch.clamp(torch.norm(F_xysp[:2, :], dim=0), 1e-8, np.inf)

                # F_xysp[:3, :] represents line equations of the form ax + by + c = 0
                # since dy/dx = -a / b
                # and after F_xysp[:3, :] has been normalised
                # the line vectors are [b, -a]
                # the line normal vectors are [a, b], i.e. F_xysp[:2, :]
                epipolar_line_offsets = torch.sum(xysn * F_xysp, dim=0) * F_xysp[:2, :]
                epipolar_line_distances = torch.stack([torch.sum(epipolar_line_offsets * xysn_weights[2:4, :], dim=0),
                                                       torch.sum(epipolar_line_offsets * xysn_weights[4:, :], dim=0)])

                # Heavy tailed bell shaped function based on Student's t-distribution with v = 1 (i.e. Cauchy distribution)
                # y=1/(1+x**2) has a knee point at x ~ 3.0 beyond which it slowly converges to 0, e.g. [1.0, 0.5], [3.0, 0.1], [9.9, 0.01]
                # d(1-y)/dx (the loss function's virtual spring tension) rises rapidly from zero, peaks at x ~ 0.58 and then tends to -2/x**3,
                # e.g. [0, 0], [0.58, 0.65], [3, 0.06], [9.9, 0.002], so the virtual spring is elastic x < 0.58 beyond which it becomes plastic.
                # Note that when x is scaled (e.g. by the threshold value) then the gradients d(1-y)/dx are naturally inversely scaled.
                # The gradients could be made invariant to x-scaling by using y=s/(1+(x/s)**2), but since that just scales the loss magnitude
                # that would not make any difference to gradient descent based optimisation. We also have to be aware that Adam's RMSProp and Momentum
                # states can be adversely disrupted by abrupt changes to the loss function.
                #threshold = min(0.5 * (1 + motion_blur), 2.0)
                threshold = threshold_target * (1 - temperature) + 6.0 * temperature
                #epipolar_line_losses = 1 - 1 / (1 + torch.pow(epipolar_line_distances / threshold * 3.0, 2))
                epipolar_line_losses = gamma_softplus(epipolar_line_distances, threshold=threshold, alpha=gamma_softplus_alpha, relative_outer_gradient=0.01)

                return torch.sum(epipolar_line_losses * xysn_weights[:2, :])

            lr = 0.005
            num_steps = 3000
            convergence_criterion = {'rtol': 1e-4, 'window_size': 100, 'min_num_steps': 300}
            optimiser = torch.optim.Adam([model_rvec, model_tvec], lr=lr)
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
                    print('optim_step', optim_step, 'loss', losses[-1], 'loss_window_std', loss_window_std,
                          'learning rate', np.round(scheduler.get_last_lr()[0], 6), 'temperature', np.round(temperature, 3))
                if optim_step >= convergence_criterion['min_num_steps'] and loss_window_std < convergence_criterion['rtol'] * losses[-1]:
                    break

            print('len(losses)', len(losses))
            print('initial, first, final loss', losses[0], losses[convergence_criterion['min_num_steps']], losses[-1])
            print('highest, lowest loss', np.max(losses[convergence_criterion['min_num_steps']:]), np.min(losses[convergence_criterion['min_num_steps']:]))

            rvec, tvec = model_rvec.numpy(force=True), model_tvec.numpy(force=True)
            R, jacobian = cv2.Rodrigues(rvec)
            t = tvec / np.linalg.norm(tvec)

            print('optimised unit length t', t.flatten())

            camera_transform = np.block([[R, t], [0, 0, 0, 1]])

            E = R @ np.cross(np.identity(3), t.T)

            # F_xysp are the epipolar lines upon which the corresponding xysn points should lie
            F_xysp = np.linalg.inv(camera_matrix).T @ E @ np.linalg.inv(camera_matrix) @ np.vstack([corres_pts_p.T, np.ones((corres_pts_p.shape[0],))])
            xysn = np.vstack([corres_pts_n.T, np.ones((corres_pts_n.shape[0],))])
            # Normalise the lines
            # https://math.stackexchange.com/questions/4585131/understanding-distance-from-point-to-line-in-homogeneous-coordinates
            # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/BEARDSLEY/node2.html
            F_xysp = F_xysp[:3, :] / np.clip(np.linalg.norm(F_xysp[:2, :], axis=0), 1e-8, np.inf)

            #threshold = min(0.5 * (1 + motion_blur), 2.0)
            threshold = 6.0
            inlier_idxs = np.where(np.abs(np.sum(xysn * F_xysp, axis=0)) < threshold)[0]
            print('num inlier points for triangulation', len(inlier_idxs))

            prev_proj_mat = camera_matrix @ np.block([np.identity(3), np.zeros((3, 1))])
            next_proj_mat = camera_matrix @ camera_transform[:3, :]
            triangulatedPoints = cv2.triangulatePoints(prev_proj_mat, next_proj_mat,
                                                       corres_pts_p[inlier_idxs, :].reshape((-1, 1, 2)),
                                                       corres_pts_n[inlier_idxs, :].reshape((-1, 1, 2)))
            new_triangulated_points = cv2.convertPointsFromHomogeneous(triangulatedPoints.T).reshape((-1, 3))

            # Estimate the scaling factor from the depth distribution of the triangulated points
            # assuming the camera is positioned at a distance of ~8mm from the target
            c = np.percentile(new_triangulated_points[:, 2], 25) / 8.0
            t = t / c
            camera_transform = np.block([[R, t], [0, 0, 0, 1]])
            new_triangulated_points = new_triangulated_points / c

            if False:
                prev_proj_mat = camera_matrix @ np.block([np.identity(3), np.zeros((3, 1))])
                next_proj_mat = camera_matrix @ camera_transform[:3, :]
                triangulatedPoints_check = cv2.triangulatePoints(prev_proj_mat, next_proj_mat,
                                                                 corres_pts_p[inlier_idxs, :].reshape((-1, 1, 2)),
                                                                 corres_pts_n[inlier_idxs, :].reshape((-1, 1, 2)))
                triangulated_points_check = cv2.convertPointsFromHomogeneous(triangulatedPoints_check.T).reshape((-1, 3))
                assert np.allclose(triangulated_points_check, new_triangulated_points, atol=1e-4)

            triangulated_z_bound = 100
            mask = new_triangulated_points[:, 2] > triangulated_z_bound
            new_triangulated_points[mask, :] *= triangulated_z_bound / new_triangulated_points[mask, 2:]

            view_frustum_mask = new_triangulated_points[:, 2] > 0
            new_triangulated_points = new_triangulated_points[view_frustum_mask, :]
            inlier_idxs = inlier_idxs[view_frustum_mask]

            # Set the initial array of triangulated points and RGB values
            triangulated_rgb = ref_img[ysp[corres_idxs[inlier_idxs]], xsp[corres_idxs[inlier_idxs]], :]
            stitched_triangulated_points = np.hstack([new_triangulated_points, triangulated_rgb])
            # Transform the triangulated points into the current / next camera frame of reference
            stitched_triangulated_points[:, :3] = (R @ stitched_triangulated_points[:, :3].T + t).T

            # Find grid points to extend the frame's set of sample points
            mask = np.isfinite(xsn) & np.isfinite(ysn)
            xsn_i = np.round(xsn[mask]).astype(np.int32)
            ysn_i = np.round(ysn[mask]).astype(np.int32)
            mask = (xsn_i >= 0) & (xsn_i < img.shape[1]) & (ysn_i >= 0) & (ysn_i < img.shape[0])
            xsn_i, ysn_i = xsn_i[mask], ysn_i[mask]
            neighbours = np.zeros(img.shape[:2], dtype=np.uint8)
            neighbours[ysn_i, xsn_i] = 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grid_step * 2 - 1, grid_step * 2 - 1))
            neighbours = cv2.dilate(neighbours, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            mask = neighbours[ysg, xsg] == 0
            xsne, ysne = xsg[mask], ysg[mask]

            # Add the set of extended image sample points to be used for the next pair of frames
            key_frame_image_sample_points.append((np.hstack([xsn, xsne], dtype=np.float32), np.hstack([ysn, ysne], dtype=np.float32)))

            # Add the index array mapping previous image points to triangulated points
            image_to_triangulated_point_idxs = np.full((len(xsp),), fill_value=-1, dtype=int)
            image_to_triangulated_point_idxs[corres_idxs[inlier_idxs]] = np.arange(len(inlier_idxs))
            key_frame_image_triangulated_point_idxs.append(image_to_triangulated_point_idxs)

            # Add the index array mapping current image points to triangulated points
            image_to_triangulated_point_idxs = np.full((len(xsn) + len(xsne),), fill_value=-1, dtype=int)
            image_to_triangulated_point_idxs[corres_idxs[inlier_idxs]] = np.arange(len(inlier_idxs))
            key_frame_image_triangulated_point_idxs.append(image_to_triangulated_point_idxs)

            key_frame_camera_extrinsics.append(np.identity(4))
            key_frame_camera_extrinsics.append(camera_transform)

        else:

            # Transform stitched_triangulated_points from the absolute origin frame back
            # to the last camera frame of reference
            last_camera_extrinsic = key_frame_camera_extrinsics[current_frame_idx - 1]
            R = last_camera_extrinsic[:3, :3]
            t = last_camera_extrinsic[:3, 3:]
            stitched_triangulated_points[:, :3] = (R @ stitched_triangulated_points[:, :3].T + t).T

            prev_image_to_triangulated_point_idxs = key_frame_image_triangulated_point_idxs[current_frame_idx - 1]
            prev_triangulated_idxs = np.where(prev_image_to_triangulated_point_idxs >= 0)[0]
            intersected_corres_idxs = np.array(list(set(list(prev_triangulated_idxs)) & set(list(corres_idxs))), dtype=int)
            next_image_points = np.vstack([xsn, ysn])[:, intersected_corres_idxs].T
            prev_corres_triangulated_points = stitched_triangulated_points[prev_image_to_triangulated_point_idxs[intersected_corres_idxs], :3]
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
                                                                 cameraMatrix=camera_matrix, distCoeffs=np.zeros((4,)),
                                                                 rvec=None, tvec=None, useExtrinsicGuess=False,
                                                                 iterationsCount=10, reprojectionError=threshold, confidence=0.999, inliers=None,
                                                                 flags=cv2.SOLVEPNP_ITERATIVE)
                #assert retval == True
                if retval:
                    R, jacobian = cv2.Rodrigues(rvec)
                    t = tvec

                    camera_transform = np.block([[R, t], [0, 0, 0, 1]])

                    object_points = camera_transform @ np.vstack([prev_corres_triangulated_points.T, np.ones((prev_corres_triangulated_points.shape[0],))])
                    projected_points = camera_matrix @ object_points[:3, :]
                    projected_points = projected_points[:2, :] / projected_points[2, :]

                    #threshold = min(0.5 * (1 + motion_blur), 2.0)
                    threshold = 6.0
                    triangulated_points_inliers = np.where(np.linalg.norm(next_image_points - projected_points.T, axis=1) < threshold)[0]

                    E = R @ np.cross(np.identity(3), t.T)

                    # F_xysp are the epipolar lines upon which the corresponding xysn points should lie
                    if True:
                        F_xysp = np.linalg.inv(camera_matrix).T @ E @ np.linalg.inv(camera_matrix) @ np.vstack([prev_new_triangulation_points.T, np.ones((prev_new_triangulation_points.shape[0],))])
                        xysn = np.vstack([next_new_triangulation_points.T, np.ones((next_new_triangulation_points.shape[0],))])
                    else:
                        F_xysp = np.linalg.inv(camera_matrix).T @ E @ np.linalg.inv(camera_matrix) @ np.vstack([corres_pts_p.T, np.ones((corres_pts_p.shape[0],))])
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

            # TODO: start a new fragment with new stitched_triangulated_points if the stitch has low confidence

            R, jacobian = cv2.Rodrigues(rvec)
            t = tvec
            E = R @ np.cross(np.identity(3), t.T)
            camera_transform = np.block([[R, t], [0, 0, 0, 1]])

            prev_points = np.vstack([xsp, ysp])[:, intersected_corres_idxs].T.astype(np.float32)
            next_points = np.vstack([xsn, ysn])[:, intersected_corres_idxs].T.astype(np.float32)

            # F_xysp are the epipolar lines upon which the corresponding xysn points should lie
            F_xysp = np.linalg.inv(camera_matrix).T @ E @ np.linalg.inv(camera_matrix) @ np.vstack([prev_points.T, np.ones((prev_points.shape[0],))])
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

            new_triangulation_idxs = np.array(list(set(list(corres_idxs)) - set(list(intersected_corres_idxs))), dtype=int)
            prev_points = np.vstack([xsp, ysp])[:, new_triangulation_idxs].T.astype(np.float32)
            next_points = np.vstack([xsn, ysn])[:, new_triangulation_idxs].T.astype(np.float32)

            # F_xysp are the epipolar lines upon which the corresponding xysn points should lie
            F_xysp = np.linalg.inv(camera_matrix).T @ E @ np.linalg.inv(camera_matrix) @ np.vstack([prev_points.T, np.ones((prev_points.shape[0],))])
            xysn = np.vstack([next_points.T, np.ones((next_points.shape[0],))])
            # Normalise the lines
            # https://math.stackexchange.com/questions/4585131/understanding-distance-from-point-to-line-in-homogeneous-coordinates
            # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/BEARDSLEY/node2.html
            F_xysp = F_xysp[:3, :] / np.clip(np.linalg.norm(F_xysp[:2, :], axis=0), 1e-8, np.inf)

            #threshold = min(0.5 * (1 + motion_blur), 2.0)
            threshold = 6.0
            inlier_idxs = np.where(np.abs(np.sum(xysn * F_xysp, axis=0)) < threshold)[0]

            if len(inlier_idxs) > 0:
                # Compute triangulated points using camera_transform
                prev_proj_mat = camera_matrix @ np.block([np.identity(3), np.zeros((3, 1))])
                next_proj_mat = camera_matrix @ camera_transform[:3, :]
                triangulatedPoints = cv2.triangulatePoints(prev_proj_mat, next_proj_mat,
                                                           prev_points[inlier_idxs, :].reshape((-1, 1, 2)), next_points[inlier_idxs, :].reshape((-1, 1, 2)))
                new_triangulated_points = cv2.convertPointsFromHomogeneous(triangulatedPoints.T).reshape((-1, 3))

                triangulated_z_bound = 100
                mask = new_triangulated_points[:, 2] > triangulated_z_bound
                new_triangulated_points[mask, :] *= triangulated_z_bound / new_triangulated_points[mask, 2:]

                view_frustum_mask = new_triangulated_points[:, 2] > 0
                new_triangulated_points = new_triangulated_points[view_frustum_mask, :]
                new_inlier_triangulation_idxs = new_triangulation_idxs[inlier_idxs][view_frustum_mask]
            else:
                new_triangulated_points = np.empty((0, 3))
                new_inlier_triangulation_idxs = np.empty((0,), dtype=int)

            print('num new triangulation points', len(new_triangulation_idxs))
            print('num new triangulation point inliers', len(new_inlier_triangulation_idxs))

            # Estimate normals for new triangulated points separately from existing stitched triangulated points at this stage
            # because prior to optimisation, the location of the new triangulated points relative to the existing stitched triangulated
            # points is unlikely to be accurate enough to produce reliable normal estimates if their point clouds are combined together.

            valid_mask = np.all(np.isfinite(stitched_triangulated_points[:, :3]), axis=1)
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(stitched_triangulated_points[valid_mask, :3]))
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
            pcd.transform(np.linalg.inv(key_frame_camera_extrinsics[current_frame_idx - 1]))
            pcd_points = np.array(pcd.points).T
            pcd_normals = np.array(pcd.normals).T
            normal_ray_alignment = np.zeros((np.sum(valid_mask),), dtype=np.float32)
            for image_to_triangulated_point_idxs, camera_extrinsic in zip(key_frame_image_triangulated_point_idxs,
                                                                          key_frame_camera_extrinsics):
                triangulated_image_idxs = np.where(image_to_triangulated_point_idxs >= 0)[0]
                triangulated_point_idxs = image_to_triangulated_point_idxs[triangulated_image_idxs]
                image_points_weights = np.zeros((stitched_triangulated_points.shape[0],), dtype=np.float32)
                image_points_weights[triangulated_point_idxs] = 1

                R = camera_extrinsic[:3, :3]
                t = camera_extrinsic[:3, 3:]
                object_points = R @ pcd_points + t
                object_normals = R @ pcd_normals

                object_point_ray = object_points / np.clip(np.linalg.norm(object_points, axis=0), 1e-6, np.inf)
                normal_ray_alignment += np.sum(object_point_ray * object_normals, axis=0) * image_points_weights[valid_mask]
            pcd.normals = o3d.utility.Vector3dVector((pcd_normals * -np.sign(normal_ray_alignment)).T)
            assert np.allclose(np.linalg.norm(np.array(pcd.normals), axis=1), 1)

            stitched_triangulated_normals = np.full((3, stitched_triangulated_points.shape[0]), fill_value=np.nan, dtype=np.float32)
            stitched_triangulated_normals[:, valid_mask] = np.array(pcd.normals).T

            if False:
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(new_triangulated_points))
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=64))
                pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))

                stitched_triangulated_normals = np.hstack([stitched_triangulated_normals, np.array(pcd.normals).T])
            else:
                stitched_triangulated_normals = np.hstack([stitched_triangulated_normals,
                                                           np.array([0, 0, -1])[:, None] * np.ones((new_triangulated_points.shape[0],))],
                                                          dtype=np.float32)

            # Extend the set of triangulated points and RGB values
            triangulated_rgb = ref_img[np.clip(np.round(ysp[new_inlier_triangulation_idxs]).astype(np.int32), 0, img.shape[0] - 1),
                                       np.clip(np.round(xsp[new_inlier_triangulation_idxs]).astype(np.int32), 0, img.shape[1] - 1), :]
            stitched_triangulated_points = np.vstack([stitched_triangulated_points, np.hstack([new_triangulated_points, triangulated_rgb])])
            # Transform the triangulated points into the current / next camera frame of reference
            R = camera_transform[:3, :3]
            t = camera_transform[:3, 3:]
            stitched_triangulated_points[:, :3] = (R @ stitched_triangulated_points[:, :3].T + t).T

            # Update the index array mapping previous image points to triangulated points
            image_to_triangulated_point_idxs = key_frame_image_triangulated_point_idxs[current_frame_idx - 1]
            image_to_triangulated_point_idxs[new_inlier_triangulation_idxs] = stitched_triangulated_points.shape[0] - new_triangulated_points.shape[0] + np.arange(new_triangulated_points.shape[0])

            camera_fxy = torch.tensor(np.diag(camera_matrix[:2, :2])[:, None], dtype=torch.float32)
            camera_cxy = torch.tensor(camera_matrix[:2, 2:], dtype=torch.float32)

            key_frame_camera_extrinsics.append(camera_transform @ key_frame_camera_extrinsics[current_frame_idx - 1])

            inv_camera_extrinsic = np.linalg.inv(key_frame_camera_extrinsics[current_frame_idx])
            R = inv_camera_extrinsic[:3, :3]
            t = inv_camera_extrinsic[:3, 3:]
            model_triangulated_points = R @ stitched_triangulated_points[:, :3].T + t

            # Although including all the camera extrinsics introduces redundant degrees of freedom,
            # this appears to help make the optimisation unbiased to each camera extrinsic.
            # Previously when the first frame was fixed to the origin/axes, it appeared to be susceptible
            # to higher projection errors especially for larger convergence criteria thresholds,
            # perhaps because the gradient graph for the first frame is far more convoluted.
            rvecs = []
            tvecs = []
            for camera_extrinsic in key_frame_camera_extrinsics:
                rvec, jacobian = cv2.Rodrigues(camera_extrinsic[:3, :3])
                tvec = camera_extrinsic[:3, 3:]
                rvecs.append(rvec.flatten())
                tvecs.append(tvec)
            model_rvecs = torch.tensor(np.array(rvecs), dtype=torch.float32, requires_grad=True)
            model_tvecs = torch.tensor(np.array(tvecs), dtype=torch.float32, requires_grad=True)

            # TODO: try higher confidence map threshold

            current_image_to_triangulated_point_idxs = np.full((len(xsn),), fill_value=-1, dtype=int)
            current_image_to_triangulated_point_idxs[intersected_corres_idxs] = prev_image_to_triangulated_point_idxs[intersected_corres_idxs]
            current_image_to_triangulated_point_idxs[new_inlier_triangulation_idxs] = stitched_triangulated_points.shape[0] - new_triangulated_points.shape[0] + np.arange(new_triangulated_points.shape[0])

            triangulated_image_points = []
            triangulated_image_points_weights = []
            for (xs, ys), image_to_triangulated_point_idxs, image_point_weights, camera_extrinsic in zip(key_frame_image_sample_points + [(xsn, ysn)],
                                                                                                         key_frame_image_triangulated_point_idxs + [current_image_to_triangulated_point_idxs],
                                                                                                         key_frame_image_point_weights,
                                                                                                         key_frame_camera_extrinsics):
                triangulated_image_idxs = np.where(image_to_triangulated_point_idxs >= 0)[0]
                triangulated_point_idxs = image_to_triangulated_point_idxs[triangulated_image_idxs]

                image_points = np.full((2, stitched_triangulated_points.shape[0]), fill_value=np.nan, dtype=np.float32)
                image_points[:, triangulated_point_idxs] = np.vstack([xs, ys])[:, triangulated_image_idxs]
                triangulated_image_points.append(image_points)

                image_points_weights = np.zeros((6, stitched_triangulated_points.shape[0]))

                primary_eig_weights, secondary_eig_weights, primary_eig_vecs = image_point_weights
                eig_weights = primary_eig_weights + secondary_eig_weights * 1j
                interp = scipy.interpolate.RegularGridInterpolator((np.arange(eig_weights.shape[0]), np.arange(eig_weights.shape[1])),
                                                                   eig_weights,
                                                                   method='linear', bounds_error=True)
                eig_weights = interp((image_points[1, triangulated_point_idxs], image_points[0, triangulated_point_idxs])).astype(np.complex64)
                eig_vecs = primary_eig_vecs[:, :, 0] + primary_eig_vecs[:, :, 1] * 1j
                interp = scipy.interpolate.RegularGridInterpolator((np.arange(eig_vecs.shape[0]), np.arange(eig_vecs.shape[1])),
                                                                   eig_vecs,
                                                                   method='linear', bounds_error=True)
                eig_vecs = interp((image_points[1, triangulated_point_idxs], image_points[0, triangulated_point_idxs])).astype(np.complex64)
                eig_weights = np.vstack([eig_weights.real, eig_weights.imag,
                                         eig_vecs.real, eig_vecs.imag,
                                         eig_vecs.imag, -eig_vecs.real])
                image_points_weights[:, triangulated_point_idxs] = eig_weights

                triangulated_points = camera_extrinsic[:3, :3] @ model_triangulated_points[:, triangulated_point_idxs] + camera_extrinsic[:3, 3:]
                triangulated_normals = camera_extrinsic[:3, :3] @ stitched_triangulated_normals[:, triangulated_point_idxs]
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

            # Optimise triangulated points with 2 or more mapped image points
            # Triangulated points with 2 mapped image points implicitly provide epipolar constraints between frames.
            # Triangulated points with more than 2 mapped image points additionally provide depth / scaling associations between frames.
            triangulated_point_mapping_counts = np.sum(np.all(np.isfinite(triangulated_image_points), axis=1), axis=0)
            # All triangulated points have at least one mapped image point
            assert np.all(triangulated_point_mapping_counts > 0)
            # Triangulated points are initialised with two mapped image points, but the second (or a later point) may
            # exit from view or be pruned if it is obscured. However the association with the initial image point is still retained.
            assert np.all(np.all(~np.isfinite(model_triangulated_points), axis=0) == (triangulated_point_mapping_counts == 1))
            assert np.all(np.all(np.isfinite(model_triangulated_points), axis=0) == (triangulated_point_mapping_counts >= 2))
            bundle_triangulated_point_idxs = np.where(triangulated_point_mapping_counts >= 2)[0]

            print('len(bundle_triangulated_point_idxs)', len(bundle_triangulated_point_idxs),
                  'of len(triangulated_point_mapping_counts)', len(triangulated_point_mapping_counts))

            model_triangulated_points = torch.tensor(model_triangulated_points[:, bundle_triangulated_point_idxs], dtype=torch.float32, requires_grad=True)
            bundle_image_points = torch.tensor(triangulated_image_points[:, :, bundle_triangulated_point_idxs], dtype=torch.float32)
            bundle_image_points[~torch.isfinite(bundle_image_points)] = 0
            bundle_image_points_weights = torch.tensor(triangulated_image_points_weights[:, :, bundle_triangulated_point_idxs], dtype=torch.float32)
            #threshold_target = torch.tensor(3.0 + np.log2(triangulated_point_mapping_counts[bundle_triangulated_point_idxs]), dtype=torch.float32)
            threshold_target = torch.tensor(10.0, dtype=torch.float32)
            #gamma_softplus_alpha = torch.tensor(2.0 + np.log2(triangulated_point_mapping_counts[bundle_triangulated_point_idxs]), dtype=torch.float32)
            gamma_softplus_alpha = torch.tensor(6, dtype=torch.float32)

            def calc_projected_image_points():
                model_Rs = pytorch3d.transforms.axis_angle_to_matrix(model_rvecs)
                object_points = model_Rs @ model_triangulated_points + model_tvecs

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
            num_steps = 3000
            convergence_criterion = {'rtol': 1e-4, 'window_size': 100, 'min_num_steps': 300}
            optimiser = torch.optim.Adam([model_rvecs, model_tvecs, model_triangulated_points], lr=lr)
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
                    print('optim_step', optim_step, 'loss', losses[-1], 'loss_window_std', loss_window_std,
                          'learning rate', np.round(scheduler.get_last_lr()[0], 6), 'temperature', np.round(temperature, 3))
                if optim_step >= convergence_criterion['min_num_steps'] and loss_window_std < convergence_criterion['rtol'] * losses[-1]:
                    break

            print('len(losses)', len(losses))
            print('initial, first, final loss', losses[0], losses[convergence_criterion['min_num_steps']], losses[-1])
            print('highest, lowest loss', np.max(losses[convergence_criterion['min_num_steps']:]), np.min(losses[convergence_criterion['min_num_steps']:]))

            del key_frame_camera_extrinsics[:]
            for rvec, tvec in zip(model_rvecs.numpy(force=True), model_tvecs.numpy(force=True)):
                R, jacobian = cv2.Rodrigues(rvec)
                t = tvec
                key_frame_camera_extrinsics.append(np.block([[R, t], [0, 0, 0, 1]]))

            camera_transform = key_frame_camera_extrinsics[current_frame_idx] @ np.linalg.inv(key_frame_camera_extrinsics[current_frame_idx - 1])
            R = camera_transform[:3, :3]
            t = camera_transform[:3, 3:]
            optim_transform_delta_vec = (R @ transform_delta_ref_vec + t - transform_delta_ref_vec).flatten()
            print('optimised transform delta vec', np.round(optim_transform_delta_vec, 3))
            torch.set_default_device(torch.device('cpu'))
            for idx in range(2):
                print(f'gmm[{idx}] log_prob(optim_transform_delta_vec)', np.round(gmm_distributions[idx].log_probability(optim_transform_delta_vec[None, :]).numpy(), 3))
            torch.set_default_device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            # TODO: calculate and record epipolar inliers as a measure of traction / slip

            if False:
                current_camera_extrinsic = key_frame_camera_extrinsics[current_frame_idx]
                R = current_camera_extrinsic[:3, :3]
                t = current_camera_extrinsic[:3, 3:]
                stitched_triangulated_points[bundle_triangulated_point_idxs, :3] = (R @ model_triangulated_points.numpy(force=True) + t).T
            else:
                first_camera_extrinsic = key_frame_camera_extrinsics[0]
                R = first_camera_extrinsic[:3, :3]
                t = first_camera_extrinsic[:3, 3:]
                model_triangulated_points = R @ model_triangulated_points.numpy(force=True) + t

                key_frame_camera_extrinsics[:] = [camera_extrinsic @ np.linalg.inv(first_camera_extrinsic)
                                                  for camera_extrinsic in key_frame_camera_extrinsics]

                scaling_factors = []
                for image_points_weights_magn, camera_extrinsic in zip(torch.norm(bundle_image_points_weights[:, :2, :], dim=1).numpy(force=True),
                                                                       key_frame_camera_extrinsics):
                    triangulated_points = camera_extrinsic[:3, :3] @ model_triangulated_points[:, image_points_weights_magn >= 1e-2] + camera_extrinsic[:3, 3:]

                    # Estimate the scaling factor from the depth distribution of the triangulated points
                    # assuming the camera is positioned at a distance of ~8mm from the target
                    c = np.percentile(triangulated_points[2, :], 25) / 8.0
                    scaling_factors.append(c)
                c = np.median(scaling_factors)

                print('scaling factor', c, np.round(scaling_factors, 2))
                model_triangulated_points /= c
                key_frame_camera_extrinsics[:] = [np.block([[camera_extrinsic[:3, :3], camera_extrinsic[:3, 3:] / c], [0, 0, 0, 1]])
                                                  for camera_extrinsic in key_frame_camera_extrinsics]

                current_camera_extrinsic = key_frame_camera_extrinsics[current_frame_idx]
                R = current_camera_extrinsic[:3, :3]
                t = current_camera_extrinsic[:3, 3:]
                stitched_triangulated_points[bundle_triangulated_point_idxs, :3] = (R @ model_triangulated_points + t).T


            # Cross stitching / loop closure between each previous frame with the current frame.
            # Retrace triangulated points that previously moved out of view and lost sequential optical flow tracking,
            # but have re-entered the current view.
            # These are the set of unmatched points between each previous cross frame and the current frame.
            # For each previous cross frame:
            #  - Calculate and apply the stereo camera rectification transforms corresponding to the camera extrinsic transform
            #    and compute the disparity map between the rectified images.
            #  - Derive the cross flow from the disparity map and apply it to map the current frame image onto the cross frame image space.
            #  - Compute the residual optical flow between the mapped current frame image and the cross frame image and apply the
            #    residual optical flow vectors to update the location of the cross mapped triangulated point projections.
            # For each triangulated point, cluster the collection of unmatched cross flow mapped coordinates across all the cross frames.
            # Stitch the triangulated point if the modal cluster has a small enough variance and its centroid is close enough
            # to the projected location of the triangulated point in the current frame.

            current_triangulated_image_idxs = np.where(current_image_to_triangulated_point_idxs >= 0)[0]
            current_triangulated_idxs = current_image_to_triangulated_point_idxs[current_triangulated_image_idxs]
            current_triangulated_idxs_to_image_idxs = dict(zip(current_triangulated_idxs, current_triangulated_image_idxs))

            valid_mask = np.all(np.isfinite(stitched_triangulated_points[:, :3]), axis=1)
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(stitched_triangulated_points[valid_mask, :3]))
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
            pcd.transform(np.linalg.inv(key_frame_camera_extrinsics[current_frame_idx]))
            pcd_points = np.array(pcd.points).T
            pcd_normals = np.array(pcd.normals).T
            normal_ray_alignment = np.zeros((np.sum(valid_mask),), dtype=np.float32)
            for image_to_triangulated_point_idxs, camera_extrinsic in zip(key_frame_image_triangulated_point_idxs + [current_image_to_triangulated_point_idxs],
                                                                          key_frame_camera_extrinsics):
                triangulated_image_idxs = np.where(image_to_triangulated_point_idxs >= 0)[0]
                triangulated_point_idxs = image_to_triangulated_point_idxs[triangulated_image_idxs]
                image_points_weights = np.zeros((stitched_triangulated_points.shape[0],), dtype=np.float32)
                image_points_weights[triangulated_point_idxs] = 1

                R = camera_extrinsic[:3, :3]
                t = camera_extrinsic[:3, 3:]
                object_points = R @ pcd_points + t
                object_normals = R @ pcd_normals

                object_point_ray = object_points / np.clip(np.linalg.norm(object_points, axis=0), 1e-6, np.inf)
                normal_ray_alignment += np.sum(object_point_ray * object_normals, axis=0) * image_points_weights[valid_mask]
            pcd.normals = o3d.utility.Vector3dVector((pcd_normals * -np.sign(normal_ray_alignment)).T)
            pcd.transform(key_frame_camera_extrinsics[current_frame_idx])
            assert np.allclose(np.linalg.norm(np.array(pcd.normals), axis=1), 1)

            #     z               object-plane
            #     z             or
            #     z           o  r
            #     z         o   r
            #     z       o     ray-from_camera
            #     z     o      r
            #     z   o        r
            #     z o         r
            # iiiioiiiiiiiiiiiriiiimage-plane
            #   o z  n       r
            # o   z     n    r
            #     z        nr
            #     z         r n
            #     z        r     normal-from-object-plane

            # The image plane, object plane, object plane normal and z-axis are coincident at oi = on = oz
            # object-plane section or-oi is projected onto image-plane section ir-oi
            # Perspective distortion is based on object_to_image_ratio = |or-oi| / |ir-oi|
            # camera_rays = (or-nr) / |or-nr| = (ir-nr) / |ir-nr|
            # camera_ray_to_object_plane = or-nr
            # camera_ray_to_image_plane = ir-nr

            # |nr-oi| = 1
            # ⇒ (nr-oi) ⋅ (or-nr) = -1
            # ⇒ |or-nr| = -1 / [(nr-oi) ⋅ (or-nr) / |or-nr|] where (or-nr) / |or-nr| is the unit length camera ray

            # Given the unit vector z on the z-axis
            # (ir-nr) ⋅ z = - (nr-oi) ⋅ z
            # ⇒ |ir-nr| = - [(nr-oi) ⋅ z] / [(ir-nr) / |ir-nr| ⋅ z] where (ir-nr) / |ir-nr| is the unit length camera ray

            triangulated_points = np.full((3, stitched_triangulated_points.shape[0]), fill_value=np.nan, dtype=np.float32)
            triangulated_points[:, valid_mask] = np.array(pcd.points).T
            triangulated_normals = np.full((3, stitched_triangulated_points.shape[0]), fill_value=np.nan, dtype=np.float32)
            triangulated_normals[:, valid_mask] = np.array(pcd.normals).T
            camera_rays = triangulated_points / np.clip(np.linalg.norm(triangulated_points, axis=0), 1e-6, np.inf)
            normal_ray_alignment = np.sum(camera_rays * triangulated_normals, axis=0)
            camera_ray_to_object_plane = camera_rays / np.clip(-normal_ray_alignment, 1e-8, 1)
            camera_ray_to_image_plane = camera_rays * -triangulated_normals[2, :] / camera_rays[2, :]

            object_to_image_ratio = (np.linalg.norm(triangulated_normals + camera_ray_to_object_plane, axis=0)
                                     / np.clip(np.linalg.norm(triangulated_normals + camera_ray_to_image_plane, axis=0), 1e-8, np.inf))
            perspective_distortion = 1 - np.min(np.stack([object_to_image_ratio,
                                                          np.clip(1 / object_to_image_ratio, 1e-8, np.inf)]), axis=0)

            triangulated_normal_weights = 0.5 * (1 - scipy.special.erf((perspective_distortion - 0.6) / 0.1))
            triangulated_normal_weights[normal_ray_alignment >= 0] = 0

            cross_warp_triangulated_points = {}

            for cross_frame_idx in range(current_frame_idx):
                print('cross_frame_idx', cross_frame_idx)

                #cross_motion_blur = np.max([key_frame_motion_blurs[idx] for idx in [cross_frame_idx, current_frame_idx]])

                cross_camera_extrinsic = key_frame_camera_extrinsics[cross_frame_idx]

                ref_img, ref_gray = key_frame_images[cross_frame_idx]
                #img, gray = key_frame_images[current_frame_idx]

                ref_img_mask = key_frame_masks[cross_frame_idx]

                # Extract the indices of the common set of triangulated points between the cross and current frames
                xsc, ysc = key_frame_image_sample_points[cross_frame_idx]
                cross_image_points = np.vstack([xsc, ysc])
                cross_image_to_triangulated_point_idxs = key_frame_image_triangulated_point_idxs[cross_frame_idx]
                cross_triangulated_image_idxs = np.where(cross_image_to_triangulated_point_idxs >= 0)[0]
                cross_triangulated_idxs = cross_image_to_triangulated_point_idxs[cross_triangulated_image_idxs]
                valid_mask = np.all(np.isfinite(stitched_triangulated_points[cross_triangulated_idxs, :3]), axis=1)
                cross_triangulated_image_idxs = cross_triangulated_image_idxs[valid_mask]
                cross_triangulated_idxs = cross_triangulated_idxs[valid_mask]
                cross_triangulated_idxs_to_image_idxs = dict(zip(cross_triangulated_idxs, cross_triangulated_image_idxs))

                common_triangulated_idxs = np.array(list(set(list(cross_triangulated_idxs)) & set(list(current_triangulated_idxs))), dtype=int)
                cross_unmatched_triangulated_idxs = np.array(list(set(list(cross_triangulated_idxs)) - set(list(current_triangulated_idxs))), dtype=int)
                print('len(common_triangulated_idxs)', len(common_triangulated_idxs))
                print('len(cross_unmatched_triangulated_idxs)', len(cross_unmatched_triangulated_idxs))

                cross_triangulated_image_idxs = list(map(cross_triangulated_idxs_to_image_idxs.get, common_triangulated_idxs))
                cross_triangulated_image_points = cross_image_points[:, cross_triangulated_image_idxs].T
                current_triangulated_image_idxs = list(map(current_triangulated_idxs_to_image_idxs.get, common_triangulated_idxs))
                current_triangulated_image_points = np.vstack([xsn, ysn])[:, current_triangulated_image_idxs].T
                cross_unmatched_triangulated_image_idxs = list(map(cross_triangulated_idxs_to_image_idxs.get, cross_unmatched_triangulated_idxs))
                cross_unmatched_triangulated_image_points = cross_image_points[:, cross_unmatched_triangulated_image_idxs].T

                inv_current_camera_extrinsic = np.linalg.inv(current_camera_extrinsic)
                cross_triangulated_points = (inv_current_camera_extrinsic[:3, :3] @ stitched_triangulated_points[common_triangulated_idxs, :3].T
                                             + inv_current_camera_extrinsic[:3, 3:]).T
                cross_triangulated_rgb = stitched_triangulated_points[common_triangulated_idxs, 3:]

                cross_object_points = cross_camera_extrinsic @ np.vstack([cross_triangulated_points.T, np.ones((cross_triangulated_points.shape[0],))])
                cross_projected_points = camera_matrix @ cross_object_points[:3, :]
                cross_projected_points = cross_projected_points[:2, :] / cross_projected_points[2, :]

                current_object_points = current_camera_extrinsic @ np.vstack([cross_triangulated_points.T, np.ones((cross_triangulated_points.shape[0],))])
                current_projected_points = camera_matrix @ current_object_points[:3, :]
                current_projected_points = current_projected_points[:2, :] / current_projected_points[2, :]

                #threshold = min(0.5 * (1 + cross_motion_blur), 2.0)
                threshold = 6.0
                cross_triangulated_inlier_idxs = np.where(np.linalg.norm(current_triangulated_image_points - current_projected_points.T, axis=1) < threshold)[0]
                print('num cross triangulated point inliers', len(cross_triangulated_inlier_idxs))

                # TODO: Resolve the different causes of low disparity confidence, in particular the case where
                #       there is only partial visibility of the object as opposed to erroneous camera extrinsic estimates.
                #       Investigate whether the triangulated points can be used to resolve this.

                if exec_mode == 'debug_key_frame' and current_frame_idx == debug_key_frame_idx:
                    plt.figure('Cross projected triangulated points', figsize=(16, 10))
                    setup_new_fig_page()
                    plt.suptitle(f'cross, current frame idxs: {cross_frame_idx}, {current_frame_idx}')
                    ax = plt.subplot(2, 3, 1)
                    plt.imshow(np.require(ref_img, dtype=np.uint8))
                    ax = plt.subplot(2, 3, 2, sharex=ax, sharey=ax)
                    plt.imshow(np.require(img, dtype=np.uint8))
                    ax3 = plt.subplot(2, 3, 3, projection='3d')
                    ax3.scatter(*cross_triangulated_points.T, s=2, c=cross_triangulated_rgb/255)
                    ax3.set_xlim((-20, 20))
                    ax3.set_ylim((-20, 20))
                    ax3.set_zlim((0, 40))
                    ax3.set_aspect('equal', adjustable='datalim')
                    ax3.set_xlabel('X')
                    ax3.set_ylabel('Y')
                    ax3.set_zlabel('Z')
                    ax3.view_init(elev=-135, azim=-90, roll=0)
                    ax = plt.subplot(2, 3, 4, sharex=ax, sharey=ax)
                    plt.scatter(*cross_triangulated_image_points.T, s=2, c='b', marker='o')
                    plt.scatter(*cross_projected_points, s=2, c='y', marker='o')
                    ax.set_aspect('equal')
                    ax = plt.subplot(2, 3, 5, sharex=ax, sharey=ax)
                    plt.scatter(*current_triangulated_image_points.T, s=2, c='b', marker='o')
                    plt.scatter(*current_projected_points, s=2, c='y', marker='o')
                    ax.set_aspect('equal')
                    plt.tight_layout()
                    stash_fig_page()


                camera_transform = current_camera_extrinsic @ np.linalg.inv(cross_camera_extrinsic)
                R = camera_transform[:3, :3]
                t = camera_transform[:3, 3:]

                # Project the current frame's camera view cone onto the cross frame's camera image and calculate
                # the intersection as a measure of the disparity that can potentially be derived
                uvs = np.array([[0, 0], [img.shape[1] - 1, 0], [img.shape[1] - 1, img.shape[0] - 1], [0, img.shape[0] - 1]]).T
                uvcs = uvs - camera_matrix[:2, 2:]
                xys = np.linalg.inv(camera_matrix[:2, :2]) @ uvcs
                xyzs = np.vstack([xys, np.ones(xys.shape[1],)])
                xyzs = np.hstack([xyzs * 5, xyzs * 30])
                xyzs = R.T @ xyzs - R.T @ t
                projected_points = camera_matrix @ xyzs
                projected_points = projected_points[:2, :] / projected_points[2, :]
                projected_points_geometry = shapely.MultiPoint(projected_points.T).convex_hull
                cross_image_geometry = shapely.MultiPoint(uvs.T).convex_hull
                projected_cone_coverage = cross_image_geometry.intersection(projected_points_geometry).area / cross_image_geometry.area
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
                R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(camera_matrix, None,
                                                                                  camera_matrix, None,
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
                #rect_map_prev, _ = cv2.initUndistortRectifyMap(camera_matrix, None, R1, P1[:3, :3], newImageSize, cv2.CV_32FC2)
                #rect_map_next, _ = cv2.initUndistortRectifyMap(camera_matrix, None, R2, P2[:3, :3], newImageSize, cv2.CV_32FC2)

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
                    uvcs = uvs - camera_matrix[:2, 2:]
                    uvcs_clip = (max(img.shape[:2]) - 1) / 2 - img_size_trim
                    uvcs = np.clip(uvcs, -uvcs_clip, uvcs_clip)
                    xys = np.linalg.inv(camera_matrix[:2, :2]) @ uvcs
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
                uvcs = uvs - camera_matrix[:2, 2:]
                xys = np.linalg.inv(camera_matrix[:2, :2]) @ uvcs
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

                disparity_map_zoom = 0.125
                fxy = (disparity_map_zoom * np.diag(camera_matrix[:2, :2])
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
                if exec_mode == 'debug_key_frame' and current_frame_idx == debug_key_frame_idx:
                    plt.figure('dx12s', figsize=(16, 10))
                    setup_new_fig_page()
                    plt.suptitle(f'cross, current frame idxs: {cross_frame_idx}, {current_frame_idx}\ndisparity_spread {disparity_spread}')
                    hist_bins = np.arange(np.floor(np.min(dx12s[common_view_frustum_inlier_mask])),
                                          np.floor(np.max(dx12s[common_view_frustum_inlier_mask])) + 1)
                    plt.hist(dx12s[common_view_frustum_inlier_mask], bins=hist_bins,
                             weights=common_view_frustum_sample_points_weights[common_view_frustum_inlier_mask])
                    plt.tight_layout()
                    stash_fig_page()
                """

                """
                if img_size_trim > max(img.shape[:2]) / 4 or rect_proximity < 1.0 or disparity_spread < 16 * disparity_map_zoom or disparity_spread > 384 * disparity_map_zoom:
                    print('img_size_trim, rect_proximity, disparity_spread', img_size_trim, rect_proximity, disparity_spread)
                    continue
                """

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


                # TODO: Weighting and error threshold dependent on depth?
                # TODO: Run regression in both cross and current frames of reference, then filter by the intersection of inliers
                # TODO: change cv2.remap to use cv2.INTER_AREA where downsampling, e.g. for disparity, but actually cv2.INTER_AREA is
                #       not supported by cv2.remap anyway, so should be manually filtering first

                camera_transform = current_camera_extrinsic @ np.linalg.inv(cross_camera_extrinsic)
                R = camera_transform[:3, :3]
                t = camera_transform[:3, 3:]

                zoom_ext = 1.5
                imageSize = img.shape[1::-1]
                imageSize_ext = tuple(np.round(np.array(imageSize) * zoom_ext).astype(np.int32))
                camera_matrix_ext = np.block([[camera_matrix[:2, :2], (camera_matrix[:2, 2:] + 0.5) * zoom_ext - 0.5], [0, 0, 1]])

                current_triangulated_points = stitched_triangulated_points[:len(triangulated_idxs_weights), :3].T
                cross_triangulated_points = R.T @ current_triangulated_points - R.T @ t
                cross_projected_points = camera_matrix_ext @ cross_triangulated_points
                cross_projected_points = cross_projected_points[:2, :] / cross_projected_points[2, :]

                inlier_mask = ((cross_projected_points[0, :] > -0.5) & (cross_projected_points[0, :] < imageSize_ext[0] - 0.5)
                               & (cross_projected_points[1, :] > -0.5) & (cross_projected_points[1, :] < imageSize_ext[1] - 0.5))

                inlier_uv = cross_projected_points[:, inlier_mask]
                inlier_z = cross_triangulated_points[2, inlier_mask]

                cross_triangulated_normals = R.T @ stitched_triangulated_normals[:, :len(triangulated_idxs_weights)][:, inlier_mask]
                camera_rays = cross_triangulated_points[:, inlier_mask] / np.clip(np.linalg.norm(cross_triangulated_points[:, inlier_mask], axis=0), 1e-6, np.inf)

                normal_ray_alignment = np.sum(camera_rays * cross_triangulated_normals, axis=0)

                camera_ray_to_object_plane = camera_rays / np.clip(-normal_ray_alignment, 1e-8, 1)
                camera_ray_to_image_plane = camera_rays * -cross_triangulated_normals[2, :] / camera_rays[2, :]

                object_to_image_ratio = (np.linalg.norm(cross_triangulated_normals + camera_ray_to_object_plane, axis=0)
                                         / np.clip(np.linalg.norm(cross_triangulated_normals + camera_ray_to_image_plane, axis=0), 1e-8, np.inf))
                perspective_distortion = 1 - np.min(np.stack([object_to_image_ratio,
                                                              np.clip(1 / object_to_image_ratio, 1e-8, np.inf)]), axis=0)

                perspective_distortion_weights = 0.5 * (1 - scipy.special.erf((perspective_distortion - 0.4) / 0.15))
                perspective_distortion_weights[normal_ray_alignment >= 0] = 0

                X = []
                uv_grid = np.mgrid[0:imageSize_ext[1], 0:imageSize_ext[0]][::-1]
                X_grid = []
                X.append(inlier_uv[0, :] / imageSize_ext[0] - 0.5)
                X.append(inlier_uv[1, :] / imageSize_ext[1] - 0.5)
                X_grid.append(uv_grid[0, :, :] / imageSize_ext[0] - 0.5)
                X_grid.append(uv_grid[1, :, :] / imageSize_ext[1] - 0.5)
                if True:
                    for wx in range(2):
                        for wy in range(2):
                            if wx == wy == 0:
                                continue
                            for phase_offset in [0, np.pi / 2]:
                                X.append(np.cos(np.sum(inlier_uv.T / imageSize_ext * (wx, wy), axis=1) + phase_offset))
                                X_grid.append(np.cos(np.sum(uv_grid.T / imageSize_ext * (wx, wy), axis=-1).T + phase_offset))

                # TODO: model and predict the disparity along the epipolar lines instead of the depth

                X = np.array(X).T
                sample_weight = triangulated_idxs_weights[inlier_mask] * perspective_distortion_weights
                #estimator = sklearn.linear_model.LinearRegression(fit_intercept=True)
                estimator = sklearn.linear_model.Ridge(alpha=0.01, fit_intercept=True)
                #estimator = sklearn.linear_model.BayesianRidge(fit_intercept=True)
                ransac_threshold = 0.5
                model = sklearn.linear_model.RANSACRegressor(estimator=estimator, min_samples=X.shape[1]+1, residual_threshold=ransac_threshold, random_state=0)
                model.fit(X, inlier_z, sample_weight=np.maximum(sample_weight, 1e-5))

                print('model intercept, coef', model.estimator_.intercept_, model.estimator_.coef_)
                model_errs = model.estimator_.intercept_ + np.sum(X * model.estimator_.coef_, axis=-1) - inlier_z
                predicted_z = model.estimator_.intercept_ + np.sum(np.stack(X_grid, axis=-1) * model.estimator_.coef_, axis=-1)

                uvs = np.vstack([uv.flatten() for uv in uv_grid])
                uvcs = uvs - camera_matrix_ext[:2, 2:]
                xys = np.linalg.inv(camera_matrix_ext[:2, :2]) @ uvcs
                xyzs = np.vstack([xys, np.ones(xys.shape[1],)])


                border_value = 127

                #current_predicted_points = xyzs * predicted_z.flatten()
                #cross_projected_predicted_points = camera_matrix @ (R.T @ current_predicted_points - R.T @ t)
                #cross_projected_predicted_points = cross_projected_predicted_points[:2, :] / cross_projected_predicted_points[2, :]

                #warped_gray = cv2.remap(ref_gray,
                #                        cross_projected_predicted_points.T.reshape(ref_gray.shape + (2,)).astype(np.float32), None, cv2.INTER_LINEAR,
                #                        borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)

                cross_predicted_points = xyzs * predicted_z.flatten()

                cross_projected_predicted_points = camera_matrix @ cross_predicted_points
                cross_projected_predicted_points = cross_projected_predicted_points[:2, :] / cross_projected_predicted_points[2, :]

                current_predicted_points = R @ cross_predicted_points + t
                current_projected_predicted_points = camera_matrix @ current_predicted_points
                current_projected_predicted_points = current_projected_predicted_points[:2, :] / current_projected_predicted_points[2, :]

                ref_gray_ext = cv2.remap(ref_gray,
                                         cross_projected_predicted_points.T.reshape(imageSize_ext[::-1] + (2,)).astype(np.float32), None, cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)

                gray_warp_map = current_projected_predicted_points.T.reshape(imageSize_ext[::-1] + (2,)).astype(np.float32)
                gray_warp = cv2.remap(gray,
                                      gray_warp_map, None, cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)

                down_step = 10
                assert imageSize_ext[0] % down_step == 0 and imageSize_ext[1] % down_step == 0
                imageSize_ext_down = (imageSize_ext[0] // down_step, imageSize_ext[1] // down_step)
                uvs_down = np.vstack([uv.flatten() for uv in np.mgrid[0:imageSize_ext_down[1], 0:imageSize_ext_down[0]][::-1]])
                uvs_down = (uvs_down + 0.5) * down_step - 0.5
                kd_tree = scipy.spatial.KDTree(inlier_uv.T)
                nn_distances, nn_point_idxs = kd_tree.query(uvs_down.T, k=8, distance_upper_bound=down_step*1.5)
                nn_model_errs = np.append(model_errs, [np.nan])[nn_point_idxs]
                nn_model_errs_mask = np.any(np.isfinite(nn_model_errs), axis=1)
                nn_model_errs[~nn_model_errs_mask, :] = 0
                cross_model_errs = np.sqrt(np.nanmin(np.power(nn_model_errs, 2), axis=1))
                cross_model_errs[~nn_model_errs_mask] = np.nan
                cross_model_errs = cv2.resize(cross_model_errs.reshape(imageSize_ext_down[::-1]), (0, 0), fx=down_step, fy=down_step, interpolation=cv2.INTER_LINEAR)

                ref_gray_ext_attenuated = np.round((ref_gray_ext.astype(np.float32) - border_value) * cauchy(np.nan_to_num(cross_model_errs, nan=np.inf), 1.0) + border_value).astype(np.uint8)
                #gray_warp = np.round((gray_warp.astype(np.float32) - border_value) * cauchy(np.nan_to_num(cross_model_errs, nan=np.inf), 1.0) + border_value).astype(np.uint8)


                """
                # prev(y, x) ~ next(y + flow(y, x)[1], x + flow(y, x)[0])
                # when cv2.OPTFLOW_FARNEBACK_GAUSSIAN is applied, flow_sigma = (winsize // 2) * 0.3
                # poly_n: size of the pixel neighborhood used to find polynomial expansion in each pixel;
                #         larger values mean that the image will be approximated with smoother surfaces,
                #         yielding more robust algorithm and more blurred motion field, typically poly_n=5 or 7.
                # poly_sigma: standard deviation of the Gaussian that is used to smooth derivatives used as a basis
                #             for the polynomial expansion;
                #             for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
                zoom = 0.25
                cross_flow = cv2.calcOpticalFlowFarneback(prev=cv2.resize(ref_gray_ext_attenuated, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA),
                                                          next=cv2.resize(gray_warp, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA),
                                                          flow=None,
                                                          #pyr_scale=0.75, levels=3, winsize=51, iterations=2,
                                                          #poly_n=13, poly_sigma=2.8, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                                                          #pyr_scale=0.95, levels=30, winsize=51, iterations=2,
                                                          #poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                                                          #pyr_scale=0.95, levels=30, winsize=101, iterations=30,
                                                          #poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                                                          # winsize=25 with zoom = 0.25 translates to a Gaussian window size of 100
                                                          # which means that each iteration takes into account up to 100 / 2 = 50 pixels
                                                          # in range at the base of the pyramid when calculating the weighted average of A.T @ A terms.
                                                          # Then accounting for pyr_scale ** levels, for pyr_scale=0.75, levels=5
                                                          # 50 / (0.75**5) = 210.7 pixels of range at the top of the pyramid
                                                          pyr_scale=0.75, levels=5, winsize=25, iterations=30,
                                                          poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                if False:
                    cross_flow = cv2.calcOpticalFlowFarneback(prev=cv2.resize(ref_gray_ext_attenuated, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA),
                                                              next=cv2.resize(gray_warp, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA),
                                                              flow=cross_flow,
                                                              #pyr_scale=0.75, levels=3, winsize=51, iterations=2,
                                                              #poly_n=13, poly_sigma=2.8, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                                                              #pyr_scale=0.95, levels=30, winsize=51, iterations=2,
                                                              #poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                                                              pyr_scale=0.95, levels=30, winsize=13, iterations=30,
                                                              poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN+cv2.OPTFLOW_USE_INITIAL_FLOW)
                cross_flow = cv2.resize(cross_flow, (0, 0), fx=1/zoom, fy=1/zoom, interpolation=cv2.INTER_NEAREST) / zoom

                cross_flow_map = (np.transpose(uv_grid, axes=(1, 2, 0)) + cross_flow).astype(np.float32)
                gray_cross_warp = cv2.remap(gray_warp,
                                            cross_flow_map, None, cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


                # prev(y, x) ~ next(y + flow(y, x)[1], x + flow(y, x)[0])
                # when cv2.OPTFLOW_FARNEBACK_GAUSSIAN is applied, flow_sigma = (winsize // 2) * 0.3
                # poly_n: size of the pixel neighborhood used to find polynomial expansion in each pixel;
                #         larger values mean that the image will be approximated with smoother surfaces,
                #         yielding more robust algorithm and more blurred motion field, typically poly_n=5 or 7.
                # poly_sigma: standard deviation of the Gaussian that is used to smooth derivatives used as a basis
                #             for the polynomial expansion;
                #             for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
                zoom = 0.25
                cross_flow_warp = cv2.calcOpticalFlowFarneback(prev=cv2.resize(gray_warp, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA),
                                                               next=cv2.resize(ref_gray_ext_attenuated, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA),
                                                               flow=None,
                                                               # winsize=25 with zoom = 0.25 translates to a Gaussian window size of 100
                                                               # which means that each iteration takes into account up to 100 / 2 = 50 pixels
                                                               # in range at the base of the pyramid when calculating the weighted average of A.T @ A terms.
                                                               # Then accounting for pyr_scale ** levels, for pyr_scale=0.75, levels=5
                                                               # 50 / (0.75**5) = 210.7 pixels of range at the top of the pyramid
                                                               pyr_scale=0.75, levels=5, winsize=25, iterations=30,
                                                               poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                cross_flow_warp = cv2.resize(cross_flow_warp, (0, 0), fx=1/zoom, fy=1/zoom, interpolation=cv2.INTER_NEAREST) / zoom

                cross_flow_warp_map = (np.transpose(uv_grid, axes=(1, 2, 0)) + cross_flow_warp).astype(np.float32)
                ref_gray_cross_warp = cv2.remap(ref_gray_ext_attenuated,
                                                cross_flow_warp_map, None, cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


                # Consistency error between cross->current and current->cross cross flow
                interp = scipy.interpolate.RegularGridInterpolator((np.arange(cross_flow_warp_map.shape[0]), np.arange(cross_flow_warp_map.shape[1])),
                                                                   cross_flow_warp_map[:, :, 0] + cross_flow_warp_map[:, :, 1] * 1j,
                                                                   method='linear', bounds_error=False, fill_value=np.nan)
                cross_flow_warp_map_interp = interp((cross_flow_map[:, :, 1], cross_flow_map[:, :, 0])).astype(np.complex64)
                cross_flow_consistency_err = np.linalg.norm(np.stack([cross_flow_warp_map_interp.real, cross_flow_warp_map_interp.imag], axis=-1)
                                                             - np.transpose(uv_grid, axes=(1, 2, 0)), axis=-1)


                sample_weight_mask = sample_weight > 0.1 * np.median(sample_weight)

                projected_points = camera_matrix @ cross_triangulated_points[:, inlier_mask]
                projected_points = projected_points[:2, :] / projected_points[2, :]

                xys_mask = (sample_weight_mask & (np.abs(model_errs) < ransac_threshold * 3)
                            & (projected_points[0, :] > -0.5) & (projected_points[0, :] < imageSize[0] - 0.5)
                            & (projected_points[1, :] > -0.5) & (projected_points[1, :] < imageSize[1] - 0.5))

                xs, ys = inlier_uv[:, xys_mask]

                interp = scipy.interpolate.RegularGridInterpolator((np.arange(cross_flow.shape[0]), np.arange(cross_flow.shape[1])),
                                                                   cross_flow[:, :, 0] + cross_flow[:, :, 1] * 1j,
                                                                   method='linear', bounds_error=False, fill_value=np.nan)
                cross_flow_interp = interp((ys, xs)).astype(np.complex64)
                cross_flow_xys = np.vstack([cross_flow_interp.real, cross_flow_interp.imag]).T

                interp = scipy.interpolate.RegularGridInterpolator((np.arange(gray_warp_map.shape[0]), np.arange(gray_warp_map.shape[1])),
                                                                   gray_warp_map[:, :, 0] + gray_warp_map[:, :, 1] * 1j,
                                                                   method='linear', bounds_error=False, fill_value=np.nan)
                gray_warp_map_interp = interp((ys, xs)).astype(np.complex64)
                gray_warp_map_xys = np.vstack([gray_warp_map_interp.real, gray_warp_map_interp.imag]).T

                gray_flow_interp = interp((ys + cross_flow_xys[:, 1], xs + cross_flow_xys[:, 0])).astype(np.complex64)
                gray_flow_xys = np.vstack([gray_flow_interp.real, gray_flow_interp.imag])

                interp = scipy.interpolate.RegularGridInterpolator((np.arange(cross_flow_consistency_err.shape[0]), np.arange(cross_flow_consistency_err.shape[1])),
                                                                   cross_flow_consistency_err,
                                                                   method='linear', bounds_error=False, fill_value=np.nan)
                cross_flow_consistency_err_xys = interp((ys, xs)).astype(np.float32)

                cross_image_to_triangulated_point_idxs = key_frame_image_triangulated_point_idxs[cross_frame_idx]
                cross_image_triangulated_point_idxs = cross_image_to_triangulated_point_idxs[(cross_image_to_triangulated_point_idxs >= 0)
                                                                                             & (cross_image_to_triangulated_point_idxs < len(triangulated_idxs_weights))]
                cross_image_triangulated_points = np.full((len(triangulated_idxs_weights),), fill_value=False, dtype=bool)
                cross_image_triangulated_points[cross_image_triangulated_point_idxs] = True
                cross_image_triangulated_points_xys = cross_image_triangulated_points[inlier_mask][xys_mask]

                cross_flow_mask = ((gray_flow_xys[0, :] > -0.5) & (gray_flow_xys[0, :] < imageSize[0] - 0.5)
                                   & (gray_flow_xys[1, :] > -0.5) & (gray_flow_xys[1, :] < imageSize[1] - 0.5)
                                   & (gray_warp_map_xys[:, 0] > -0.5) & (gray_warp_map_xys[:, 0] < imageSize[0] - 0.5)
                                   & (gray_warp_map_xys[:, 1] > -0.5) & (gray_warp_map_xys[:, 1] < imageSize[1] - 0.5)
                                   & (cross_flow_consistency_err_xys < 6.0)
                                   & cross_image_triangulated_points_xys)
                """

                # TODO: Filter image edges and then blur to produce gradients throughout the image
                # that will help convergence
                if False:
                    ref_img_ext = cv2.remap(np.mean(ref_img, axis=-1).astype(np.uint8),
                                            cross_projected_predicted_points.T.reshape(imageSize_ext[::-1] + (2,)).astype(np.float32), None, cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
                    img_warp = cv2.remap(np.mean(img, axis=-1).astype(np.uint8),
                                         gray_warp_map, None, cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
                elif False:
                    ref_img_ext = cv2.remap(ref_gray,
                                            cross_projected_predicted_points.T.reshape(imageSize_ext[::-1] + (2,)).astype(np.float32), None, cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
                    img_warp = cv2.remap(gray,
                                         gray_warp_map, None, cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
                elif True:
                    # Using the sqrt of the image intensity values appears to reduce and moderate the ECC update step size
                    ref_img_ext = cv2.remap(np.sqrt(np.mean(ref_img.astype(np.float32), axis=-1)) * 16,
                                            cross_projected_predicted_points.T.reshape(imageSize_ext[::-1] + (2,)).astype(np.float32), None, cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=np.sqrt(border_value)*16)
                    img_warp = cv2.remap(np.sqrt(np.mean(img.astype(np.float32), axis=-1)) * 16,
                                         gray_warp_map, None, cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=np.sqrt(border_value)*16)
                else:
                    ref_img_ext = cv2.remap(np.power(np.mean(ref_img.astype(np.float32), axis=-1), 2) / 255,
                                            cross_projected_predicted_points.T.reshape(imageSize_ext[::-1] + (2,)).astype(np.float32), None, cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=border_value**2/255)
                    img_warp = cv2.remap(np.power(np.mean(img.astype(np.float32), axis=-1), 2) / 255,
                                         gray_warp_map, None, cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=border_value**2/255)


                """
                projected_points = camera_matrix @ xyzs
                projected_points = projected_points[:2, :] / projected_points[2, :]

                ref_img_ext_mask = ((projected_points[0, :] > -0.5) & (projected_points[0, :] < imageSize[0] - 0.5)
                                    & (projected_points[1, :] > -0.5) & (projected_points[1, :] < imageSize[1] - 0.5)).reshape(imageSize_ext[::-1])
                """

                # Execute a few cv2.MOTION_AFFINE iterations for all the cross frames,
                # and then rerun the pytorch optimiser to update the camera extrinsics with the loss function
                # using existing projection errors for existing triangulated points, plus the new
                # projection errors for unmatched triangulated points given the ECC warp between cross frames and current frame.

                if False:
                    ref_img_ext_base_mask = cv2.remap(np.full(ref_img.shape[:2], fill_value=255, dtype=np.uint8),
                                                      cross_projected_predicted_points.T.reshape(imageSize_ext[::-1] + (2,)).astype(np.float32), None, cv2.INTER_LINEAR,
                                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)

                    img_ext_base_mask = cv2.remap(np.full(img.shape[:2], fill_value=255, dtype=np.uint8),
                                                  gray_warp_map, None, cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                else:
                    ref_img_ext_base_mask = cv2.remap((~ref_img_mask).astype(np.uint8) * 255,
                                                      cross_projected_predicted_points.T.reshape(imageSize_ext[::-1] + (2,)).astype(np.float32), None, cv2.INTER_LINEAR,
                                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)

                    img_ext_base_mask = cv2.remap((~img_mask).astype(np.uint8) * 255,
                                                  gray_warp_map, None, cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)


                def zoom_matrix(zoom):
                    return (np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]], dtype=np.float32)
                            @ np.array([[zoom, 0, 0], [0, zoom, 0], [0, 0, 1]], dtype=np.float32)
                            @ np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], dtype=np.float32))

                def decompose_warp_matrix(warpMatrix):
                    translation = warpMatrix[:2, 2]
                    A = warpMatrix[:2, :2]
                    detA = np.linalg.det(A)
                    if detA == 0:
                        rotation_angle, scaling, shear = 0, np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32)
                    else:
                        reflection = np.sign(detA)
                        U, S, Vh = np.linalg.svd(A @ np.diag([reflection, 1]))
                        R = U @ Vh
                        rvec, jacobian = cv2.Rodrigues(np.vstack([np.pad(R, ((0, 0), (0, 1))), [0, 0, 1]]))
                        assert np.allclose(rvec[:2, :], 0)
                        rotation_angle = rvec[2, 0]
                        P = R.T @ A
                        scaling = np.linalg.norm(P, axis=1)
                        shear = np.array([np.arctan2(*P[::-1, 0]), np.arctan2(*P[:, 1])])
                        out_of_bounds = (np.linalg.norm(translation) > 150
                                         or reflection < 0
                                         or np.abs(rotation_angle) > np.pi / 4
                                         or np.any(np.max(np.vstack([scaling, 1 / np.maximum(scaling, 1e-5)]), axis=0) > 1.5)
                                         or np.any(np.abs(shear) > np.pi / 6))
                    return translation, reflection, rotation_angle, scaling, shear, out_of_bounds


                warpMatrix = np.identity(3, dtype=np.float32)
                for motionType, zoom, gaussFiltSize in [(cv2.MOTION_TRANSLATION, 1 / 16, 1),
                                                        (cv2.MOTION_EUCLIDEAN, 1 / 16, 1),
                                                        (cv2.MOTION_AFFINE, 1 / 8, 1),
                                                        (cv2.MOTION_AFFINE, 1 / 8, 1),
                                                        (cv2.MOTION_AFFINE, 1 / 8, 1),
                                                        (cv2.MOTION_AFFINE, 1 / 8, 1),
                                                        (cv2.MOTION_AFFINE, 1 / 8, 1),
                                                        (cv2.MOTION_AFFINE, 1 / 8, 1),
                                                        (cv2.MOTION_HOMOGRAPHY, 1 / 32, 1),
                                                        (cv2.MOTION_HOMOGRAPHY, 1 / 16, 1),
                                                        (cv2.MOTION_HOMOGRAPHY, 1 / 8, 1)][2:-3]:
                    img_ext_mask = cv2.warpPerspective(img_ext_base_mask, M=warpMatrix, dsize=img_ext_base_mask.shape[::-1],
                                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    input_mask = ((cv2.resize(cross_model_errs, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA) < 2.0)
                                  & (cv2.resize(ref_img_ext_base_mask, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA) > 240)
                                  & (cv2.resize(img_ext_mask, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA) > 240)).astype(np.uint8)
                    warpMatrix_zoom = zoom_matrix(zoom) @ warpMatrix @ zoom_matrix(1 / zoom)
                    if motionType != cv2.MOTION_HOMOGRAPHY:
                        warpMatrix_zoom = warpMatrix_zoom[:2, :]
                    #print(warpMatrix_zoom)
                    # https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/ecc.cpp#L409
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, -1)
                    try:
                        retval, _ = cv2.findTransformECC(#templateImage=cv2.resize(gray_warp, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA),
                                                         #inputImage=cv2.resize(ref_gray_ext_attenuated, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA),
                                                         templateImage=cv2.resize(img_warp, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA),
                                                         inputImage=cv2.resize(ref_img_ext, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA),
                                                         warpMatrix=warpMatrix_zoom,
                                                         motionType=motionType,
                                                         criteria=criteria,
                                                         inputMask=input_mask,
                                                         gaussFiltSize=gaussFiltSize)
                    except cv2.error as e:
                        print(e)
                    if warpMatrix_zoom.shape[0] < 3:
                        warpMatrix_zoom = np.vstack([warpMatrix_zoom, [0, 0, 1]], dtype=np.float32)
                    #print(warpMatrix_zoom)

                    # https://docs.opencv.org/4.10.0/dc/d6b/group__video__track.html#ga1aa357007eaec11e9ed03500ecbcbe47
                    # https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/ecc.cpp#L530
                    # cv2.findTransformECC() applies warpPerspective() or warpAffine() with warpMatrix to inputImage
                    # using cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
                    # This means that warpMatrix is the homography matrix that maps templateImage coordinates to inputImage coordinates
                    # and conversely np.linalg.inv(warpMatrix) maps inputImage coordinates to templateImage coordinates
                    warpMatrix = zoom_matrix(1 / zoom) @ warpMatrix_zoom @ zoom_matrix(zoom)
                    translation, reflection, rotation_angle, scaling, shear, out_of_bounds = decompose_warp_matrix(warpMatrix)
                    ecc_uvs = np.linalg.inv(warpMatrix) @ np.vstack([uvs, np.ones((uvs.shape[1],), dtype=np.float32)])
                    ecc_uvs = ecc_uvs[:2, :] / ecc_uvs[2, :]
                    ecc_uv_grid = ecc_uvs.reshape(uv_grid.shape).astype(np.float32)

                    cross_flow_map = np.transpose(ecc_uv_grid, axes=(1, 2, 0))

                    border_value = 127
                    gray_cross_warp = cv2.remap(gray_warp,
                                                cross_flow_map, None, cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)

                    if exec_mode == 'debug_key_frame' and current_frame_idx == debug_key_frame_idx:
                        plt.figure('findTransformECC', figsize=(24, 12))
                        setup_new_fig_page()
                        plt.suptitle('\n'.join([f'cross, current frame idxs: {cross_frame_idx}, {current_frame_idx}',
                                                f'motionType, zoom, gaussFiltSize: {(motionType, zoom, gaussFiltSize)}',
                                                f'translation, reflection, rotation_angle: {(tuple(translation), reflection, rotation_angle)}',
                                                f'scaling, shear, out_of_bounds: {(tuple(scaling), tuple(shear), out_of_bounds)}']))
                        ax = plt.subplot(2, 3, 1)
                        plt.imshow(gray_warp, cmap='gray')
                        plt.title('gray_warp')
                        xys_extent = (-0.5, gray_warp.shape[1] - 0.5, gray_warp.shape[0] - 0.5, -0.5)
                        ax = plt.subplot(2, 3, 2, sharex=ax, sharey=ax)
                        #plt.imshow(cv2.resize(gray_warp, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA), cmap='gray', extent=xys_extent)
                        plt.imshow(cv2.resize(img_warp, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA), cmap='gray', extent=xys_extent)
                        plt.title('img_warp')
                        ax = plt.subplot(2, 3, 3, sharex=ax, sharey=ax)
                        plt.imshow(ref_gray_ext_attenuated, cmap='gray')
                        plt.title('ref_gray_ext_attenuated')
                        ax = plt.subplot(2, 3, 4, sharex=ax, sharey=ax)
                        #plt.imshow(cv2.resize(ref_gray_ext_attenuated, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA), cmap='gray', extent=xys_extent)
                        plt.imshow(cv2.resize(ref_img_ext, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA), cmap='gray', extent=xys_extent)
                        plt.title('ref_img_ext')
                        ax = plt.subplot(2, 3, 5, sharex=ax, sharey=ax)
                        plt.imshow(input_mask, extent=xys_extent)
                        plt.title('input_mask')
                        ax = plt.subplot(2, 3, 6, sharex=ax, sharey=ax)
                        plt.imshow(gray_cross_warp, cmap='gray')
                        plt.title('gray_cross_warp')
                        plt.tight_layout()
                        stash_fig_page()

                    if out_of_bounds:
                        break

                translation, reflection, rotation_angle, scaling, shear, out_of_bounds = decompose_warp_matrix(warpMatrix)
                if out_of_bounds:
                    continue

                sample_weight_mask = sample_weight > 0.1 * np.median(sample_weight)

                projected_points = camera_matrix @ cross_triangulated_points[:, inlier_mask]
                projected_points = projected_points[:2, :] / projected_points[2, :]

                xys_mask = (sample_weight_mask & (np.abs(model_errs) < ransac_threshold * 3)
                            & (projected_points[0, :] > -0.5) & (projected_points[0, :] < imageSize[0] - 0.5)
                            & (projected_points[1, :] > -0.5) & (projected_points[1, :] < imageSize[1] - 0.5))

                xs, ys = inlier_uv[:, xys_mask]

                interp = scipy.interpolate.RegularGridInterpolator((np.arange(cross_flow_map.shape[0]), np.arange(cross_flow_map.shape[1])),
                                                                   cross_flow_map[:, :, 0] + cross_flow_map[:, :, 1] * 1j,
                                                                   method='linear', bounds_error=False, fill_value=np.nan)
                cross_flow_map_interp = interp((ys, xs)).astype(np.complex64)
                cross_flow_map_xys = np.vstack([cross_flow_map_interp.real, cross_flow_map_interp.imag]).T

                interp = scipy.interpolate.RegularGridInterpolator((np.arange(gray_warp_map.shape[0]), np.arange(gray_warp_map.shape[1])),
                                                                   gray_warp_map[:, :, 0] + gray_warp_map[:, :, 1] * 1j,
                                                                   method='linear', bounds_error=False, fill_value=np.nan)
                gray_warp_map_interp = interp((ys, xs)).astype(np.complex64)
                gray_warp_map_xys = np.vstack([gray_warp_map_interp.real, gray_warp_map_interp.imag]).T

                gray_flow_interp = interp((cross_flow_map_xys[:, 1], cross_flow_map_xys[:, 0])).astype(np.complex64)
                gray_flow_xys = np.vstack([gray_flow_interp.real, gray_flow_interp.imag])

                cross_image_to_triangulated_point_idxs = key_frame_image_triangulated_point_idxs[cross_frame_idx]
                cross_image_triangulated_point_idxs = cross_image_to_triangulated_point_idxs[(cross_image_to_triangulated_point_idxs >= 0)
                                                                                             & (cross_image_to_triangulated_point_idxs < len(triangulated_idxs_weights))]
                cross_image_triangulated_points = np.full((len(triangulated_idxs_weights),), fill_value=False, dtype=bool)
                cross_image_triangulated_points[cross_image_triangulated_point_idxs] = True
                cross_image_triangulated_points_xys = cross_image_triangulated_points[inlier_mask][xys_mask]

                cross_flow_mask = ((gray_flow_xys[0, :] > -0.5) & (gray_flow_xys[0, :] < imageSize[0] - 0.5)
                                   & (gray_flow_xys[1, :] > -0.5) & (gray_flow_xys[1, :] < imageSize[1] - 0.5)
                                   & (gray_warp_map_xys[:, 0] > -0.5) & (gray_warp_map_xys[:, 0] < imageSize[0] - 0.5)
                                   & (gray_warp_map_xys[:, 1] > -0.5) & (gray_warp_map_xys[:, 1] < imageSize[1] - 0.5)
                                   & cross_image_triangulated_points_xys)

                xs = xs[cross_flow_mask]
                ys = ys[cross_flow_mask]
                gray_flow_xys = gray_flow_xys[:, cross_flow_mask]
                gray_warp_map_xys = gray_warp_map_xys[cross_flow_mask, :]
                cross_warp_triangulated_point_idxs = np.arange(len(triangulated_idxs_weights))[inlier_mask][xys_mask][cross_flow_mask]
                ref_gray_points = cross_triangulated_points[:, cross_warp_triangulated_point_idxs]
                ref_gray_triangulated_idxs_weights = triangulated_idxs_weights[cross_warp_triangulated_point_idxs]

                cross_warp_triangulated_points[cross_frame_idx] = (gray_flow_xys, cross_warp_triangulated_point_idxs)

                if False:
                    current_projected_points = camera_matrix @ current_triangulated_points[:, inlier_mask][:, xys_mask][:, cross_flow_mask]
                    current_projected_points = current_projected_points[:2, :] / current_projected_points[2, :]

                    if gray_flow_xys.shape[1] >= 30:

                        results = []
                        for iter_idx in range(5000):
                            np.random.seed(iter_idx)
                            sample_fraction = np.random.uniform(1 / 3, 3 / 4)
                            sample_size = np.clip(int(np.ceil(gray_flow_xys.shape[1] * sample_fraction)), 50, gray_flow_xys.shape[1])
                            pts_idxs = np.random.choice(np.arange(gray_flow_xys.shape[1]), size=(sample_size,), replace=False)

                            # Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
                            #threshold = min(0.5 * (1 + motion_blur), 2.0)
                            threshold = 3.0
                            camera_transform = current_camera_extrinsic @ np.linalg.inv(cross_camera_extrinsic)
                            rvec, jacobian = cv2.Rodrigues(camera_transform[:3, :3])
                            tvec = camera_transform[:3, 3:]
                            retval, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints=ref_gray_points[:, pts_idxs].T,
                                                                             imagePoints=gray_flow_xys[:, pts_idxs].T,
                                                                             cameraMatrix=camera_matrix, distCoeffs=np.zeros((4,)),
                                                                             rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
                                                                             iterationsCount=10, reprojectionError=threshold, confidence=0.999, inliers=None,
                                                                             flags=cv2.SOLVEPNP_ITERATIVE)
                            #assert retval == True
                            if retval:
                                R, jacobian = cv2.Rodrigues(rvec)
                                t = tvec

                                camera_transform = np.block([[R, t], [0, 0, 0, 1]])

                                object_points = camera_transform @ np.vstack([ref_gray_points, np.ones((ref_gray_points.shape[1],))])
                                projected_points = camera_matrix @ object_points[:3, :]
                                projected_points = projected_points[:2, :] / projected_points[2, :]

                                #threshold = min(0.5 * (1 + motion_blur), 2.0)
                                threshold = 6.0
                                triangulated_points_inliers = np.where(np.linalg.norm(gray_flow_xys.T - projected_points.T, axis=1) < threshold)[0]

                                # TODO: include in score the entropy of the distribution of errors in space, in terms of whether
                                # errors are smaller in one region than another, or whether errors are more negative in one region
                                # than another

                                sigma_prev = 3.0
                                score = (np.sum((ref_gray_triangulated_idxs_weights + 1)
                                                * np.exp(-0.5 * np.power(np.linalg.norm(gray_flow_xys.T - projected_points.T, axis=1) / sigma_prev, 2)))
                                         / np.sum(ref_gray_triangulated_idxs_weights + 1))

                                results.append((rvec, tvec, triangulated_points_inliers, score))

                            if sum([score for _, _, _, score in results]) >= 100:
                                break

                        results.sort(key=lambda item: item[3], reverse=True)
                        print('cv2.solvePnPRansac len(results)', len(results))

                        transform_delta_ref_vec = np.array([(1, 0, 8), (0, 0, 8), (0, 0, 9)]).T
                        transform_delta_vecs = []
                        result_scores = []
                        for rvec, tvec, _, score in results:
                            R, jacobian = cv2.Rodrigues(rvec)
                            t = tvec
                            transform_delta_vecs.append((R @ transform_delta_ref_vec + t - transform_delta_ref_vec).flatten())
                            result_scores.append(score)
                        transform_delta_vecs = np.array(transform_delta_vecs)
                        result_scores = np.array(result_scores)[:, None]



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

                        rvec, tvec, triangulated_points_inliers, score = results[0]

                        print('cv2.solvePnPRansac extrinsic rotation', rvec.flatten())
                        print('cv2.solvePnPRansac extrinsic translation', tvec.flatten())
                        print('cv2.solvePnPRansac len(triangulated_points_inliers)', len(triangulated_points_inliers))
                        print('cv2.solvePnPRansac inliers score', score)

                        R, jacobian = cv2.Rodrigues(rvec)
                        t = tvec
                        E = R @ np.cross(np.identity(3), t.T)
                        camera_transform = np.block([[R, t], [0, 0, 0, 1]])




                    err_inlier_mask = sample_weight_mask & (np.abs(model_errs) < ransac_threshold)
                    err_outlier_mask = sample_weight_mask & (np.abs(model_errs) >= ransac_threshold)


                    plt.figure('Cross flow refinement experiment 1', figsize=(24, 12))
                    setup_new_fig_page()
                    plt.suptitle(f'cross, current frame idxs: {cross_frame_idx}, {current_frame_idx}')
                    ax = plt.subplot(3, 4, 1)
                    plt.imshow(ref_gray_ext, cmap='gray', vmin=0, vmax=255)
                    plt.scatter(*inlier_uv, s=1.0*np.power(sample_weight, 2), c=inlier_z, cmap='jet', alpha=0.2)
                    plt.colorbar()
                    ax = plt.subplot(3, 4, 2, sharex=ax, sharey=ax)
                    plt.imshow(ref_gray_ext, cmap='gray', vmin=0, vmax=255)
                    plt.scatter(*inlier_uv[:, sample_weight_mask], s=1.0*np.power(sample_weight[sample_weight_mask], 2), c=sample_weight[sample_weight_mask], cmap='jet', alpha=0.2)
                    plt.colorbar()
                    ax = plt.subplot(3, 4, 3, sharex=ax, sharey=ax)
                    plt.imshow(predicted_z)
                    plt.colorbar()
                    ax = plt.subplot(3, 4, 4, sharex=ax, sharey=ax)
                    plt.imshow(ref_gray_ext, cmap='gray', vmin=0, vmax=255)
                    plt.scatter(*inlier_uv[:, sample_weight_mask], s=0.1*np.power(model_errs[sample_weight_mask], 2), c=np.abs(model_errs[sample_weight_mask]), cmap='jet', alpha=0.2)
                    plt.colorbar()
                    ax = plt.subplot(3, 4, 5, sharex=ax, sharey=ax)
                    plt.imshow(ref_gray_ext, cmap='gray', vmin=0, vmax=255)
                    plt.scatter(*inlier_uv[:, err_inlier_mask], s=5.0, c='blue', alpha=0.2)
                    plt.scatter(*inlier_uv[:, err_outlier_mask], s=5.0, c='red', alpha=0.2)
                    ax = plt.subplot(3, 4, 6, sharex=ax, sharey=ax)
                    plt.imshow(cross_model_errs, vmin=0, vmax=3)
                    plt.colorbar()
                    ax = plt.subplot(3, 4, 7, sharex=ax, sharey=ax)
                    plt.imshow(ref_gray_ext_attenuated, cmap='gray', vmin=0, vmax=255)
                    plt.subplot(3, 4, 8)
                    plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
                    ax = plt.subplot(3, 4, 9, sharex=ax, sharey=ax)
                    plt.imshow(gray_warp, cmap='gray', vmin=0, vmax=255)
                    ax = plt.subplot(3, 4, 10, sharex=ax, sharey=ax)
                    plt.imshow(gray_cross_warp, cmap='gray', vmin=0, vmax=255)
                    ax = plt.subplot(3, 4, 11, sharex=ax, sharey=ax)
                    #plt.imshow(ref_gray_cross_warp, cmap='gray', vmin=0, vmax=255)
                    ax = plt.subplot(3, 4, 12, sharex=ax, sharey=ax)
                    #plt.imshow(cross_flow_consistency_err)
                    plt.tight_layout()
                    stash_fig_page()

                    plt.figure('Cross flow refinement experiment 2', figsize=(24, 12))
                    setup_new_fig_page()
                    plt.suptitle(f'cross, current frame idxs: {cross_frame_idx}, {current_frame_idx}')
                    plt.subplot(2, 3, 1)
                    plt.imshow(ref_gray_ext_attenuated, cmap='gray', vmin=0, vmax=255)
                    plt.scatter(xs, ys, s=3, c='blue', alpha=0.2)
                    ax = plt.subplot(2, 3, 2)
                    plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
                    plt.scatter(*current_projected_points, s=3, c='blue', alpha=0.2)
                    ax = plt.subplot(2, 3, 3, sharex=ax, sharey=ax)
                    plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
                    plt.scatter(*gray_warp_map_xys.T, s=3, c='blue', alpha=0.2)
                    ax = plt.subplot(2, 3, 4, sharex=ax, sharey=ax)
                    plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
                    plt.scatter(*gray_flow_xys, s=3, c='yellow', alpha=0.2)
                    ax = plt.subplot(2, 3, 5, sharex=ax, sharey=ax)
                    plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
                    plt.scatter(*gray_flow_xys, s=3, c='yellow', alpha=0.2)
                    plt.scatter(*gray_warp_map_xys.T, s=3, c='blue', alpha=0.2)
                    segments = list(zip(gray_flow_xys.T, gray_warp_map_xys))
                    ax.add_collection(matplotlib.collections.LineCollection(segments, colors='red', linewidths=1, alpha=0.2))
                    plt.tight_layout()
                    stash_fig_page()


            inv_camera_extrinsic = np.linalg.inv(key_frame_camera_extrinsics[current_frame_idx])
            R = inv_camera_extrinsic[:3, :3]
            t = inv_camera_extrinsic[:3, 3:]
            model_triangulated_points = R @ stitched_triangulated_points[:, :3].T + t

            warp_image_points = []
            warp_image_points_weights = []
            for cross_frame_idx in range(current_frame_idx):

                image_point_weights = key_frame_image_point_weights[current_frame_idx]

                image_points = np.full((2, stitched_triangulated_points.shape[0]), fill_value=np.nan, dtype=np.float32)
                image_points_weights = np.zeros((6, stitched_triangulated_points.shape[0]))

                if cross_frame_idx in cross_warp_triangulated_points:
                    gray_flow_xys, triangulated_point_idxs = cross_warp_triangulated_points[cross_frame_idx]

                    image_points[:, triangulated_point_idxs] = gray_flow_xys

                    primary_eig_weights, secondary_eig_weights, primary_eig_vecs = image_point_weights
                    eig_weights = primary_eig_weights + secondary_eig_weights * 1j
                    interp = scipy.interpolate.RegularGridInterpolator((np.arange(eig_weights.shape[0]), np.arange(eig_weights.shape[1])),
                                                                       eig_weights,
                                                                       method='linear', bounds_error=False, fill_value=None)
                    eig_weights = interp((image_points[1, triangulated_point_idxs], image_points[0, triangulated_point_idxs])).astype(np.complex64)
                    eig_vecs = primary_eig_vecs[:, :, 0] + primary_eig_vecs[:, :, 1] * 1j
                    interp = scipy.interpolate.RegularGridInterpolator((np.arange(eig_vecs.shape[0]), np.arange(eig_vecs.shape[1])),
                                                                       eig_vecs,
                                                                       method='linear', bounds_error=False, fill_value=None)
                    eig_vecs = interp((image_points[1, triangulated_point_idxs], image_points[0, triangulated_point_idxs])).astype(np.complex64)
                    eig_weights = np.vstack([eig_weights.real, eig_weights.imag,
                                             eig_vecs.real, eig_vecs.imag,
                                             eig_vecs.imag, -eig_vecs.real])
                    image_points_weights[:, triangulated_point_idxs] = eig_weights

                    triangulated_points = camera_extrinsic[:3, :3] @ model_triangulated_points[:, triangulated_point_idxs] + camera_extrinsic[:3, 3:]
                    triangulated_normals = camera_extrinsic[:3, :3] @ stitched_triangulated_normals[:, triangulated_point_idxs]
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

                warp_image_points.append(image_points)
                warp_image_points_weights.append(image_points_weights)
            warp_image_points = np.array(warp_image_points)
            warp_image_points_weights = np.array(warp_image_points_weights)

            model_triangulated_points = torch.tensor(model_triangulated_points[:, bundle_triangulated_point_idxs], dtype=torch.float32, requires_grad=True)
            warp_image_points = torch.tensor(warp_image_points[:, :, bundle_triangulated_point_idxs], dtype=torch.float32)
            warp_image_points[~torch.isfinite(warp_image_points)] = 0
            warp_image_points_weights = torch.tensor(warp_image_points_weights[:, :, bundle_triangulated_point_idxs], dtype=torch.float32)

            def calc_projected_image_points():
                model_Rs = pytorch3d.transforms.axis_angle_to_matrix(model_rvecs)
                object_points = model_Rs @ model_triangulated_points + model_tvecs

                xd = object_points[:, 0:1, :] / torch.clamp(object_points[:, 2:3, :], 1e-3, np.inf)
                yd = object_points[:, 1:2, :] / torch.clamp(object_points[:, 2:3, :], 1e-3, np.inf)

                return torch.cat([xd, yd], dim=1) * camera_fxy + camera_cxy

            def loss_fn(temperature):
                bundle_projected_points = calc_projected_image_points()
                projection_offset = bundle_projected_points - bundle_image_points
                projection_err = torch.stack([torch.sum(projection_offset * bundle_image_points_weights[:, 2:4, :], dim=1),
                                              torch.sum(projection_offset * bundle_image_points_weights[:, 4:, :], dim=1)], dim=1)
                warp_projection_offset = bundle_projected_points[current_frame_idx, :, :] - warp_image_points
                warp_projection_err = torch.stack([torch.sum(warp_projection_offset * warp_image_points_weights[:, 2:4, :], dim=1),
                                                   torch.sum(warp_projection_offset * warp_image_points_weights[:, 4:, :], dim=1)], dim=1)
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
                    warp_projection_err_losses = gamma_softplus(warp_projection_err, threshold=threshold, alpha=gamma_softplus_alpha, relative_outer_gradient=0.01)

                return (torch.sum(projection_err_losses * bundle_image_points_weights[:, :2, :])
                        + torch.sum(warp_projection_err_losses * warp_image_points_weights[:, :2, :]))

            lr = 0.005
            num_steps = 3000
            convergence_criterion = {'rtol': 1e-4, 'window_size': 100, 'min_num_steps': 300}
            optimiser = torch.optim.Adam([model_rvecs, model_tvecs, model_triangulated_points], lr=lr)
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
                    print('optim_step', optim_step, 'loss', losses[-1], 'loss_window_std', loss_window_std,
                          'learning rate', np.round(scheduler.get_last_lr()[0], 6), 'temperature', np.round(temperature, 3))
                if optim_step >= convergence_criterion['min_num_steps'] and loss_window_std < convergence_criterion['rtol'] * losses[-1]:
                    break

            print('len(losses)', len(losses))
            print('initial, first, final loss', losses[0], losses[convergence_criterion['min_num_steps']], losses[-1])
            print('highest, lowest loss', np.max(losses[convergence_criterion['min_num_steps']:]), np.min(losses[convergence_criterion['min_num_steps']:]))

            del key_frame_camera_extrinsics[:]
            for rvec, tvec in zip(model_rvecs.numpy(force=True), model_tvecs.numpy(force=True)):
                R, jacobian = cv2.Rodrigues(rvec)
                t = tvec
                key_frame_camera_extrinsics.append(np.block([[R, t], [0, 0, 0, 1]]))

            camera_transform = key_frame_camera_extrinsics[current_frame_idx] @ np.linalg.inv(key_frame_camera_extrinsics[current_frame_idx - 1])
            R = camera_transform[:3, :3]
            t = camera_transform[:3, 3:]
            optim_transform_delta_vec = (R @ transform_delta_ref_vec + t - transform_delta_ref_vec).flatten()
            print('optimised transform delta vec', np.round(optim_transform_delta_vec, 3))
            torch.set_default_device(torch.device('cpu'))
            for idx in range(2):
                print(f'gmm[{idx}] log_prob(optim_transform_delta_vec)', np.round(gmm_distributions[idx].log_probability(optim_transform_delta_vec[None, :]).numpy(), 3))
            torch.set_default_device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            # TODO: calculate and record epipolar inliers as a measure of traction / slip

            if False:
                current_camera_extrinsic = key_frame_camera_extrinsics[current_frame_idx]
                R = current_camera_extrinsic[:3, :3]
                t = current_camera_extrinsic[:3, 3:]
                stitched_triangulated_points[bundle_triangulated_point_idxs, :3] = (R @ model_triangulated_points.numpy(force=True) + t).T
            else:
                first_camera_extrinsic = key_frame_camera_extrinsics[0]
                R = first_camera_extrinsic[:3, :3]
                t = first_camera_extrinsic[:3, 3:]
                model_triangulated_points = R @ model_triangulated_points.numpy(force=True) + t

                key_frame_camera_extrinsics[:] = [camera_extrinsic @ np.linalg.inv(first_camera_extrinsic)
                                                  for camera_extrinsic in key_frame_camera_extrinsics]

                scaling_factors = []
                for image_points_weights_magn, camera_extrinsic in zip(torch.norm(bundle_image_points_weights[:, :2, :], dim=1).numpy(force=True),
                                                                       key_frame_camera_extrinsics):
                    triangulated_points = camera_extrinsic[:3, :3] @ model_triangulated_points[:, image_points_weights_magn >= 1e-2] + camera_extrinsic[:3, 3:]

                    # Estimate the scaling factor from the depth distribution of the triangulated points
                    # assuming the camera is positioned at a distance of ~8mm from the target
                    c = np.percentile(triangulated_points[2, :], 25) / 8.0
                    scaling_factors.append(c)
                c = np.median(scaling_factors)

                print('scaling factor', c, np.round(scaling_factors, 2))
                model_triangulated_points /= c
                key_frame_camera_extrinsics[:] = [np.block([[camera_extrinsic[:3, :3], camera_extrinsic[:3, 3:] / c], [0, 0, 0, 1]])
                                                  for camera_extrinsic in key_frame_camera_extrinsics]

                current_camera_extrinsic = key_frame_camera_extrinsics[current_frame_idx]
                R = current_camera_extrinsic[:3, :3]
                t = current_camera_extrinsic[:3, 3:]
                stitched_triangulated_points[bundle_triangulated_point_idxs, :3] = (R @ model_triangulated_points + t).T


            # Cross stitching / loop closure between each previous frame with the current frame.
            # Retrace triangulated points that previously moved out of view and lost sequential optical flow tracking,
            # but have re-entered the current view.
            # These are the set of unmatched points between each previous cross frame and the current frame.
            # For each previous cross frame:
            #  - Calculate and apply the stereo camera rectification transforms corresponding to the camera extrinsic transform
            #    and compute the disparity map between the rectified images.
            #  - Derive the cross flow from the disparity map and apply it to map the current frame image onto the cross frame image space.
            #  - Compute the residual optical flow between the mapped current frame image and the cross frame image and apply the
            #    residual optical flow vectors to update the location of the cross mapped triangulated point projections.
            # For each triangulated point, cluster the collection of unmatched cross flow mapped coordinates across all the cross frames.
            # Stitch the triangulated point if the modal cluster has a small enough variance and its centroid is close enough
            # to the projected location of the triangulated point in the current frame.

            current_triangulated_image_idxs = np.where(current_image_to_triangulated_point_idxs >= 0)[0]
            current_triangulated_idxs = current_image_to_triangulated_point_idxs[current_triangulated_image_idxs]
            current_triangulated_idxs_to_image_idxs = dict(zip(current_triangulated_idxs, current_triangulated_image_idxs))

            valid_mask = np.all(np.isfinite(stitched_triangulated_points[:, :3]), axis=1)
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(stitched_triangulated_points[valid_mask, :3]))
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
            pcd.transform(np.linalg.inv(key_frame_camera_extrinsics[current_frame_idx]))
            pcd_points = np.array(pcd.points).T
            pcd_normals = np.array(pcd.normals).T
            normal_ray_alignment = np.zeros((np.sum(valid_mask),), dtype=np.float32)
            for image_to_triangulated_point_idxs, camera_extrinsic in zip(key_frame_image_triangulated_point_idxs + [current_image_to_triangulated_point_idxs],
                                                                          key_frame_camera_extrinsics):
                triangulated_image_idxs = np.where(image_to_triangulated_point_idxs >= 0)[0]
                triangulated_point_idxs = image_to_triangulated_point_idxs[triangulated_image_idxs]
                image_points_weights = np.zeros((stitched_triangulated_points.shape[0],), dtype=np.float32)
                image_points_weights[triangulated_point_idxs] = 1

                R = camera_extrinsic[:3, :3]
                t = camera_extrinsic[:3, 3:]
                object_points = R @ pcd_points + t
                object_normals = R @ pcd_normals

                object_point_ray = object_points / np.clip(np.linalg.norm(object_points, axis=0), 1e-6, np.inf)
                normal_ray_alignment += np.sum(object_point_ray * object_normals, axis=0) * image_points_weights[valid_mask]
            pcd.normals = o3d.utility.Vector3dVector((pcd_normals * -np.sign(normal_ray_alignment)).T)
            pcd.transform(key_frame_camera_extrinsics[current_frame_idx])
            assert np.allclose(np.linalg.norm(np.array(pcd.normals), axis=1), 1)

            #     z               object-plane
            #     z             or
            #     z           o  r
            #     z         o   r
            #     z       o     ray-from_camera
            #     z     o      r
            #     z   o        r
            #     z o         r
            # iiiioiiiiiiiiiiiriiiimage-plane
            #   o z  n       r
            # o   z     n    r
            #     z        nr
            #     z         r n
            #     z        r     normal-from-object-plane

            # The image plane, object plane, object plane normal and z-axis are coincident at oi = on = oz
            # object-plane section or-oi is projected onto image-plane section ir-oi
            # Perspective distortion is based on object_to_image_ratio = |or-oi| / |ir-oi|
            # camera_rays = (or-nr) / |or-nr| = (ir-nr) / |ir-nr|
            # camera_ray_to_object_plane = or-nr
            # camera_ray_to_image_plane = ir-nr

            # |nr-oi| = 1
            # ⇒ (nr-oi) ⋅ (or-nr) = -1
            # ⇒ |or-nr| = -1 / [(nr-oi) ⋅ (or-nr) / |or-nr|] where (or-nr) / |or-nr| is the unit length camera ray

            # Given the unit vector z on the z-axis
            # (ir-nr) ⋅ z = - (nr-oi) ⋅ z
            # ⇒ |ir-nr| = - [(nr-oi) ⋅ z] / [(ir-nr) / |ir-nr| ⋅ z] where (ir-nr) / |ir-nr| is the unit length camera ray

            triangulated_points = np.full((3, stitched_triangulated_points.shape[0]), fill_value=np.nan, dtype=np.float32)
            triangulated_points[:, valid_mask] = np.array(pcd.points).T
            triangulated_normals = np.full((3, stitched_triangulated_points.shape[0]), fill_value=np.nan, dtype=np.float32)
            triangulated_normals[:, valid_mask] = np.array(pcd.normals).T
            camera_rays = triangulated_points / np.clip(np.linalg.norm(triangulated_points, axis=0), 1e-6, np.inf)
            normal_ray_alignment = np.sum(camera_rays * triangulated_normals, axis=0)
            camera_ray_to_object_plane = camera_rays / np.clip(-normal_ray_alignment, 1e-8, 1)
            camera_ray_to_image_plane = camera_rays * -triangulated_normals[2, :] / camera_rays[2, :]

            object_to_image_ratio = (np.linalg.norm(triangulated_normals + camera_ray_to_object_plane, axis=0)
                                     / np.clip(np.linalg.norm(triangulated_normals + camera_ray_to_image_plane, axis=0), 1e-8, np.inf))
            perspective_distortion = 1 - np.min(np.stack([object_to_image_ratio,
                                                          np.clip(1 / object_to_image_ratio, 1e-8, np.inf)]), axis=0)

            triangulated_normal_weights = 0.5 * (1 - scipy.special.erf((perspective_distortion - 0.6) / 0.1))
            triangulated_normal_weights[normal_ray_alignment >= 0] = 0

            retrace_triangulated_candidates = collections.defaultdict(dict)

            for cross_frame_idx in range(current_frame_idx):
                print('cross_frame_idx', cross_frame_idx)

                #cross_motion_blur = np.max([key_frame_motion_blurs[idx] for idx in [cross_frame_idx, current_frame_idx]])

                cross_camera_extrinsic = key_frame_camera_extrinsics[cross_frame_idx]

                ref_img, ref_gray = key_frame_images[cross_frame_idx]
                #img, gray = key_frame_images[current_frame_idx]

                ref_img_mask = key_frame_masks[cross_frame_idx]

                # Extract the indices of the common set of triangulated points between the cross and current frames
                xsc, ysc = key_frame_image_sample_points[cross_frame_idx]
                cross_image_points = np.vstack([xsc, ysc])
                cross_image_to_triangulated_point_idxs = key_frame_image_triangulated_point_idxs[cross_frame_idx]
                cross_triangulated_image_idxs = np.where(cross_image_to_triangulated_point_idxs >= 0)[0]
                cross_triangulated_idxs = cross_image_to_triangulated_point_idxs[cross_triangulated_image_idxs]
                valid_mask = np.all(np.isfinite(stitched_triangulated_points[cross_triangulated_idxs, :3]), axis=1)
                cross_triangulated_image_idxs = cross_triangulated_image_idxs[valid_mask]
                cross_triangulated_idxs = cross_triangulated_idxs[valid_mask]
                cross_triangulated_idxs_to_image_idxs = dict(zip(cross_triangulated_idxs, cross_triangulated_image_idxs))

                common_triangulated_idxs = np.array(list(set(list(cross_triangulated_idxs)) & set(list(current_triangulated_idxs))), dtype=int)
                cross_unmatched_triangulated_idxs = np.array(list(set(list(cross_triangulated_idxs)) - set(list(current_triangulated_idxs))), dtype=int)
                print('len(common_triangulated_idxs)', len(common_triangulated_idxs))
                print('len(cross_unmatched_triangulated_idxs)', len(cross_unmatched_triangulated_idxs))

                cross_triangulated_image_idxs = list(map(cross_triangulated_idxs_to_image_idxs.get, common_triangulated_idxs))
                cross_triangulated_image_points = cross_image_points[:, cross_triangulated_image_idxs].T
                current_triangulated_image_idxs = list(map(current_triangulated_idxs_to_image_idxs.get, common_triangulated_idxs))
                current_triangulated_image_points = np.vstack([xsn, ysn])[:, current_triangulated_image_idxs].T
                cross_unmatched_triangulated_image_idxs = list(map(cross_triangulated_idxs_to_image_idxs.get, cross_unmatched_triangulated_idxs))
                cross_unmatched_triangulated_image_points = cross_image_points[:, cross_unmatched_triangulated_image_idxs].T

                inv_current_camera_extrinsic = np.linalg.inv(current_camera_extrinsic)
                cross_triangulated_points = (inv_current_camera_extrinsic[:3, :3] @ stitched_triangulated_points[common_triangulated_idxs, :3].T
                                             + inv_current_camera_extrinsic[:3, 3:]).T
                cross_triangulated_rgb = stitched_triangulated_points[common_triangulated_idxs, 3:]

                cross_object_points = cross_camera_extrinsic @ np.vstack([cross_triangulated_points.T, np.ones((cross_triangulated_points.shape[0],))])
                cross_projected_points = camera_matrix @ cross_object_points[:3, :]
                cross_projected_points = cross_projected_points[:2, :] / cross_projected_points[2, :]

                current_object_points = current_camera_extrinsic @ np.vstack([cross_triangulated_points.T, np.ones((cross_triangulated_points.shape[0],))])
                current_projected_points = camera_matrix @ current_object_points[:3, :]
                current_projected_points = current_projected_points[:2, :] / current_projected_points[2, :]

                #threshold = min(0.5 * (1 + cross_motion_blur), 2.0)
                threshold = 6.0
                cross_triangulated_inlier_idxs = np.where(np.linalg.norm(current_triangulated_image_points - current_projected_points.T, axis=1) < threshold)[0]
                print('num cross triangulated point inliers', len(cross_triangulated_inlier_idxs))

                # TODO: Resolve the different causes of low disparity confidence, in particular the case where
                #       there is only partial visibility of the object as opposed to erroneous camera extrinsic estimates.
                #       Investigate whether the triangulated points can be used to resolve this.

                if exec_mode == 'debug_key_frame' and current_frame_idx == debug_key_frame_idx:
                    plt.figure('Cross projected triangulated points', figsize=(16, 10))
                    setup_new_fig_page()
                    plt.suptitle(f'cross, current frame idxs: {cross_frame_idx}, {current_frame_idx}')
                    ax = plt.subplot(2, 3, 1)
                    plt.imshow(np.require(ref_img, dtype=np.uint8))
                    ax = plt.subplot(2, 3, 2, sharex=ax, sharey=ax)
                    plt.imshow(np.require(img, dtype=np.uint8))
                    ax3 = plt.subplot(2, 3, 3, projection='3d')
                    ax3.scatter(*cross_triangulated_points.T, s=2, c=cross_triangulated_rgb/255)
                    ax3.set_xlim((-20, 20))
                    ax3.set_ylim((-20, 20))
                    ax3.set_zlim((0, 40))
                    ax3.set_aspect('equal', adjustable='datalim')
                    ax3.set_xlabel('X')
                    ax3.set_ylabel('Y')
                    ax3.set_zlabel('Z')
                    ax3.view_init(elev=-135, azim=-90, roll=0)
                    ax = plt.subplot(2, 3, 4, sharex=ax, sharey=ax)
                    plt.scatter(*cross_triangulated_image_points.T, s=2, c='b', marker='o')
                    plt.scatter(*cross_projected_points, s=2, c='y', marker='o')
                    ax.set_aspect('equal')
                    ax = plt.subplot(2, 3, 5, sharex=ax, sharey=ax)
                    plt.scatter(*current_triangulated_image_points.T, s=2, c='b', marker='o')
                    plt.scatter(*current_projected_points, s=2, c='y', marker='o')
                    ax.set_aspect('equal')
                    plt.tight_layout()
                    stash_fig_page()


                camera_transform = current_camera_extrinsic @ np.linalg.inv(cross_camera_extrinsic)
                R = camera_transform[:3, :3]
                t = camera_transform[:3, 3:]

                # Project the current frame's camera view cone onto the cross frame's camera image and calculate
                # the intersection as a measure of the disparity that can potentially be derived
                uvs = np.array([[0, 0], [img.shape[1] - 1, 0], [img.shape[1] - 1, img.shape[0] - 1], [0, img.shape[0] - 1]]).T
                uvcs = uvs - camera_matrix[:2, 2:]
                xys = np.linalg.inv(camera_matrix[:2, :2]) @ uvcs
                xyzs = np.vstack([xys, np.ones(xys.shape[1],)])
                xyzs = np.hstack([xyzs * 5, xyzs * 30])
                xyzs = R.T @ xyzs - R.T @ t
                projected_points = camera_matrix @ xyzs
                projected_points = projected_points[:2, :] / projected_points[2, :]
                projected_points_geometry = shapely.MultiPoint(projected_points.T).convex_hull
                cross_image_geometry = shapely.MultiPoint(uvs.T).convex_hull
                projected_cone_coverage = cross_image_geometry.intersection(projected_points_geometry).area / cross_image_geometry.area
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
                R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(camera_matrix, None,
                                                                                  camera_matrix, None,
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
                #rect_map_prev, _ = cv2.initUndistortRectifyMap(camera_matrix, None, R1, P1[:3, :3], newImageSize, cv2.CV_32FC2)
                #rect_map_next, _ = cv2.initUndistortRectifyMap(camera_matrix, None, R2, P2[:3, :3], newImageSize, cv2.CV_32FC2)

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
                    uvcs = uvs - camera_matrix[:2, 2:]
                    uvcs_clip = (max(img.shape[:2]) - 1) / 2 - img_size_trim
                    uvcs = np.clip(uvcs, -uvcs_clip, uvcs_clip)
                    xys = np.linalg.inv(camera_matrix[:2, :2]) @ uvcs
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
                uvcs = uvs - camera_matrix[:2, 2:]
                xys = np.linalg.inv(camera_matrix[:2, :2]) @ uvcs
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

                disparity_map_zoom = 0.125
                fxy = (disparity_map_zoom * np.diag(camera_matrix[:2, :2])
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
                if exec_mode == 'debug_key_frame' and current_frame_idx == debug_key_frame_idx:
                    plt.figure('dx12s', figsize=(16, 10))
                    setup_new_fig_page()
                    plt.suptitle(f'cross, current frame idxs: {cross_frame_idx}, {current_frame_idx}\ndisparity_spread {disparity_spread}')
                    hist_bins = np.arange(np.floor(np.min(dx12s[common_view_frustum_inlier_mask])),
                                          np.floor(np.max(dx12s[common_view_frustum_inlier_mask])) + 1)
                    plt.hist(dx12s[common_view_frustum_inlier_mask], bins=hist_bins,
                             weights=common_view_frustum_sample_points_weights[common_view_frustum_inlier_mask])
                    plt.tight_layout()
                    stash_fig_page()
                """

                """
                if img_size_trim > max(img.shape[:2]) / 4 or rect_proximity < 1.0 or disparity_spread < 16 * disparity_map_zoom or disparity_spread > 384 * disparity_map_zoom:
                    print('img_size_trim, rect_proximity, disparity_spread', img_size_trim, rect_proximity, disparity_spread)
                    continue
                """

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

                rect_map_prev, _ = cv2.initUndistortRectifyMap(camera_matrix, None, R1, P1, newImageSize, cv2.CV_32FC2)
                rect_map_next, _ = cv2.initUndistortRectifyMap(camera_matrix, None, R2, P2, newImageSize, cv2.CV_32FC2)

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
                                      'uniquenessRatio': 5,
                                      'speckleWindowSize': 50,
                                      'speckleRange': 2,
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
                wls_filter.setLRCthresh(48)
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
                uvcs = uvs - camera_matrix[:2, 2:]
                xys = np.linalg.inv(camera_matrix[:2, :2]) @ uvcs
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
                uvs12 = camera_matrix[:2, :2] @ xys12 + camera_matrix[:2, 2:]

                # Apply the current frame's stereo camera rectification transformation to the
                # cross frame's disparity mapped grid of image coordinates
                zs12_map = (R2 @ (xyzs12 / xyzs12[2, :]))[2, :].reshape(img.shape[:2])

                if exec_mode == 'debug_key_frame' and current_frame_idx == debug_key_frame_idx:
                    plt.figure('Cross disparity', figsize=(24, 12))
                    setup_new_fig_page()
                    rvec1, _ = cv2.Rodrigues(R1)
                    rvec2, _ = cv2.Rodrigues(R2)
                    title = '\n'.join([f'cross, current frame idxs: {cross_frame_idx}, {current_frame_idx}',
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
                    plt.imshow(ref_img_rect_padded)
                    plt.title('ref_img_rect_padded')
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

                # Calculate residual optical flow between key frames to refine the cumulative vector values
                xfp, yfp = uvs
                xfn, yfn = uvs12
                mask = np.all(np.isfinite(uvs12), axis=0)
                xfp, yfp = xfp[mask], yfp[mask]
                xfn, yfn = xfn[mask], yfn[mask]
                mask = (xfn > -0.5) & (xfn < img.shape[1] - 0.5) & (yfn > -0.5) & (yfn < img.shape[0] - 0.5)
                xfp, yfp = xfp[mask], yfp[mask]
                xfn, yfn = xfn[mask], yfn[mask]

                for iter_idx in range(1):
                    cross_flow = np.full(img.shape[:2] + (2,), fill_value=np.nan, dtype=np.float32)
                    cross_flow[yfp, xfp] = np.vstack([xfn, yfn]).T

                    # Mask the pixels in the reference image that do not intersect with the current image
                    border_value = 127
                    ref_gray_masked = np.array(ref_gray)
                    ref_gray_masked[np.any(~np.isfinite(cross_flow), axis=2)] = border_value

                    gray_warp = cv2.remap(np.array(gray), cross_flow, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)

                    # prev(y, x) ~ next(y + flow(y, x)[1], x + flow(y, x)[0])
                    # when cv2.OPTFLOW_FARNEBACK_GAUSSIAN is applied, flow_sigma = (winsize // 2) * 0.3
                    # poly_n: size of the pixel neighborhood used to find polynomial expansion in each pixel;
                    #         larger values mean that the image will be approximated with smoother surfaces,
                    #         yielding more robust algorithm and more blurred motion field, typically poly_n=5 or 7.
                    # poly_sigma: standard deviation of the Gaussian that is used to smooth derivatives used as a basis
                    #             for the polynomial expansion;
                    #             for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
                    cross_flow_warp = cv2.calcOpticalFlowFarneback(prev=cv2.resize(ref_gray_masked, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                                                                   next=cv2.resize(gray_warp, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                                                                   flow=None,
                                                                   pyr_scale=0.5, levels=2, winsize=101, iterations=2,
                                                                   poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                    cross_flow_warp = cv2.resize(cross_flow_warp, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST) * 2

                    cross_flow_fp = cross_flow_warp[yfp, xfp]
                    interp = scipy.interpolate.RegularGridInterpolator((np.arange(cross_flow.shape[0]), np.arange(cross_flow.shape[1])),
                                                                       cross_flow[:, :, 0] + cross_flow[:, :, 1] * 1j,
                                                                       method='linear', bounds_error=False, fill_value=np.nan)
                    cross_flow_interp = interp((yfp + cross_flow_fp[:, 1], xfp + cross_flow_fp[:, 0])).astype(np.complex64)
                    xfn, yfn = cross_flow_interp.real, cross_flow_interp.imag

                    dx, dy = cross_flow_warp[ysg, xsg].T
                    print('iter_idx', iter_idx, 'root [50, 90] percentiles squared residual dxy',
                          np.round(np.sqrt(np.percentile(dx * dx + dy * dy, [50, 90])), 2))

                    if exec_mode == 'debug_key_frame' and current_frame_idx == debug_key_frame_idx:
                        plt.figure('Cross flow refinement', figsize=(16, 10))
                        setup_new_fig_page()
                        plt.suptitle('\n'.join([f'cross, current frame idxs: {cross_frame_idx}, {current_frame_idx}',
                                                f'iter_idx {iter_idx}']))
                        ax = plt.subplot(2, 2, 1)
                        plt.imshow(ref_gray_masked, cmap='gray', vmin=0, vmax=255)
                        ax = plt.subplot(2, 2, 2, sharex=ax, sharey=ax)
                        plt.imshow(gray_warp, cmap='gray', vmin=0, vmax=255)
                        ax = plt.subplot(2, 2, 3, sharex=ax, sharey=ax)
                        plt.imshow(cross_flow_warp[:, : ,0])
                        ax = plt.subplot(2, 2, 4, sharex=ax, sharey=ax)
                        plt.imshow(cross_flow_warp[:, : ,1])
                        plt.tight_layout()
                        stash_fig_page()

                # Construct the mapping from the cross frame to the current frame coordinate space
                #cross_flow = uvs12.T.reshape(img.shape[:2] + (2,)).astype(np.float32)
                cross_flow = np.full(img.shape[:2] + (2,), fill_value=np.nan, dtype=np.float32)
                cross_flow[yfp, xfp] = np.vstack([xfn, yfn]).T

                # Mask out points where the local stereo rectification transformation scaling makes disparity less reliable
                cross_flow[zs1_map < 0.3] = np.nan
                cross_flow[zs12_map < 0.3] = np.nan
                """
                # Mask out a margin around the edge of the depth image where the disparity is less reliable
                margin = int(np.ceil(block_size / 2 / disparity_map_zoom))
                cross_flow[:margin, :] = np.nan
                cross_flow[-margin:, :] = np.nan
                cross_flow[:, :margin] = np.nan
                cross_flow[:, -margin:] = np.nan
                """

                # Record candidates for retraced triangulated points
                interp = scipy.interpolate.RegularGridInterpolator((np.arange(cross_flow.shape[0]), np.arange(cross_flow.shape[1])),
                                                                   cross_flow[:, :, 0] + cross_flow[:, :, 1] * 1j,
                                                                   method='linear', bounds_error=False, fill_value=np.nan)
                cross_unmatched_triangulated_current_image_points = interp((cross_unmatched_triangulated_image_points[:, 1],
                                                                            cross_unmatched_triangulated_image_points[:, 0])).astype(np.complex64)

                confidence_floor = -1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                pad_width = 151
                uvs12_filtered_confidence_map = cv2.copyMakeBorder(uvs12_confidence_map,
                                                                   top=pad_width, bottom=pad_width, left=pad_width, right=pad_width,
                                                                   borderType=cv2.BORDER_CONSTANT, value=confidence_floor)

                def set_confidence_floor():
                    uvs12_filtered_confidence_map[pad_width:-pad_width, pad_width:-pad_width][~np.isfinite(uvs12_confidence_map)] = confidence_floor
                    uvs12_filtered_confidence_map[pad_width:-pad_width, pad_width:-pad_width][~np.all(np.isfinite(cross_flow), axis=2)
                                                                                              | (cross_flow[:, :, 0] < 0)
                                                                                              | (cross_flow[:, :, 0] >= img.shape[1] - 1)
                                                                                              | (cross_flow[:, :, 1] < 0)
                                                                                              | (cross_flow[:, :, 1] >= img.shape[0] - 1)] = confidence_floor

                set_confidence_floor()
                uvs12_filtered_confidence_map = cv2.morphologyEx(uvs12_filtered_confidence_map, op=cv2.MORPH_CLOSE, kernel=kernel,
                                                                 iterations=5, borderType=cv2.BORDER_CONSTANT, borderValue=confidence_floor)
                set_confidence_floor()
                uvs12_filtered_confidence_map = cv2.morphologyEx(uvs12_filtered_confidence_map, op=cv2.MORPH_OPEN, kernel=kernel,
                                                                 iterations=10, borderType=cv2.BORDER_CONSTANT, borderValue=confidence_floor)
                set_confidence_floor()
                ksize = (51, 51)
                uvs12_filtered_confidence_map = (cv2.GaussianBlur(uvs12_filtered_confidence_map, ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_CONSTANT)
                                                 / cv2.GaussianBlur(np.ones_like(uvs12_filtered_confidence_map), ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_CONSTANT))
                uvs12_filtered_confidence_map = uvs12_filtered_confidence_map[pad_width:-pad_width, pad_width:-pad_width]

                rvec, _ = cv2.Rodrigues(camera_transform[:3, :3])
                # Ignore rotation around the z-axis (in the primary frame of reference)
                interframe_angle = np.linalg.norm(rvec[:2])

                filtered_cross_flow = nan_gaussian_filter(cross_flow, ksize=(9, 9), unfiltered_point_value=np.nan)
                grid_step = 5
                down_filtered_cross_flow = cv2.resize(filtered_cross_flow, dsize=None, fx=1/grid_step, fy=1/grid_step, interpolation=cv2.INTER_LINEAR)

                flow_gradient_x = np.gradient(down_filtered_cross_flow, grid_step, edge_order=2, axis=1)
                flow_gradient_y = np.gradient(down_filtered_cross_flow, grid_step, edge_order=2, axis=0)

                flow_grid_scaling_aspect = np.linalg.norm(flow_gradient_x, axis=-1) / np.linalg.norm(flow_gradient_y, axis=-1)
                flow_grid_scaling_aspect_score = cauchy(np.clip(np.max(np.stack([flow_grid_scaling_aspect, 1 / flow_grid_scaling_aspect]) - 1, axis=0), 0, np.inf), 2.0)

                flow_grid_gradient_cross = np.cross(flow_gradient_x, flow_gradient_y, axis=-1)
                flow_grid_gradient_cross[flow_grid_gradient_cross <= 0] = np.nan
                # Note that the acute/obtuse angle ambiguity between flow_gradient_x and flow_gradient_y is not
                # relevant providing we only consider the magnitude of the offset from np.pi / 2
                assert ~np.any(np.abs(flow_grid_gradient_cross) > (1 + 1e-6) * np.linalg.norm(flow_gradient_x, axis=-1) * np.linalg.norm(flow_gradient_y, axis=-1))
                flow_grid_orthogonality_score = cauchy(np.pi / 2 - np.arcsin(np.clip(flow_grid_gradient_cross / np.linalg.norm(flow_gradient_x, axis=-1) / np.linalg.norm(flow_gradient_y, axis=-1), -1, 1)),
                                                       #np.pi / 2 * cauchy(interframe_angle, np.pi / 2))
                                                       np.pi / 4)

                flow_laplacian_x = np.gradient(flow_gradient_x, grid_step, edge_order=2, axis=1)
                flow_laplacian_x_transverse_normed = np.abs(np.sum(np.stack([-flow_laplacian_x[:, :, 1], flow_laplacian_x[:, :, 0]], axis=-1) * flow_gradient_x, axis=-1)) / np.sum(np.power(flow_gradient_x, 2), axis=-1)
                flow_laplacian_y = np.gradient(flow_gradient_y, grid_step, edge_order=2, axis=0)
                flow_laplacian_y_transverse_normed = np.abs(np.sum(np.stack([-flow_laplacian_y[:, :, 0], flow_laplacian_y[:, :, 1]], axis=-1) * flow_gradient_y, axis=-1)) / np.sum(np.power(flow_gradient_y, 2), axis=-1)
                flow_laplacian_score = cauchy(np.linalg.norm(np.stack([flow_laplacian_x_transverse_normed, flow_laplacian_y_transverse_normed]), axis=0), 0.1)

                filtered_flow_grid_scaling_aspect_score = cv2.resize(nan_gaussian_filter(flow_grid_scaling_aspect_score, ksize=(3, 3)),
                                                                     dsize=None, fx=grid_step, fy=grid_step, interpolation=cv2.INTER_LINEAR)
                filtered_flow_grid_orthogonality_score = cv2.resize(nan_gaussian_filter(flow_grid_orthogonality_score, ksize=(3, 3)),
                                                                    dsize=None, fx=grid_step, fy=grid_step, interpolation=cv2.INTER_LINEAR)
                filtered_flow_laplacian_score = cv2.resize(nan_gaussian_filter(flow_laplacian_score, ksize=(3, 3)),
                                                           dsize=None, fx=grid_step, fy=grid_step, interpolation=cv2.INTER_LINEAR)
                if True:
                    filtered_flow_score = filtered_flow_grid_scaling_aspect_score * filtered_flow_grid_orthogonality_score * filtered_flow_laplacian_score
                else:
                    filtered_flow_score = np.power(3 / (1 / np.maximum(filtered_flow_grid_scaling_aspect_score, 1e-8)
                                                        + 1 / np.maximum(filtered_flow_grid_orthogonality_score, 1e-8)
                                                        + 1 / np.maximum(filtered_flow_laplacian_score, 1e-8)), 2)

                filtered_flow_confidence_map = np.array(uvs12_filtered_confidence_map)
                filtered_flow_confidence_map[filtered_flow_confidence_map > 0] *= filtered_flow_score[filtered_flow_confidence_map > 0]

                interp = scipy.interpolate.RegularGridInterpolator((np.arange(filtered_flow_confidence_map.shape[0]), np.arange(filtered_flow_confidence_map.shape[1])),
                                                                   filtered_flow_confidence_map,
                                                                   method='linear', bounds_error=False, fill_value=np.nan)
                cross_unmatched_triangulated_confidence = interp((cross_unmatched_triangulated_image_points[:, 1],
                                                                  cross_unmatched_triangulated_image_points[:, 0])) * triangulated_normal_weights[cross_unmatched_triangulated_idxs]

                for triangulated_idx, current_image_point, confidence in zip(cross_unmatched_triangulated_idxs,
                                                                             cross_unmatched_triangulated_current_image_points,
                                                                             cross_unmatched_triangulated_confidence):
                    record = np.array([confidence, current_image_point.real, current_image_point.imag])
                    if np.all(np.isfinite(record)):
                        retrace_triangulated_candidates[triangulated_idx][cross_frame_idx] = record

                cross_stitch_disparity_confidence_maps[(cross_frame_idx, current_frame_idx)] = (img_size_trim, rect_proximity,
                                                                                                min_disparity, max_disparity, num_disparities,
                                                                                                uvs12_confidence_map, uvs12_filtered_confidence_map,
                                                                                                cross_triangulated_image_points)

                if exec_mode == 'debug_key_frame' and current_frame_idx == debug_key_frame_idx:
                    # Map the current frame image onto the cross frame
                    uvs12_gray = cv2.remap(np.array(gray), cross_flow, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)

                    cross_triangulated_idxs = [triangulated_idx for triangulated_idx in retrace_triangulated_candidates
                                               if cross_frame_idx in retrace_triangulated_candidates[triangulated_idx]
                                               and triangulated_idx in cross_triangulated_idxs_to_image_idxs]
                    cross_retrace_triangulated_candidates_image_idxs = [cross_triangulated_idxs_to_image_idxs[triangulated_idx]
                                                                        for triangulated_idx in cross_triangulated_idxs]
                    current_retrace_triangulated_candidate_records = np.array([retrace_triangulated_candidates[triangulated_idx][cross_frame_idx]
                                                                               for triangulated_idx in cross_triangulated_idxs]).reshape((-1, 3))

                    current_projected_points = camera_matrix @ stitched_triangulated_points[cross_triangulated_idxs, :3].T
                    current_projected_points = current_projected_points[:2, :] / current_projected_points[2, :]

                    R = camera_transform[:3, :3]
                    t = camera_transform[:3, 3:]
                    cross_projected_points = camera_matrix @ (R.T @ stitched_triangulated_points[cross_triangulated_idxs, :3].T - R.T @ t)
                    cross_projected_points = cross_projected_points[:2, :] / cross_projected_points[2, :]

                    plt.figure('Cross flow from disparity', figsize=(24, 12))
                    setup_new_fig_page()
                    plt.suptitle(f'cross, current frame idxs: {cross_frame_idx}, {current_frame_idx}')
                    ax = plt.subplot(3, 4, 1)
                    plt.imshow(ref_gray, cmap='gray', vmin=0, vmax=255)
                    edge_colours = plt.get_cmap('seismic')((current_retrace_triangulated_candidate_records[:, 0] + 1) / 2)
                    plt.scatter(*cross_projected_points, s=8, facecolors='none', edgecolors=edge_colours, marker='o')
                    segments = list(zip(cross_projected_points.T, cross_image_points[:, cross_retrace_triangulated_candidates_image_idxs].T))
                    ax.add_collection(matplotlib.collections.LineCollection(segments, colors=edge_colours, linewidths=1))
                    plt.title('ref_gray')
                    ax = plt.subplot(3, 4, 2, sharex=ax, sharey=ax)
                    plt.imshow(uvs12_gray, cmap='gray', vmin=0, vmax=255)
                    plt.title('uvs12_gray')
                    ax = plt.subplot(3, 4, 3, sharex=ax, sharey=ax)
                    plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
                    edge_colours = plt.get_cmap('seismic')((current_retrace_triangulated_candidate_records[:, 0] + 1) / 2)
                    plt.scatter(*current_projected_points, s=8, facecolors='none', edgecolors=edge_colours, marker='o')
                    segments = list(zip(current_projected_points.T, current_retrace_triangulated_candidate_records[:, 1:]))
                    ax.add_collection(matplotlib.collections.LineCollection(segments, colors=edge_colours, linewidths=1))
                    ax.set_xlim((-100, img.shape[1] + 100))
                    ax.set_ylim((img.shape[0] + 100, -100))
                    plt.title('gray')
                    ax = plt.subplot(3, 4, 5, sharex=ax, sharey=ax)
                    plt.imshow(uvs12_disparity)
                    plt.title('uvs12_disparity')
                    ax = plt.subplot(3, 4, 6, sharex=ax, sharey=ax)
                    ax.set_facecolor('magenta')
                    plt.imshow(uvs12_confidence_map, vmin=-1, vmax=1, cmap='seismic')
                    plt.title('uvs12_confidence_map')
                    ax = plt.subplot(3, 4, 7, sharex=ax, sharey=ax)
                    plt.imshow(uvs12_filtered_confidence_map, vmin=-1, vmax=1, cmap='seismic')
                    plt.title('uvs12_filtered_confidence_map')
                    ax = plt.subplot(3, 4, 8, sharex=ax, sharey=ax)
                    plt.imshow(filtered_flow_grid_scaling_aspect_score)
                    plt.title('filtered_flow_grid_scaling_aspect_score')
                    ax = plt.subplot(3, 4, 9, sharex=ax, sharey=ax)
                    plt.imshow(filtered_flow_grid_orthogonality_score)
                    plt.title('filtered_flow_grid_orthogonality_score')
                    ax = plt.subplot(3, 4, 10, sharex=ax, sharey=ax)
                    plt.imshow(filtered_flow_laplacian_score)
                    plt.title('filtered_flow_laplacian_score')
                    ax = plt.subplot(3, 4, 11, sharex=ax, sharey=ax)
                    plt.imshow(filtered_flow_score)
                    plt.title('filtered_flow_score')
                    ax = plt.subplot(3, 4, 12, sharex=ax, sharey=ax)
                    plt.imshow(filtered_flow_confidence_map, vmin=-1, vmax=1, cmap='seismic')
                    plt.title('filtered_flow_confidence_map')
                    ax = plt.subplot(3, 4, 4, sharex=ax, sharey=ax)
                    plt.scatter(*cross_triangulated_image_points.T, s=2, c='b', marker='o')
                    plt.scatter(*cross_unmatched_triangulated_image_points.T, s=2, c='y', marker='o')
                    ax.set_aspect('equal')
                    plt.tight_layout()
                    stash_fig_page()

            # Select retraced triangulated points
            retrace_triangulated_main_clusters = []
            retraced_triangulated_idxs = []
            retraced_triangulated_points = []
            current_projected_points = camera_matrix @ stitched_triangulated_points[:, :3].T
            current_projected_points = current_projected_points[:2, :] / current_projected_points[2, :]
            for triangulated_idx, cross_candidates in retrace_triangulated_candidates.items():
                candidates = np.array(list(cross_candidates.values()))
                candidates = candidates[candidates[:, 0] >= 1e-3]
                #threshold = min(0.5 * (1 + motion_blur), 2.0)
                #threshold = 50.0
                #candidates = candidates[np.linalg.norm(candidates[:, 1:] - current_projected_points[:, triangulated_idx], axis=1) < threshold, :]
                weights = candidates[:, :1]
                if candidates.shape[0] >= 2:
                    # TODO: determine if there is a better way of unifying/clustering the collection of candidates into a single retraced point
                    # Cluster the candidate points, and take the heaviest weighted cluster if it is either the only cluster or
                    # it is over twice as heavy as the next heaviest cluster.
                    # The default affinity metric uses the negative squared euclidean distance between points.
                    # Points with larger values of preferences are more likely to be chosen as exemplars.
                    # The number of exemplars / clusters is influenced by the input preferences value.
                    # Preference values close to the minimum possible similarity produces fewer classes,
                    # while values close to or larger than the maximum possible similarity produces many classes.
                    af_preference = -np.power(15 + 85 * (1 - candidates[:, 0]), 2)
                    # Affinity Propagation occasionally converges to an incorrect set of clusters.
                    # Execute fitting with different random states to derive a modal result.
                    af_clusters = collections.Counter()
                    af_consensus_found = False
                    for iter_idx in range(10):
                        with warnings.catch_warnings(record=True) as ws:
                            af = sklearn.cluster.AffinityPropagation(damping=0.5, convergence_iter=15,
                                                                     preference=af_preference, random_state=iter_idx).fit(candidates[:, 1:])
                            for w in ws:
                                # This warning is raised if all the candidates are equispaced and the preferences are all equal
                                # However, the returned labels appear to be as expected, e.g.
                                #   In: sklearn.cluster.AffinityPropagation(preference=[-1, -1], random_state=0).fit([[0], [0]]).labels_
                                #   Out: array([0, 0])
                                #   In: sklearn.cluster.AffinityPropagation(preference=[-1, -1], random_state=0).fit([[0], [1.1]]).labels_
                                #   Out: array([0, 1])
                                if (issubclass(w.category, UserWarning)
                                    and w.message.args == ('All samples have mutually equal similarities. Returning arbitrary cluster center(s).',)):
                                    continue
                                if (issubclass(w.category, sklearn.exceptions.ConvergenceWarning)
                                    and w.message.args == ('Affinity propagation did not converge, this model may return degenerate cluster centers and labels.',)):
                                    continue
                                if (issubclass(w.category, sklearn.exceptions.ConvergenceWarning)
                                    and w.message.args == ('Affinity propagation did not converge and this model will not have any cluster centers.',)):
                                    continue
                                raise w.message
                        if len(af.cluster_centers_indices_) > 0:
                            label_sets = frozenset(frozenset(np.where(af.labels_ == label)[0]) for label in np.unique(af.labels_))
                            af_clusters.update([label_sets])
                            most_common_af_clusters = af_clusters.most_common()
                            if ((len(most_common_af_clusters) == 1 and most_common_af_clusters[0][1] > 2)
                                or (len(most_common_af_clusters) > 1 and most_common_af_clusters[0][1] > 2 * most_common_af_clusters[1][1])):
                                af_consensus_found = True
                                break

                    if af_consensus_found:
                        cluster_sets, _ = af_clusters.most_common()[0]
                        cluster_sets = [np.array(list(cluster_set)) for cluster_set in cluster_sets]
                        cluster_weights = np.array([np.sum(weights[cluster_idxs]) for cluster_idxs in cluster_sets])
                        sort_idxs = np.argsort(cluster_weights)[::-1]
                        sorted_cluster_weights = cluster_weights[sort_idxs]
                        if (len(sorted_cluster_weights) > 0 and sorted_cluster_weights[0] > 1.5
                            and (len(sorted_cluster_weights) == 1 or sorted_cluster_weights[0] > 2 * sorted_cluster_weights[1])):
                            cluster_idxs = cluster_sets[sort_idxs[0]]
                            cluster_points = candidates[cluster_idxs, 1:]
                            weights = weights[cluster_idxs, :]
                            normed_weights = weights / np.sum(weights)
                            weighted_retraced_point_mean = np.sum(cluster_points * normed_weights, axis=0)
                            # Variance of mean(X) calculated for N samples of random variable X = var(X) / N
                            weighted_retraced_point_mean_std = np.sqrt(np.sum(np.power(np.linalg.norm(cluster_points - weighted_retraced_point_mean), 2) * normed_weights)
                                                                       / np.sum(weights))
                            retrace_triangulated_main_clusters.append((weighted_retraced_point_mean,
                                                                       current_projected_points[:, triangulated_idx],
                                                                       weighted_retraced_point_mean_std,
                                                                       sorted_cluster_weights,
                                                                       len(candidates)))
                            if (np.linalg.norm(weighted_retraced_point_mean - current_projected_points[:, triangulated_idx]) < 100.0
                                and weighted_retraced_point_mean_std < 15.0):
                                retraced_triangulated_idxs.append(triangulated_idx)
                                retraced_triangulated_points.append(weighted_retraced_point_mean)

            retraced_triangulated_idxs = np.array(retraced_triangulated_idxs, dtype=int)
            retraced_triangulated_points = np.array(retraced_triangulated_points).reshape((-1, 2))
            print('len(retraced_triangulated_idxs)', len(retraced_triangulated_idxs))

            # Augment sample points with retraced triangulated points
            xsna, ysna = np.hstack([np.vstack([xsn, ysn]), retraced_triangulated_points.T])

            # Prune obscured triangulated sample points (both existing and new points) based on their normal angles in the current view
            current_triangulated_image_idxs = np.where(current_image_to_triangulated_point_idxs >= 0)[0]
            current_triangulated_idxs = current_image_to_triangulated_point_idxs[current_triangulated_image_idxs]

            mask = triangulated_normal_weights[current_triangulated_idxs] < 1e-2
            obscure_image_idxs = current_triangulated_image_idxs[mask]

            # Invalidate image coordinates of obscured triangulated sample points
            xsna[obscure_image_idxs] = np.nan
            ysna[obscure_image_idxs] = np.nan
            # Invalidate the obscured triangulation mapping from both intersected_corres_idxs and new_inlier_triangulation_idxs
            current_image_to_triangulated_point_idxs[obscure_image_idxs] = -1

            print('len(obscure_image_idxs)', len(obscure_image_idxs))
            print('num pruned intersected_corres_idxs', np.sum(current_image_to_triangulated_point_idxs[intersected_corres_idxs] < 0))
            print('num pruned new_inlier_triangulation_idxs', np.sum(current_image_to_triangulated_point_idxs[new_inlier_triangulation_idxs] < 0))

            # Find grid points to extend the frame's set of sample points
            mask = np.isfinite(xsna) & np.isfinite(ysna)
            xsna_i = np.round(xsna[mask]).astype(np.int32)
            ysna_i = np.round(ysna[mask]).astype(np.int32)
            mask = (xsna_i >= 0) & (xsna_i < img.shape[1]) & (ysna_i >= 0) & (ysna_i < img.shape[0])
            xsna_i, ysna_i = xsna_i[mask], ysna_i[mask]
            neighbours = np.zeros(img.shape[:2], dtype=np.uint8)
            neighbours[ysna_i, xsna_i] = 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grid_step * 2 - 1, grid_step * 2 - 1))
            neighbours = cv2.dilate(neighbours, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            mask = neighbours[ysg, xsg] == 0
            xsne, ysne = xsg[mask], ysg[mask]

            # Add the set of extended image sample points to be used for the next pair of frames
            key_frame_image_sample_points.append((np.hstack([xsna, xsne], dtype=np.float32), np.hstack([ysna, ysne], dtype=np.float32)))

            # Add the index array mapping current image points to triangulated points
            image_to_triangulated_point_idxs = np.hstack([current_image_to_triangulated_point_idxs,
                                                          retraced_triangulated_idxs,
                                                          np.full((len(xsne),), fill_value=-1, dtype=int)])
            key_frame_image_triangulated_point_idxs.append(image_to_triangulated_point_idxs)

            # Invalidate triangulated points with less than two mapped image points
            triangulated_idxs_counts = np.zeros((stitched_triangulated_points.shape[0],), dtype=int)
            for image_to_triangulated_point_idxs in key_frame_image_triangulated_point_idxs:
                triangulated_image_idxs = np.where(image_to_triangulated_point_idxs >= 0)[0]
                triangulated_idxs = image_to_triangulated_point_idxs[triangulated_image_idxs]
                triangulated_idxs_counts[triangulated_idxs] = triangulated_idxs_counts[triangulated_idxs] + 1

            print('num triangulated points with less than two mapped image points to invalidate',
                  np.sum(np.all(np.isfinite(stitched_triangulated_points[triangulated_idxs_counts < 2, :3]), axis=1)))
            stitched_triangulated_points[triangulated_idxs_counts < 2, :] = np.nan
            print('total invalidated triangulated points',
                  np.sum(np.all(~np.isfinite(stitched_triangulated_points[:, :3]), axis=1)),
                  'of total triangulated points', stitched_triangulated_points.shape[0])

            plt_data_path = output_dirpath / 'Retraced cross triangulated points' / f'{current_frame_idx:04d}.pickle'
            plt_data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(plt_data_path, 'wb') as pickle_file:
                pickle.dump({'xsn': xsn,
                             'ysn': ysn,
                             'xsne': xsne,
                             'ysne': ysne,
                             'current_image_to_triangulated_point_idxs': current_image_to_triangulated_point_idxs,
                             'current_triangulated_image_points': current_triangulated_image_points,
                             'retrace_triangulated_main_clusters': retrace_triangulated_main_clusters,
                             'retraced_triangulated_points': retraced_triangulated_points},
                            pickle_file)

            """
            plt.figure(figsize=(16, 10))
            ax = plt.subplot(1, 1, 1, projection='3d')
            ax.scatter(*stitched_triangulated_points[:-new_triangulated_points.shape[0], :3].T, s=2, c='b', alpha=0.2)
            ax.scatter(*stitched_triangulated_points[-new_triangulated_points.shape[0]:, :3].T, s=2, c='r', alpha=0.2)
            ax.set_xlim((-20, 20))
            ax.set_ylim((-20, 20))
            ax.set_zlim((0, 40))
            ax.set_aspect('equal', adjustable='datalim')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=-135, azim=-90, roll=0)
            plt.tight_layout()
            """


        triangulated_point_projection_errs = collections.defaultdict(list)
        for frame_idx in range(len(key_frame_image_sample_points)):
            xsc, ysc = key_frame_image_sample_points[frame_idx]
            frame_image_points = np.vstack([xsc, ysc])
            frame_image_to_triangulated_point_idxs = key_frame_image_triangulated_point_idxs[frame_idx]
            frame_triangulated_image_idxs = np.where(frame_image_to_triangulated_point_idxs >= 0)[0]
            frame_triangulated_idxs = frame_image_to_triangulated_point_idxs[frame_triangulated_image_idxs]

            valid_mask = np.all(np.isfinite(stitched_triangulated_points[frame_triangulated_idxs, :3]), axis=1)
            frame_triangulated_image_idxs = frame_triangulated_image_idxs[valid_mask]
            frame_triangulated_idxs = frame_triangulated_idxs[valid_mask]

            frame_triangulated_image_points = frame_image_points[:, frame_triangulated_image_idxs].T
            frame_triangulated_points = stitched_triangulated_points[frame_triangulated_idxs, :3]

            camera_transform = key_frame_camera_extrinsics[frame_idx] @ np.linalg.inv(key_frame_camera_extrinsics[current_frame_idx])
            object_points = camera_transform @ np.vstack([frame_triangulated_points.T, np.ones((frame_triangulated_points.shape[0],))])
            projected_points = camera_matrix @ object_points[:3, :]
            projected_points = projected_points[:2, :] / projected_points[2, :]
            projection_errs = np.linalg.norm(frame_triangulated_image_points - projected_points.T, axis=1)

            for triangulated_idx, projection_err in zip(frame_triangulated_idxs, projection_errs):
                triangulated_point_projection_errs[triangulated_idx].append(projection_err)

        # Transform stitched_triangulated_points from the current camera frame back
        # to the absolute origin frame of reference
        inv_camera_extrinsic = np.linalg.inv(key_frame_camera_extrinsics[current_frame_idx])
        R = inv_camera_extrinsic[:3, :3]
        t = inv_camera_extrinsic[:3, 3:]
        stitched_triangulated_points[:, :3] = (R @ stitched_triangulated_points[:, :3].T + t).T

        triangulated_idxs_weights = np.zeros((stitched_triangulated_points.shape[0],))
        for triangulated_idx, projection_errs in triangulated_point_projection_errs.items():
            projection_err_sigma = 3.0
            triangulated_idxs_weights[triangulated_idx] = np.sum(np.exp(-0.5 * np.power(np.array(projection_errs) / projection_err_sigma, 2)))

        stitched_key_frame_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stitched_key_frame_path, 'wb') as pickle_file:
            pickle.dump({'stitched_triangulated_points': stitched_triangulated_points,
                         'triangulated_idxs_weights': triangulated_idxs_weights,
                         'key_frame_image_sample_points': (key_frame_image_sample_points[current_frame_idx:]
                                                           if current_frame_idx > 1 else key_frame_image_sample_points),
                         'key_frame_image_triangulated_point_idxs': key_frame_image_triangulated_point_idxs[current_frame_idx-1:],
                         'key_frame_camera_extrinsics': key_frame_camera_extrinsics,
                         'cross_stitch_disparity_confidence_maps': {(_cross_frame_idx, _current_frame_idx): value
                                                                    for (_cross_frame_idx, _current_frame_idx), value in cross_stitch_disparity_confidence_maps.items()
                                                                    if _current_frame_idx == current_frame_idx}},
                        pickle_file)

        processed_frame_idxs.append(current_frame_idx)

        plt_data_path = output_dirpath / 'Sequential point correspondences' / f'{current_frame_idx:04d}.pickle'
        plt_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(plt_data_path, 'wb') as pickle_file:
            pickle.dump({'xyfn_flow': xyfn_flow,
                         'xsp': xsp,
                         'ysp': ysp,
                         'xsn': xsn,
                         'ysn': ysn,
                         'corres_idxs': corres_idxs},
                        pickle_file)

        if exec_mode == 'next_key_frame' or (exec_mode == 'debug_key_frame' and current_frame_idx == debug_key_frame_idx):
            break

    # %%

    if exec_mode != 'debug_key_frame' and len(key_frames_filepaths) in processed_frame_idxs:

        plt_data_dirpath = output_dirpath / 'Retraced cross triangulated points'

        for plt_data_filepath in sorted(plt_data_dirpath.glob('*.pickle')):
            current_frame_idx, filename_ext = plt_data_filepath.name.split('.')
            current_frame_idx = int(current_frame_idx)

            with open(plt_data_filepath, 'rb') as pickle_file:
                data = pickle.load(pickle_file)
                xsn = data['xsn']
                ysn = data['ysn']
                xsne = data['xsne']
                ysne = data['ysne']
                current_image_to_triangulated_point_idxs = data['current_image_to_triangulated_point_idxs']
                current_triangulated_image_points = data['current_triangulated_image_points']
                retrace_triangulated_main_clusters = data['retrace_triangulated_main_clusters']
                retraced_triangulated_points = data['retraced_triangulated_points']

            current_triangulated_image_idxs = np.where(current_image_to_triangulated_point_idxs >= 0)[0]
            current_triangulated_image_points = np.vstack([xsn, ysn])[:, current_triangulated_image_idxs].T

            fig = plt.figure('Retraced cross triangulated points', figsize=(16, 10))
            setup_new_fig_page()
            plt.suptitle(f'Current frame idx: {current_frame_idx}')
            ax = plt.subplot(3, 3, 1)
            plt.imshow(np.require(key_frame_images[current_frame_idx - 1][0], dtype=np.uint8))
            plt.title('previous frame')
            ax = plt.subplot(3, 3, 2, sharex=ax, sharey=ax)
            plt.imshow(np.require(key_frame_images[current_frame_idx][0], dtype=np.uint8))
            plt.title('current frame')
            ax = plt.subplot(3, 3, 3, sharex=ax, sharey=ax)
            plt.scatter(xsn, ysn, s=8, c='k', marker='o', alpha=0.2)
            plt.scatter(*current_triangulated_image_points.T, s=8, c='b', marker='o')
            plt.scatter(*np.array([record[1] for record in retrace_triangulated_main_clusters]).reshape((-1, 2)).T,
                        s=8, facecolors='none', edgecolors='y', marker='o')
            segments = [record[:2] for record in retrace_triangulated_main_clusters]
            ax.add_collection(matplotlib.collections.LineCollection(segments, colors='g', linewidths=1))
            plt.title('projected -> image points')
            ax.set_aspect('equal')
            ax.set_xlim((-50, image_size[0] + 50))
            ax.set_ylim((image_size[1] + 50, -50))
            ax = plt.subplot(3, 3, 4, sharex=ax, sharey=ax)
            plt.scatter(xsn, ysn, s=8, c='k', marker='o', alpha=0.2)
            plt.scatter(*current_triangulated_image_points.T, s=8, c='b', marker='o')
            patches = [plt.Circle(record[0], record[2]) for record in retrace_triangulated_main_clusters]
            ax.add_collection(matplotlib.collections.PatchCollection(patches, facecolors='y', edgecolors='g', alpha=0.2))
            plt.title('standard deviation')
            ax.set_aspect('equal')
            ax = plt.subplot(3, 3, 5, sharex=ax, sharey=ax)
            plt.scatter(xsn, ysn, s=8, c='k', marker='o', alpha=0.2)
            plt.scatter(*current_triangulated_image_points.T, s=8, c='b', marker='o')
            plt.scatter(*np.array([record[0] for record in retrace_triangulated_main_clusters]).reshape((-1, 2)).T,
                        s=np.array([record[3][0] ** 2 for record in retrace_triangulated_main_clusters]), c='y', marker='o', alpha=0.2)
            plt.title('main cluster weight')
            ax.set_aspect('equal')
            ax = plt.subplot(3, 3, 6, sharex=ax, sharey=ax)
            plt.scatter(xsn, ysn, s=8, c='k', marker='o', alpha=0.2)
            plt.scatter(*current_triangulated_image_points.T, s=8, c='b', marker='o')
            plt.scatter(*np.array([record[0] for record in retrace_triangulated_main_clusters]).reshape((-1, 2)).T,
                        s=np.array([record[4] ** 2 for record in retrace_triangulated_main_clusters]), c='y', marker='o', alpha=0.2)
            plt.title('candidate count')
            ax.set_aspect('equal')
            ax = plt.subplot(3, 3, 8, sharex=ax, sharey=ax)
            # Note that current_triangulated_image_points are plotted over xsn, ysn points, so the visible
            # xsn, ysn points are all the sequential flow points that are not triangulated
            plt.scatter(xsn, ysn, s=8, c='k', marker='o', alpha=0.2, label='points outside valid flow or epipolar constraint')
            plt.scatter(xsne, ysne, s=8, c='w', marker='o', edgecolor='k', alpha=0.2, label='new points to find matches for in following frames')
            plt.scatter(*current_triangulated_image_points.T, s=8, c='b', marker='o', label='triangulated sequential points')
            plt.scatter(*retraced_triangulated_points.T, s=8, c='y', marker='o', label='triangulated cross points')
            plt.title('retraced triangulated points')
            ax.set_aspect('equal')
            fig.legend(loc='lower right')
            plt.tight_layout()
            stash_fig_page()

        # %%

        plt_data_dirpath = output_dirpath / 'Sequential point correspondences'

        for plt_data_filepath in sorted(plt_data_dirpath.glob('*.pickle')):
            current_frame_idx, filename_ext = plt_data_filepath.name.split('.')
            current_frame_idx = int(current_frame_idx)

            with open(plt_data_filepath, 'rb') as pickle_file:
                data = pickle.load(pickle_file)
                xyfn_flow = data['xyfn_flow']
                xsp = data['xsp']
                ysp = data['ysp']
                xsn = data['xsn']
                ysn = data['ysn']
                corres_idxs = data['corres_idxs']

            img_warp = cv2.remap(np.array(key_frame_images[current_frame_idx][0]).astype(np.float32), xyfn_flow, None, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(np.nan, np.nan, np.nan))

            plt.figure('Sequential point correspondences', figsize=(16, 10))
            setup_new_fig_page()
            plt.suptitle(f'Previous, current frame idxs: {current_frame_idx - 1}, {current_frame_idx}')
            ax = plt.subplot(2, 2, 1)
            plt.imshow(np.require(key_frame_images[current_frame_idx - 1][0], dtype=np.uint8))
            plt.scatter(xsp[corres_idxs], ysp[corres_idxs], s=2, c='b', alpha=0.5)
            ax = plt.subplot(2, 2, 2, sharex=ax, sharey=ax)
            plt.imshow(np.require(key_frame_images[current_frame_idx][0], dtype=np.uint8))
            plt.scatter(xsn[corres_idxs], ysn[corres_idxs], s=2, c='b', alpha=0.5)
            ax = plt.subplot(2, 2, 3, sharex=ax, sharey=ax)
            plt.imshow(img_warp / 255)
            plt.scatter(xsp[corres_idxs], ysp[corres_idxs], s=2, c='b', alpha=0.5)
            ax = plt.subplot(2, 2, 4, sharex=ax, sharey=ax)
            plt.imshow(np.clip((key_frame_images[current_frame_idx - 1][0].astype(np.float32) - img_warp) / 255 + 0.5, 0, 1))
            plt.scatter(xsp[corres_idxs], ysp[corres_idxs], s=2, c='b', alpha=0.5)
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

        for prev_camera_extrinsic, camera_extrinsic in zip([None] + key_frame_camera_extrinsics[:-1], key_frame_camera_extrinsics):
            # The extrinsic matrix transforms from world coordinates to camera coordinates
            camera_lines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=image_size[0], view_height_px=image_size[1],
                                                                            intrinsic=camera_matrix,
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

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(stitched_triangulated_points[:, :3]))
        pcd.colors = o3d.utility.Vector3dVector(stitched_triangulated_points[:, 3:] / 255)
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

        nrows = int(np.ceil(np.sqrt(len(key_frame_images) / 1.6)))
        ncols = int(np.ceil(nrows * 1.6))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(16, 10))
        plt.suptitle('cross disparity confidences for current frames')

        for plot_idx, current_frame_idx in enumerate(range(2, len(key_frame_images))):
            ax = axs.flatten()[plot_idx]
            stats = []
            for cross_frame_idx in range(current_frame_idx):
                if (cross_frame_idx, current_frame_idx) in cross_stitch_disparity_confidence_maps:
                    (img_size_trim, rect_proximity,
                     min_disparity, max_disparity, num_disparities,
                     disparity_confidence_map, filtered_disparity_confidence_map,
                     cross_triangulated_image_points) = cross_stitch_disparity_confidence_maps[(cross_frame_idx, current_frame_idx)]

                    #if img_size_trim > 0 or rect_proximity < 2.0:
                    #    continue

                    pad_width = 16
                    image_mask = np.full(np.array(image_size[::-1]) + pad_width * 2, fill_value=0, dtype=np.uint8)
                    image_points = np.round(cross_triangulated_image_points).astype(int) + pad_width
                    image_mask[image_points[:, 1], image_points[:, 0]] = 1

                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
                    image_mask = cv2.morphologyEx(image_mask, op=cv2.MORPH_CLOSE, kernel=kernel,
                                                  iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)

                    image_mask_idxs = np.where(image_mask[pad_width:-pad_width, pad_width:-pad_width])

                    if np.array(image_mask_idxs).size > 0:
                        stats.append([cross_frame_idx,
                                      np.sum(~np.isfinite(filtered_disparity_confidence_map[*image_mask_idxs])),
                                      np.sum(filtered_disparity_confidence_map[*image_mask_idxs] < 0.5),
                                      np.sum(filtered_disparity_confidence_map[*image_mask_idxs] >= 0.5)])

            if len(stats) > 0:
                stats = np.array(stats).T
                """
                cross_frame_idxs = stats[0, :]
                cumulative_stats = np.cumsum(stats[1:, :], axis=0)
                #cumulative_stats = cumulative_stats / cumulative_stats[-1, :]
                for height, bottom in zip(np.diff(cumulative_stats, axis=0, prepend=0), [None] + list(cumulative_stats[:-1])):
                    ax.bar(cross_frame_idxs, height, bottom=bottom)
                """
                cumulative_stats = np.cumsum(stats[1:, :], axis=0)
                cumulative_stats = cumulative_stats[:, np.argsort(cumulative_stats[-1, :])[::-1]]
                for height, bottom in zip(np.diff(cumulative_stats, axis=0, prepend=0), [None] + list(cumulative_stats[:-1])):
                    ax.bar(np.arange(cumulative_stats.shape[1]), height, bottom=bottom)
            ax.axvline(current_frame_idx, linestyle='--')
            ax.set_title(f'{current_frame_idx}')

        fig.tight_layout()

        # %%

        # Estimate traction to detect slippage
        # TODO: Alternatively track cross_unmatched_triangulated_idxs that would project into the current frame, but have invalid or low confidence
        # TODO: Alternatively track rejected retrace candidates, and the creation & position of new triangulated points
        cross_traction = np.zeros((len(key_frame_images),))
        for (cross_frame_idx, current_frame_idx), (img_size_trim, rect_proximity,
                                                   min_disparity, max_disparity, num_disparities,
                                                   disparity_confidence_map, filtered_disparity_confidence_map,
                                                   cross_triangulated_image_points) in cross_stitch_disparity_confidence_maps.items():

            pad_width = 16
            image_mask = np.full(np.array(image_size[::-1]) + pad_width * 2, fill_value=0, dtype=np.uint8)
            image_points = np.round(cross_triangulated_image_points).astype(int) + pad_width
            image_mask[image_points[:, 1], image_points[:, 0]] = 1

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            image_mask = cv2.morphologyEx(image_mask, op=cv2.MORPH_CLOSE, kernel=kernel,
                                          iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)

            image_mask_idxs = np.where(image_mask[pad_width:-pad_width, pad_width:-pad_width])

            if np.array(image_mask_idxs).size > 0:
                filtered_confidence_values = filtered_disparity_confidence_map[*image_mask_idxs]
                #cross_traction[current_frame_idx] += np.sum(filtered_confidence_values[filtered_confidence_values > 0])
                cross_traction[current_frame_idx] += np.nansum(filtered_confidence_values)

        plt.figure(figsize=(16, 10))
        ax = plt.subplot(2, 1, 1)
        plt.plot(cross_traction)
        plt.title('Current frame traction / slippage')
        ax = plt.subplot(2, 1, 2, sharex=ax)
        plt.plot(key_frame_motion_blurs)
        plt.title('Frame motion blur')

        # %%

        """
        # TODO: Resolve the different causes of low disparity confidence, in particular the case where
        #       there is only partial visibility of the object as opposed to erroneous camera extrinsic estimates.
        #       Investigate whether the triangulated points can be used to resolve this.
        precomputed_confidences = np.full((len(key_frame_images),) * 2, fill_value=np.nan)
        for (cross_frame_idx, current_frame_idx), (img_size_trim, rect_proximity,
                                                   min_disparity, max_disparity, num_disparities,
                                                   disparity_confidence_map, filtered_disparity_confidence_map,
                                                   cross_triangulated_image_points) in cross_stitch_disparity_confidence_maps.items():

            pad_width = 16
            image_mask = np.full(np.array(image_size[::-1]) + pad_width * 2, fill_value=0, dtype=np.uint8)
            image_points = np.round(cross_triangulated_image_points).astype(int) + pad_width
            image_mask[image_points[:, 1], image_points[:, 0]] = 1

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            image_mask = cv2.morphologyEx(image_mask, op=cv2.MORPH_CLOSE, kernel=kernel,
                                          iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)

            image_mask_idxs = np.where(image_mask[pad_width:-pad_width, pad_width:-pad_width])

            if np.array(image_mask_idxs).size > 0:
                filtered_confidence_values = filtered_disparity_confidence_map[*image_mask_idxs]
                precomputed_confidences[cross_frame_idx, current_frame_idx] = np.nansum(filtered_confidence_values)
                precomputed_confidences[current_frame_idx, cross_frame_idx] = np.nansum(filtered_confidence_values)

        # %%

        plt.figure(figsize=(16, 10))
        plt.suptitle('Affinity propagation silhouette scores')

        for plot_idx, af_preference in enumerate([-2, -3, -4]):

            cluster_stats = []
            for current_frame_idx in range(2, len(key_frame_images)):
                print('AffinityPropagation', current_frame_idx)
                current_precomputed_confidences = precomputed_confidences[:current_frame_idx, :current_frame_idx]

                precomputed_affinity = current_precomputed_confidences / np.prod(image_size) - 1
                np.fill_diagonal(precomputed_affinity, 1)
                #af_preference = np.nanpercentile(precomputed_affinity, 5)
                #af_preference = -1.5
                precomputed_affinity[~np.isfinite(precomputed_affinity)] = -1
                af = sklearn.cluster.AffinityPropagation(damping=0.9, convergence_iter=15,
                                                         preference=af_preference, affinity='precomputed', random_state=0).fit(precomputed_affinity)
                label_sets = [list(np.where(af.labels_ == label)[0]) for label in np.unique(af.labels_)]
                for label_set in label_sets:
                    print(label_set)
                print('af.cluster_centers_indices_', af.cluster_centers_indices_)

                precomputed_metrics = 1 - current_precomputed_confidences / np.prod(image_size)
                np.fill_diagonal(precomputed_metrics, 0)
                precomputed_metrics[~np.isfinite(precomputed_metrics)] = 1
                print('mean of distances to all other samples excluding self', 1 - np.sum(precomputed_metrics) / (current_frame_idx * (current_frame_idx - 1)))
                print('mean of distances to all other samples including self', 1 - np.mean(precomputed_metrics))
                label_stats = []
                if len(label_sets) > 1:
                    print('silhouette_score', sklearn.metrics.silhouette_score(precomputed_metrics, af.labels_, metric='precomputed'))
                    sample_silhouettes = sklearn.metrics.silhouette_samples(precomputed_metrics, af.labels_, metric='precomputed')
                    for label_set in label_sets:
                        print(np.mean(sample_silhouettes[label_set]), np.percentile(sample_silhouettes[label_set], [0, 50, 100]), label_set)
                        label_stats.append(np.mean(sample_silhouettes[label_set]))

                cluster_stats.append((current_frame_idx, 1 - np.mean(precomputed_metrics), label_stats))

            plt.subplot(3, 1, plot_idx + 1)
            plt.scatter([current_frame_idx for current_frame_idx, mean_intra_distance, label_stats in cluster_stats],
                        [mean_intra_distance for current_frame_idx, mean_intra_distance, label_stats in cluster_stats],
                        marker='x')
            for current_frame_idx, mean_intra_distance, label_stats in cluster_stats:
                plt.plot([current_frame_idx] * len(label_stats), label_stats, marker='.')

        # %%

        plt.figure(figsize=(16, 10))
        plt.suptitle('HDBSCAN silhouette scores')

        for plot_idx, min_cluster_size in enumerate([2, 3, 4]):

            cluster_stats = []
            for current_frame_idx in range(min_cluster_size, len(key_frame_images)):
                print('HDBSCAN', current_frame_idx)
                current_precomputed_confidences = precomputed_confidences[:current_frame_idx, :current_frame_idx]

                precomputed_metrics = 1 - current_precomputed_confidences / np.prod(image_size)
                np.fill_diagonal(precomputed_metrics, 0)
                precomputed_metrics[~np.isfinite(precomputed_metrics)] = 1
                hdb = sklearn.cluster.HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed', allow_single_cluster=True).fit(precomputed_metrics)
                label_sets = [list(np.where(hdb.labels_ == label)[0]) for label in np.unique(hdb.labels_)]
                for label_set in label_sets:
                    print(label_set)
                print(hdb.probabilities_)

                precomputed_metrics = 1 - current_precomputed_confidences / np.prod(image_size)
                np.fill_diagonal(precomputed_metrics, 0)
                precomputed_metrics[~np.isfinite(precomputed_metrics)] = 1
                print('mean of distances to all other samples excluding self', 1 - np.sum(precomputed_metrics) / (current_frame_idx * (current_frame_idx - 1)))
                print('mean of distances to all other samples including self', 1 - np.mean(precomputed_metrics))
                label_stats = []
                if len(label_sets) > 1:
                    print('silhouette_score', sklearn.metrics.silhouette_score(precomputed_metrics, hdb.labels_, metric='precomputed'))
                    sample_silhouettes = sklearn.metrics.silhouette_samples(precomputed_metrics, hdb.labels_, metric='precomputed')
                    for label_set in label_sets:
                        print(np.mean(sample_silhouettes[label_set]), np.percentile(sample_silhouettes[label_set], [0, 50, 100]), label_set)
                        label_stats.append(np.mean(sample_silhouettes[label_set]))

                cluster_stats.append((current_frame_idx, 1 - np.mean(precomputed_metrics), label_stats))

            plt.subplot(3, 1, plot_idx + 1)
            plt.scatter([current_frame_idx for current_frame_idx, mean_intra_distance, label_stats in cluster_stats],
                        [mean_intra_distance for current_frame_idx, mean_intra_distance, label_stats in cluster_stats],
                        marker='x')
            for current_frame_idx, mean_intra_distance, label_stats in cluster_stats:
                plt.plot([current_frame_idx] * len(label_stats), label_stats, marker='.')

        # %%

        plt.figure(figsize=(16, 10))
        plt.suptitle('Spectral clustering silhouette scores')

        for plot_idx, n_clusters in enumerate([2, 3, 4]):

            cluster_stats = []
            for current_frame_idx in range(n_clusters + 1, len(key_frame_images)):
                print('SpectralClustering', current_frame_idx)
                current_precomputed_confidences = precomputed_confidences[:current_frame_idx, :current_frame_idx]

                precomputed_affinity = current_precomputed_confidences / np.prod(image_size)
                np.fill_diagonal(precomputed_affinity, 1)
                precomputed_affinity[~np.isfinite(precomputed_affinity)] = 0
                precomputed_affinity = np.exp(-(1 - precomputed_affinity) / 0.5)
                for assign_labels in ['kmeans', 'discretize', 'cluster_qr'][0:1]:
                    spcl = sklearn.cluster.SpectralClustering(n_clusters=n_clusters, random_state=0, affinity='precomputed', assign_labels=assign_labels).fit(precomputed_affinity)
                    label_sets = [list(np.where(spcl.labels_ == label)[0]) for label in np.unique(spcl.labels_)]
                    for label_set in label_sets:
                        print(label_set)

                precomputed_metrics = 1 - current_precomputed_confidences / np.prod(image_size)
                np.fill_diagonal(precomputed_metrics, 0)
                precomputed_metrics[~np.isfinite(precomputed_metrics)] = 1
                print('mean of distances to all other samples excluding self', 1 - np.sum(precomputed_metrics) / (current_frame_idx * (current_frame_idx - 1)))
                print('mean of distances to all other samples including self', 1 - np.mean(precomputed_metrics))
                label_stats = []
                if len(label_sets) > 1:
                    print('silhouette_score', sklearn.metrics.silhouette_score(precomputed_metrics, spcl.labels_, metric='precomputed'))
                    sample_silhouettes = sklearn.metrics.silhouette_samples(precomputed_metrics, spcl.labels_, metric='precomputed')
                    for label_set in label_sets:
                        print(np.mean(sample_silhouettes[label_set]), np.percentile(sample_silhouettes[label_set], [0, 50, 100]), label_set)
                        label_stats.append(np.mean(sample_silhouettes[label_set]))

                cluster_stats.append((current_frame_idx, 1 - np.mean(precomputed_metrics), label_stats))

            plt.subplot(3, 1, plot_idx + 1)
            plt.scatter([current_frame_idx for current_frame_idx, mean_intra_distance, label_stats in cluster_stats],
                        [mean_intra_distance for current_frame_idx, mean_intra_distance, label_stats in cluster_stats],
                        marker='x')
            for current_frame_idx, mean_intra_distance, label_stats in cluster_stats:
                plt.plot([current_frame_idx] * len(label_stats), label_stats, marker='.')
        """

        # %%

        interframe_confidences = np.full((len(key_frame_indices),) * 2, fill_value=np.nan, dtype=np.float32)
        for (cross_frame_idx, current_frame_idx), (img_size_trim, rect_proximity,
                                                   min_disparity, max_disparity, num_disparities,
                                                   disparity_confidence_map, filtered_disparity_confidence_map,
                                                   cross_triangulated_image_points) in cross_stitch_disparity_confidence_maps.items():

            pad_width = 16
            image_mask = np.full(np.array(image_size[::-1]) + pad_width * 2, fill_value=0, dtype=np.uint8)
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

        # TODO: some of the interframe_confidences may need to be recalculated depending on whether
        #       the original camera transform estimate, reprojection of points, epipolar lines or rectification
        #       changed significantly after optimisation
        path_distances = -np.log(np.clip(interframe_confidences, 1e-2, 1))
        path_distances[~np.isfinite(path_distances)] = -np.log(1e-2)

        dist_matrix, predecessors = scipy.sparse.csgraph.shortest_path(path_distances, directed=False, return_predecessors=True)

        median_dist_matrix = np.median(dist_matrix, axis=0)

        plt.figure('Interframe confidences and path distances', figsize=(16, 10))
        plt.clf()
        ax = plt.subplot(3, 3, 2)
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
        plt.tight_layout()

        # %%

        # Counts of associated image points for each 3D triangulated point
        triangulated_idxs_counts = np.zeros((stitched_triangulated_points.shape[0],), dtype=int)
        for image_to_triangulated_point_idxs in key_frame_image_triangulated_point_idxs:
            triangulated_image_idxs = np.where(image_to_triangulated_point_idxs >= 0)[0]
            triangulated_idxs = image_to_triangulated_point_idxs[triangulated_image_idxs]
            triangulated_idxs_counts[triangulated_idxs] = triangulated_idxs_counts[triangulated_idxs] + 1

        plt.figure(figsize=(16, 10))
        ax = plt.subplot(4, 1, 1)
        plt.plot(triangulated_idxs_counts)
        plt.title('triangulated_idxs_counts')
        ax = plt.subplot(4, 1, 2, sharex=ax, sharey=ax)
        plt.plot(triangulated_idxs_weights)
        plt.title('triangulated_idxs_weights')
        ax = plt.subplot(4, 1, 3)
        hist_bins = np.arange(np.max(triangulated_idxs_counts) + 2) - 0.5
        plt.hist(triangulated_idxs_counts, bins=hist_bins)
        plt.title('triangulated_idxs_counts')
        ax = plt.subplot(4, 1, 4, sharex=ax, sharey=ax)
        plt.hist(triangulated_idxs_weights, bins=hist_bins)
        plt.title('triangulated_idxs_weights')
        plt.tight_layout()

        """
        plt.figure(figsize=(16, 10))
        ax3 = plt.subplot(1, 1, 1, projection='3d')
        ax3.scatter(*stitched_triangulated_points[:, :3].T, s=triangulated_idxs_counts, c=triangulated_idxs_counts)
        ax3.set_xlim((-20, 20))
        ax3.set_ylim((-20, 20))
        ax3.set_zlim((0, 40))
        ax3.set_aspect('equal', adjustable='datalim')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.view_init(elev=-135, azim=-90, roll=0)
        plt.tight_layout()
        """

        plt.figure(figsize=(16, 10))
        ax3 = plt.subplot(1, 1, 1, projection='3d')
        ax3.scatter(*stitched_triangulated_points[:, :3].T, s=triangulated_idxs_weights, c=triangulated_idxs_weights)
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

        for prev_camera_extrinsic, camera_extrinsic in zip([None] + key_frame_camera_extrinsics[:-1], key_frame_camera_extrinsics):
            # The extrinsic matrix transforms from world coordinates to camera coordinates
            camera_lines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=image_size[0], view_height_px=image_size[1],
                                                                            intrinsic=camera_matrix,
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

        view_status = vis.get_view_status()
        view_status_time = time.time()
        visualisation_idle_timeout = 60 if IPython.get_ipython() is not None else 10
        while True:
            close_vis = False
            geometries = []
            for triangulated_idxs_count in sorted(np.unique(triangulated_idxs_counts), reverse=True):
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(stitched_triangulated_points[triangulated_idxs_counts == triangulated_idxs_count, :3]))
                pcd.colors = o3d.utility.Vector3dVector(stitched_triangulated_points[triangulated_idxs_counts == triangulated_idxs_count, 3:] / 255)
                pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox([-100, -100, -100], [100, 100, 100]))
                vis.add_geometry(pcd, reset_bounding_box=False)
                geometries.append(pcd)

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

                if close_vis:
                    break

            if close_vis:
                break

            for geometry in geometries:
                vis.remove_geometry(geometry, reset_bounding_box=False)

        vis.destroy_window()
        """

        # %%

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1024, height=768, left=200, top=200)

        opt = vis.get_render_option()
        opt.point_show_normal = True
        opt.point_size = 2.0
        #opt.show_coordinate_frame = True

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

        for prev_camera_extrinsic, camera_extrinsic in zip([None] + key_frame_camera_extrinsics[:-1], key_frame_camera_extrinsics):
            # The extrinsic matrix transforms from world coordinates to camera coordinates
            camera_lines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=image_size[0], view_height_px=image_size[1],
                                                                            intrinsic=camera_matrix,
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

        valid_mask = np.all(np.isfinite(stitched_triangulated_points[:, :3]), axis=1)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(stitched_triangulated_points[valid_mask, :3]))
        pcd.colors = o3d.utility.Vector3dVector(stitched_triangulated_points[valid_mask, 3:] / 255)
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
                                                                      key_frame_camera_extrinsics):
            triangulated_image_idxs = np.where(image_to_triangulated_point_idxs >= 0)[0]
            triangulated_point_idxs = image_to_triangulated_point_idxs[triangulated_image_idxs]
            image_points_weights = np.zeros((stitched_triangulated_points.shape[0],), dtype=np.float32)
            image_points_weights[triangulated_point_idxs] = 1

            R = camera_extrinsic[:3, :3]
            t = camera_extrinsic[:3, 3:]
            object_points = R @ pcd_points + t
            object_normals = R @ pcd_normals

            object_point_ray = object_points / np.clip(np.linalg.norm(object_points, axis=0), 1e-6, np.inf)
            normal_ray_alignment += np.sum(object_point_ray * object_normals, axis=0) * image_points_weights[valid_mask]
        pcd.normals = o3d.utility.Vector3dVector((pcd_normals * -np.sign(normal_ray_alignment)).T)
        assert np.allclose(np.linalg.norm(np.array(pcd.normals), axis=1), 1)

        stitched_triangulated_normals = np.full((3, stitched_triangulated_points.shape[0]), fill_value=np.nan, dtype=np.float32)
        stitched_triangulated_normals[:, valid_mask] = np.array(pcd.normals).T

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

        model_camera_fxy = torch.tensor(np.diag(camera_matrix[:2, :2])[:, None], dtype=torch.float32, requires_grad=True)
        model_camera_cxy = torch.tensor(camera_matrix[:2, 2:], dtype=torch.float32, requires_grad=True)
        model_camera_k1 = torch.tensor(0, dtype=torch.float32)
        model_camera_k2 = torch.tensor(0, dtype=torch.float32)
        model_camera_p1 = torch.tensor(0, dtype=torch.float32)
        model_camera_p2 = torch.tensor(0, dtype=torch.float32)

        # Although including all the camera extrinsics introduces redundant degrees of freedom,
        # this appears to help make the optimisation unbiased to each camera extrinsic.
        # Previously when the first frame was fixed to the origin/axes, it appeared to be susceptible
        # to higher projection errors especially for larger convergence criteria thresholds,
        # perhaps because the gradient graph for the first frame is far more convoluted.
        rvecs = []
        tvecs = []
        for camera_extrinsic in key_frame_camera_extrinsics:
            rvec, jacobian = cv2.Rodrigues(camera_extrinsic[:3, :3])
            tvec = camera_extrinsic[:3, 3:]
            rvecs.append(rvec.flatten())
            tvecs.append(tvec)
        model_rvecs = torch.tensor(np.array(rvecs), dtype=torch.float32, requires_grad=True)
        model_tvecs = torch.tensor(np.array(tvecs), dtype=torch.float32, requires_grad=True)

        model_triangulated_points = stitched_triangulated_points[:, :3].T

        triangulated_image_points = []
        triangulated_image_points_weights = []
        for (xs, ys), image_to_triangulated_point_idxs, image_point_weights, camera_extrinsic in zip(key_frame_image_sample_points,
                                                                                                     key_frame_image_triangulated_point_idxs,
                                                                                                     key_frame_image_point_weights,
                                                                                                     key_frame_camera_extrinsics):
            triangulated_image_idxs = np.where(image_to_triangulated_point_idxs >= 0)[0]
            triangulated_point_idxs = image_to_triangulated_point_idxs[triangulated_image_idxs]

            image_points = np.full((2, stitched_triangulated_points.shape[0]), fill_value=np.nan, dtype=np.float32)
            image_points[:, triangulated_point_idxs] = np.vstack([xs, ys])[:, triangulated_image_idxs]
            triangulated_image_points.append(image_points)

            image_points_weights = np.zeros((6, stitched_triangulated_points.shape[0]))

            primary_eig_weights, secondary_eig_weights, primary_eig_vecs = image_point_weights
            eig_weights = primary_eig_weights + secondary_eig_weights * 1j
            interp = scipy.interpolate.RegularGridInterpolator((np.arange(eig_weights.shape[0]), np.arange(eig_weights.shape[1])),
                                                               eig_weights,
                                                               method='linear', bounds_error=True)
            eig_weights = interp((image_points[1, triangulated_point_idxs], image_points[0, triangulated_point_idxs])).astype(np.complex64)
            eig_vecs = primary_eig_vecs[:, :, 0] + primary_eig_vecs[:, :, 1] * 1j
            interp = scipy.interpolate.RegularGridInterpolator((np.arange(eig_vecs.shape[0]), np.arange(eig_vecs.shape[1])),
                                                               eig_vecs,
                                                               method='linear', bounds_error=True)
            eig_vecs = interp((image_points[1, triangulated_point_idxs], image_points[0, triangulated_point_idxs])).astype(np.complex64)
            eig_weights = np.vstack([eig_weights.real, eig_weights.imag,
                                     eig_vecs.real, eig_vecs.imag,
                                     eig_vecs.imag, -eig_vecs.real])
            image_points_weights[:, triangulated_point_idxs] = eig_weights

            triangulated_points = camera_extrinsic[:3, :3] @ model_triangulated_points[:, triangulated_point_idxs] + camera_extrinsic[:3, 3:]
            triangulated_normals = camera_extrinsic[:3, :3] @ stitched_triangulated_normals[:, triangulated_point_idxs]
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

        # Optimise triangulated points with 2 or more mapped image points
        # Triangulated points with 2 mapped image points implicitly provide epipolar constraints between frames.
        # Triangulated points with more than 2 mapped image points additionally provide depth / scaling associations between frames.
        triangulated_point_mapping_counts = np.sum(np.all(np.isfinite(triangulated_image_points), axis=1), axis=0)
        # All triangulated points have at least one mapped image point
        assert np.all(triangulated_point_mapping_counts > 0)
        # Triangulated points are initialised with two mapped image points, but the second (or a later point) may
        # exit from view or be pruned if it is obscured. However the association with the initial image point is still retained.
        assert np.all(np.all(~np.isfinite(model_triangulated_points), axis=0) == (triangulated_point_mapping_counts == 1))
        assert np.all(np.all(np.isfinite(model_triangulated_points), axis=0) == (triangulated_point_mapping_counts >= 2))
        bundle_triangulated_point_idxs = np.where(triangulated_point_mapping_counts >= 2)[0]

        print('len(bundle_triangulated_point_idxs)', len(bundle_triangulated_point_idxs),
              'of len(triangulated_point_mapping_counts)', len(triangulated_point_mapping_counts))

        model_triangulated_points = torch.tensor(model_triangulated_points[:, bundle_triangulated_point_idxs], dtype=torch.float32, requires_grad=True)
        bundle_image_points = torch.tensor(triangulated_image_points[:, :, bundle_triangulated_point_idxs], dtype=torch.float32)
        bundle_image_points[~torch.isfinite(bundle_image_points)] = 0
        bundle_image_points_weights = torch.tensor(triangulated_image_points_weights[:, :, bundle_triangulated_point_idxs], dtype=torch.float32)
        # TODO: Consider if threshold and alpha should be reduced for the final optimisation
        #threshold_target = torch.tensor(3.0 + np.log2(triangulated_point_mapping_counts[bundle_triangulated_point_idxs]), dtype=torch.float32)
        threshold_target = torch.tensor(10.0, dtype=torch.float32)
        #gamma_softplus_alpha = torch.tensor(2.0 + np.log2(triangulated_point_mapping_counts[bundle_triangulated_point_idxs]), dtype=torch.float32)
        gamma_softplus_alpha = torch.tensor(6, dtype=torch.float32)

        def calc_projected_image_points_with_camera_model():
            model_Rs = pytorch3d.transforms.axis_angle_to_matrix(model_rvecs)
            object_points = model_Rs @ model_triangulated_points + model_tvecs

            xd = object_points[:, 0:1, :] / torch.clamp(object_points[:, 2:3, :], 1e-3, np.inf)
            yd = object_points[:, 1:2, :] / torch.clamp(object_points[:, 2:3, :], 1e-3, np.inf)
            r2 = xd * xd + yd * yd
            r4 = r2 * r2
            k = 1 + model_camera_k1 * r2 + model_camera_k2 * r4
            xdyd = xd * yd
            xdd = xd * k + 2 * model_camera_p1 * xdyd + model_camera_p2 * (r2 + 2 * xd * xd)
            ydd = yd * k + 2 * model_camera_p2 * xdyd + model_camera_p1 * (r2 + 2 * yd * yd)

            return torch.cat([xdd, ydd], dim=1) * model_camera_fxy + model_camera_cxy

        def loss_fn(temperature):
            bundle_projected_points = calc_projected_image_points_with_camera_model()
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
        num_steps = 5000
        convergence_criterion = {'rtol': 1e-4, 'window_size': 100, 'min_num_steps': 300}
        optimiser = torch.optim.Adam([model_camera_fxy, model_camera_cxy, model_rvecs, model_tvecs, model_triangulated_points], lr=lr)
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
                print('optim_step', optim_step, 'loss', losses[-1], 'loss_window_std', loss_window_std,
                      'learning rate', np.round(scheduler.get_last_lr()[0], 6), 'temperature', np.round(temperature, 3))
            if optim_step >= convergence_criterion['min_num_steps'] and loss_window_std < convergence_criterion['rtol'] * losses[-1]:
                break

        print('len(losses)', len(losses))
        print('initial, first, final loss', losses[0], losses[convergence_criterion['min_num_steps']], losses[-1])
        print('highest, lowest loss', np.max(losses[convergence_criterion['min_num_steps']:]), np.min(losses[convergence_criterion['min_num_steps']:]))

        print('model_camera_fxy', model_camera_fxy.numpy(force=True).flatten())
        print('model_camera_cxy', model_camera_cxy.numpy(force=True).flatten())
        print('camera k1, k2, p1, p2', model_camera_k1.numpy(force=True), model_camera_k2.numpy(force=True), model_camera_p1.numpy(force=True), model_camera_p2.numpy(force=True))

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

        with torch.no_grad():
            bundle_projected_points = np.full(triangulated_image_points.shape, fill_value=np.nan, dtype=np.float32)
            bundle_projected_points[:, :, bundle_triangulated_point_idxs] = calc_projected_image_points_with_camera_model().numpy(force=True)

        camera_intrinsic = np.block([[np.diag(model_camera_fxy.numpy(force=True).flatten()), model_camera_cxy.numpy(force=True)], [0, 0, 1]])

        camera_extrinsics = []
        for rvec, tvec in zip(model_rvecs.numpy(force=True), model_tvecs.numpy(force=True)):
            R, jacobian = cv2.Rodrigues(rvec)
            t = tvec
            camera_extrinsics.append(np.block([[R, t], [0, 0, 0, 1]]))

        first_camera_extrinsic = camera_extrinsics[0]
        R = first_camera_extrinsic[:3, :3]
        t = first_camera_extrinsic[:3, 3:]
        model_triangulated_points_numpy = R @ model_triangulated_points.numpy(force=True) + t

        camera_extrinsics = [camera_extrinsic @ np.linalg.inv(first_camera_extrinsic)
                             for camera_extrinsic in camera_extrinsics]

        scaling_factors = []
        for image_points_weights_magn, camera_extrinsic in zip(torch.norm(bundle_image_points_weights[:, :2, :], dim=1).numpy(force=True),
                                                               camera_extrinsics):
            triangulated_points = camera_extrinsic[:3, :3] @ model_triangulated_points_numpy[:, image_points_weights_magn >= 1e-2] + camera_extrinsic[:3, 3:]

            # Estimate the scaling factor from the depth distribution of the triangulated points
            # assuming the camera is positioned at a distance of ~8mm from the target
            c = np.percentile(triangulated_points[2, :], 25) / 8.0
            scaling_factors.append(c)
        c = np.median(scaling_factors)

        print('scaling factor', c, np.round(scaling_factors, 2))
        model_triangulated_points_numpy /= c
        camera_extrinsics = [np.block([[camera_extrinsic[:3, :3], camera_extrinsic[:3, 3:] / c], [0, 0, 0, 1]])
                             for camera_extrinsic in camera_extrinsics]

        model_triangulated_points = np.array(stitched_triangulated_points.T)
        model_triangulated_points[:3, bundle_triangulated_point_idxs] = model_triangulated_points_numpy

        bundle_image_points_numpy = bundle_image_points.numpy(force=True)
        bundle_image_points = np.full(triangulated_image_points.shape, fill_value=np.nan, dtype=np.float32)
        bundle_image_points[:, :, bundle_triangulated_point_idxs] = bundle_image_points_numpy
        bundle_image_points_weights_numpy = bundle_image_points_weights.numpy(force=True)
        bundle_image_points_weights = np.full(triangulated_image_points_weights.shape, fill_value=np.nan, dtype=np.float32)
        bundle_image_points_weights[:, :, bundle_triangulated_point_idxs] = bundle_image_points_weights_numpy
        bundle_image_points_weights_magn = np.linalg.norm(bundle_image_points_weights[:, :2, :], axis=1)

        for frame_idx, (projected_points, image_points, image_points_weights_magn, (img, gray)) in enumerate(zip(bundle_projected_points,
                                                                                                                 bundle_image_points,
                                                                                                                 bundle_image_points_weights_magn,
                                                                                                                 key_frame_images)):
            image_points_mask = image_points_weights_magn >= 1e-2
            projected_points = projected_points[:, image_points_mask]
            image_points = image_points[:, image_points_mask]
            image_points_weights_magn = image_points_weights_magn[image_points_mask]
            object_points = model_triangulated_points[:3, image_points_mask]
            triangulated_rgb = model_triangulated_points[3:, image_points_mask].T

            plt.figure('Optimised projected point correspondences', figsize=(16, 10))
            setup_new_fig_page()
            plt.suptitle(f'Frame idx: {frame_idx}')
            ax = plt.subplot(2, 2, 1)
            plt.imshow(np.require(img, dtype=np.uint8))
            ax3 = plt.subplot(2, 2, 2, projection='3d')
            ax3.scatter(*object_points, s=2, c=triangulated_rgb/255)
            ax3.set_xlim((-20, 20))
            ax3.set_ylim((-20, 20))
            ax3.set_zlim((0, 40))
            ax3.set_aspect('equal', adjustable='datalim')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            ax3.view_init(elev=-135, azim=-90, roll=0)
            ax = plt.subplot(2, 2, 3, sharex=ax, sharey=ax)
            plt.scatter(*image_points, s=3, c=triangulated_rgb/255, marker='o')
            plt.scatter(*projected_points, s=12, facecolors='none', edgecolors=triangulated_rgb/255, marker='o')
            ax.set_aspect('equal')
            ax.set_xlim((-20, img.shape[1] + 20))
            ax.set_ylim((img.shape[0] + 20, -20))
            ax = plt.subplot(2, 2, 4, sharex=ax, sharey=ax)
            plt.scatter(*image_points, s=15*image_points_weights_magn, c='b', marker='o')
            plt.scatter(*projected_points, s=15*image_points_weights_magn, c='y', marker='o')
            ax.set_aspect('equal')
            ax.set_xlim((-20, img.shape[1] + 20))
            ax.set_ylim((img.shape[0] + 20, -20))
            plt.tight_layout()
            stash_fig_page()

        # %%

        fig = plt.figure('Optimised triangulated points', figsize=(20, 10))
        plt.clf()
        ax0 = plt.subplot(3, 3, 1)
        ax1 = plt.subplot(3, 3, 4, sharex=ax0, sharey=ax0)
        ax2 = plt.subplot(3, 3, 7, sharex=ax1, sharey=ax1)
        ax3a = plt.subplot(1, 3, 2, projection='3d')
        ax3a.set_xlim((-20, 20))
        ax3a.set_ylim((-20, 20))
        ax3a.set_zlim((0, 40))
        ax3a.set_aspect('equal', adjustable='datalim')
        ax3a.set_xlabel('X')
        ax3a.set_ylabel('Y')
        ax3a.set_zlabel('Z')
        ax3a.view_init(elev=-90, azim=-90, roll=0)
        ax3b = plt.subplot(1, 3, 3, projection='3d')
        ax3b.set_xlim((-20, 20))
        ax3b.set_ylim((-20, 20))
        ax3b.set_zlim((0, 40))
        ax3b.set_aspect('equal', adjustable='datalim')
        ax3b.set_xlabel('X')
        ax3b.set_ylabel('Y')
        ax3b.set_zlabel('Z')
        ax3b.view_init(elev=-180, azim=-90, roll=0)

        fig_state = {'frame_idx': 0}
        def plot_new_fig_state(frame_idx_delta=0):
            frame_idx = np.clip(fig_state['frame_idx'] + frame_idx_delta, 0, len(key_frame_images) - 1)
            fig_state['frame_idx'] = frame_idx
            img, gray = key_frame_images[frame_idx]
            image_points_mask = bundle_image_points_weights_magn[frame_idx] >= 1e-2
            projected_points = bundle_projected_points[frame_idx][:, image_points_mask]
            image_points = bundle_image_points[frame_idx][:, image_points_mask]
            image_points_weights_magn = bundle_image_points_weights_magn[frame_idx][image_points_mask]
            object_points = model_triangulated_points[:3, image_points_mask]
            triangulated_rgb = model_triangulated_points[3:, image_points_mask].T

            fig.suptitle(f'Frame idx: {frame_idx}')
            ax0.cla()
            ax0.imshow(np.require(img, dtype=np.uint8))
            ax1.cla()
            ax1.scatter(*image_points, s=2, c=triangulated_rgb/255, marker='o')
            ax1.scatter(*projected_points, s=8, facecolors='none', edgecolors=triangulated_rgb/255, marker='o')
            ax1.set_aspect('equal')
            ax1.set_xlim((-20, img.shape[1] + 20))
            ax1.set_ylim((img.shape[0] + 20, -20))
            ax2.cla()
            ax2.scatter(*image_points, s=5*image_points_weights_magn, c='b', marker='o')
            ax2.scatter(*projected_points, s=5*image_points_weights_magn, c='y', marker='o')
            ax2.set_aspect('equal')
            ax2.set_xlim((-20, img.shape[1] + 20))
            ax2.set_ylim((img.shape[0] + 20, -20))

            xlim = ax3a.get_xlim()
            ylim = ax3a.get_ylim()
            zlim = ax3a.get_zlim()
            ax3a.cla()
            ax3a.scatter(*object_points, s=2, c=triangulated_rgb/255)
            ax3a.set_xlim(xlim)
            ax3a.set_ylim(ylim)
            ax3a.set_zlim(zlim)

            xlim = ax3b.get_xlim()
            ylim = ax3b.get_ylim()
            zlim = ax3b.get_zlim()
            ax3b.cla()
            ax3b.scatter(*object_points, s=2, c=triangulated_rgb/255)
            ax3b.set_xlim(xlim)
            ax3b.set_ylim(ylim)
            ax3b.set_zlim(zlim)

            fig.canvas.draw_idle()

        plot_new_fig_state()

        ax_prev = fig.add_axes([0.7, 0.05, 0.1, 0.075])
        ax_next = fig.add_axes([0.81, 0.05, 0.1, 0.075])
        button_prev = matplotlib.widgets.Button(ax_prev, 'Prev')
        button_prev.on_clicked(lambda event: plot_new_fig_state(frame_idx_delta=-1))
        button_next = matplotlib.widgets.Button(ax_next, 'Next')
        button_next.on_clicked(lambda event: plot_new_fig_state(frame_idx_delta=1))

        def disconnect_on_ax_clear(widget):
            old_ax_clear_fn = widget.ax.clear
            def new_ax_clear_fn():
                widget.disconnect_events()
                old_ax_clear_fn()
            widget.ax.clear = new_ax_clear_fn
        disconnect_on_ax_clear(button_prev)
        disconnect_on_ax_clear(button_next)

        # %%

        # Filter post optimisation triangulated points based on the distribution of projection errors
        triangulated_point_projection_errors = collections.defaultdict(list)
        for projected_points, image_points, image_points_weights_magn in zip(bundle_projected_points, bundle_image_points, bundle_image_points_weights_magn):
            image_points_mask = image_points_weights_magn >= 1e-2
            projection_errors = np.linalg.norm(projected_points - image_points, axis=0)[image_points_mask]
            for triangulated_point_idx, projection_error in zip(np.where(image_points_mask)[0], projection_errors):
                triangulated_point_projection_errors[triangulated_point_idx].append(projection_error)

        post_optimise_triangulated_idxs = np.array([triangulated_point_idx
                                                    for triangulated_point_idx, projection_errors in triangulated_point_projection_errors.items()
                                                    if len(projection_errors) >= 4 and np.percentile(projection_errors, 75) < 10.0], dtype=int)
        post_optimise_triangulated_idxs_mask = np.full((model_triangulated_points.shape[1],), fill_value=False, dtype=bool)
        post_optimise_triangulated_idxs_mask[post_optimise_triangulated_idxs] = True

        # %%

        output_path = output_dirpath / 'stitched_key_frames.pickle'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as pickle_file:
            pickle.dump({'key_frame_indices': key_frame_indices,
                         'key_frame_motion_blurs': key_frame_motion_blurs,
                         'triangulated_idxs_weights': triangulated_idxs_weights,
                         'key_frame_image_sample_points': key_frame_image_sample_points,
                         'key_frame_image_triangulated_point_idxs': key_frame_image_triangulated_point_idxs,
                         'cross_stitch_disparity_confidence_maps': cross_stitch_disparity_confidence_maps,
                         'camera_extrinsics': camera_extrinsics,
                         'camera_intrinsic': camera_intrinsic,
                         'model_triangulated_points': model_triangulated_points,
                         'post_optimise_triangulated_idxs_mask': post_optimise_triangulated_idxs_mask},
                        pickle_file)

        # %%

        plt.figure(figsize=(16, 10))
        ax = plt.subplot(1, 1, 1, projection='3d')
        ax.scatter(*stitched_triangulated_points[post_optimise_triangulated_idxs, :3].T, s=2, c='b', alpha=0.2)
        ax.scatter(*model_triangulated_points[:3, post_optimise_triangulated_idxs], s=2, c='r', alpha=0.2)
        ax.set_xlim((-20, 20))
        ax.set_ylim((-20, 20))
        ax.set_zlim((0, 40))
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=-135, azim=-90, roll=0)
        plt.tight_layout()

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

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(model_triangulated_points[:3, post_optimise_triangulated_idxs].T))
        pcd.colors = o3d.utility.Vector3dVector(model_triangulated_points[3:, post_optimise_triangulated_idxs].T / 255)
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

        # Cumulative counts of 3D triangulated points after stitching each key frame
        key_frame_cumul_num_stitched_triangulated_points = [0]
        for stitched_key_frame_path in sorted(output_dirpath.glob('*.stitch.pickle')):
            with open(stitched_key_frame_path, 'rb') as pickle_file:
                data = pickle.load(pickle_file)
                stitched_triangulated_points = data['stitched_triangulated_points']
                key_frame_cumul_num_stitched_triangulated_points.append(stitched_triangulated_points.shape[0])

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
            for prev_camera_extrinsic, camera_extrinsic, start_idx, end_idx in zip([None] + camera_extrinsics[:-1], camera_extrinsics,
                                                                                   [None] + key_frame_cumul_num_stitched_triangulated_points[:-1],
                                                                                   key_frame_cumul_num_stitched_triangulated_points):
                grayed_level = 0.9
                for geometry in geometries:
                    geometry.colors = o3d.utility.Vector3dVector(grayed_level - 0.95 * (grayed_level - np.array(geometry.colors)))
                    vis.update_geometry(geometry)

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

                if prev_camera_extrinsic is not None:
                    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(model_triangulated_points[:3, start_idx:end_idx].T))
                    pcd.colors = o3d.utility.Vector3dVector(model_triangulated_points[3:, start_idx:end_idx].T / 255)
                    pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox([-100, -100, -100], [100, 100, 100]))
                    vis.add_geometry(pcd, reset_bounding_box=False)
                    geometries.append(pcd)

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

                if close_vis:
                    break

            if close_vis:
                break

            for geometry in geometries:
                vis.remove_geometry(geometry, reset_bounding_box=False)

        vis.destroy_window()

    # %%

    if exec_mode == 'all_key_frames' or (exec_mode == 'next_key_frame' and len(key_frames_filepaths) in processed_frame_idxs):
        post_to_pipeline_server((f'{working_subdir} / {output_dirpath.name}', None))
