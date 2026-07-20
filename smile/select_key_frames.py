"""
Select key frames for image stitching by tracking cumulative optical
flow and using a cost function based on estimated flow displacement,
flow distortion and motion blur to compare the suitability of each frame.

This is one stage of the processing pipeline for https://github.com/mcmhsieh/Smile

SPDX-FileCopyrightText: 2026 Mark Hsieh
SPDX-License-Identifier: MIT
"""

import sys
import time
import datetime
import pathlib
import shutil
import pickle

import numpy as np
import scipy
import skimage
import cv2

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

from image_filtering import img_to_normed_gray, nan_gaussian_filter
from weighting_functions import cauchy
from fig_paging import setup_new_fig_page, stash_fig_page

from pipeline_server import start_pipeline_server, post_to_pipeline_server, get_queue_from_pipeline_server


if __name__ == '__main__':

    working_subdir_config_path = pathlib.Path(r'../pipeline-workspace/working_subdir.txt')
    with open(working_subdir_config_path, 'r') as config_file:
        working_subdir = config_file.read().rstrip('\n')

    workspace_dirpath = pathlib.Path(r'../pipeline-workspace') / working_subdir
    input_source_dirpath = workspace_dirpath / 'calc_sequential_flow_and_blur'
    output_dirpath = workspace_dirpath / 'select_key_frames'

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

    input_path = input_source_dirpath / 'camera_metadata.pickle'
    with open(input_path, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
        make_model = data['make_model']
        image_size = data['image_size']
        camera_matrix = data['camera_matrix']

    # %%

    # Sequence of all ordered frame indices
    frame_indices = []
    # Dictionary of filtered frame images
    filtered_images = {}
    # Dictionary of optical flow maps between sequential frames
    flow_maps = {}
    # Dictionary of frame blur scores
    frame_blur_scores = {}

    for input_filepath in sorted(input_source_dirpath.glob('*.flow.pickle')):
        with open(input_filepath, 'rb') as pickle_file:
            flow_data = pickle.load(pickle_file)

        prev_img_indices, next_img_indices, flow, next_img_blur_score = [
            flow_data[name]
            for name in ['prev_img_indices', 'next_img_indices', 'flow', 'next_img_blur_score']]

        if len(frame_indices) == 0:
            frame_indices.append(prev_img_indices)
        frame_indices.append(next_img_indices)

        for frame_index, frame_time in [prev_img_indices, next_img_indices]:
            if (frame_index, frame_time) not in filtered_images:
                filename = f'{frame_time.strftime("%Y%m%d-%H%M%S%f")}.{frame_index:03d}.filtered.png'
                img = cv2.cvtColor(cv2.imread(input_source_dirpath / filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB)
                filtered_images[(frame_index, frame_time)] = img

        flow_maps[(prev_img_indices, next_img_indices)] = flow

        # The blur scores are associated with the next frame, and therefore the first frame has no blur score
        frame_blur_scores[next_img_indices] = next_img_blur_score

    # %%

    # List of image frame indices being processed from the previous key frame to the current frame
    frame_image_stack = []
    # List of optical flow maps being processed from the previous key frame to the current frame
    flow_map_stack = []
    # List of cumulative optical flow correspondences being processed from the previous key frame to the current frame
    correspondences_stack = []
    # List of all computed and recomputed optical flow displacement magnitudes from the previous key frame to the current frame,
    # including values calculated for an extended window of frames to enable retrospective key frame selection,
    # together with an index referencing frame_indices
    frame_flow_displacements = []
    # Indices of selected key frames referencing frame_indices
    selected_key_frame_idxs = []

    # Derive pseudo motion blur estimates from the blur scores using the Softplus function which has characteristic points
    # (-3, 0.05), (0, 0.69), (3, 3.05) around the knee point (0, 0.69).
    # Note that directly calculating np.log(1 + np.exp(x)) is unstable for large x.
    # The blur scores are associated with the next frame, and therefore the first frame has no blur score
    blur_scores = np.array([frame_blur_scores[next_frame_indices] for next_frame_indices in frame_indices[1:]])
    frame_motion_blurs = np.pad(np.sqrt(np.logaddexp(0, blur_scores)), pad_width=(1, 0), constant_values=np.nan)

    # Key frames for pose estimation ideally have low motion blur, low flow distortion
    # and useful translation displacement from the previous key frame.
    # For a change in depth Δd = 1mm at a depth z = 15mm, with a baseline of b and fx = 360
    # the change in disparity Δu = b * fx * (1 / z - 1 / (z + Δz)) = 1.5 * b
    # Considering disparity_map_zoom = 0.125 and 0.25 for stitching and view synthesis,
    # for Δu in range [4, 8] then b = [2.667mm, 5.333mm]
    # With the target object at z = 15mm, [2.667mm, 5.333mm] of relative translation is visible as
    # [2.667, 5.333] / 15 * 360 = [64 pixels, 128 pixels] of flow displacement
    # TODO: consider selecting key frames to optimise disparity computation?

    # The flow costs model profile possesses the following characteristics:
    #  - for each flow displacement cross section, the cost values against motion blur + flow distortion:
    #    - are U-shaped
    #    - have a minimum located at blur + flow distortion == 0 when flow displacement == target_flow_displacement
    #  - for each motion blur + flow distortion cross section, the cost values against flow displacement:
    #    - are U-shaped
    #    - have a broadening U-shape for increasing motion blur + flow distortion
    #    - increase to infinity approaching zero flow displacement
    #    - increase with a power law like relationship to flow displacement beyond target_flow_displacement
    #    - have a minimum that:
    #      - is located at flow displacement == target_flow_displacement when motion blur + flow distortion == blur_and_distortion_key_frame_target
    #      - decreases with increasing motion blur + flow distortion
    blur_and_distortion_key_frame_target = 2.0
    target_flow_displacement = 128

    def calc_flow_costs(flow_displacements, blurs_and_distortions):
        # The gradient of this component wrt flow_displacements at flow_displacements = target_flow_displacement
        #   = 2 / target_flow_displacement
        cost_exponent = blurs_and_distortions / blur_and_distortion_key_frame_target + 3
        cost_magnitude = 1.1 / (blurs_and_distortions / blur_and_distortion_key_frame_target + 0.1)
        #cost_magnitude = cauchy(blurs_and_distortions, blur_and_distortion_key_frame_target * 3)
        flow_costs = cost_magnitude * 2 / cost_exponent * (np.power(np.clip(flow_displacements / target_flow_displacement, 0, np.inf), cost_exponent) - 1)
        # The gradient of this component wrt flow_displacements at flow_displacements = target_flow_displacement
        #   = -2 * (target_flow_displacement ** 0.5) / (target_flow_displacement ** 1.5)
        #   = -2 / target_flow_displacement
        flow_costs += cost_magnitude * 4 * (np.sqrt(target_flow_displacement / np.maximum(flow_displacements, 1e-8)) - 1)
        flow_costs += 0.2 * np.power(blurs_and_distortions / blur_and_distortion_key_frame_target, 2)
        return flow_costs

    plt.figure('Flow costs model', figsize=(16, 10))
    plt.clf()
    flow_displacements_grid, blurs_and_distortions_grid = np.meshgrid(np.arange(0.2, target_flow_displacement * 3, 0.2), np.arange(0.01, 9, 0.01))
    flow_costs_grid = calc_flow_costs(flow_displacements_grid, blurs_and_distortions_grid)
    ax = plt.subplot(1, 2, 1, projection='3d')
    flow_costs_grid_masked = np.array(flow_costs_grid)
    flow_costs_grid_masked[(flow_costs_grid_masked < 0) | (flow_costs_grid_masked > 5)] = np.nan
    ax.plot_wireframe(flow_displacements_grid, blurs_and_distortions_grid, flow_costs_grid_masked)
    ax.set_zlim((-0.5, 5.5))
    ax.set_xlabel('flow_displacements')
    ax.set_ylabel('motion_blurs + flow_distortions')
    ax.set_zlabel('flow_costs')
    ax.view_init(elev=80, azim=-90, roll=0)
    ax = plt.subplot(1, 2, 2, projection='3d')
    ax.contour(flow_displacements_grid, blurs_and_distortions_grid, flow_costs_grid, levels=np.arange(-0.1, 5.1, 0.1))
    ax.set_zlim((-0.5, 5.5))
    ax.set_xlabel('flow_displacements')
    ax.set_ylabel('motion_blurs + flow_distortions')
    ax.set_zlabel('flow_costs')
    ax.view_init(elev=80, azim=-90, roll=0)
    plt.tight_layout()

    def calc_flow_displacement_and_distortion(img, yfp, xfp, xfn, yfn, ysg, xsg):
        xyfn_flow = np.full(img.shape[:2] + (2,), fill_value=np.nan, dtype=np.float32)
        xyfn_flow[yfp, xfp] = np.vstack([xfn, yfn]).T

        # Correspondences based on a subsampled regular grid in the previous image coordinate space,
        # for feeding (after residual optical flow adjustment) to cv2.findEssentialMat() and cv2.recoverPose()
        # to estimate camera motion.
        xsn, ysn = xyfn_flow[ysg, xsg].T
        # Mask out a margin around the edge of the image where the flow estimation is less reliable
        margin = 5
        mask = (np.isfinite(xsn) & np.isfinite(ysn)
                & (xsg >= margin) & (xsg < img.shape[1] - margin) & (ysg >= margin) & (ysg < img.shape[0] - margin)
                & (xsn >= margin) & (xsn < img.shape[1] - margin) & (ysn >= margin) & (ysn < img.shape[0] - margin))
        xsp, ysp, xsn, ysn = xsg[mask], ysg[mask], xsn[mask], ysn[mask]

        dx, dy = xsn - xsp, ysn - ysp
        # Weight the displacement in each direction according to the image aspect ratio
        # to reflect the difference in stitching window areas available under motion along each dimension
        aspect_ratio = img.shape[1] / img.shape[0]
        flow_displacement = np.linalg.norm([np.median(dx) / np.sqrt(aspect_ratio),
                                            np.median(dy) * np.sqrt(aspect_ratio)])

        filtered_cross_flow = nan_gaussian_filter(xyfn_flow, ksize=(9, 9), unfiltered_point_value=np.nan)
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
                                               #np.pi / 2)
                                               np.pi / 4)

        #flow_grid_scaling_score = cauchy(np.sqrt(np.max(np.stack([flow_grid_gradient_cross, 1 / flow_grid_gradient_cross]), axis=0)) - 1, 0.2)
        #flow_grid_scaling_score = cauchy(np.max(np.stack([flow_grid_gradient_cross, 1 / flow_grid_gradient_cross]), axis=0) - 1, 0.3)
        flow_grid_scaling_score = np.exp(-0.5 * np.power((np.max(np.stack([flow_grid_gradient_cross, 1 / flow_grid_gradient_cross]), axis=0) - 1) / (0.1 / 3), 2))

        flow_laplacian_x = np.gradient(flow_gradient_x, grid_step, edge_order=2, axis=1)
        flow_laplacian_x_transverse_normed = np.abs(np.sum(np.stack([-flow_laplacian_x[:, :, 1], flow_laplacian_x[:, :, 0]], axis=-1) * flow_gradient_x, axis=-1)) / np.sum(np.power(flow_gradient_x, 2), axis=-1)
        flow_laplacian_y = np.gradient(flow_gradient_y, grid_step, edge_order=2, axis=0)
        flow_laplacian_y_transverse_normed = np.abs(np.sum(np.stack([-flow_laplacian_y[:, :, 0], flow_laplacian_y[:, :, 1]], axis=-1) * flow_gradient_y, axis=-1)) / np.sum(np.power(flow_gradient_y, 2), axis=-1)
        flow_laplacian_score = cauchy(np.linalg.norm(np.stack([flow_laplacian_x_transverse_normed, flow_laplacian_y_transverse_normed]), axis=0), 0.1)

        if True:
            mapping_score = (nan_gaussian_filter(flow_grid_scaling_score)
                             * nan_gaussian_filter(flow_grid_scaling_aspect_score)
                             * nan_gaussian_filter(flow_grid_orthogonality_score)
                             * nan_gaussian_filter(flow_laplacian_score))
            #flow_distortion = np.power(np.nanmean(1 - np.sqrt(mapping_score)) / 0.25, 2)
            flow_distortion = -2.0 * np.log(np.maximum(np.nanmean(mapping_score), 1e-8))
        elif True:
            mapping_score = 9 / (6 / np.maximum(nan_gaussian_filter(flow_grid_scaling_score), 1e-8)
                                 + 1 / np.maximum(nan_gaussian_filter(flow_grid_scaling_aspect_score), 1e-8)
                                 + 1 / np.maximum(nan_gaussian_filter(flow_grid_orthogonality_score), 1e-8)
                                 + 1 / np.maximum(nan_gaussian_filter(flow_laplacian_score), 1e-8))
            flow_distortion = -2.0 * np.log(np.maximum(np.nanmean(np.power(mapping_score, 2)), 1e-8))
            #flow_distortion = -6.0 * np.log(np.maximum(np.sqrt(np.nanmean(np.power(mapping_score, 2))), 1e-8))
        elif False:
            mapping_score = 3 / (1 / np.maximum(nan_gaussian_filter(flow_grid_scaling_aspect_score), 1e-8)
                                 + 1 / np.maximum(nan_gaussian_filter(flow_grid_orthogonality_score), 1e-8)
                                 + 1 / np.maximum(nan_gaussian_filter(flow_laplacian_score), 1e-8))
            flow_distortion = np.nanmean(1 - np.power(mapping_score, 3)) / 0.25
        elif False:
            mapping_score = 3 / (1 / np.maximum(nan_gaussian_filter(flow_grid_scaling_aspect_score), 1e-8)
                                 + 1 / np.maximum(nan_gaussian_filter(flow_grid_orthogonality_score), 1e-8)
                                 + 1 / np.maximum(nan_gaussian_filter(flow_laplacian_score), 1e-8))
            flow_distortion = np.nanmean(1 - np.power(mapping_score, 2)) / 0.25
        else:
            mapping_score = 3 / (1 / np.maximum(nan_gaussian_filter(flow_grid_scaling_aspect_score), 1e-8)
                                 + 1 / np.maximum(nan_gaussian_filter(flow_grid_orthogonality_score), 1e-8)
                                 + 1 / np.maximum(nan_gaussian_filter(flow_laplacian_score), 1e-8))
            flow_distortion = np.nanmean(1 - mapping_score) / 0.25

        corres_pts_p = np.vstack([xsp, ysp]).T.astype(np.float32)
        corres_pts_n = np.vstack([xsn, ysn]).T.astype(np.float32)

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

        epipolar_rmse = []
        for E, t in results:
            # F_xysp are the epipolar lines upon which the corresponding xysn points should lie
            F_xysp = np.linalg.inv(camera_matrix).T @ E @ np.linalg.inv(camera_matrix) @ np.vstack([corres_pts_p.T, np.ones((corres_pts_p.shape[0],))])
            xysn = np.vstack([corres_pts_n.T, np.ones((corres_pts_n.shape[0],))])
            # Normalise the lines
            # https://math.stackexchange.com/questions/4585131/understanding-distance-from-point-to-line-in-homogeneous-coordinates
            # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/BEARDSLEY/node2.html
            F_xysp = F_xysp[:3, :] / np.clip(np.linalg.norm(F_xysp[:2, :], axis=0), 1e-8, np.inf)

            epipolar_rmse.append(np.sqrt(np.mean(np.power(np.sum(xysn * F_xysp, axis=0), 2))))

        return flow_displacement, 0.25 * flow_distortion + 0.75 * np.min(epipolar_rmse)


    frame_image_stack.append(frame_indices[0])

    for frame_indices_idx, ((prev_frame_index, prev_frame_time), (frame_index, frame_time)) in enumerate(zip(frame_indices[:-1], frame_indices[1:]), start=1):

        frame_image_stack.append((frame_index, frame_time))
        flow = flow_maps[((prev_frame_index, prev_frame_time), (frame_index, frame_time))]
        flow_map_stack.append(flow)

        img = filtered_images[frame_image_stack[-1]]

        grid_step = 8
        h, w = img.shape[:2]
        yfg, xfg = np.mgrid[0:h, 0:w].reshape((2, -1)).astype(int)
        ysg, xsg = np.mgrid[grid_step/2:h:grid_step, grid_step/2:w:grid_step].reshape((2, -1)).astype(int)

        # Track the cumulative optical flow correspondences based on a full regular reference grid in the previous key frame's
        # image coordinate space
        if len(selected_key_frame_idxs) == 0:
            # Select the previous frame as the first key frame if its motion blur
            # meets the target key frame motion blur value
            prev_frame_indices_idx = frame_indices_idx - 1
            if frame_motion_blurs[prev_frame_indices_idx] <= blur_and_distortion_key_frame_target:
                assert len(frame_image_stack) >= 2

                xfp, yfp = xfg, yfg
                xfn, yfn = flow[yfp, xfp].T + np.vstack([xfp, yfp], dtype=np.float32)
                #mask = np.all(np.isfinite(np.vstack([xfn, yfn])), axis=0)
                #xfp, yfp, xfn, yfn = xfp[mask], yfp[mask], xfn[mask], yfn[mask]
                correspondences_stack.append((xfp, yfp, xfn, yfn))
                selected_key_frame_idxs.append(prev_frame_indices_idx)
                del frame_image_stack[:-2]
                del flow_map_stack[:-1]

                print('-' * 80)
                print('selected key frame', prev_frame_indices_idx, *frame_indices[prev_frame_indices_idx])
                print('motion blur', frame_motion_blurs[prev_frame_indices_idx])
            else:
                del frame_image_stack[:-1]
        else:
            interp = scipy.interpolate.RegularGridInterpolator((np.arange(flow.shape[0]), np.arange(flow.shape[1])),
                                                               flow[:, :, 0] + flow[:, :, 1] * 1j,
                                                               method='linear', bounds_error=False, fill_value=np.nan)
            if len(correspondences_stack) > 0:
                xfp, yfp, xfn, yfn = correspondences_stack[-1]
            else:
                xfp, yfp, xfn, yfn = xfg, yfg, xfg.astype(np.float32), yfg.astype(np.float32)
            dxyn = interp((yfn, xfn)).astype(np.complex64)
            mask = np.isfinite(dxyn)
            xfp, yfp = xfp[mask], yfp[mask]
            xfn, yfn = xfn[mask] + dxyn[mask].real, yfn[mask] + dxyn[mask].imag
            correspondences_stack.append((xfp, yfp, xfn, yfn))

        if len(selected_key_frame_idxs) > 0:
            # Calculate cumulative key frame flow displacement and flow distortion
            flow_displacement, flow_distortion = calc_flow_displacement_and_distortion(img, yfp, xfp, xfn, yfn, ysg, xsg)
            frame_flow_displacements.append((frame_indices_idx, flow_displacement, flow_distortion))

        while (len(frame_image_stack) >= 2 and len(frame_flow_displacements) > 0 and len(correspondences_stack) > 0
               and ((frame_index, frame_time) == frame_indices[-1]
                    or frame_flow_displacements[-1][1] > 200
                    or frame_flow_displacements[-1][2] > 9
                    or len(correspondences_stack[-1][0]) < 2 / 3 * np.prod(img.shape[:2]))):

            prev_key_frame_idx = selected_key_frame_idxs[-1]
            motion_blurs = frame_motion_blurs[prev_key_frame_idx+1:frame_indices_idx+1]
            flow_displacements, flow_distortions = np.array(frame_flow_displacements)[-len(motion_blurs):, 1:].T
            flow_costs = calc_flow_costs(flow_displacements, motion_blurs + flow_distortions)

            next_key_frame_offset = np.argmin(flow_costs)
            if ((frame_index, frame_time) == frame_indices[-1]
                and flow_costs[next_key_frame_offset] > calc_flow_costs(target_flow_displacement, 3 * blur_and_distortion_key_frame_target)):
                break
            next_key_frame_indices_idx = prev_key_frame_idx + 1 + next_key_frame_offset

            if len(selected_key_frame_idxs) < 2:
                plt.close('Flow costs')

            plt.figure('Flow costs', figsize=(16, 10))
            setup_new_fig_page()
            plt.suptitle('Ref key frame: {} {}'.format(*frame_image_stack[0]))
            ax = plt.subplot(4, 2, 1)
            plt.plot(prev_key_frame_idx + 1 + np.arange(len(flow_displacements)), flow_displacements, 'o-')
            plt.axvline(next_key_frame_indices_idx, linestyle='--', color='black')
            plt.ylim((0, plt.ylim()[1]))
            plt.title('flow_displacements')
            plt.subplot(4, 2, 3, sharex=ax)
            plt.plot(prev_key_frame_idx + 1 + np.arange(len(flow_distortions)), flow_distortions, 'o-')
            plt.axvline(next_key_frame_indices_idx, linestyle='--', color='black')
            plt.ylim((0, plt.ylim()[1]))
            plt.title('flow_distortions')
            plt.subplot(4, 2, 5, sharex=ax)
            plt.plot(prev_key_frame_idx + 1 + np.arange(len(motion_blurs)), motion_blurs, 'o-')
            plt.axvline(next_key_frame_indices_idx, linestyle='--', color='black')
            plt.ylim((0, plt.ylim()[1]))
            plt.title('motion_blurs')
            plt.subplot(4, 2, 7, sharex=ax)
            plt.plot(prev_key_frame_idx + 1 + np.arange(len(flow_costs)), flow_costs, 'o-')
            plt.axvline(next_key_frame_indices_idx, linestyle='--', color='black')
            plt.ylim((-0.5, 5.5))
            plt.title('flow_costs')
            ax = plt.subplot(1, 2, 2, projection='3d')
            ax.plot(flow_displacements, motion_blurs + flow_distortions, flow_costs, 'ko:')
            flow_displacements_grid, blurs_and_distortions_grid = np.meshgrid(np.arange(0.2, target_flow_displacement * 3, 0.2), np.arange(0.01, 9, 0.01))
            flow_costs_grid = calc_flow_costs(flow_displacements_grid, blurs_and_distortions_grid)
            ax.contour(flow_displacements_grid, blurs_and_distortions_grid, flow_costs_grid, levels=np.arange(-0.1, 5.1, 0.1))
            ax.set_zlim((-0.5, 5.5))
            ax.set_xlabel('flow_displacements')
            ax.set_ylabel('motion_blurs + flow_distortions')
            ax.set_zlabel('flow_costs')
            ax.view_init(elev=80, azim=-90, roll=0)
            plt.tight_layout()
            stash_fig_page()

            for frame_offset_idx in range(next_key_frame_offset + 1):
                current_frame_indices_idx = prev_key_frame_idx + 1 + frame_offset_idx
                current_frame_index, current_frame_time = frame_indices[current_frame_indices_idx]

                ref_motion_blur = frame_motion_blurs[prev_key_frame_idx]
                motion_blur = frame_motion_blurs[current_frame_indices_idx]

                flow_displacement = flow_displacements[frame_offset_idx]

                ref_img_indices = frame_image_stack[0]
                img_indices = frame_image_stack[1 + frame_offset_idx]

                ref_img = filtered_images[ref_img_indices]
                img = filtered_images[img_indices]

                ref_gray = img_to_normed_gray(ref_img)
                gray = img_to_normed_gray(img)

                # Calculate residual optical flow between key frames to refine the cumulative vector values
                xfp, yfp, xfn, yfn = correspondences_stack[frame_offset_idx]

                # TODO: calculate the next frame's flow by interpolating from the previous frame's refined flow
                for iter_idx in range(1):
                    xyfn_flow = np.full(img.shape[:2] + (2,), fill_value=np.nan, dtype=np.float32)
                    xyfn_flow[yfp, xfp] = np.vstack([xfn, yfn]).T

                    # Mask the pixels in the reference image that do not intersect with the current image
                    border_value = 127
                    ref_gray_masked = np.array(ref_gray)
                    ref_gray_masked[np.any(~np.isfinite(xyfn_flow), axis=2)] = border_value

                    gray_warp = cv2.remap(np.array(gray), xyfn_flow, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)

                    # prev(y, x) ~ next(y + flow(y, x)[1], x + flow(y, x)[0])
                    # when cv2.OPTFLOW_FARNEBACK_GAUSSIAN is applied, flow_sigma = (winsize // 2) * 0.3
                    # poly_n: size of the pixel neighborhood used to find polynomial expansion in each pixel;
                    #         larger values mean that the image will be approximated with smoother surfaces,
                    #         yielding more robust algorithm and more blurred motion field, typically poly_n=5 or 7.
                    # poly_sigma: standard deviation of the Gaussian that is used to smooth derivatives used as a basis
                    #             for the polynomial expansion;
                    #             for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
                    flow_warp = cv2.calcOpticalFlowFarneback(prev=cv2.resize(ref_gray_masked, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                                                             next=cv2.resize(gray_warp, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                                                             flow=None,
                                                             pyr_scale=0.5, levels=3, winsize=51, iterations=3,
                                                             poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                    flow_warp = cv2.resize(flow_warp, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST) * 2

                    flow_fp = flow_warp[yfp, xfp]
                    interp = scipy.interpolate.RegularGridInterpolator((np.arange(xyfn_flow.shape[0]), np.arange(xyfn_flow.shape[1])),
                                                                       xyfn_flow[:, :, 0] + xyfn_flow[:, :, 1] * 1j,
                                                                       method='linear', bounds_error=False, fill_value=np.nan)
                    xyfn_flow_interp = interp((yfp + flow_fp[:, 1], xfp + flow_fp[:, 0])).astype(np.complex64)
                    xfn, yfn = xyfn_flow_interp.real, xyfn_flow_interp.imag

                    dx, dy = flow_warp[ysg, xsg].T
                    print('iter_idx', iter_idx, 'root median squared residual dxy', np.sqrt(np.median(dx * dx + dy * dy)))

                if frame_offset_idx < next_key_frame_offset:
                    current_frame_type = 'aux'
                else:
                    print('-' * 80)
                    print('selected key frame', current_frame_indices_idx, current_frame_index, current_frame_time)
                    print('flow_displacement', flow_displacement)
                    print('motion_blur', motion_blur)
                    current_frame_type = 'key'

                    xyfn_flow = np.full(img.shape[:2] + (2,), fill_value=np.nan, dtype=np.float32)
                    xyfn_flow[yfp, xfp] = np.vstack([xfn, yfn]).T

                    xsn, ysn = xyfn_flow[ysg, xsg].T
                    # Mask out a margin around the edge of the image where the flow estimation is less reliable
                    margin = 5
                    mask = (np.isfinite(xsn) & np.isfinite(ysn)
                            & (xsg >= margin) & (xsg < img.shape[1] - margin) & (ysg >= margin) & (ysg < img.shape[0] - margin)
                            & (xsn >= margin) & (xsn < img.shape[1] - margin) & (ysn >= margin) & (ysn < img.shape[0] - margin))
                    xsp, ysp, xsn, ysn = xsg[mask], ysg[mask], xsn[mask], ysn[mask]

                    img_warp = cv2.remap(np.array(img).astype(np.float32), xyfn_flow, None, cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(np.nan, np.nan, np.nan))

                    if len(selected_key_frame_idxs) < 2:
                        plt.close('Key frame point correspondences')

                    fig = plt.figure('Key frame point correspondences', figsize=(16, 10))
                    setup_new_fig_page()
                    plt.suptitle(f'Previous, current frame idxs: {prev_key_frame_idx}, {current_frame_indices_idx}')
                    ax = plt.subplot(2, 2, 1)
                    plt.imshow(np.require(ref_img, dtype=np.uint8))
                    plt.scatter(xsp, ysp, s=2, c='b', alpha=0.5)
                    ax = plt.subplot(2, 2, 2, sharex=ax, sharey=ax)
                    plt.imshow(np.require(img, dtype=np.uint8))
                    plt.scatter(xsn, ysn, s=2, c='b', alpha=0.5)
                    ax = plt.subplot(2, 2, 3, sharex=ax, sharey=ax)
                    plt.imshow(img_warp / 255)
                    plt.scatter(xsp, ysp, s=2, c='b', alpha=0.5)
                    ax = plt.subplot(2, 2, 4, sharex=ax, sharey=ax)
                    plt.imshow(np.clip((ref_img.astype(np.float32) - img_warp) / 255 + 0.5, 0, 1))
                    plt.scatter(xsp, ysp, s=2, c='b', alpha=0.5)
                    plt.tight_layout()
                    stash_fig_page()

                filename = f'{current_frame_time.strftime("%Y%m%d-%H%M%S%f")}.{current_frame_index:03d}.flow.{current_frame_type}.pickle'
                output_path = output_dirpath / filename
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as pickle_file:
                    pickle.dump({'ref_img_indices': ref_img_indices,
                                 'img_indices': img_indices,
                                 'camera_matrix': camera_matrix,
                                 'flow_displacement': flow_displacement,
                                 'ref_motion_blur': ref_motion_blur,
                                 'motion_blur': motion_blur,
                                 'flow_vectors': (xfp, yfp, xfn, yfn)},
                                pickle_file)

            selected_key_frame_idxs.append(next_key_frame_indices_idx)

            # Delete the images, masks and flow maps up to the new key frame
            del frame_image_stack[:next_key_frame_offset + 1]
            del flow_map_stack[:next_key_frame_offset + 1]
            # Delete the cumulative optical flow correspondences after the new key frame
            del correspondences_stack[next_key_frame_offset+1:]

            xyfn_flow = np.full(img.shape[:2] + (2,), fill_value=np.nan, dtype=np.float32)
            xyfn_flow[yfp, xfp] = np.vstack([xfn, yfn]).T

            # Correspondences based on a subsampled regular grid in the previous image coordinate space,
            # for feeding (after residual optical flow adjustment) to cv2.findEssentialMat() and cv2.recoverPose()
            # to estimate camera motion.
            xsn, ysn = xyfn_flow[ysg, xsg].T
            # Mask out a margin around the edge of the image where the flow estimation is less reliable
            margin = 5
            mask = (np.isfinite(xsn) & np.isfinite(ysn)
                    & (xsg >= margin) & (xsg < img.shape[1] - margin) & (ysg >= margin) & (ysg < img.shape[0] - margin)
                    & (xsn >= margin) & (xsn < img.shape[1] - margin) & (ysn >= margin) & (ysn < img.shape[0] - margin))
            xsp, ysp, xsn, ysn = xsg[mask], ysg[mask], xsn[mask], ysn[mask]

            print('num corresponding flow points', len(xsp))


            # Having set a new key frame, recalculate the cumulative optical flow correspondences and flow displacements
            del correspondences_stack[:]
            for flow_idx, flow in enumerate(flow_map_stack, start=next_key_frame_indices_idx+1):
                interp = scipy.interpolate.RegularGridInterpolator((np.arange(flow.shape[0]), np.arange(flow.shape[1])),
                                                                   flow[:, :, 0] + flow[:, :, 1] * 1j,
                                                                   method='linear', bounds_error=False, fill_value=np.nan)
                if len(correspondences_stack) > 0:
                    xfp, yfp, xfn, yfn = correspondences_stack[-1]
                else:
                    xfp, yfp, xfn, yfn = xfg, yfg, xfg.astype(np.float32), yfg.astype(np.float32)
                dxyn = interp((yfn, xfn)).astype(np.complex64)
                mask = np.isfinite(dxyn)
                xfp, yfp = xfp[mask], yfp[mask]
                xfn, yfn = xfn[mask] + dxyn[mask].real, yfn[mask] + dxyn[mask].imag
                correspondences_stack.append((xfp, yfp, xfn, yfn))

                # Calculate cumulative key frame flow displacement and flow distortion
                flow_displacement, flow_distortion = calc_flow_displacement_and_distortion(img, yfp, xfp, xfn, yfn, ysg, xsg)
                frame_flow_displacements.append((flow_idx, flow_displacement, flow_distortion))

    # %%

    fig = plt.figure('Flow displacement', figsize=(16, 10))
    plt.clf()
    ax = plt.subplot(3, 1, 1)
    plt.plot(*np.array(frame_flow_displacements)[:, [0, 1]].T)
    flow_displacements = np.full((max(idx for idx, _, _ in frame_flow_displacements) + 1,), fill_value=np.nan)
    for idx, value, _ in frame_flow_displacements:
        flow_displacements[idx] = value
    plt.plot(selected_key_frame_idxs, flow_displacements[selected_key_frame_idxs], color='g', linestyle='--', marker='o')
    plt.ylabel('flow displacement')
    ax = plt.subplot(3, 1, 2, sharex=ax)
    plt.plot(*np.array(frame_flow_displacements)[:, [0, 2]].T)
    flow_distortions = np.full((max(idx for idx, _, _ in frame_flow_displacements) + 1,), fill_value=np.nan)
    for idx, _, value in frame_flow_displacements:
        flow_distortions[idx] = value
    plt.plot(selected_key_frame_idxs, flow_distortions[selected_key_frame_idxs], color='g', linestyle='--', marker='o')
    plt.ylabel('flow distortion')
    ax = plt.subplot(3, 1, 3, sharex=ax)
    plt.plot(frame_motion_blurs)
    plt.plot(selected_key_frame_idxs, frame_motion_blurs[selected_key_frame_idxs], color='g', linestyle='--', marker='o')
    plt.ylabel('motion blur')
    plt.tight_layout()

    # %%

    post_to_pipeline_server((f'{working_subdir} / {output_dirpath.name}', None))
