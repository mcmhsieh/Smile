"""
Filter each input frame image to reduce JPEG compression artifacts.
Detect, mask and inpaint specular reflection.
Calculate the optical flow between sequential frame images.
Use the optical flow maps to estimate the magnitude of motion blur for each frame.

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
import PIL

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

from image_filtering import img_to_normed_gray

from pipeline_server import start_pipeline_server, post_to_pipeline_server, get_queue_from_pipeline_server


if __name__ == '__main__':

    working_subdir_config_path = pathlib.Path(r'../pipeline-workspace/working_subdir.txt')
    if not working_subdir_config_path.exists():
        print('The working sub-directory configuration file does not exist')
        dirpaths = sorted([(dirpath.name, len(list(dirpath.iterdir()))) for dirpath in pathlib.Path(r'../pipeline-input').iterdir() if dirpath.is_dir()],
                          key=lambda item: item[1])
        selected_dirpath, _ = dirpaths[0]
        print('Defaulting to the sub-directory containing the shortest sequence of frames', selected_dirpath)
        with open(working_subdir_config_path, 'w') as config_file:
            config_file.write(selected_dirpath)

    with open(working_subdir_config_path, 'r') as config_file:
        working_subdir = config_file.read().rstrip('\n')

    input_source_dirpath = pathlib.Path(r'../pipeline-input') / working_subdir
    workspace_dirpath = pathlib.Path(r'../pipeline-workspace') / working_subdir
    output_dirpath = workspace_dirpath / 'calc_sequential_flow_and_blur'

    if output_dirpath.exists():
        shutil.rmtree(output_dirpath)

    start_pipeline_server()
    post_to_pipeline_server((f'{working_subdir} / {output_dirpath.name}', 'waiting'))
    while True:
        pipeline_queue = get_queue_from_pipeline_server()
        print(pipeline_queue)
        if f'{working_subdir} / ~' not in pipeline_queue:
            break
        time.sleep(10)
    post_to_pipeline_server((f'{working_subdir} / {output_dirpath.name}', 'running'))
    print(get_queue_from_pipeline_server())

    # %%

    # Sequence of all ordered frame indices
    frame_indices = []
    # Dictionary of frame image metadata
    frame_images_metadata = {}

    for input_filepath in sorted(input_source_dirpath.iterdir()):
        frame_time, frame_index, filename_ext = input_filepath.name.split('.')
        frame_time = datetime.datetime.strptime(frame_time, '%Y%m%d-%H%M%S%f')
        frame_index = int(frame_index)

        frame_indices.append((frame_index, frame_time))
        if len(frame_indices) >= 2 and (frame_indices[-1][0] - frame_indices[-2][0]) % 256 != 1:
            print(f'Skipped from frame {frame_indices[-2][0]} to frame {frame_indices[-1][0]}')

        image = PIL.Image.open(input_filepath)
        exif = image.getexif()
        exif_subifd = exif.get_ifd(PIL.ExifTags.IFD.Exif)

        frame_images_metadata[(frame_index, frame_time)] = {'filepath': input_filepath,
                                                            'size': image.size,
                                                            'exif': {'make': exif[PIL.ExifTags.Base.Make],
                                                                     'model': exif[PIL.ExifTags.Base.Model],
                                                                     'datetime': exif[PIL.ExifTags.Base.DateTime],
                                                                     'subsec_time': exif_subifd[PIL.ExifTags.Base.SubsecTime],
                                                                     'offset_time': exif_subifd[PIL.ExifTags.Base.OffsetTime],
                                                                     'datetime_original': exif_subifd[PIL.ExifTags.Base.DateTimeOriginal],
                                                                     'subsec_time_original': exif_subifd[PIL.ExifTags.Base.SubsecTimeOriginal],
                                                                     'offset_time_original': exif_subifd[PIL.ExifTags.Base.OffsetTimeOriginal],}}

    if len(frame_indices) >= 2:
        frame_indices_at_fps_intervals = frame_indices[:1] + frame_indices[-1:]
        for frame_index, frame_time in frame_indices[1:-1]:
            if (frame_time >= frame_indices_at_fps_intervals[-2][1] + datetime.timedelta(seconds=1)
                and frame_time <= frame_indices_at_fps_intervals[-1][1] - datetime.timedelta(seconds=1)):
                frame_indices_at_fps_intervals.insert(-1, (frame_index, frame_time))

        fps = []
        for (start_frame_index, start_frame_time), (end_frame_index, end_frame_time) in zip(frame_indices_at_fps_intervals[:-1],
                                                                                            frame_indices_at_fps_intervals[1:]):
            fps.append(((end_frame_index - start_frame_index) % 256) / (end_frame_time - start_frame_time).total_seconds())
        print('FPS [10, 50, 90] percentiles', np.percentile(fps, [10, 50, 90]))

    image_sizes = set(metadata['size'] for metadata in frame_images_metadata.values())
    assert len(image_sizes) == 1
    image_size = image_sizes.pop()
    print('image size', image_size)

    make_models = set((metadata['exif']['make'], metadata['exif']['model']) for metadata in frame_images_metadata.values())
    assert len(make_models) == 1
    make_model = make_models.pop()
    print('make, model', make_model)

    assert make_model in [('YPC', 'TX806-XRH-401'), ('MoLink Technology', 'iTiMO-0877')]
    assert image_size in [(1280, 720), (480, 640)]

    # %%

    if make_model == ('YPC', 'TX806-XRH-401'):
        # u = fx*X/Z + cx, v = fy*Y/Z + cy
        # at Z~10mm, the height of view is ~10mm,
        # i.e. 360 = fy * 10 / 10
        # so, the angle of view is ~83° horizontally and ~53° degrees vertically (2 * arctan([320, 180] / 360))
        w, h = 0.5 * np.array(image_size)
        camera_matrix = np.array([[360, 0,   (w - 1) / 2],
                                  [0,   360, (h - 1) / 2],
                                  [0,   0,   1]], dtype=np.float32)
    else:
        # u = fx*X/Z + cx, v = fy*Y/Z + cy
        # at Z~15mm, the 480px width of the view ~23mm
        # i.e. 480 = fx * 23 / 15
        # so, the angle of view is ~75° horizontally and ~91° degrees vertically (2 * arctan([240, 320] / 315))
        w, h = image_size
        camera_matrix = np.array([[315, 0,   (w - 1) / 2],
                                  [0,   315, (h - 1) / 2],
                                  [0,   0,   1]], dtype=np.float32)

    output_path = output_dirpath / 'camera_metadata.pickle'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as pickle_file:
        pickle.dump({'make_model': make_model,
                     'image_size': image_size,
                     'camera_matrix': camera_matrix},
                    pickle_file)

    # %%

    # Dictionary of frame images
    frame_images = {}
    # Dictionary of filtered frame images
    filtered_images = {}
    # Dictionary of specular reflection masks associated with the images from the previous key frame to the current frame
    image_masks = {}

    for frame_index, frame_time in frame_indices:
        print('filtering frame', frame_index, frame_time)

        input_filepath = frame_images_metadata[(frame_index, frame_time)]['filepath']

        frame_data = np.fromfile(input_filepath, dtype=np.uint8)
        image = cv2.imdecode(frame_data, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert image.shape[1::-1] == image_size

        # Filter JPEG compression artifacts
        # smallest resolvable object is approximately 5x5 pixels for YPC TX806-XRH-401 1280x720 images
        if make_model == ('YPC', 'TX806-XRH-401'):
            ksize = (7, 7)
            image = np.clip(cv2.GaussianBlur(image.astype(np.float32), ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_CONSTANT)
                            / cv2.GaussianBlur(np.ones(image.shape, dtype=np.float32), ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_CONSTANT), 0, 255).astype(np.uint8)
            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        else:
            ksize = (5, 5)
            image = np.clip(cv2.GaussianBlur(image.astype(np.float32), ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_CONSTANT)
                            / cv2.GaussianBlur(np.ones(image.shape, dtype=np.float32), ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_CONSTANT), 0, 255).astype(np.uint8)
        assert np.allclose(image.shape[1::-1], (camera_matrix[:2, 2] + 0.5) * 2)

        frame_images[(frame_index, frame_time)] = np.array(image)

        filename = f'{frame_time.strftime("%Y%m%d-%H%M%S%f")}.{frame_index:03d}.resized.png'
        output_path = output_dirpath / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), params=[cv2.IMWRITE_PNG_COMPRESSION, 1])

        # Mask and inpaint specular reflection
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(7, 7))
        kernel = kernel | kernel.T
        pad_width = 16
        img_padded = cv2.copyMakeBorder(image, top=pad_width, bottom=pad_width, left=pad_width, right=pad_width, borderType=cv2.BORDER_REPLICATE)
        img_opened = cv2.morphologyEx(img_padded, op=cv2.MORPH_OPEN, kernel=kernel, iterations=4, borderType=cv2.BORDER_REPLICATE)
        img_opened = img_opened[pad_width:-pad_width, pad_width:-pad_width]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(5, 5))
        kernel = kernel | kernel.T
        gray_rel = np.mean(image - img_opened, axis=-1)
        gray_rel_padded = cv2.copyMakeBorder(gray_rel, top=pad_width, bottom=pad_width, left=pad_width, right=pad_width, borderType=cv2.BORDER_REPLICATE)
        gray_rel_closed = cv2.morphologyEx(gray_rel_padded, op=cv2.MORPH_CLOSE, kernel=kernel, iterations=1, borderType=cv2.BORDER_REPLICATE)
        gray_rel_closed = gray_rel_closed[pad_width:-pad_width, pad_width:-pad_width]
        mask_rel = skimage.filters.apply_hysteresis_threshold(gray_rel_closed, low=16, high=24)

        gray_padded = cv2.copyMakeBorder(np.mean(image, axis=-1), top=pad_width, bottom=pad_width, left=pad_width, right=pad_width, borderType=cv2.BORDER_REPLICATE)
        gray_closed = cv2.morphologyEx(gray_padded, op=cv2.MORPH_CLOSE, kernel=kernel, iterations=1, borderType=cv2.BORDER_REPLICATE)
        gray_closed = gray_closed[pad_width:-pad_width, pad_width:-pad_width]
        mask_abs = skimage.filters.apply_hysteresis_threshold(gray_closed, low=232, high=248)

        mask_img = cv2.morphologyEx((mask_abs | mask_rel).astype(np.uint8), op=cv2.MORPH_DILATE, kernel=kernel, iterations=1, borderType=cv2.BORDER_REPLICATE)
        img_mask = mask_img.astype(bool)
        mask_img[img_mask] = 255

        image_masks[(frame_index, frame_time)] = img_mask

        filename = f'{frame_time.strftime("%Y%m%d-%H%M%S%f")}.{frame_index:03d}.mask.png'
        output_path = output_dirpath / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, mask_img, params=[cv2.IMWRITE_PNG_COMPRESSION, 1])

        inpainted_img = cv2.inpaint(image, inpaintMask=mask_img, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        ksize = (51, 51)
        filtered_inpainted_img = np.clip(cv2.GaussianBlur(inpainted_img.astype(np.float32), ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_CONSTANT)
                                         / cv2.GaussianBlur(np.ones(inpainted_img.shape, dtype=np.float32), ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_CONSTANT), 0, 255).astype(np.uint8)
        filtered_inpainted_img[~img_mask] = image[~img_mask]

        if len(filtered_images) == 0:
            print(f'filtered image shape {filtered_inpainted_img.shape}')

        filtered_images[(frame_index, frame_time)] = filtered_inpainted_img

        filename = f'{frame_time.strftime("%Y%m%d-%H%M%S%f")}.{frame_index:03d}.filtered.png'
        output_path = output_dirpath / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(filtered_inpainted_img, cv2.COLOR_RGB2BGR), params=[cv2.IMWRITE_PNG_COMPRESSION, 1])

    # %%

    # Dictionary of optical flow maps between sequential frames
    # TODO: consider generating flow masks based on the consistency between forward and reverse flow vectors
    flow_maps = {}

    for prev_img_indices, next_img_indices in zip(frame_indices[:-1], frame_indices[1:]):

        print('calculating optical flow between frames', *prev_img_indices, *next_img_indices)

        prev_img = filtered_images[prev_img_indices]
        img = filtered_images[next_img_indices]

        prev_gray = img_to_normed_gray(prev_img)
        gray = img_to_normed_gray(img)

        # prev(y, x) ~ next(y + flow(y, x)[1], x + flow(y, x)[0])
        # when cv2.OPTFLOW_FARNEBACK_GAUSSIAN is applied, flow_sigma = (winsize // 2) * 0.3
        # poly_n: size of the pixel neighborhood used to find polynomial expansion in each pixel;
        #         larger values mean that the image will be approximated with smoother surfaces,
        #         yielding more robust algorithm and more blurred motion field, typically poly_n=5 or 7.
        # poly_sigma: standard deviation of the Gaussian that is used to smooth derivatives used as a basis
        #             for the polynomial expansion;
        #             for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
        flow = cv2.calcOpticalFlowFarneback(prev=cv2.resize(prev_gray, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                                            next=cv2.resize(gray, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                                            flow=None,
                                            pyr_scale=0.5, levels=3, winsize=51, iterations=3,
                                            poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        flow = cv2.resize(flow, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST) * 2

        flow_maps[(prev_img_indices, next_img_indices)] = flow

    # %%

    # The blur scores are associated with the next frame, and therefore the first frame has no blur score
    relative_blur_scores = []
    frame_blur_results = []
    for prev_img_indices, next_img_indices in zip(frame_indices[:-1], frame_indices[1:]):

        print('calculating blur score for frame', *next_img_indices)

        prev_gray = img_to_normed_gray(filtered_images[prev_img_indices]).astype(np.float32) / 255
        next_gray = img_to_normed_gray(filtered_images[next_img_indices]).astype(np.float32) / 255

        prev_img_mask = image_masks[prev_img_indices]
        next_img_mask = image_masks[next_img_indices]

        prev_gray[prev_img_mask] = np.nan
        next_gray[next_img_mask] = np.nan

        flow = flow_maps[(prev_img_indices, next_img_indices)]

        h, w = prev_gray.shape
        yfp, xfp = np.mgrid[0:h, 0:w].reshape((2, -1)).astype(int)
        xfn, yfn = flow[yfp, xfp].T + np.vstack([xfp, yfp], dtype=np.float32)

        xyfn_flow = np.full(next_gray.shape + (2,), fill_value=np.nan, dtype=np.float32)
        xyfn_flow[yfp, xfp] = np.vstack([xfn, yfn]).T

        next_gray_warp = cv2.remap(np.array(next_gray), xyfn_flow, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)

        ksize = 7
        prev_gray_grad2 = (np.power(cv2.Sobel(prev_gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize), 2)
                           + np.power(cv2.Sobel(prev_gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize), 2))
        next_gray_warp_grad2 = (np.power(cv2.Sobel(next_gray_warp, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize), 2)
                                + np.power(cv2.Sobel(next_gray_warp, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize), 2))

        prev_gray_grad2[~np.isfinite(next_gray_warp_grad2)] = np.nan
        next_gray_warp_grad2[~np.isfinite(prev_gray_grad2)] = np.nan

        prev_grad2 = np.sqrt(np.nanmean(prev_gray_grad2))
        next_grad2 = np.sqrt(np.nanmean(next_gray_warp_grad2))

        relative_blur_scores.append(prev_grad2 - next_grad2)
        frame_blur_results.append((prev_img_indices, next_img_indices, prev_gray, next_gray, next_gray_warp,
                                   prev_gray_grad2, next_gray_warp_grad2, prev_grad2, next_grad2))

    relative_blur_scores = np.array(relative_blur_scores)

    # %%

    sigma = 8
    relative_blur_scores_lp = (scipy.ndimage.gaussian_filter1d(relative_blur_scores, sigma=sigma, mode='constant', cval=0)
                               / scipy.ndimage.gaussian_filter1d(np.ones_like(relative_blur_scores), sigma=sigma, mode='constant', cval=0))

    blur_scores = np.cumsum(relative_blur_scores - relative_blur_scores_lp)

    for (prev_img_indices, next_img_indices, _, _, _, _, _, _, _), blur_score in zip(frame_blur_results, blur_scores):
        # The blur scores are associated with the next frame, and therefore the first frame has no blur score
        frame_index, frame_time = next_img_indices
        filename = f'{frame_time.strftime("%Y%m%d-%H%M%S%f")}.{frame_index:03d}.flow.pickle'
        output_path = output_dirpath / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as pickle_file:
            pickle.dump({'prev_img_indices': prev_img_indices,
                         'next_img_indices': next_img_indices,
                         'flow': flow_maps[(prev_img_indices, next_img_indices)],
                         'next_img_blur_score': blur_score},
                        pickle_file)

    plt.figure('Frame blur scores', figsize=(16, 10))
    plt.clf()
    ax = plt.subplot(3, 1, 1)
    # The blur scores are associated with the next frame, and therefore the first frame has no blur score
    plt.plot(np.arange(len(relative_blur_scores)) + 1, relative_blur_scores)
    plt.plot(np.arange(len(relative_blur_scores)) + 1, relative_blur_scores_lp)
    plt.subplot(3, 1, 2, sharex=ax)
    plt.plot(np.arange(len(relative_blur_scores)) + 1, blur_scores)
    plt.subplot(3, 1, 3)
    plt.hist(blur_scores, bins=100)
    plt.tight_layout()

    # %%

    sorted_frame_idx = np.argsort(blur_scores)

    if False:
        # Save frame images with filenames formatted to include sort indices and blur scores so that they can be
        # ordered and reviewed to check the relationship between score value and image motion blur
        for sorted_score_idx, frame_idx in enumerate(sorted_frame_idx):
            prev_img_indices, next_img_indices, _, _, _, _, _, _, _ = frame_blur_results[frame_idx]
            output_path = output_dirpath / 'blur_score_ordered_frames' / f'{sorted_score_idx:06d} ({blur_scores[frame_idx]:.3f}).jpg'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, cv2.cvtColor(frame_images[next_img_indices], cv2.COLOR_RGB2BGR))

    fig = plt.figure('Sorted frame blur scores', figsize=(24, 12))
    fig.clf()

    axs = fig.subplots(2, 3, sharex=True, sharey=True)

    fig_state = {'sorted_score_idx': 0, 'set_tight_layout': True}
    def plot_new_fig_state(frame_idx_delta=0):
        sorted_score_idx = (fig_state['sorted_score_idx'] + frame_idx_delta) % len(sorted_frame_idx)
        fig_state['sorted_score_idx'] = sorted_score_idx

        frame_idx = sorted_frame_idx[sorted_score_idx]
        (prev_img_indices, next_img_indices, prev_gray, next_gray, next_gray_warp,
         prev_gray_grad2, next_gray_warp_grad2, prev_grad2, next_grad2) = frame_blur_results[frame_idx]

        for ax in axs.flatten():
            ax.cla()

        title = '\n'.join([f'Sorted frame score index {sorted_score_idx} / {len(sorted_frame_idx)}',
                           'frames {} {}, {} {}'.format(*prev_img_indices, *next_img_indices),
                           f'next frame blur score {blur_scores[frame_idx]:.3f}'])
        fig.suptitle(title)
        axs[0, 0].imshow(frame_images[prev_img_indices])
        axs[0, 0].set_title('prev frame image')
        axs[1, 0].imshow(frame_images[next_img_indices])
        axs[1, 0].set_title('next frame image')
        axs[0, 1].imshow(prev_gray, cmap='gray', vmin=0, vmax=1)
        axs[0, 1].set_title('prev frame normed gray')
        axs[1, 1].imshow(next_gray_warp, cmap='gray', vmin=0, vmax=1)
        axs[1, 1].set_title('next frame warped normed gray')
        vmax = np.nanmax(np.stack([prev_gray_grad2, next_gray_warp_grad2]))
        axs[0, 2].imshow(prev_gray_grad2, vmin=0, vmax=vmax)
        axs[0, 2].set_title(f'prev_grad2 {prev_grad2}')
        axs[1, 2].imshow(next_gray_warp_grad2, vmin=0, vmax=vmax)
        axs[1, 2].set_title(f'next_grad2 {next_grad2}')

        if fig_state['set_tight_layout']:
            fig.tight_layout()
            fig_state['set_tight_layout'] = False

        fig.canvas.draw_idle()

    plot_new_fig_state()

    ax_prev = fig.add_axes([0.1, 0.9, 0.1, 0.075])
    ax_next = fig.add_axes([0.21, 0.9, 0.1, 0.075])
    button_prev = matplotlib.widgets.Button(ax_prev, 'Prev')
    button_prev.on_clicked(lambda event: plot_new_fig_state(frame_idx_delta=-1))
    button_next = matplotlib.widgets.Button(ax_next, 'Next')
    button_next.on_clicked(lambda event: plot_new_fig_state(frame_idx_delta=1))

    ax_prev_50 = fig.add_axes([0.7, 0.9, 0.1, 0.075])
    ax_next_50 = fig.add_axes([0.81, 0.9, 0.1, 0.075])
    button_prev_50 = matplotlib.widgets.Button(ax_prev_50, 'Prev 50')
    button_prev_50.on_clicked(lambda event: plot_new_fig_state(frame_idx_delta=-50))
    button_next_50 = matplotlib.widgets.Button(ax_next_50, 'Next 50')
    button_next_50.on_clicked(lambda event: plot_new_fig_state(frame_idx_delta=50))

    def disconnect_on_ax_clear(widget):
        old_ax_clear_fn = widget.ax.clear
        def new_ax_clear_fn():
            widget.disconnect_events()
            old_ax_clear_fn()
        widget.ax.clear = new_ax_clear_fn
    disconnect_on_ax_clear(button_prev)
    disconnect_on_ax_clear(button_next)
    disconnect_on_ax_clear(button_prev_50)
    disconnect_on_ax_clear(button_next_50)

    # %%

    post_to_pipeline_server((f'{working_subdir} / {output_dirpath.name}', None))
