"""
Image filtering functions used by the processing pipeline for
https://github.com/mcmhsieh/Smile

SPDX-FileCopyrightText: 2026 Mark Hsieh
SPDX-License-Identifier: MIT
"""

import numpy as np
import scipy
import cv2


def rgb_to_gray(img, dtype=np.uint8):
    # TODO: Determine and apply a RGB weighting vector that provides the most contrast between neighbouring image regions
    gray = np.mean(img, axis=-1)
    if np.issubdtype(dtype, np.integer):
        gray = np.round(gray)
    return gray.astype(dtype)

def img_to_normed_gray(img):
    gray = rgb_to_gray(img, dtype=np.float32)
    ksize = (7, 7)
    gray = (cv2.GaussianBlur(gray, ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_CONSTANT)
            / cv2.GaussianBlur(np.ones(gray.shape, dtype=np.float32), ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_CONSTANT))

    ksize = (51, 51)
    ones_lp = cv2.GaussianBlur(np.ones(gray.shape, dtype=np.float32), ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_CONSTANT)
    gray_lp = cv2.GaussianBlur(gray, ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_CONSTANT) / ones_lp
    gray2_lp = cv2.GaussianBlur(np.power(gray, 2), ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_CONSTANT) / ones_lp
    gray_var = np.clip(gray2_lp - np.power(gray_lp, 2), 10.0 ** 2, np.inf)
    gray = (gray - gray_lp) / (np.sqrt(gray_var) * 2.5)
    gray = np.clip(scipy.special.expit(gray * 6) * 256, 0, 255).astype(np.uint8)
    return gray

def nan_gaussian_filter(arr, ksize=(15, 15), unfiltered_point_value=None):
    ones_lp = np.ones(arr.shape, dtype=np.float32)
    ones_lp[~np.isfinite(arr)] = 0
    ones_lp = cv2.GaussianBlur(ones_lp, ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_CONSTANT)
    arr_lp = np.array(arr, dtype=np.float32)
    arr_lp[~np.isfinite(arr)] = 0
    arr_lp = cv2.GaussianBlur(arr_lp, ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_CONSTANT) / np.clip(ones_lp, 1e-8, np.inf)
    if unfiltered_point_value is not None:
        arr_lp[ones_lp < 1e-8] = unfiltered_point_value
    return arr_lp

def normalise_img_intensities(img, disparity_map_zoom):
    # Normalise image intensity and contrast.
    # SGBM measures pixel dissimilarity using absolute intensity values, with only a small local range of spatial search minimisation.
    # Depth Discontinuities by Pixel-to-Pixel Stereo https://cecas.clemson.edu/~stb/publications/p2p_iccv1998.pdf
    ksize = {0.0625: (13, 13), 0.125: (25, 25), 0.25: (51, 51), 0.5: (101, 101)}[disparity_map_zoom]
    ones_lp = cv2.GaussianBlur(np.ones(img.shape, dtype=np.float32), ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_CONSTANT)
    img_lp = cv2.GaussianBlur(img.astype(np.float32), ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_CONSTANT) / ones_lp
    img2_lp = cv2.GaussianBlur(np.power(img.astype(np.float32), 2), ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_CONSTANT) / ones_lp
    img_var = np.clip(np.mean(img2_lp - np.power(img_lp, 2), axis=2), 16.0 ** 2, np.inf)
    img = (img.astype(np.float32) - img_lp) / (np.sqrt(img_var) * 6.0)[:, :, None]
    img = np.clip(scipy.special.expit(img * 6) * 256, 0, 255).astype(np.uint8)
    return img

def calc_image_point_weights(filtered_img):
    gray_down = cv2.resize(img_to_normed_gray(filtered_img), dsize=None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    eig_vals_and_vecs = cv2.cornerEigenValsAndVecs(gray_down, blockSize=9, ksize=5, borderType=cv2.BORDER_REPLICATE)

    sort_idxs = np.argsort(eig_vals_and_vecs[:, :, :2], axis=-1)
    eig_vals = np.take_along_axis(eig_vals_and_vecs, sort_idxs, axis=-1)
    primary_eig_vals = eig_vals[:, :, 1]
    secondary_eig_vals = eig_vals[:, :, 0]
    assert np.all(primary_eig_vals >= secondary_eig_vals)

    primary_eig_vecs = np.concatenate([np.take_along_axis(eig_vals_and_vecs[:, :, [2, 4]], sort_idxs[:, :, 1:], axis=-1),
                                       np.take_along_axis(eig_vals_and_vecs[:, :, [3, 5]], sort_idxs[:, :, 1:], axis=-1)], axis=-1)
    secondary_eig_vecs = np.concatenate([np.take_along_axis(eig_vals_and_vecs[:, :, [2, 4]], sort_idxs[:, :, :1], axis=-1),
                                         np.take_along_axis(eig_vals_and_vecs[:, :, [3, 5]], sort_idxs[:, :, :1], axis=-1)], axis=-1)
    assert np.allclose(np.sum(primary_eig_vecs * secondary_eig_vecs, axis=-1), 0, atol=1e-6)

    h, w = filtered_img.shape[:2]

    primary_eig_vecs = cv2.resize(primary_eig_vecs, dsize=(w, h), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
    secondary_eig_vecs = np.stack([primary_eig_vecs[:, :, 1], -primary_eig_vecs[:, :, 0]], axis=-1)
    assert np.allclose(np.sum(primary_eig_vecs * secondary_eig_vecs, axis=-1), 0, atol=1e-6)

    if True:
        primary_eig_weights = scipy.special.expit((np.sqrt(primary_eig_vals) - 0.4) * 4.5)
        secondary_eig_weights = scipy.special.expit((np.sqrt(secondary_eig_vals) - 0.3) * 6.0)
    else:
        primary_eig_weights = scipy.special.expit((np.sqrt(primary_eig_vals) - 0.4) * 3.0)
        secondary_eig_weights = scipy.special.expit((np.sqrt(secondary_eig_vals) - 0.3) * 4.0)

    primary_eig_weights = cv2.resize(primary_eig_weights, dsize=(w, h), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
    secondary_eig_weights = cv2.resize(secondary_eig_weights, dsize=(w, h), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)

    ksize = (17, 17)
    primary_eig_weights = (cv2.GaussianBlur(primary_eig_weights, ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_CONSTANT)
                           / cv2.GaussianBlur(np.ones_like(primary_eig_weights), ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_CONSTANT))
    secondary_eig_weights = (cv2.GaussianBlur(secondary_eig_weights, ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_CONSTANT)
                             / cv2.GaussianBlur(np.ones_like(secondary_eig_weights), ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_CONSTANT))

    #eig_vals_magn = np.linalg.norm(eig_vals, axis=-1)

    """
    margin = 5
    eig_vals_magn[:margin, :] = np.nan
    eig_vals_magn[-margin:, :] = np.nan
    eig_vals_magn[:, :margin] = np.nan
    eig_vals_magn[:, -margin:] = np.nan
    """

    """
    eig_vals_balance = eig_vals[:, :, 0] / eig_vals[:, :, 1]

    duniformity_dx = np.abs(np.sum(primary_eig_vecs[:, :-1, :] * primary_eig_vecs[:, 1:, :], axis=-1))
    duniformity_dx = (np.pad(duniformity_dx, pad_width=((0, 0), (1, 0)), mode='constant', constant_values=0)
                      + np.pad(duniformity_dx, pad_width=((0, 0), (0, 1)), mode='constant', constant_values=0))
    duniformity_dx[:, 1:-1] /= 2

    duniformity_dy = np.abs(np.sum(primary_eig_vecs[:-1, :, :] * primary_eig_vecs[1:, :, :], axis=-1))
    duniformity_dy = (np.pad(duniformity_dy, pad_width=((1, 0), (0, 0)), mode='constant', constant_values=0)
                      + np.pad(duniformity_dy, pad_width=((0, 1), (0, 0)), mode='constant', constant_values=0))
    duniformity_dy[1:-1, :] /= 2

    eig_vec_curvature = np.sqrt(1 - np.power(duniformity_dx * duniformity_dy, 2))

    image_eigv_weights_down = np.sqrt(eig_vals_magn * eig_vals_balance)

    image_eigv_weights = cv2.resize(image_eigv_weights_down, dsize=(w, h), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
    #image_eigv_weights[frame_img_masks[frame_idx]] *= 0.2
    image_eigv_weights = cv2.GaussianBlur(image_eigv_weights, (17, 17), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_REFLECT101)
    frame_image_eigv_weights.append(image_eigv_weights)
    """

    return primary_eig_vals, secondary_eig_vals, primary_eig_weights, secondary_eig_weights, primary_eig_vecs, secondary_eig_vecs
