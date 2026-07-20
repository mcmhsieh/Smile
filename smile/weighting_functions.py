"""
Weighting functions used by the processing pipeline for
https://github.com/mcmhsieh/Smile

SPDX-FileCopyrightText: 2026 Mark Hsieh
SPDX-License-Identifier: MIT
"""

import numpy as np
import torch

def cauchy(x, sigma):
    # Heavy tailed bell shaped function based on Student's t-distribution with v = 1 (i.e. Cauchy distribution)
    # y=1/(1+x**2) has a knee point at x ~ 3.0 beyond which it slowly converges to 0, e.g. [1.0, 0.5], [3.0, 0.1], [9.9, 0.01]
    return 1 / (1 + np.power(x / (sigma / 3.0), 2))

def gamma_softplus(x, threshold, alpha, relative_outer_gradient):
    # S-shaped curve based on the Gamma cumulative distribution and Softplus functions.
    # The threshold parameter controls scaling along x and is located approximately at the outer/upper knee point.
    # The alpha shape parameter controls the polynomial order near zero and the location of the inner/lower knee point.
    # The relative_outer_gradient parameter controls the far outer asymptotic gradient as an approximate proportion
    # of the peak gradient around the mid-point of the S-shaped curve.
    x_normed = torch.abs(x) * (10 / threshold)
    outer_gradient_normed = relative_outer_gradient / 10 * torch.sqrt(alpha)
    return (torch.special.gammainc(alpha, x_normed * (alpha / 6))
            + outer_gradient_normed * torch.nn.Softplus()(x_normed - (6 + alpha)))
