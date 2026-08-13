#!/usr/bin/env python3
"""Module that defines the NST class for neural style transfer"""
import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for neural style transfer"""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1',
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor

        parameters:
            style_image [numpy.ndarray of shape (h, w, 3)]:
                the image used as a style reference
            content_image [numpy.ndarray of shape (h, w, 3)]:
                the image used as a content reference
            alpha [number]: the weight for content cost
            beta [number]: the weight for style cost
        """
        style_valid = (
            isinstance(style_image, np.ndarray) and
            len(style_image.shape) == 3 and
            style_image.shape[2] == 3
        )
        if not style_valid:
            raise TypeError(
                "style_image must be a numpy.ndarray with
        