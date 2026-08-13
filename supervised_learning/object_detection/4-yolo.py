#!/usr/bin/env python3
"""Defines the Yolo class for performing object detection using YOLOv3"""
import numpy as np
import tensorflow.keras as K
import glob
import cv2
import os


class Yolo:
    """Uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        model_path: path to Darknet Keras model
        classes_path: path to list of class names used for the Darknet model
