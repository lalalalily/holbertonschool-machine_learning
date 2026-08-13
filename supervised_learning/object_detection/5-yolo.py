#!/usr/bin/env python3
"""Yolo class - preprocess images"""
import numpy as np
import tensorflow.keras as K
import glob
import cv2
import os


class Yolo:
    """Uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class constructor"""
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """Process Darknet outputs"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            box = output[..., :4]
            box_confidence = self.sigmoid(output[..., 4:5])
            box_class_prob = self.sigmoid(output[..., 5:])

            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

            tx = box[..., 0]
            ty = box[..., 1]
            tw = box[..., 2]
            th = box[..., 3]
