#!/usr/bin/env python3
"""
Contains the Yolo class definition for YOLOv3 object detection.
"""
import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    Uses the YOLO v3 algorithm to perform object detection.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor for Yolo.

        Parameters:
            model_path (str): Path to where a Darknet Keras model is stored.
            classes_path (str): Path to where the list of class names used for
                                the Darknet model can be found.
            class_t (float): Box score threshold for initial filtering step.
            nms_t (float): IOU threshold for non-max suppression.
            anchors (numpy.ndarray): Array of shape (outputs, anchor_boxes, 2)
                                     containing all anchor boxes.
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes predictions from the Darknet model for a single image.

        Parameters:
            outputs (list of numpy.ndarray): Model output predictions.
                Each array shape: (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
            image_size (numpy.ndarray):Original image size[image_height, image_width]

        Returns:
            tuple of (boxes, box_confidences, box_class_probs):
                boxes: list of numpy.ndarrays of shape (grid_height, grid_width,
                       anchor_boxes, 4) containing boundary boxes (x1, y1)
                       relative to original image.
                box_confidences: list of numpy.ndarrays containing box confidences.
                box_class_probs: list of numpy.ndarrays containing class.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size[0], image_size[1]
        input_width = int(self.model.input.shape[1])
        input_height = int(self.model.input.shape[2])

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Predicted bounding box offsets
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            # Sigmoid activation on center offsets
            sig_tx = 1 / (1 + np.exp(-t_x))
            sig_ty = 1 / (1 + np.exp(-t_y))

            # Grid coordinates
            cx = np.tile(np.arange(0, grid_width), (grid_height, 1))
            cx = np.tile(cx[:, :, np.newaxis], (1, 1, anchor_boxes))

            cy = np.tile(np.arange(0, grid_height), (grid_width, 1)).T
            cy = np.tile(cy[:, :, np.newaxis], (1, 1, anchor_boxes))

            # Anchor dimensions
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            # Center coordinates & dimensions normalized relative to grid/model
            bx = (sig_tx + cx) / grid_width
            by = (sig_ty + cy) / grid_height

            bw = (pw * np.exp(t_w)) / input_width
            bh = (ph * np.exp(t_h)) / input_height

            # Corner coordinates scaled to original image dimensions
            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            box = np.zeros(output[..., :4].shape)
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2
            boxes.append(box)

            # Box confidence
            confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(confidence)

            # Box class probabilities
            class_prob = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(class_prob)

        return boxes, box_confidences, box_class_probs
