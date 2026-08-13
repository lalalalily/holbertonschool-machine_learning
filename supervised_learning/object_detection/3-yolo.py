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
            image_size (numpy.ndarray): Original image size [image_height, image_width].

        Returns:
            tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size[0], image_size[1]
        input_width = int(self.model.input.shape[1])
        input_height = int(self.model.input.shape[2])

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            sig_tx = 1 / (1 + np.exp(-t_x))
            sig_ty = 1 / (1 + np.exp(-t_y))

            cx = np.tile(np.arange(0, grid_width), (grid_height, 1))
            cx = np.tile(cx[:, :, np.newaxis], (1, 1, anchor_boxes))

            cy = np.tile(np.arange(0, grid_height), (grid_width, 1)).T
            cy = np.tile(cy[:, :, np.newaxis], (1, 1, anchor_boxes))

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            bx = (sig_tx + cx) / grid_width
            by = (sig_ty + cy) / grid_height

            bw = (pw * np.exp(t_w)) / input_width
            bh = (ph * np.exp(t_h)) / input_height

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

            confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(confidence)

            class_prob = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters bounding boxes based on box scores (confidence * class_probability).

        Parameters:
            boxes (list of numpy.ndarray): Bounding boxes for each output.
            box_confidences (list of numpy.ndarray): Box confidences for each output.
            box_class_probs (list of numpy.ndarray): Class probabilities for each output.

        Returns:
            tuple of (filtered_boxes, box_classes, box_scores)
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]

            b_classes = np.argmax(scores, axis=-1)
            b_class_scores = np.max(scores, axis=-1)

            mask = b_class_scores >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(b_classes[mask])
            box_scores.append(b_class_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies Non-Max Suppression (NMS) to the filtered bounding boxes.

        Parameters:
            filtered_boxes (numpy.ndarray): Shape (?, 4) containing filtered boxes.
            box_classes (numpy.ndarray): Shape (?,) containing predicted class numbers.
            box_scores (numpy.ndarray): Shape (?,) containing box scores.

        Returns:
            tuple of (box_predictions, predicted_box_classes, predicted_box_scores):
                box_predictions (numpy.ndarray): Bounding boxes ordered by class and score.
                predicted_box_classes (numpy.ndarray): Class numbers for box predictions.
                predicted_box_scores (numpy.ndarray): Box scores for box predictions.
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            cls_mask = box_classes == cls
            cls_boxes = filtered_boxes[cls_mask]
            cls_scores = box_scores[cls_mask]

            x1 = cls_boxes[:, 0]
            y1 = cls_boxes[:, 1]
            x2 = cls_boxes[:, 2]
            y2 = cls_boxes[:, 3]

            areas = (x2 - x1) * (y2 - y1)
            order = cls_scores.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)

                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h

                iou = inter / (areas[i] + areas[order[1:]] - inter)

                inds = np.where(iou <= self.nms_t)[0]
                order = order[inds + 1]

            box_predictions.append(cls_boxes[keep])
            predicted_box_classes.append(np.full(len(keep), cls))
            predicted_box_scores.append(cls_scores[keep])

        if box_predictions:
            box_predictions = np.concatenate(box_predictions, axis=0)
            predicted_box_classes = np.concatenate(
                predicted_box_classes, axis=0
            )
            predicted_box_scores = np.concatenate(
                predicted_box_scores, axis=0
            )
        else:
            box_predictions = np.empty((0, 4))
            predicted_box_classes = np.empty((0,))
            predicted_box_scores = np.empty((0,))

        return box_predictions, predicted_box_classes, predicted_box_scores
