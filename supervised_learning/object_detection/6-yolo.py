def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        image: a numpy.ndarray containing an unprocessed image
        boxes: a numpy.ndarray containing the boundary boxes for the image
        box_classes: a numpy.ndarray containing the class indices for each
            box
        box_scores: a numpy.ndarray containing the box scores for each box
        file_name: the file path where the original image is stored

        Displays the image with all boundary boxes, class names, and box
        scores
        """
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            class_name = self.class_names[box_classes[i]]
            score = str(round(box_scores[i], 2))
            text = "{} {}".format(class_name, score)

            cv2.putText(
                image,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')
            cv2.imwrite(os.path.join('detections', file_name), image)

        cv2.destroyAllWindows()
