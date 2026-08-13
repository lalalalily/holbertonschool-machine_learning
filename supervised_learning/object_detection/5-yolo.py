def preprocess_images(self, images):
        """
        images: a list of images as numpy.ndarrays

        Returns: (pimages, image_shapes)
            pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3)
                containing all of the preprocessed images
            image_shapes: a numpy.ndarray of shape (ni, 2) containing the
                original height and width of the images
        """
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages_list = []
        image_shapes_list = []

        for img in images:
            image_shapes_list.append(img.shape[:2])

            resized = cv2.resize(
                img, (input_w, input_h), interpolation=cv2.INTER_CUBIC
            )
            resized = resized / 255

            pimages_list.append(resized)

        pimages = np.array(pimages_list)
        image_shapes = np.array(image_shapes_list)

        return pimages, image_shapes
