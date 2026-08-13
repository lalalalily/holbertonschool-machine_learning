@staticmethod
    def load_images(folder_path):
        """
        folder_path: a string representing the path to the folder holding
            all the images to load

        Returns: (images, image_paths)
            images: a list of images as numpy.ndarrays
            image_paths: a list of paths to the individual images in
                images
        """
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')

        image_paths = sorted(
            p for p in glob.glob(os.path.join(folder_path, '*'))
            if os.path.isfile(p) and p.lower().endswith(valid_ext)
        )
        images = [cv2.imread(image_path) for image_path in image_paths]

        return images, image_paths
    