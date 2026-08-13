@staticmethod
def load_images(folder_path):
    """
    folder_path: a string representing the path to the folder holding
        all the images to load

    Returns: (images, image_paths)
        images: a list of images as numpy.ndarrays
        image_paths: a list of paths to the individual images in images
    """
    image_paths = sorted(glob.glob(os.path.join(folder_path, '*')))
    images = [cv2.imread(image_path) for image_path in image_paths]

    return images, image_paths
