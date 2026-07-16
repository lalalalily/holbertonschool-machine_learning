import numpy as np

def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.
    
    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple grayscale images.
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel for the convolution.
        
    Returns:
        numpy.ndarray containing the convolved images.
    """
    # Extract dimensions from the inputs
    m, h, w = images.shape
    kh, kw = kernel.shape
    
    # Calculate the dimensions of the output image for a 'valid' convolution
    output_h = h - kh + 1
    output_w = w - kw + 1
    
    # Initialize an array of zeros to store the convolved images
    convolved_images = np.zeros((m, output_h, output_w))
    
    # Loop over the spatial dimensions of the output image
    for i in range(output_h):
        for j in range(output_w):
            # Slice the current region of interest (patch) across all 'm' images
            # image_patch shape: (m, kh, kw)
            image_patch = images[:, i:i+kh, j:j+kw]
            
            # Multiply the patch by the kernel (broadcasting applies the kernel to each image in the batch)
            # Sum over axis 1 (height) and axis 2 (width) to collapse the kernel dimensions
            convolved_images[:, i, j] = np.sum(image_patch * kernel, axis=(1, 2))
            
    return convolved_images
