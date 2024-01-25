import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(image1, image2):
    # Load the images as grayscale
    image1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)
    
    # Normalize the images to the range [0, 1]
    image1 = image1.astype(np.float64) / 255.0
    image2 = image2.astype(np.float64) / 255.0
    
    # Calculate the SSIM
    score, _ = ssim(image1, image2, full=True, data_range=1.0)
    return score
# Example usage:
image1 = "2.bmp"  # Load the first image
image2 = "2.bmp"  # Load the second image
#ssim_score = calculate_ssim(image1, image2)
#print("SSIM:", ssim_score)
