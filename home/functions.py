# Import lib
import cv2
import matplotlib.pyplot as plt
import numpy as np

#------------------------------------------------------
# Read image

def read_img(img_path):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

#------------------------------------------------------
# Make noise

# Gausian Noise
def gaussian_noise(img):
    mean = 0
    sigma = 0.7
    gaussian_noise = np.random.normal(mean, sigma, img.shape).astype('uint8')
    gaussian_noise_img = cv2.add(img, gaussian_noise)
    return gaussian_noise_img

# Salt-and-pepper Noise
def snp_noise(img):
    # Make noise probability
    salt_prob = 0.01
    pepper_prob = 0.01
    snp_noise_img = np.copy(img)
    
    # Add salt noise (white pixel)
    num_salt = np.ceil(salt_prob * img.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    snp_noise_img[coords[0], coords[1]] = 255

    # Add pepper noise (black pixel)
    num_pepper = np.ceil(pepper_prob * img.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    snp_noise_img[coords[0], coords[1]] = 0
    
    return snp_noise_img

#--------------------------------------------------------------
# Denoising/Smoothing

# Mean blur
def mean_blur(img):
    filter_size = (5, 5)
    return cv2.blur(img, filter_size)

# Median blur
def median_blur(img):
    return cv2.medianBlur(img, 5)

#---------------------------------------------------------------
# Sharpening

# Laplacian filter
def laplacian_sharp(img):
    laplacian_gaussian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = img.astype(np.float64) - laplacian_gaussian
    laplacian = cv2.convertScaleAbs(laplacian)
    return laplacian

# High pass filter
def high_pass_sharp(img):
    blurred = cv2.GaussianBlur(img, (21, 21), 0)
    high_pass = cv2.subtract(img, blurred)
    high_pass = cv2.add(img, high_pass)
    return high_pass

# Kernel-based sharpening
def kernel_based_sharpening(img):
    # create kernel
    kernel = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])

    sharp_kernel = cv2.filter2D(img, -1, kernel)
    return sharp_kernel

#---------------------------------------------------------------------
# Function extract Edge features

# Sobel
def sobel_features(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Đạo hàm theo X
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Đạo hàm theo Y
    
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    
    return sobel_combined

# Prewitt Filter
def prewitt_extract(img):
    prewitt_x = np.array([[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]])

    prewitt_y = np.array([[1, 1, 1],
                        [0, 0, 0],
                        [-1, -1, -1]])

    prewitt_x_img = cv2.filter2D(img, -1, prewitt_x)
    prewitt_y_img = cv2.filter2D(img, -1, prewitt_y)
    
    prewitt_x_img = prewitt_x_img.astype(np.float32)
    prewitt_y_img = prewitt_y_img.astype(np.float32)
    
    prewitt_combined = cv2.magnitude(prewitt_x_img, prewitt_y_img)
    
    return prewitt_combined

# Canny
def canny_extract(img):
    return cv2.Canny(img, 125, 125)