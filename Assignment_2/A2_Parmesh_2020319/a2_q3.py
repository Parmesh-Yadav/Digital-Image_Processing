import math
import matplotlib.pyplot as plt
import numpy as np
import cv2

def display_image_graph(title,data):
    plt.figure()
    plt.bar(range(256), data)
    plt.title(title)
    plt.show()

def log_transform():
    image = cv2.imread('x5.bmp')
    c = 255 / math.log(1 + np.max(image))#c = 255/log(1+max)
    l_t_image = c * np.log(1 + image)#log transform
    l_t_image = np.array(l_t_image, dtype = np.uint8)
    cv2.imwrite('log_transformed.bmp', l_t_image)

def normalized_histogram_with_cdf(image):
    h = [0 for i in range(256)] #array of 256 zeros
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            h[image[i][j]] += 1
    h = np.array(h)
    normalized_hist = h/h.sum()
    #CDF
    cdf = np.zeros((256,))
    for i in range(256):
        cdf[i] = normalized_hist[i] + cdf[i-1]
    return normalized_hist, cdf

def histogram_matching(H_r,G_s):
    argsMin = np.zeros((256,))
    for i in range(256):
        argsMin[i] = np.argmin(np.abs(H_r[i] - G_s))
    return argsMin

def histogram_matching_image(Orig_image,argsMin):
    histo_matched_image = np.zeros(Orig_image.shape)
    for i in range(Orig_image.shape[0]):
        for j in range(Orig_image.shape[1]):
            histo_matched_image[i][j] = argsMin[Orig_image[i][j]]
    histo_matched_image = np.array(histo_matched_image, dtype = np.uint8)
    cv2.imwrite('histogram_matched_image.bmp', histo_matched_image)


if __name__ == "__main__":
    log_transform()
    norm_his_orig , cdf_orig = normalized_histogram_with_cdf(cv2.imread('x5.bmp',0))
    norm_his_logT , cdf_logT = normalized_histogram_with_cdf(cv2.imread('log_transformed.bmp',0))
    display_image_graph('Original Image Normalized Histogram',norm_his_orig)
    display_image_graph('Log Transformed Image Normalized Histogram',norm_his_logT)
    display_image_graph('Original Image CDF',cdf_orig)
    display_image_graph('Log Transformed Image CDF',cdf_logT)
    argsMin = histogram_matching(cdf_orig,cdf_logT)
    histogram_matching_image(cv2.imread('x5.bmp',0),argsMin)
