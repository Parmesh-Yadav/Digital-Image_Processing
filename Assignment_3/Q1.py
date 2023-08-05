import numpy as np
import scipy as sp
import cv2
import matplotlib.pyplot as plt

def start(image_i):
    #display the original image
    plt.subplot(1,4,1), plt.imshow(image_i, "gray"), plt.title("Original Image")
    #create a 5x5 box filter
    box_filter = np.ones((5,5),np.float32)/25
    #apply the filter to the image
    image_o = sp.signal.convolve2d(image_i, box_filter, mode='same')
    #display the filtered image
    plt.subplot(1,4,2), plt.imshow(image_o, "gray"), plt.title("Blurred Image")
    #compute the mask
    mask = image_i - image_o
    #display the mask
    # plt.subplot(1,5,3), plt.imshow(mask, "gray"), plt.title("Mask")
    #compute the sharpened image
    image_s = image_i + mask
    #clip the values to be between 0 and 255
    image_s = np.clip(image_s, 0, 255)
    #display the sharpened image
    plt.subplot(1,4,3), plt.imshow(image_s, "gray"), plt.title("Unsharp Masked Image")
    #highboost add k times to mask
    k = 2
    image_s = image_i + k*mask
    #clip the values to be between 0 and 255
    image_s = np.clip(image_s, 0, 255)
    #display the sharpened image
    plt.subplot(1,4,4), plt.imshow(image_s, "gray"), plt.title("HighBoost image")
    plt.show()

if __name__ == '__main__':
    # Read image
    img_i = cv2.imread('x5.bmp', 0)
    start(img_i)