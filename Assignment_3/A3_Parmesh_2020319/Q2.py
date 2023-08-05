import numpy as np
import cv2
import matplotlib.pyplot as plt

def start(image_i):
    #display the original image
    plt.subplot(1,3,1), plt.imshow(image_i, "gray"), plt.title("Original Image")
    #create a 5x5 box filter
    box_filter = np.ones((5,5),np.float32)/25
    #zero padding the image and the filter so they are of same dimension
    shape_image_i = np.shape(image_i)
    shape_box_filter = np.shape(box_filter)
    padded_image_i = np.zeros((516,516))
    padded_image_i[:shape_image_i[0],:shape_image_i[1]] = image_i
    padded_box_filter = np.zeros((516,516))
    padded_box_filter[:shape_box_filter[0],:shape_box_filter[1]] = box_filter
    #compute 2d DFT of image and filter
    F_image_i = np.fft.fftshift(np.fft.fft2(padded_image_i))
    F_box_filter = np.fft.fftshift(np.fft.fft2(padded_box_filter))
    #apply the filter to the image
    #multiply the DFT of image and filter
    F_image_o = F_image_i * F_box_filter
    #compute the mask
    F_mask = F_image_i - F_image_o
    #compute the sharpened image
    F_image_s = F_image_i + F_mask
    #compute the 2d inverse DFT of the sharpened image and take the real part
    image_s = np.real(np.fft.ifft2(np.fft.ifftshift(F_image_s)))
    #clip the values to be between 0 and 255
    image_s = np.clip(image_s, 0, 255)
    #display the sharpened image
    plt.subplot(1,3,2), plt.imshow(image_s, "gray"), plt.title("Unsharp Masked Image")
    #highboost add k times to mask
    k = 4
    F_image_s = F_image_i + k*F_mask
    #compute the 2d inverse DFT of the sharpened image and take the real part
    image_s = np.real(np.fft.ifft2(np.fft.ifftshift(F_image_s)))
    #clip the values to be between 0 and 255
    image_s = np.clip(image_s, 0, 255)
    #display the sharpened image
    plt.subplot(1,3,3), plt.imshow(image_s, "gray"), plt.title("HighBoost image")
    plt.show()

if __name__ == '__main__':
    # Read image
    img_i = cv2.imread('x5.bmp', 0)
    start(img_i)