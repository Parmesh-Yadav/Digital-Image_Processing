import numpy as np
import cv2
import matplotlib.pyplot as plt


def magnitude_spectrum(image_i):
    #compute the magnitude spectrum of the image
    fft = np.fft.fft2(image_i)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = 20*np.log(np.abs(fft_shift))
    return magnitude_spectrum

def add_noise_t_pixel(image_i,image_o,m,n,k):
    #add the following noise to the image ∑(i=0 to 511) ∑(j=0 to5)*K*δ(n −100j, m − i)
    for i in range(511):
        for j in range(5):
            image_o[m-100*j][n-i] = image_i[m-100*j][n-i] + k
    

def add_noise(image_i):
    #add the following noise to the image ∑(i=0 to 511) ∑(j=0 to5)*K*δ(n −100j, m − i)
    im_shape = image_i.shape
    #copy the original image to the noisy image
    image_n_20 = image_i.copy()
    image_n_30 = image_i.copy()
    image_n_50 = image_i.copy()
    m, n = im_shape[0] - 1, im_shape[1] - 1
    #add noise to the image
    add_noise_t_pixel(image_i,image_n_20,m,n,20)
    add_noise_t_pixel(image_i,image_n_30,m,n,30)
    add_noise_t_pixel(image_i,image_n_50,m,n,50)

    return image_n_20, image_n_30, image_n_50

def filter_image_to_remove_noise(image_n):
    shape_image_i = np.shape(image_n)
    box_filter = np.ones((7,7),np.float32)/49
    shape_box_filter = np.shape(box_filter)
    padded_image_i = np.zeros((516,516))
    padded_image_i[:shape_image_i[0],:shape_image_i[1]] = image_n
    padded_box_filter = np.zeros((516,516))
    padded_box_filter[:shape_box_filter[0],:shape_box_filter[1]] = box_filter
    #filter the noisy image to remove the noise
    F_image_i = np.fft.fftshift(np.fft.fft2(padded_image_i))
    #apply the filter
    #convert the filter to frequency domain
    F_box_filter = np.fft.fftshift(np.fft.fft2(padded_box_filter))
    plt.imshow(magnitude_spectrum(F_box_filter), cmap='gray')
    plt.show()
    #multiply the filter with the noisy image
    F_image_o = F_image_i * F_box_filter
    #convert the filtered image to spatial domain
    image_o = np.real(np.fft.ifft2(np.fft.ifftshift(F_image_o)))
    #clip the values to 0-255
    image_o = np.clip(image_o,0,255)
    #ceop the image to the original size
    image_o = image_o[:shape_image_i[0],:shape_image_i[1]]
    return image_o


def start(image_i):
    #display the original image
    plt.subplot(1,2,1), plt.imshow(image_i, "gray"), plt.title("Original Image")
    mag_im_i = magnitude_spectrum(image_i)
    plt.subplot(1,2,2), plt.imshow(mag_im_i, cmap = 'gray'), plt.title('Magnitude Spectrum of Image')
    plt.show()
    #add noise to the image
    image_n_20, image_n_30, image_n_50 = add_noise(image_i)
    #display the noisy image
    plt.subplot(2,3,1), plt.imshow(image_n_20, "gray"), plt.title("Noisy Image with k = 20")
    plt.subplot(2,3,2), plt.imshow(image_n_30, "gray"), plt.title("Noisy Image with k = 30")
    plt.subplot(2,3,3), plt.imshow(image_n_50, "gray"), plt.title("Noisy Image with k = 50")
    mag_im_n_20 = magnitude_spectrum(image_n_20)
    mag_im_n_30 = magnitude_spectrum(image_n_30)
    mag_im_n_50 = magnitude_spectrum(image_n_50)
    plt.subplot(2,3,4), plt.imshow(mag_im_n_20, cmap = 'gray'), plt.title('Magnitude Spectrum of Noisy Image with k = 20')
    plt.subplot(2,3,5), plt.imshow(mag_im_n_30, cmap = 'gray'), plt.title('Magnitude Spectrum of Noisy Image with k = 30')
    plt.subplot(2,3,6), plt.imshow(mag_im_n_50, cmap = 'gray'), plt.title('Magnitude Spectrum of Noisy Image with k = 50')
    plt.show()
    #filter the noisy image
    image_filtered_20 = filter_image_to_remove_noise(image_n_20)
    image_filtered_30 = filter_image_to_remove_noise(image_n_30)
    image_filtered_50 = filter_image_to_remove_noise(image_n_50)
    #display the filtered image
    plt.subplot(2,3,1), plt.imshow(image_filtered_20, "gray"), plt.title("Filtered Image with k = 20")
    plt.subplot(2,3,2), plt.imshow(image_filtered_30, "gray"), plt.title("Filtered Image with k = 30")
    plt.subplot(2,3,3), plt.imshow(image_filtered_50, "gray"), plt.title("Filtered Image with k = 50")
    mag_im_f_20 = magnitude_spectrum(image_filtered_20)
    mag_im_f_30 = magnitude_spectrum(image_filtered_30)
    mag_im_f_50 = magnitude_spectrum(image_filtered_50)
    plt.subplot(2,3,4), plt.imshow(mag_im_f_20, cmap = 'gray'), plt.title('Magnitude Spectrum of Filtered Image with k = 20')
    plt.subplot(2,3,5), plt.imshow(mag_im_f_30, cmap = 'gray'), plt.title('Magnitude Spectrum of Filtered Image with k = 30')
    plt.subplot(2,3,6), plt.imshow(mag_im_f_50, cmap = 'gray'), plt.title('Magnitude Spectrum of Filtered Image with k = 50')
    plt.show()

if __name__ == '__main__':
    # Read image
    img_i = cv2.imread('cameraman.jpg', 0)
    start(img_i)