import cv2 as cv
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

def z_c(log_image):
    zc_image = np.zeros(log_image.shape)
    for m in range(1,log_image.shape[0]-1):
        for n in range(1, log_image.shape[1]-1):
            neg,pos =0,0
            nh = [log_image[m+1, n-1],log_image[m+1, n],log_image[m+1, n+1],log_image[m, n-1],log_image[m, n+1],log_image[m-1, n-1],log_image[m-1, n],log_image[m-1, n+1]]
            mx = max(nh)
            mn = min(nh)
            for neh in nh:
                if neh > 0:
                    pos += 1
                elif neh < 0:
                    neg += 1
            out = ((neg>0) and (pos>0))
            if out:
                if log_image[m,n] > 0:
                    zc_image[m,n] = log_image[m,n] + np.abs(mn)
                elif log_image[m,n] < 0:
                    zc_image[m,n] = np.abs(log_image[m,n]) + mx
    # ret, thresh1 = cv.threshold(zc_image, 4, 255, cv.THRESH_BINARY)
    return zc_image
    # return thresh1

def log(i_image,c_image):#Question 2A
    g_image = sc.ndimage.gaussian_filter(i_image, sigma=2)
    log_image = sc.ndimage.laplace(g_image)
    zero_crossed = z_c(log_image)
    # zero_crossed = log_image/log_image.max()
    plt.subplot(2,2,1), plt.imshow(c_image, cmap='gray'), plt.title('Clean Image')
    plt.subplot(2,2,2), plt.imshow(i_image, cmap='gray'), plt.title('Noisy Image')
    plt.subplot(2,2,3), plt.imshow(log_image, cmap='gray'), plt.title('LOG before zero crossing')
    plt.subplot(2,2,4), plt.imshow(zero_crossed, cmap='gray'), plt.title('LOG after zero crossing')
    plt.show()
    #return log image
    return log_image

def log_using_1d(i_image,c_image):#Question 2B
    plt.subplot(2,3,1), plt.imshow(c_image, cmap='gray'), plt.title('Clean Image')
    plt.subplot(2,3,2), plt.imshow(i_image, cmap='gray'), plt.title('Noisy Image')
    g_image = sc.ndimage.gaussian_filter1d(i_image, sigma=2,axis=1)
    g_image = sc.ndimage.gaussian_filter1d(g_image, sigma=2,axis=0)
    plt.subplot(2,3,3), plt.imshow(g_image, cmap='gray'), plt.title('Gaussian Filtered Image')
    l = np.array([[1,-2,1]])
    l_image_h = sc.signal.convolve2d(g_image,l,mode='same')
    l_image_h = np.clip(l_image_h,0,255)
    plt.subplot(2,3,4), plt.imshow(l_image_h, cmap='gray'), plt.title('Horizontal response only')
    l_image_v = sc.signal.convolve2d(g_image,l.T,mode='same')
    l_image_v = np.clip(l_image_v,0,255)
    plt.subplot(2,3,5), plt.imshow(l_image_v, cmap='gray'), plt.title('Vertical response only')
    plt.subplot(2,3,6), plt.imshow(l_image_h+l_image_v, cmap='gray'), plt.title('Laplacian Filtered Image(LOG)')
    plt.show()
    #return log image
    return l_image_h+l_image_v


def start(i_image,c_image):
    q2a_log_image = log(i_image,c_image)#Question 2A
    q2b_log_image = log_using_1d(i_image,c_image)#Question 2B
    #compute difference
    diff = q2a_log_image - q2b_log_image
    #take sum of absolute difference
    sum_abs_diff = np.sum(np.abs(diff))
    print("Sum of absolute difference between the two LOG images is: ",sum_abs_diff)
    #convert to [0,255]
    #subtract the min
    diff = diff - np.min(diff)
    #divide by max
    diff = diff/np.max(diff)
    #multiply by 255
    diff = diff*255
    print("Max value of difference image is: ",np.max(diff))

if __name__ == '__main__':
    cimg = cv.imread('x5.bmp',0)
    img = cv.imread('noisy.jpg',0)
    start(img,cimg)