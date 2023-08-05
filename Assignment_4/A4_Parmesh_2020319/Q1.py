import cv2 as cv
import numpy as np
import scipy as sp

#add gaussian noise with mean=0 and variance of order of 100
def add_awgn(i_image):
    row,col= i_image.shape
    m = 0
    v = 100 #variance of order of 100
    s = v**0.5
    g = np.random.normal(m,s,(row,col))
    g = g.reshape(row,col)
    noisy = i_image + g
    cv.imwrite('noisy.jpg',noisy)

def weiner_filter(): #apply weiner filter on the noisy image
    noisy = cv.imread('noisy.jpg',0)
    weiner = sp.signal.wiener(noisy, mysize=None, noise=None)
    cv.imshow('noisy',noisy)
    cv.imshow('weiner',weiner)
    cv.waitKey(0)

def start(i_image):
    add_awgn(i_image)
    # weiner_filter()
    # return 0

if __name__ == '__main__':
    img = cv.imread('x5.bmp',0)
    # print(img)
    start(img)