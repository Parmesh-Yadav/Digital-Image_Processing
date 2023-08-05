import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def convert_to_lab(i_image):
    lab = cv.cvtColor(i_image, cv.COLOR_BGR2Lab)
    L, A, B = cv.split(lab)
    plt.subplot(1, 3, 1), plt.imshow(L, cmap='gray'), plt.title('L')
    plt.subplot(1, 3, 2), plt.imshow(A, cmap='gray'), plt.title('A')
    plt.subplot(1, 3, 3), plt.imshow(B, cmap='gray'), plt.title('B')
    plt.show()
    return L


def class_mean(f, l, p, h, max):
    mean = 0
    for j in range(f, l):
        mean = mean + (h[j] * (j/max))
    mean = mean/p
    return mean


def class_variance(f, l, p, h, max, m):
    var = 0
    for j in range(f, l):
        var = var + (h[j] * ((j/max)-m)**2)
    var = var/p
    return var


def otsu_thresholding(intensity):
    intensity_copy = intensity.copy()
    max = np.max(intensity)
    print("Max intensity: ", max)
    m, n = intensity.shape
    h, bins = np.histogram(intensity.flatten(), range(255))
    h = h*1./h.sum()
    within_class_variance = np.zeros(2)
    t1 = 0.25
    t2 = 0.5
    z = 0
    for i in [max*t1, max*t2]:
        print("Threshold: ", i)
        # class probability
        p1 = h[:int(i)].sum()
        p2 = h[int(i):].sum()
        print("Class probability 1: ", p1)
        print("Class probability 2: ", p2)
        # class means
        m1 = class_mean(0, int(i), p1, h, max)
        m2 = class_mean(int(i), len(h), p2, h, max)
        print("Class mean 1: ", m1)
        print("Class mean 2: ", m2)
        # class variance
        v1 = class_variance(0, int(i), p1, h, max, m1)
        v2 = class_variance(int(i), len(h), p2, h, max, m2)
        print("Class variance 1: ", v1)
        print("Class variance 2: ", v2)
        # wihin class variance
        wcv = p1*v1 + p2*v2
        print("Within class variance: ", wcv)
        within_class_variance[z] = wcv
        z = z+1
    print("Within class variance: ", within_class_variance)
    thresh_arg = np.argmin(within_class_variance)
    if thresh_arg == 0:
        thresh = max*t1
    else:
        thresh = max*t2
    print("Threshold Calculated: ", thresh)
    # thresholding the image binary
    _, bin = cv.threshold(intensity_copy, thresh, 255, cv.THRESH_BINARY)
    plt.subplot(1, 2, 1), plt.imshow(intensity_copy,
                                     cmap='gray'), plt.title('Original')
    plt.subplot(1, 2, 2), plt.imshow(
        bin, cmap='gray'), plt.title('Otsu Thresholding')
    plt.show()


def start(i_image):
    # print(i_image.shape)# (128,128)
    intensity = convert_to_lab(i_image)
    otsu_thresholding(intensity)


if __name__ == "__main__":
    image = cv.imread('lena.jpg')
    start(image)
