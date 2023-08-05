import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def convert_to_hsi(i_image):
    i_image = cv.cvtColor(i_image, cv.COLOR_BGR2RGB)
    hsi = cv.cvtColor(i_image, cv.COLOR_RGB2HLS)
    H, S, I = cv.split(hsi)
    plt.subplot(1, 3, 1), plt.imshow(H, cmap='gray'), plt.title('Hue')
    plt.subplot(1, 3, 2), plt.imshow(S, cmap='gray'), plt.title('Saturation')
    plt.subplot(1, 3, 3), plt.imshow(I, cmap='gray'), plt.title('Intensity')
    plt.show()
    return I


def canny_edge_with_sobel(intensity):
    s_x = cv.Sobel(intensity, cv.CV_64F, 1, 0, ksize=3)
    s_y = cv.Sobel(intensity, cv.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(s_x**2 + s_y**2)
    direction = np.arctan2(s_y, s_x)
    magnitude = np.clip(magnitude, 0, 255)
    return magnitude, direction


def non_max_suppression(magnitude, direction):
    new_magnitude = np.zeros(magnitude.shape)
    for i in range(1, magnitude.shape[0]-1):
        for j in range(1, magnitude.shape[1]-1):
            d = direction[i, j]
            if (0 <= d < np.pi / 8) or (15 * np.pi / 8 <= d <= 2 * np.pi):
                p1 = magnitude[i, j - 1]
                p2 = magnitude[i, j + 1]
            elif (np.pi / 8 <= d < 3 * np.pi / 8) or (9 * np.pi / 8 <= d < 11 * np.pi / 8):
                p1 = magnitude[i + 1, j - 1]
                p2 = magnitude[i - 1, j + 1]
            elif (3 * np.pi / 8 <= d < 5 * np.pi / 8) or (11 * np.pi / 8 <= d < 13 * np.pi / 8):
                p1 = magnitude[i - 1, j]
                p2 = magnitude[i + 1, j]
            else:
                p1 = magnitude[i - 1, j - 1]
                p2 = magnitude[i + 1, j + 1]
            if magnitude[i, j] >= p1 and magnitude[i, j] >= p2:
                new_magnitude[i, j] = magnitude[i, j]
    return new_magnitude


def start(i_image):
    # print(i_image.shape) (151,157)
    intensity = convert_to_hsi(i_image)
    magnitude, direction = canny_edge_with_sobel(intensity)
    new_magnitude = non_max_suppression(magnitude, direction)
    i_image = cv.cvtColor(i_image, cv.COLOR_BGR2RGB)
    plt.subplot(1, 3, 1), plt.imshow(
        magnitude, cmap='gray'), plt.title('Magnitude')
    plt.subplot(1, 3, 2), plt.imshow(
        direction, cmap='gray'), plt.title('Direction')
    plt.subplot(1, 3, 3), plt.imshow(new_magnitude,
                                     cmap='gray'), plt.title('Non Max Suppression')
    plt.show()


if __name__ == "__main__":
    image = cv.imread('palette-1c-8b.tiff')
    start(image)
