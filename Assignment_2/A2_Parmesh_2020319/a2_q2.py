import math
import numpy as np
import cv2

def get_t(tx, ty, theta):
    return np.array([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [tx*math.cos(theta) + ty*math.sin(theta), -tx*math.sin(theta) + ty*math.cos(theta), 1]])


def get_o(x_out, y_out):
    return np.array([[x_out, y_out, 1]])


def get_v(v1, v2, v3, v4):  # matrix of pixel values
    return np.array([[v1], [v2], [v3], [v4]])


def get_x(x1, y1, x2, y2, x3, y3, x4, y4):  # matrix of x,y values at 4 closest points
    return np.array([[x1, y1, x1*y1, 1], [x2, y2, x2*y2, 1], [x3, y3, x3*y3, 1], [x4, y4, x4*y4, 1]])


def get_a(v, x):  # matrix of coefficients
    return np.linalg.inv(x)@v


def start(image_i, image_o):
    i_h, i_w = image_o.shape[:2]  # input image dimensions
    t_m = get_t(50, 50, np.radians(10))  # 10 degrees into radians 0.174533
    # i_t_m = np.linalg.inv(t_m)  # inverse of t_m
    h = 200
    w = 200
    new_image = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            #mapping coordinates
            o_c = get_o(i, j)  # output coordinates of image to be regsietered
            i_c = np.matmul(o_c,t_m)  # input coordinates of output.bmp because Ireg * T = O
            # print(i_c)
            X = i_c[0][0]
            Y = i_c[0][1]
            # print([X,Y])
            if (X < 0 or X > i_h or Y < 0 or Y > i_w):
                continue
            #interpolation
            x1 = math.floor(X)
            y1 = math.floor(Y)
            x2 = min(i_h - 1, math.ceil(X))
            y2 = min(i_w - 1, math.ceil(Y))
            # print([x1,y1,x2,y2])
            if (x1 == x2 and y1 == y2):#nearest neighbour
                p_v = image_o[int(X)][int(Y)]
            elif (x1 == x2):#linear interpolation
                p_v_1 = image_o[int(X)][int(y1)]
                p_v_2 = image_o[int(X)][int(y2)]
                p_v = p_v_1 * (y2 - Y) + p_v_2 * (Y - y1)
            elif (y1 == y2):#linear interpolation
                p_v_1 = image_o[int(x1)][int(Y)]
                p_v_2 = image_o[int(x2)][int(Y)]
                p_v = p_v_1 * (x2 - X) + p_v_2 * (X - x2)
            else:#bi-linear interpolation
                v1 = image_o[x1][y1]
                v2 = image_o[x1][y2]
                v3 = image_o[x2][y1]
                v4 = image_o[x2][y2]
                v = get_v(v1, v2, v3, v4)
                # print(v)
                x = get_x(x1, y1, x1, y2, x2, y1, x2, y2)
                # print(x)
                a = get_a(v, x)
                # print(a)
                p_v = a[0][0]*i + a[1][0]*j + a[2][0]*i*j + a[3][0]  # V = Ax + By + Cxy + D
            if (p_v > 255):
                p_v = 255
            elif (p_v < 0):
                p_v = 0
            new_image[i][j] = p_v
    cv2.imwrite('registered.bmp',new_image)

if __name__ == "__main__":
    img_i = cv2.imread('x5.bmp', 0)
    img_o = cv2.imread('output.bmp', 0)
    image_i = np.asarray(img_i)
    image_o = np.asarray(img_o)
    start(image_i, image_o)
