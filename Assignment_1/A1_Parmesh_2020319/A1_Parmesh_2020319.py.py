#name : PARMESH YADAV
#roll no. : 2020319

import math
import numpy as np
from PIL import Image

def transpose(array):
    return np.transpose(array)

def get_v(v1,v2,v3,v4):#matrix of pixel values
    return np.array([[v1,v2,v3,v4]])

def get_x(x1,y1,x2,y2,x3,y3,x4,y4):#matrix of x,y values at 4 closest points
    return np.array([[x1,y1,x1*y1,1],[x2,y2,x2*y2,1],[x3,y3,x3*y3,1],[x4,y4,x4*y4,1]])

def get_a(v,x):#matrix of coefficients
    return (np.linalg.inv(x))*v

#THE ABOVE CODE IS REQUIRED FOR BOTH PART B and C OF QUESTION 2.
#THE BELOW START AND MAIN FUNCTION ARE FOR QUESTION 2 PART B.

# def start(image):
#     i_h,i_w = image.shape[:2]#input image dimensions
#     new_image = np.zeros((i_h, i_w))
#     interpolation_factor = 1
#     h = math.ceil(i_h * interpolation_factor)
#     w = math.ceil(i_w * interpolation_factor)
#     for i in range(h):
#         for j in range(w):
#             X = i/interpolation_factor#mapping x to the input grid
#             Y = j/interpolation_factor#mapping y to the input grid
#             x1 = math.floor(X)
#             y1 = math.floor(Y)
#             x2 = min(i_h - 1,math.ceil(X))
#             y2 = min(i_w - 1,math.ceil(Y))
#             if(x1 == x2 and y1 == y2):
#                 p_v = image[int(X)][int(Y)]
#             elif(x1 == x2):
#                 p_v_1 = image[int(X)][int(y1)]
#                 p_v_2 = image[int(X)][int(y2)]
#                 p_v = p_v_1 * (y2 - Y) + p_v_2 * (Y - y1)
#             elif(y1 == y2):
#                 p_v_1 = image[int(x1)][int(Y)]
#                 p_v_2 = image[int(x2)][int(Y)]
#                 p_v = p_v_1 * (x2 - X) + p_v_2 * (X - x2)
#             else:
#                 v1 = image[x1][y1]
#                 v2 = image[x1][y2]
#                 v3 = image[x2][y1]
#                 v4 = image[x2][y2]
#                 v = get_v(v1,v2,v3,v4)
#                 x = get_x(x1,y1,x1,y2,x2,y1,x2,y2)
#                 a = get_a(v,x)
#                 p_v = a[0][0]*j + a[0][1]*i + a[0][2]*i*j + a[0][3]#V = Ax + By + Cxy + D
#             new_image[i][j] = p_v
#     print(new_image)

# if __name__ == "__main__":
#     image = np.array([[5,0,0,0],[0,1,3,1],[3,0,2,0]])
#     start(image)

#THE BELOW START AND MAIN FUNCTION ARE FOR QUESTION 2 PART C
#COMMENT THE ABOVE START AND MAIN FUNCTION AND UNCOMMENT THE BELOW START AND MAIN FUNCTION TO RUN THE CODE FOR QUESTION 2 PART C

def start(image):
    i_h,i_w = image.shape[:2]#input image dimensions
    interpolation_factor = 0.5
    h = math.ceil(i_h * interpolation_factor)
    w = math.ceil(i_w * interpolation_factor)
    new_image = np.zeros((i_h, i_w))
    for i in range(h):
        for j in range(w):
            X = i/interpolation_factor#mapping x to the input grid
            Y = j/interpolation_factor#mapping y to the input grid
            x1 = math.floor(X)
            y1 = math.floor(Y)
            x2 = min(i_h - 1,math.ceil(X))
            y2 = min(i_w - 1,math.ceil(Y))
            if(x1 == x2 and y1 == y2):
                p_v = image[int(X)][int(Y)]
            elif(x1 == x2):
                p_v_1 = image[int(X)][int(y1)]
                p_v_2 = image[int(X)][int(y2)]
                p_v = p_v_1 * (y2 - Y) + p_v_2 * (Y - y1)
            elif(y1 == y2):
                p_v_1 = image[int(x1)][int(Y)]
                p_v_2 = image[int(x2)][int(Y)]
                p_v = p_v_1 * (x2 - X) + p_v_2 * (X - x2)
            else:
                v1 = image[x1][y1]
                v2 = image[x1][y2]
                v3 = image[x2][y1]
                v4 = image[x2][y2]
                v = get_v(v1,v2,v3,v4)
                x = get_x(x1,y1,x1,y2,x2,y1,x2,y2)
                a = get_a(v,x)
                p_v = a[0][0]*j + a[0][1]*i + a[0][2]*i*j + a[0][3]#V = Ax + By + Cxy + D
            new_image[i][j] = p_v
    output_image = Image.fromarray(new_image)
    output_image = output_image.convert('RGB')
    output_image.save("output.bmp")

if __name__ == "__main__":
    img = Image.open("x5.bmp")
    image = np.asarray(img)
    start(image)


