import cv2
import numpy as np
import glob
from lucas_kanade import *
import matplotlib.pyplot as plt

BOX_START = (63, 49)
BOX_END = (179, 139)
BOX_COLOR = (0, 255, 0)

P_init = np.array([0, 0, 0, 0, 0, 0])
rect = np.array([[BOX_START[0], BOX_START[1], 1],
                 [BOX_END[0], BOX_START[1], 1],
                 [BOX_START[0], BOX_END[1], 1],
                 [BOX_END[0], BOX_END[1], 1]]).reshape(4, 3)
file_path = "/home/aditya/Car4/img/*"
img1 = cv2.imread("/home/aditya/Car4/img/0001.jpg")
temp = img1[BOX_START[1]:BOX_END[1], BOX_START[0]:BOX_END[0]]
Frame = 0
param = P_init
for fname in sorted(glob.glob(file_path)):
    img = cv2.imread(fname)
    #cv2.imshow('Input', img)
    # if Frame == 0:
    #     temp = img[BOX_START[1]:BOX_END[1], BOX_START[0]:BOX_END[0]]
    #     print('This is Frame 0')
        #cv2.imshow('Template ROI ', tmp)
        # param, rect = affine_LK_tracker(img, tmp, rect, P_init)
        # bounding_box = cv2.rectangle(img, tuple(rect[0, 0:2]), tuple(rect[3, 0:2]), BOX_COLOR, 2)
        # cv2.imshow('framed', img)
    #print('Image dims: ',img.shape)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_param, new_rect = affine_LK_tracker(img, temp, rect, param)
    param, rect = new_param, new_rect
    bounding_box = cv2.rectangle(img, tuple(rect[0, 0:2]), tuple(rect[3, 0:2]), BOX_COLOR, 2)
    cv2.imshow('framed', img)
    # plt.hist(img.ravel(), 256, [0, 256])
    # plt.show()
    Frame += 1
    #cv2.waitKey(0)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cv2.destroyAllWindows()
