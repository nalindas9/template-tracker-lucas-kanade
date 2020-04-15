"""
Lucas Kanade Algorithm

Authors:
Nalin Das (nalindas9@gmail.com)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College 
"""
import cv2
import numpy as np

"""

"""
def affine_LK_tracker(img, tmp, rect, pprev):
  x1,y1,x2,y2 = rect[0], rect[1], rect[2], rect[3]
  # Calculating the Gradient of the template, returns gradient along rows and columns
  grad_temp = np.gradient(tmp)

