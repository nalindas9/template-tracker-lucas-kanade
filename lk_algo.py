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
  p1,p2,p3,p4 = np.transpose(np.array(rect[0])), np.transpose(np.array(rect[1])), np.transpose(np.array(rect[2])), np.transpose(np.array(rect[3]))
  print('Template corners:', (p1,p2,p3,p4))
  delta_p = [0,0,0,0,0,0]
  p_matrix = np.array([[1+delta_p[0], delta_p[2], delta_p[4]],[delta_p[1], 1+delta_p[3], delta_p[5]], [0,0,1]])
  print('P Matrix:', p_matrix)
  # Step 1 -  Getting new ROI in current frame by multiplying p_matrix with Template 
  p1new, p2new, p3new, p4new = np.dot(p_matrix,p1), np.dot(p_matrix,p2), np.dot(p_matrix,p3), np.dot(p_matrix,p4)  
  cv2.imshow('template', tmp)
  cv2.waitKey(0) 
  print('New ROI:', (p1new, p2new, p3new, p4new))
  img_warp = img[p1new[1]:p4new[1], p1new[0]:p4new[0]]
  cv2.imshow('Warped Image', img_warp)
  cv2.waitKey(0)
  
