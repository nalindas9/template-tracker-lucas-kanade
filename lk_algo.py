"""
Lucas Kanade Algorithm

Authors:
Nalin Das (nalindas9@gmail.com)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College 
"""
import cv2
import numpy as np

def jacobian(x, y):
  jacob = np.array([[x,0,y,0,1,0],[0, x, 0, y, 0, 1]])
  return jacob
  
def affine_LK_tracker(img, tmp, rect, pprev):
  p1,p2,p3,p4 = np.transpose(np.array(rect[0])), np.transpose(np.array(rect[1])), np.transpose(np.array(rect[2])), np.transpose(np.array(rect[3]))
  print('Template corners:', (p1,p2,p3,p4))
  delta_p = [0,0,0,0,0,0]
  p_matrix = np.array([[1+delta_p[0], delta_p[2], delta_p[4]],[delta_p[1], 1+delta_p[3], delta_p[5]], [0,0,1]])
  print('P Matrix:', p_matrix)
  # Step 1 - Getting new ROI in current frame by multiplying p_matrix with Template 
  p1new, p2new, p3new, p4new = np.dot(p_matrix,p1), np.dot(p_matrix,p2), np.dot(p_matrix,p3), np.dot(p_matrix,p4)  
  cv2.imshow('template', tmp)
  cv2.waitKey(0) 
  print('New ROI:', (p1new, p2new, p3new, p4new))
  img_warp = img[p1new[1]:p4new[1], p1new[0]:p4new[0]]
  cv2.imshow('Warped Image', img_warp)
  cv2.waitKey(0)
  # Step 2 - Compute the error Image: Template - Warped image
  error_img = tmp - img_warp
  cv2.imshow('Error Image', error_img)
  cv2.waitKey(0)
  # Step 3 - Compute the gradient of the current frame
  x_grad, y_grad = np.gradient(img)
  cv2.imshow('X gradient current frame', x_grad)
  cv2.waitKey(0)
  cv2.imshow('Y gradient current frame', y_grad)
  cv2.waitKey(0)
  # Step 4 - Compute the Jacobian of the warp
  jacob_func = np.vectorize(jacobian)
  jacobian_map = []
  for i in range(img_warp.shape[0]):
    for j in range(img_warp.shape[1]):
      jacobian_map.append(jacobian(i, j))
       
  jacobian_map = np.array(jacobian_map)
  print('Jacobian:', jacobian_map)
