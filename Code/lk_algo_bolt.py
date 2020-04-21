"""
Lucas Kanade Algorithm

Authors:
Nalin Das (nalindas9@gmail.com)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""
import cv2
import numpy as np
from numpy.linalg import multi_dot
import math


# Function to calculate the Jacobian
def jacobian(x, y):
  jacob = np.array([[x,0,y,0,1,0],[0, x, 0, y, 0, 1]])
  return jacob
  
# Affine KLT tracker function
def affine_LK_tracker(img, frame, tmp, rect, pprev):
  p1,p2,p3,p4 = np.transpose(np.array(rect[0])), np.transpose(np.array(rect[1])), np.transpose(np.array(rect[2])), np.transpose(np.array(rect[3]))
  p = pprev
  norm = 5
  itr=0
  
  for i in range(30):
    if norm >= 0.075:
      p_matrix = np.array([[1+p[0], p[2], p[4]],[p[1], 1+p[3], p[5]], [0,0,1]])
      #print('P Matrix:', p_matrix)
      # Step 1 - Getting the new ROI in current frame and warping template to it
      p1new, p2new, p3new, p4new = np.dot(p_matrix,p1)[0:2], np.dot(p_matrix,p2)[0:2], np.dot(p_matrix,p3)[0:2], np.dot(p_matrix,p4)[0:2] 
      M = np.array([[1+p[0], p[2], p[4]],[p[1], 1+p[3], p[5]]], dtype=np.float32)
      img_warp = cv2.warpAffine(img, M, (0,0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
      img_warp = img_warp[int(p1[1]):int(p4[1]), int(p1[0]):int(p4[0])]
      # Step 2 - Compute the error Image: Template - Warped image
      error_img = tmp - img_warp
      # Step 3 - Compute the gradient of the current frame
      x_grad = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
      y_grad = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
      x_grad = cv2.warpAffine(x_grad, M, (0,0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
      y_grad = cv2.warpAffine(y_grad, M, (0,0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
      x_grad = x_grad[int(p1[1]):int(p4[1]), int(p1[0]):int(p4[0])]
      y_grad = y_grad[int(p1[1]):int(p4[1]), int(p1[0]):int(p4[0])]
      gradient_map = []     
      # Step 4 - Compute the Jacobian of the warp
      jacobian_map = []
      steepest_descent = []
      W = []
      sigma = 4.4556
      for i in range(x_grad.shape[0]):
        for j in range(x_grad.shape[1]):
          steepest_descent.append(np.dot(np.array([x_grad[i,j], y_grad[i,j]]), np.array(jacobian(i, j))))   
          weight = (1/sigma*math.sqrt(2*np.pi))*math.exp((-(math.sqrt((i - x_grad.shape[0]/2)**2 + (j - x_grad.shape[1]/2)**2))**2)/(2*(sigma**2)))
          W.append(weight)   
      jacobian_map = np.array(jacobian_map)
      gradient_map = np.array(gradient_map)
      # Step 5 - Compute Steepest descent
      steepest_descent = np.array(steepest_descent)
      # Huber Loss - Weighted Window Intermidiate step- Calculate diagonal weight matrix
      W = np.diag(W)
      # Step 6 - Compute the Hessian matrix
      H = multi_dot([np.transpose(steepest_descent), W, steepest_descent])
      # Step 7 - Compute updated delta P 
      delta_p = multi_dot([np.linalg.pinv(H), np.transpose(steepest_descent), W, error_img.flatten()])
      # Step 8 - Compute Norm of delta P
      norm = np.linalg.norm(delta_p)
      # Step 9 - Update P matrix
      delta_p = np.dot(delta_p, 1)
      p = np.add(p, delta_p)
      itr= itr+1
    else:
      break

  tracked_frame = cv2.rectangle(frame , (int(p1new[0]), int(p1new[1])), (int(p4new[0]), int(p4new[1])), (255,0,0), 4)
  #print('Shape:', tracked_frame.shape)
  #cv2.imshow('tracked_frame', frame)
  #cv2.waitKey(0)   
  return p, tracked_frame 

