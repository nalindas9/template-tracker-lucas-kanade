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

def jacobian(x, y):
  jacob = np.array([[x,0,y,0,1,0],[0, x, 0, y, 0, 1]])
  return jacob
  
def affine_LK_tracker(img, tmp, rect, pprev):
  p1,p2,p3,p4 = np.transpose(np.array(rect[0])), np.transpose(np.array(rect[1])), np.transpose(np.array(rect[2])), np.transpose(np.array(rect[3]))
  print('Template corners:', (p1,p2,p3,p4))
  p = pprev
  norm = 5
  itr=0
  
  for i in range(30):
    print('iteration:', itr)
    p_matrix = np.array([[1+p[0], p[2], p[4]],[p[1], 1+p[3], p[5]], [0,0,1]])
    print('P Matrix:', p_matrix)
    # Step 1 - Getting the new ROI in current frame and warping template to it
    
    """
    img_warp = []
    img_warp_coords = []
    for i in range(tmp.shape[0]):
      for j in range(tmp.shape[1]):
        new_coords = np.dot(p_matrix, np.transpose(np.array([i,j,1])))[0:2]
        img_warp.append(img[new_coords[0], new_coords[1]])
        img_warp_coords.append(new_coords)
    img_warp = np.array(img_warp)
    img_warp = img_warp.flatten()
    """
    p1new, p2new, p3new, p4new = np.dot(p_matrix,p1)[0:2], np.dot(p_matrix,p2)[0:2], np.dot(p_matrix,p3)[0:2], np.dot(p_matrix,p4)[0:2] 
    #pts1 = np.float32([p1[0:2],p2[0:2],p3[0:2]])
    #pts2 = np.float32([p1new, p2new, p3new])
    #M = cv2.getAffineTransform(pts1,pts2)
    M = np.array([[1+p[0], p[2], p[4]],[p[1], 1+p[3], p[5]]], dtype=np.float32)
    img_warp = cv2.warpAffine(img, M, (0,0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
    #cv2.imshow('Warped image', img_warp)
    #cv2.waitKey(0) 
    img_warp = img_warp[int(p1[1]):int(p4[1]), int(p1[0]):int(p4[0])]
    #cv2.imshow('template', tmp)
    #cv2.waitKey(0) 
    print('New ROI:', (p1new, p2new, p3new, p4new))
    #input_pts = np.float32([[p1[0],p1[1]],[p2[0], p2[1]],[p3[0], p3[1]],[p4[0], p4[1]]])
    #output_pts = np.float32([[p1new[0],p1new[1]],[p2new[0], p2new[1]],[p3new[0], p3new[1]],[p4new[0], p4new[1]]])
    # Compute the perspective transform M
    #M = cv2.getPerspectiveTransform(input_pts,output_pts)
    #img_warp = img[int(p1new[1]):int(p4new[1]), int(p1new[0]):int(p4new[0])]
    #img_warp = cv2.warpPerspective(tmp,M,(tmp.shape[1], tmp.shape[0]),flags=cv2.INTER_LINEAR)
    #cv2.imshow('Warped Image', img_warp)
    #cv2.waitKey(0)
    #img_warp = np.resize(img_warp, (tmp.shape))
    print('template size:', tmp.shape, 'warp size:', img_warp.shape)
    # Step 2 - Compute the error Image: Template - Warped image
    error_img = tmp - img_warp
    #error_img = np.reshape(error_img, (tmp.shape[0], tmp.shape[1]))
    #cv2.imshow('Error Image', error_img)
    #cv2.waitKey(0)
    # Step 3 - Compute the gradient of the current frame
    x_grad = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    y_grad = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    #cv2.imshow('X gradient current frame', x_grad)
    #cv2.waitKey(0)
    #cv2.imshow('Y gradient current frame', y_grad)
    #cv2.waitKey(0)
    x_grad = cv2.warpAffine(x_grad, M, (0,0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
    y_grad = cv2.warpAffine(y_grad, M, (0,0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
    x_grad = x_grad[int(p1[1]):int(p4[1]), int(p1[0]):int(p4[0])]
    y_grad = y_grad[int(p1[1]):int(p4[1]), int(p1[0]):int(p4[0])]
    gradient_map = []     
    #cv2.imshow('X gradient current frame', x_grad)
    #cv2.waitKey(0)
    #cv2.imshow('Y gradient current frame', y_grad)
    #cv2.waitKey(0)
    # Step 4 - Compute the Jacobian of the warp
    jacobian_map = []
    steepest_descent = []
    for i in range(x_grad.shape[0]):
      for j in range(x_grad.shape[1]):
        #jacobian_map.append(jacobian(i, j))
        #gradient_map.append([x_grad[i,j], y_grad[i,j]]) 
        steepest_descent.append(np.dot(np.array([x_grad[i,j], y_grad[i,j]]), np.array(jacobian(i, j))))  
    jacobian_map = np.array(jacobian_map)
    gradient_map = np.array(gradient_map)
    print('Jacobian:', jacobian_map)
    print('Gradient map:', gradient_map)
    # Step 5 - Compute Steepest descent
    steepest_descent = np.array(steepest_descent)
    print('Steepest descent', steepest_descent)
    # Step 6 - Compute the Hessian matrix
    H = np.dot(np.transpose(steepest_descent), steepest_descent)
    print('Hessian Matrix:', H, 'Shape:', H.shape)
    # Step 7 - Compute updated delta P 
    delta_p = multi_dot([np.linalg.pinv(H), np.transpose(steepest_descent), error_img.flatten()])
    print('Delta p:', delta_p)
    # Step 8 - Compute Norm of delta P
    norm = np.linalg.norm(delta_p)
    print('Norm:', norm)
    # Step 9 - Update P matrix
    p = np.add(p, delta_p)
    itr= itr+1
  tracked_frame = cv2.rectangle(img, (int(p1new[0]), int(p1new[1])), (int(p4new[0]), int(p4new[1])), 255, 2)
  cv2.imshow('tracked_frame', tracked_frame)
  cv2.waitKey(0)   
  return p

