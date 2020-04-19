import numpy as np
import cv2

'''
Pre - computes the jacobian of the image (dW/dp)
params: x-coordinate, y-coordinate of the pixels in the image
return: Jacobian Matrix
'''


def jacobian(x, y):
    return np.array([[x, 0, y, 0, 1, 0], [0, x, 0, y, 0, 1]])


'''
Get new rectangle corners for the updated rectangle
params: rectangle corners, parameters
return: new corners
'''


def getNewCorners(rect, P):
    p1, p2, p3, p4 = rect[0], rect[1], rect[2], rect[3]
    p_matrix = np.array([[1 + P[0], P[2], P[5]], [P[1], 1 + P[3], P[5]], [0, 0, 1]])
    p1new, p2new, p3new, p4new = np.dot(p_matrix, p1), np.dot(p_matrix, p2), np.dot(p_matrix, p3), np.dot(p_matrix, p4)
    return p1new, p2new, p3new, p4new


'''
Warp affine transformation
params: img, bounding box, pprev
return: Warped Image
'''


def warpImg(img, rect, P):
    # p1, p2, p3, p4 = rect[0], rect[1], rect[2], rect[3]
    # p_matrix = np.array([[1 + P[0], P[2], P[5]], [P[1], 1 + P[3], P[5]], [0, 0, 1]])
    # p1new, p2new, p3new, p4new = np.dot(p_matrix, p1), np.dot(p_matrix, p2), np.dot(p_matrix, p3), np.dot(p_matrix, p4)
    p1new, p2new, p3new, p4new = getNewCorners(rect, P)
    img_warp = img[p1new[1]:p4new[1], p1new[0]:p4new[0]]
    return img_warp


'''
Finds the error image wrt to the template imgae
params: warped image, template image
return: error image
'''


def errImg(img_warp, temp):
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    error_img = temp - img_warp
    return error_img


'''
Calculates the x gradient and y gradient of the grayscale image
params: warped image(grayscale)
return: x-grad and y-grad of the image
'''


def image_gradient(img):
    #x_grad, y_grad = np.gradient(img)
    x_grad = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    y_grad = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    return x_grad, y_grad


'''
This function calculates the steepest descent of the image, based on the image gradients and the jacobian
params: warped image, x-gradient, y-gradient
returns: steepest descent
'''


def getSteepestDescent(img, x_grad, y_grad):
    steepest_descent = []
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            img_gradient = np.array([x_grad[x,y], y_grad[x,y]])
            s_descent = np.matmul(img_gradient.T, jacobian(x, y))
            steepest_descent.append(s_descent)
    return np.array(steepest_descent)


'''
Computes the Hessian Matrix 
params: Steepest Descent
return: Hessian Matrix
'''


def computeHmatrix(descent):
    H = np.dot(descent.T, descent)
    return H


'''
Update Parameters Gradient descent
params: Parameters, delta_parameters
return: New parameters
'''


def updateParameters(pprev, delta):
    p = np.zeros_like(pprev)
    for i in range(len(pprev)):
        p[i] = pprev[i] + delta[i]
    return p


'''
Lucas Kanade Tracker Algorithm implementation
params: img, template image, bounding box, previous parameters
return: updated maximized parameters for the frame
'''


def affine_LK_tracker(img, temp, rect, pprev):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray_temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    parameters = pprev
    thresh = 0.01
    err = 1
    while True:
        img_warped = warpImg(gray_img, rect, parameters)
        # cv2.imshow('Warped frame', img_warped)
        error_img = errImg(img_warped, temp)
        error_img = error_img.reshape((error_img.shape[0]*error_img.shape[1]),1)
        img_x, img_y = image_gradient(img_warped)
        # cv2.imshow('x-gradient', img_x)
        # cv2.imshow('y-gradient', img_y)
        steepest_descent = getSteepestDescent(img_warped, img_x, img_y)
        H_matrix = computeHmatrix(steepest_descent)
        delta_p = np.matmul(np.linalg.inv(H_matrix), np.dot(steepest_descent.T, error_img))
        err = np.square(delta_p).sum()
        parameters = updateParameters(parameters, delta_p)
        if err < thresh:
            break
    new_parameters = parameters
    p1, p2, p3, p4 = getNewCorners(rect, new_parameters)
    new_rect = np.array([p1, p2, p3, p4]).reshape(4, 3)
    return new_parameters, new_rect



#################### TRIAL CODE - CAR ##############################


BOX_START = (63, 49)
BOX_END = (179, 139)
BOX_COLOR = (0, 255, 0)
rect = np.array([[BOX_START[0], BOX_START[1], 1],
                 [BOX_END[0], BOX_START[1], 1],
                 [BOX_START[0], BOX_END[1], 1],
                 [BOX_END[0], BOX_END[1], 1]]).reshape(4, 3)

P_init = np.array([0, 0, 0, 0, 0, 0])

fname = "/home/aditya/Car4/img/0001.jpg"
img = cv2.imread(fname)
#cv2.imshow('input', img)

temp = img[BOX_START[1]:BOX_END[1], BOX_START[0]:BOX_END[0]] #Template Image

newP, newRect = affine_LK_tracker(img, temp, rect, P_init)
print(newP)
print(newRect[0, 0:2])
bounding_box = cv2.rectangle(img, tuple(newRect[0, 0:2]), tuple(newRect[3, 0:2]), BOX_COLOR, 2)
cv2.imshow('framed', img)
#cv2.imshow('new warp', newWarp)
#cv2.imshow('grade',steepest_descent)
#
# p1, p2, p3, p4 = getNewCorners(rect, P_init)
# print(rect)
# print(np.array([p1, p2, p3, p4]).reshape(4,3))
cv2.waitKey(0)
cv2.destroyAllWindows()


