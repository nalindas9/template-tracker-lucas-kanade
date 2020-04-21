"""
Lucas Kanade tracker for tracking car in video

Authors:
Nalin Das (nalindas9@gmail.com)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""
import glob
import cv2
import lk_algo_baby
import numpy as np
print('Headers Loaded!')

# Specify the dataset path here
IMAGES_PATH = "/home/nalindas9/Documents/Courses/Spring_2020_Semester_2/ENPM673_Perception_for_Autonomous_Robots/Github/enpm673/template-tracker-lucas-kanade/Dataset/DragonBaby/img"

# Main function
def main():
  # Bounding box parameters
  BOX_START = (160,83)
  BOX_END = (160+56,83+65)
  BOX_COLOR = (0,255,0)
  BOX_THICKNESS = 2
  pprev = np.array([0,0,0,0,0,0])
  out = cv2.VideoWriter('dragonbaby.avi',cv2.VideoWriter_fourcc(*'XVID'), 10, (640,360))
  for frame in sorted(glob.glob(IMAGES_PATH + "/*")):
    print('Image:', frame.split("img/", 1)[1])
    img = cv2.imread(frame)
    color_img = cv2.imread(frame)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(img)
    img = cv2.GaussianBlur(img,(3,3),0)
    img_brightness = np.mean(img)
    if frame.split("img/", 1)[1] == '0001.jpg':
      template = img[BOX_START[1]:BOX_END[1], BOX_START[0]:BOX_END[0]]
      template = cv2.imwrite('dragonbaby_template.jpg', template)
      temp_brightness = np.mean(img)
      rect = [[BOX_START[0], BOX_START[1], 1], [BOX_END[0],BOX_START[1], 1], [BOX_START[0], BOX_END[1],1], [BOX_END[0], BOX_END[1], 1]]
    
    brightness_sf = temp_brightness/img_brightness
    img = cv2.multiply(img, brightness_sf)
    p, tracked_frame = lk_algo_baby.affine_LK_tracker(img, color_img, template, rect, pprev) 
    out.write(np.uint8(tracked_frame))
    pprev = p  
    cv2.destroyAllWindows()
  
if __name__ == '__main__':
  main()
