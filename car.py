"""
Lucas Kanade tracker for tracking car in video

Authors:
Nalin Das (nalindas9@gmail.com)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""
import glob
import cv2
print('Headers Loaded!')

IMAGES_PATH = "/home/nalindas9/Documents/Courses/Spring_2020_Semester_2/ENPM673_Perception_for_Autonomous_Robots/Github/enpm673/template-tracker-lucas-kanade/Dataset/Car4/img"

BOX_START = (63,49)
BOX_END = (179,139)
BOX_COLOR = (0,255,0)
BOX_THICKNESS = 2
def main():
  for frame in sorted(glob.glob(IMAGES_PATH + "/*")):
    print('Image:', frame.split("img/", 1)[1])
    img = cv2.imread(frame)
    if frame.split("img/", 1)[1] == '0001.jpg':
      template_frame = cv2.rectangle(img, BOX_START, BOX_END, BOX_COLOR, BOX_THICKNESS)  
    cv2.imshow('Frame', img)
    cv2.waitKey(0)
  cv2.destroyAllWindows()
  
if __name__ == '__main__':
  main()
