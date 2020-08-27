import cv2
import numpy as np
import glob

img_array = []
for filename in glob.glob('./big_calibrate/f*.bmp'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('project.avi',0, 5, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
