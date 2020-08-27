import numpy as np
import cv2
src1 = cv2.imread('s8.bmp',0)   # load first image in grayscale
src2 = cv2.imread('s9.bmp',0)      # load second image in grayscale
src1 = np.float32(src1)             # convert first into float32
src2 = np.float32(src2)             # convert second into float32  


h,w = src1.shape
cX, cY = w//2, h//2

prev_polar = cv2.linearPolar(src1,(cX, cY), min(cX, cY), 0)
cur_polar = cv2.linearPolar(src2,(cX, cY), min(cX, cY), 0) 

#what is df?
(dx, dy), df = cv2.phaseCorrelate(src1,src2) # now calculate the phase correlation

print(dx, dy)

(sx, sy), sf = cv2.phaseCorrelate(prev_polar, cur_polar)

rotation = -sy / h * 360;
print(rotation)

