import cv2
import numpy as np
import glob
import math
import time
# Load previously saved data
with np.load('cam_param.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

#roll = pitch = yaw = 0
#%%
path = './calibrate_old/'
for fname in glob.glob(path + 'f4*.bmp'):

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        start = time.time()

        #something dodgy
#        objp = np.zeros((corners.shape[0],3), np.float32)
#        objp[:,:2] = corners.reshape(-1,2)
        
        # Find the rotation and translation vectors.
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        #find camera pose
        Rt, _ = cv2.Rodrigues(rvecs)
        R = Rt.T
        pos = -R*tvecs   #pose of camera in global frame
        
        #get pictch and yaw and stuff
        roll = math.atan2(-R[2][1], R[2][2])*180/math.pi
        pitch = math.asin(R[2][0])*180/math.pi
        yaw = math.atan2(-R[1][0], R[0][0])*180/math.pi
        print(fname)
        print('roll:{}, pitch: {}. yaw: {}'.format(roll, pitch, yaw))

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        
        end = time.time()
        print('time: ', end-start)
        img = draw(img,corners,imgpts)
        cv2.imshow(fname,img)
        k = cv2.waitKey(0) & 0xff
        if k == 's':
            cv2.imwrite(fname[:6]+'.png', img)

cv2.destroyAllWindows()
