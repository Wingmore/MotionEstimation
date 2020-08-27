'''
RTSP Camera global motion estimation
Code adapted from:
/samples/python2/lk_track.py
'''

# Python 2/3 compatibility
from __future__ import print_function
import cv2 as cv

import numpy as np
import time

from collections import namedtuple
#import video
#from common import anorm2, draw_str
SMOOTHING_RADIUS = 3

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 200,
                       qualityLevel = 0.3,
                       minDistance = 47,
                       blockSize = 47 )

def textp(vis, text, pos):
    cv.putText(vis, text, pos,cv.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 1, cv.LINE_4)

def movingAverage(curve, radius): 
  window_size = 2 * radius + 1
  # Define the filter 
  f = np.ones(window_size)/window_size 
  # Add padding to the boundaries 
  curve_pad = np.lib.pad(curve, (radius, radius), 'edge') 
  # Apply convolution 
  curve_smoothed = np.convolve(curve_pad, f, mode='same') 
  # Remove padding 
  curve_smoothed = curve_smoothed[radius:-radius]
  # return smoothed curve
  return curve_smoothed 

def smooth(trajectory): 
  smoothed_trajectory = np.copy(trajectory) 
  # Filter the x, y and angle curves
  for i in range(3):
    smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)

  return smoothed_trajectory


class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv.VideoCapture(video_src)
        self.frame_idx = 0
        self.X = self.Y = 0
        self.X1 = self.Y1 = self.rot = 0
#        self.transforms = []
        
    #WAY TOO SLOW - takes .3s per frame
    def phase_cor_f(self, img0, img1, vis):
        start = time.time()
        
#        img0, img1 = self.prev_gray, frame_gray
        img0 = np.float32(img0)             # convert first into float32
        img1 = np.float32(img1)             # convert second into float32  
        
        h,w = img0.shape
        cX, cY = w//2, h//2
        
        prev_polar = cv.linearPolar(img0,(cX, cY), min(cX, cY), 0)
        cur_polar = cv.linearPolar(img1,(cX, cY), min(cX, cY), 0)
        
        #what is df?
        (dx, dy), df = cv.phaseCorrelate(img0,img1) # now calculate the phase correlation
        self.X1 += dx
        self.Y1 += dy
        textp(vis, 'X, Y: {:.2f} {:.2f}'.format(self.X1, self.Y1),(20, 200))

        (sx, sy), sf = cv.phaseCorrelate(prev_polar, cur_polar)
        rotation = -sy / h * 360;
        self.rot += rotation
        textp(vis, 'rot: %d' % self.rot,(20, 250))

        end = time.time()
        print('time elapsed: ',end-start)
        return vis


    def run(self):
        #Buffer For a moving average filter
        Mystruct = namedtuple("Mystruct", "dx dy da")
        m_tr = Mystruct(np.zeros((self.track_len,1)),np.zeros((self.track_len,1)),np.zeros((self.track_len,1)))
        while True:
            _ret, frame = self.cam.read()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                Xt = 0  #sum of change in X
                Yt = 0  #sum of change in X
  
                #Find transformation matrix
                # Opencv 3
                #m = cv.estimateRigidTransform(p0, p1, fullAffine=False) #will only work with OpenCV-3 or less
                m = cv.estimateAffinePartial2D(p0, p1)  #For openCV4.4
                if 'm' in locals():
                    m = m[0]    #for openCV4.4
                    try:
                        # Extract traslation
                        dx = m[0,2]
                        dy = m[1,2]
                        # Extract rotation angle
                        da = np.arctan2(m[1,0], m[0,0])
                        
                        #add to buffer
                        i = self.frame_idx%self.track_len
                        m_tr.dx[i] = dx
                        m_tr.dy[i] = dy
                        m_tr.da[i] = da
                        
                        #Show average
                        self.X1 += np.sum(m_tr.dx)/self.track_len
                        self.Y1 += np.sum(m_tr.dy)/self.track_len
                        self.rot += np.sum(m_tr.da)/self.track_len
                        
                    except:
                        print('find new points')
                # Store transformation
#                transforms[i] = [dx,dy,da]

#                trajectory = np.cumsum(transforms, axis=0)

                # following 
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        x = tr[-1][0] - tr[0][0]
                        y = tr[-1][1] - tr[0][1]
                        Xt += x
                        Yt += y
                        del tr[0]
                    new_tracks.append(tr)
                    cv.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                if not (len(self.tracks) == 0):
                    self.X +=  Xt/len(self.tracks)
                    self.Y += Yt/len(self.tracks)
                cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                
                textp(vis, 'track count: %d' % len(self.tracks),(20, 50))
                textp(vis, 'X: %d' % self.X,(20, 100))
                textp(vis, 'Y: %d' % self.Y,(20, 150))
                textp(vis, 'XYZ: {:.2f} {:.2f} {:.2f}'.format(self.X1, self.Y1, self.rot*180/np.pi), (20, 200))
                
                # WAY TOO SLOW RIP
                # vis = self.phase_cor_f(img0, img1, vis)

            # Pretend its the first frame again, find new features to track
            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)
                #find (new?) features (corners) to track
                p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])



            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv.imshow('lk_track', vis)

            ch = cv.waitKey(1)
            if ch == 27:
                break

def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 'rtsp://192.168.1.88:554/'

    App(video_src).run()
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    print(cv.__version__)
    main()
    cv.destroyAllWindows()
