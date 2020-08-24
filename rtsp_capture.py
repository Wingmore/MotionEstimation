import cv2 as cv
vcap = cv.VideoCapture("rtsp://192.168.1.88:554/")
i = 0
while(1):
    ret, frame = vcap.read()
    cv.imshow('VIDEO', frame)
    ch = cv.waitKey(1)
    if ch == 27:
        break
    if cv.waitKey(33) == ord('a'):
        print('writing to', i)
        cv.imwrite('thing'+str(i)+'.bmp', frame)
        i = i+1
