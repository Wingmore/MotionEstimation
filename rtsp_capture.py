import cv2 as cv
i = 0
vcap = cv.VideoCapture("rtsp://192.168.1.88:554/")
while(1):
    ret, frame = vcap.read()
    cv.imshow('VIDEO', frame)
    c = cv.waitKey(1)
    if cv.waitKey(1) == ord('a'):
	print('writing')
	cv.imwrite('f'+str(i)+'.bmp',frame)
	i = i+1
    if c == 27:
        break
