"We are gonna create an application for fiding the size of the box using opencv "
#so lets import libraries and load the camera

import cv2

#now load the camera

camera = cv2.VideoCapture(0)
#so camera is loaded
#now lets capture the images
while (True):
    ret, frame = camera.read()
    #we have frames now lets display that
    cv2.imshow("original_video",frame)
    #now we are converting it into grayscale and displaying it
    gray_video = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray_video",gray_video)
    #now we have to convert into binary color


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()