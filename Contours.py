"We are gonna create an application for fiding the size of the box using opencv "
#so lets import libraries and load the camera

import cv2
import numpy as np


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
    #before that we are going to apply thresholding
    ret,threshold_video = cv2.threshold(gray_video,150,250,cv2.THRESH_BINARY_INV)
    cv2.imshow("Thrsh_video",threshold_video)
    #so we applied threshold and got good result with simple thresholdin technique
    kernel = np.ones((5,5),np.uint8)
    dialated_video = cv2.dilate(threshold_video,kernel,iterations=2)
    opening_video = cv2.morphologyEx(dialated_video,cv2.MORPH_CLOSE,kernel)
    # now we have to draw contours around it
    contor_image = cv2.Canny(dialated_video,150,255)
    cv2.imshow("contour_video",contor_image)
    #with open_filter to rectify the noise
    contor_image1 = cv2.Canny(opening_video, 150, 255)
    cv2.imshow("open video", contor_image1)
    #the edge detection was good, still we have to give little dialation to the video,after thresholding



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()