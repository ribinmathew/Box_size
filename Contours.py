"We are gonna create an application for fiding the size of the box using opencv "
#so lets import libraries and load the camera

import cv2
import numpy as np


#now load the camera

camera = cv2.VideoCapture(1)
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
    ret,threshold_video = cv2.threshold(gray_video,155,250,cv2.THRESH_BINARY_INV)
    cv2.imshow("Thrsh_video",threshold_video)
    #so we applied threshold and got good result with simple thresholdin technique
    kernel = np.ones((5,5),np.uint8)
    dialated_video = cv2.dilate(threshold_video,kernel,iterations=2)
    opening_video = cv2.morphologyEx(dialated_video,cv2.MORPH_CLOSE,kernel)
    # now we have to draw contours around it
    #contor_image = cv2.Canny(dialated_video,150,255)
    #cv2.imshow("contour_video",contor_image)
    #with open_filter to rectify the noise
    edge_image1 = cv2.Canny(opening_video, 155, 255)
    cv2.imshow("open video", edge_image1)
    #now we have to draw contour around the object
    im2,contours,heirarchy = cv2.findContours(edge_image1,1,2)
    for c in contours:
        #here we are avoiding small errors by eliminating small  boxes
        if cv2.contourArea(c) > 3000:
            #we are drawing a rectangle around the counter and displaying it

            x,y,w,h = cv2.boundingRect(c)
            #print(x,y,w,h)
            #image_with_box = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
            #cv2.imshow("image_wih_box",image_with_box)
            #the image we have created with box is not a rotatable box. so lets create a rotating box
            rotating_rectangle_image = cv2.minAreaRect(c)
            box = cv2.boxPoints(rotating_rectangle_image)
            #print(box)
            box = np.int0(box)
            rotating_box_image= cv2.drawContours(frame,[box],0,(0,0,255),2)
            cv2.imshow("box",rotating_box_image)
            #now lets draw a small circles on the box points
            for (x,y) in box:
                box_circle = cv2.circle(frame,(int(x),int(y)),3,(255,0,0),-1)
                cv2.imshow("box_circle",box_circle)
            #here we are gonna add text to the points

            box= np.int0(box)
            X,Y = box[0]
            point1 = cv2.putText(frame,"A",(X,Y),cv2.FONT_ITALIC,.8,(50,200,100),2,cv2.LINE_AA)
            X1, Y1 = box[1]
            point2 = cv2.putText(frame, "B", (X1, Y1), cv2.FONT_ITALIC, .8, (50, 200, 100), 2, cv2.LINE_AA)
            X2, Y2 = box[2]
            point3 = cv2.putText(frame, "C", (X2, Y2), cv2.FONT_ITALIC, .8, (50, 200, 100), 2, cv2.LINE_AA)
            X3, Y3 = box[3]
            point4 = cv2.putText(frame, "D", (X3, Y3), cv2.FONT_ITALIC, .8, (50, 200, 100), 2, cv2.LINE_AA)
            #now we are gonna Define the points A,B,C,D and we are gonna give the co-oirdinates to the same
            P1,Q1 = box[0] #A
            P2,Q2 = box[1] #B
            P3,Q3 = box[2] #C
            P4,Q4 = box[3] #D
            # now we got the p1 and q1, co-ordinataes for all the points
            # we are gonna find the euclidean distance using formula
            # the euclidean distance formula
            # d(p,q) = Squareroot((q1-p1)**2 + (q2-p2)**2)
            # we are gonna find the distance between A,B and A,D
            # The reason for A,B and A,D is its a rectangle so we need only 2 side :-)
            A_to_B = np.sqrt(np.sum((P1-P2)**2 + (Q1-Q2)**2))
            A_to_D = np.sqrt(np.sum((P1-P4)**2 + (Q1-Q4)**2))
            #print(A_to_B)
            #print(A_to_D)
            #now we have to find the pixel per ration
            #pixel per ratio is the ratio between the pixel and the original size of the image
            #so the equation for pixel per metric = object_width /know_width
            if A_to_B > A_to_D:
                pixel_per_metric = A_to_B/66.91
            if A_to_D > A_to_B:
                pixel_per_metric = A_to_D/31.45

            #print(pixel_per_metric)
            #now we have the pixel per metric ratio
            #lets simply print the diamensions of the small box already we are having
            if A_to_B > A_to_D:
                length =  A_to_B/pixel_per_metric
                bredth =  A_to_D/pixel_per_metric
                print("lenght",length)
                print("bredth",bredth)
            if A_to_D > A_to_B:
                length =  A_to_D/pixel_per_metric
                bredth = A_to_B /pixel_per_metric
                print("lenght", length)
                print("bredth", bredth)


# So we are at a stage where we can find the pixel per metric ratio of the image
    # now we have to make that small thing as reference and then find the size of the other boxes
    #so for that we have to make sure that
    #print(cv2.contourArea(c))

    #print(cv2.contourArea(contor_image1))

    #the edge detection was good, still we have to give little dialation to the video,after thresholding



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()