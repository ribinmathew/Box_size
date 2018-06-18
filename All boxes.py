#in this we are gonna draw bounding box to every box present in the figure
import cv2
import numpy as np

camera = cv2.VideoCapture(1)


#here we are gonna define a function to detect all the boxes and number it

def boxes(image):
    ret, contour, heirarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    from imutils import contours
    (contours,_) = contours.sort_contours(contour)

    boxes = []
    for c in contours:
        if cv2.contourArea(c) > 3000:
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append((x, y, w, h))

    return boxes



while (True):
    ret,frame = camera.read()
    cv2.imshow("original",frame)

    gray_image = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    cv2.imshow("Gray_image",gray_image)

    #we are gonna threshold the image
    ret,threshold_image = cv2.threshold(gray_image,155,255,cv2.THRESH_BINARY_INV)
    cv2.imshow("thresh",threshold_image)
    edge_image = cv2.Canny(threshold_image, 155, 255)
    cv2.imshow("Edge_image", edge_image)
    image= boxes(edge_image)
    print(len(image))
    if len(image)>0:
        #print(image[0])
        images = np.array(image[0])
        rectangle= cv2.rectangle(frame,(images[0],images[1]),(images[0] + images[2],images[1]+images[3]),(0,255,0))
        cv2.imshow("rectangle",rectangle)
        print(images)
        if len(image) > 1:
            images1 = np.array(image[1])
            rectangle = cv2.rectangle(frame,(images1[0],images1[1]),(images1[0]+images1[2], images1[1]+images[3]),(255,0,0))
            cv2.imshow("2nd rectangle",rectangle)

   # print(image)


    #image_with_box = cv2.drawContours(threshold_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
   # cv2.imshow("box_image",image_with_box)

  #  print(image)





        #print(len(contours))





    #print(len(contour_array))
       # print(c)

       #

       # print(x,y,w,h)
        #


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()

