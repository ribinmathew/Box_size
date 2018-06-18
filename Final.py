import cv2
import numpy as np


def boxes(image):
    ret, contour, heirarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    from imutils import contours
    (contours,_) = contours.sort_contours(contour)

    boxes = []
    for c in contours:
        if cv2.contourArea(c) > 3000:
            rotating_rectangle_image = cv2.minAreaRect(c)
            box = cv2.boxPoints(rotating_rectangle_image)
            #box = np.int0(box)
            #(x, y, w, h) = cv2.boundingRect(c)
            box = np.int0(box)
            boxes.append(box)

    return boxes

camera = cv2.VideoCapture(1)
while(True):
    ret, frame = camera.read()
    cv2.imshow("original_video", frame)
    gray_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray_video", gray_video)
    ret, threshold_video = cv2.threshold(gray_video, 155, 250, cv2.THRESH_BINARY_INV)
    cv2.imshow("Thrsh_video", threshold_video)
    kernel = np.ones((5, 5), np.uint8)
    dialated_video = cv2.dilate(threshold_video, kernel, iterations=2)
    opening_video = cv2.morphologyEx(dialated_video, cv2.MORPH_CLOSE, kernel)
    edge_image1 = cv2.Canny(opening_video, 155, 255)
    cv2.imshow("open video", edge_image1)
    image = boxes(edge_image1)

    if len(image)>0:
        #print(image[0])
        for (x, y) in image[0]:
           # print(x,y)
            box_circle = cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)
            cv2.imshow("box_circle", box_circle)
        rotating_box_image = cv2.drawContours(frame, [image[0]], 0, (0, 0, 255), 2)
        cv2.imshow("box", rotating_box_image)
        X, Y = image[0][0]
        point1 = cv2.putText(frame, "A", (X, Y), cv2.FONT_ITALIC, .8, (50, 200, 100), 2, cv2.LINE_AA)
        X1, Y1 = image[0][1]
        point2 = cv2.putText(frame, "B", (X1, Y1), cv2.FONT_ITALIC, .8, (50, 200, 100), 2, cv2.LINE_AA)
        X2, Y2 = image[0][2]
        point3 = cv2.putText(frame, "C", (X2, Y2), cv2.FONT_ITALIC, .8, (50, 200, 100), 2, cv2.LINE_AA)
        X3, Y3 = image[0][3]
        point4 = cv2.putText(frame, "D", (X3, Y3), cv2.FONT_ITALIC, .8, (50, 200, 100), 2, cv2.LINE_AA)

        P1, Q1 = image[0][0]  # A
        P2, Q2 = image[0][1]  # B
        P3, Q3 = image[0][2]  # C
        P4, Q4 = image[0][3]  # D

        A_to_B = np.sqrt(np.sum((P1 - P2) ** 2 + (Q1 - Q2) ** 2))
        A_to_D = np.sqrt(np.sum((P1 - P4) ** 2 + (Q1 - Q4) ** 2))
        print("A_to_B",A_to_B)
        print("A_to_D",A_to_D)

        if A_to_B > A_to_D:
            pixel_per_metric = A_to_B / 66.91

        if A_to_D > A_to_B:
            pixel_per_metric = A_to_D / 31.45

        if A_to_B > A_to_D:
            length = A_to_B / pixel_per_metric
            bredth = A_to_D / pixel_per_metric
            print("lenght", length)
            print("bredth", bredth)
        if A_to_D > A_to_B:
            length = A_to_D / pixel_per_metric
            bredth = A_to_B / pixel_per_metric
            print("lenght", length)
            print("bredth", bredth)

        if len(image) > 1:
            for (x, y) in image[1]:
                box_circle = cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
                cv2.imshow("box_circle", box_circle)
           # print("2nd image", image[1])
            rotating_box_image1 = cv2.drawContours(frame, [image[1]], 0, (0, 255, 0), 2)
            cv2.imshow("box1", rotating_box_image1)
            x, y = image[1][0]
            point1 = cv2.putText(frame, "E", (x, y), cv2.FONT_ITALIC, .8, (0, 0, 200), 2, cv2.LINE_AA)
            x1, y1 = image[1][1]
            point2 = cv2.putText(frame, "F", (x1, y1), cv2.FONT_ITALIC, .8, (0, 0, 200), 2, cv2.LINE_AA)
            x2, y2 = image[1][2]
            point3 = cv2.putText(frame, "G", (x2, y2), cv2.FONT_ITALIC, .8, (0, 0, 200), 2, cv2.LINE_AA)
            x3, y3 = image[1][3]
            point4 = cv2.putText(frame, "H", (x3, y3), cv2.FONT_ITALIC, .8, (0, 0, 200), 2, cv2.LINE_AA)
            cv2.imshow("text", rotating_box_image1)

            Pp1, Qq1 = image[1][0]  # E
            Pp2, Qq2 = image[1][1]  # F
            Pp3, Qq3 = image[1][2]  # G
            Pp4, Qq4 = image[1][3]  # H

            Aa_to_Bb = np.sqrt(np.sum((Pp1 - Pp2) ** 2 + (Qq1 - Qq2) ** 2))
            print("Aa_to_Bb",Aa_to_Bb)
            Aa_to_Dd = np.sqrt(np.sum((Pp1 - Pp4) ** 2 + (Qq1 - Qq4) ** 2))
            print("Aa_to_Bd", Aa_to_Dd)
            print("Pixel_per_metric",pixel_per_metric)
            if Aa_to_Bb > Aa_to_Dd:

                length = Aa_to_Bb / pixel_per_metric
                bredth = Aa_to_Dd / pixel_per_metric
                print("lenght_of_the _box", length)
                print("bredth_of_the _box", bredth)
            if Aa_to_Dd > Aa_to_Bb:
                length = Aa_to_Dd / pixel_per_metric
                bredth = Aa_to_Bb / pixel_per_metric
                print("lenght_of_the _box", length)
                print("bredth_of_the _box", bredth)





















    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()