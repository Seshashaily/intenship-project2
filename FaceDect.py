

#import numpy as np
#import cv2
from time import sleep

print("Labraries Updated")


detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

cap = cv2.VideoCapture(0);


while(True):
   # ret, img = cap.read()
    img = cv2.imread('4.jpg', 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('grayscale',gray)
    faces = detector.detectMultiScale(gray, 1.3, 5);
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        area = (x, y, x+w, y+h)
        

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
    cv2.resizeWindow("frame", 1024, 768)              # Resize window to specified dimensions
    cv2.imshow('frame',img)
    img1 = img
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('Output Image Store.jpg',img1)
        break
    
cap.release()
cv2.destroyAllWindows()

print("Project End")


