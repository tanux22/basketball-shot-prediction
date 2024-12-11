import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
import math

myColorFinder=ColorFinder(False)
hsvVals={'hmin': 0, 'smin': 116, 'vmin': 110, 'hmax': 16, 'smax': 255, 'vmax': 255}

posListX=[]
posListY=[]
Listx=[item for item in range(0,1300)]
prediction=False


cap=cv2.VideoCapture('videos/vid (4).mp4')

while True:
    success,img=cap.read()
    # img=cv2.imread("Ball.png")
    img=img[0:900, : ]
    
    #find color of the ball
    imgColor,mask=myColorFinder.update(img,hsvVals)
    #find the location of the ball
    imgContours,contours=cvzone.findContours(img,mask,minArea=200)
        
    if contours:
        posListX.append(contours[0]['center'][0])
        posListY.append(contours[0]['center'][1])
          
    if posListX:
        #Polynomial Regression(Ax^2+Bx+C)
        #find the coefficints
        A,B,C=np.polyfit(posListX,posListY,2)
        
        for i,(posX,posY) in enumerate(zip(posListX,posListY)):
            pos=(posX,posY)
            cv2.circle(imgContours,pos,10,(0,255,0),cv2.FILLED)
            if i==0:
                cv2.line(imgContours,pos,pos,(0,255,0),5)
            else:
                cv2.line(imgContours,pos,(posListX[i-1],posListY[i-1]),(0,255,0),5)
                
        for x in Listx:
            y=int(A*x**2+B*x+C)
            cv2.circle(imgContours,(x,y),2,(255,0,255),cv2.FILLED)        
    
        if len(posListX)<10:
            #prediction
            #x values=330 to 430
            #y values=590   
            a=A
            b=B
            c=C-590
            
            x=int((-b- math.sqrt(b**2-(4*a*c)))/(2*a))
            prediction=330<x<430
                    
        if prediction:
            cvzone.putTextRect(imgContours,"Basket",(50,100),scale=7,thickness=5,colorR=(0,255,0),offset=20)
        else:
            cvzone.putTextRect(imgContours," NO Basket",(50,100),scale=7,thickness=5,colorR=(0,0,255),offset=20)
    
    imgContours=cv2.resize(imgContours,(0,0),None,0.7,0.7)
    #display
    cv2.imshow("ImageColor",imgContours)  
    cv2.waitKey(100)

