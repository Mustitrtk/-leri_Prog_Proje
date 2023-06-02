#https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md

import cv2 as cv
import mediapipe as mp
import time
import ElTakipTanimaModul as ettm
import sys

ptime =0
ctime =0

cap = cv.VideoCapture(0) #video yakalama opencv komutu

detector = ettm.Detector() #el takip tanıma modülünü çağırdık

tipId=[4,8,12,16,20] #Parmak id lerimiz

while True:
    isTrue, frame = cap.read() #frame bizim görüntümüz

    frame = cv.flip(frame,1) #aynaladık
    
    frame = detector.findHands(frame,draw=True) #eli tanıdı istersen eli çizdirebiliriz

    lmlist = detector.findPosition(frame, draw=False) #parmak id konumlarımızı tutan komut

    if len(lmlist) !=0: 

        fingers=[]

        #Baş Parmak için (istisna)
        if lmlist[tipId[0]][1] < lmlist[tipId[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1,5):
            if lmlist[tipId[id]][2] < lmlist[tipId[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        totalf = sum(fingers)

        cv.putText(frame, str(totalf), (30,200), cv.FONT_HERSHEY_COMPLEX_SMALL, 3, (0,0,255), 2)   


    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime=ctime

    cv.putText(frame, str(int(fps)), (30,60), cv.FONT_HERSHEY_COMPLEX_SMALL, 3, (255,0,255), 2)

    cv.imshow("frame",frame)

    if cv.waitKey(1)  & 0xFF == ord('q'):
        break

cap.release() #yazılımı ve sistemi yeniler
cv.destroyAllWindows()