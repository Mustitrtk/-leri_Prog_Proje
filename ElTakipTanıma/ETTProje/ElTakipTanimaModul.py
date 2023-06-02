import cv2 as cv
import mediapipe as mp
import time


class Detector():
    def __init__(self, mode=False, maxHands=2, complexy=1, detection = 0.5, tracking=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexy = complexy
        self.detection = detection
        self.tracking = tracking

        self.mpHand = mp.solutions.hands    #Mediapipe hazır kütüphanesi
        self.hand = self.mpHand.Hands(self.mode, self.maxHands, self.complexy, self.detection, self.tracking) 
        self.mpDraw = mp.solutions.drawing_utils    #Tanımlanan eli işaretler


    def findHands(self, frame, draw = True): #Eli tanıma

        frameRgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #Aldığmız görüntüyü(frame) renklerini bgr dan rgb ye çeviriyoruz
        self.results = self.hand.process(frameRgb) #işlediğimiz görüntütü üzerinden işlemlerimiz yapılacak 

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame,handlms, self.mpHand.HAND_CONNECTIONS, 
                                               self.mpDraw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                                self.mpDraw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        
        return frame
    
    
    def findPosition(self, frame,draw=True):
        lmlist=[]   #mediapipe üzerinden nokta koordinatlarını almak için dizimizi oluşturduk
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                for id,lm in enumerate(handlms.landmark): #handlms.landmark enumerate
                    h,w,c = frame.shape
                    cx,cy = int(lm.x*w), int(lm.y*h)
                    lmlist.append([id,cx,cy])
                    if draw:
                        cv.putText(frame, str(id), (cx,cy), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.75 ,(0,255,255), 1)

        return lmlist

