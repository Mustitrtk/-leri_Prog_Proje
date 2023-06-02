import cv2 as cv
import mediapipe as mp
import time

#video cap
capture = cv.VideoCapture(0)

#PIPE utulities
mpHand = mp.solutions.hands
hand = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils

#times
cTime=0
pTime=0

#video cap loop
while True:
    isTrue, frame = capture.read()

    frame = cv.flip(frame,1)

    #results are only uses with RGB
    frameRgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hand.process(frameRgb)
    
    #Multiple hand landmarks
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame,handlms, mpHand.HAND_CONNECTIONS)
            for id,lm in enumerate(handlms.landmark):
                h,w,c = frame.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                cv.putText(frame, str(id), (cx,cy), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.75 ,(0,255,255), 1)


    #our FPS 
    cTime = time.time()
    fps = int(1/(cTime-pTime))
    pTime=cTime

    cv.putText(frame, str(fps), (10,70), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 2)

    #Our window
    cv.imshow('frame',frame)

    #Pres 'q' to stop
    if cv.waitKey(1)  & 0xFF == ord('q'):
        break
    
#refresh 
capture.release()
cv.destroyAllWindows()