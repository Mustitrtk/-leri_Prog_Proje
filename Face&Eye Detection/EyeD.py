import cv2 as cv

cap = cv.VideoCapture(0)


#XML file path
eye_cascade = cv.CascadeClassifier('haar_eye.xml')

while (True):

    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #Use Haar_eye fonc.
    eyes = eye_cascade.detectMultiScale(gray, 1.1 , 8)

    #Take the sizes and draw.
    for (x,y,w,h) in eyes:

        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    frame= cv.flip(frame,1)

    cv.imshow('me',frame)

    if cv.waitKey(1)&0xFF == ord('q'):
        break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv.destroyAllWindows()