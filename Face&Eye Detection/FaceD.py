import cv2 as cv

cap = cv.VideoCapture(0)


#XML file path
haar_Cascade = cv.CascadeClassifier('haar_face.xml')

while (True):

    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #Use Haar_face fonc.
    face = haar_Cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

    #Take the sizes and draw.
    for (x,y,w,h) in face:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),thickness=2)

    frame= cv.flip(frame,1)

    cv.imshow('me',frame)

    if cv.waitKey(1)&0xFF == ord('q'):
        break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv.destroyAllWindows()