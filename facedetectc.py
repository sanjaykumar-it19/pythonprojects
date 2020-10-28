import cv2

haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cam = cv2.VideoCapture(0)

while True :

    _, img = cam.read()
    text = "No Person Detected"
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(grayImg, 1.3, 4)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        text = "Person detected"

    cv2.putText(img,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    cv2.imshow("Person detection",img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") :
        break
cam.release()
cv2.destroyAllWindows()