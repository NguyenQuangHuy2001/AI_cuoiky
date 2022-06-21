import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('gender.h5')
model_age = load_model('age.h5')
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#ranges = ['1-2','3-9','10-15','16-20','21-25','26-30','31-45','46-65','66-116']
ranges = ['1-4','5-12','13-25','26-40','41-50','51-70','71-90','91-100','101-116']
gender_dict = {0:'Male', 1:'Female'}
cam = cv2.VideoCapture(0)

while 1:
    ret,frame = cam.read()
    if ret:
        faces = detector.detectMultiScale(frame,1.3,5)
        for x,y,w,h in faces:
            face = frame[y:y+h,x:x+w]
            face_gen = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            face_gen = cv2.resize(face_gen,(128,128))
            face_gen = face_gen.reshape(1,128,128,1)

            pred = model.predict(face_gen)
            pred_gender = gender_dict[round(pred[0][0])]
            
            face_age = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            face_age = cv2.resize(face_age,(200,200))
            face_age = face_age.reshape(1,200,200,1)
            age = model_age.predict(face_age)

            age_gen=pred_gender+" "+str(ranges[np.argmax(age)])

            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(frame,(x,y+h),(x+w,y+h+50),(255,0,0),-1)
            cv2.putText(frame, age_gen, (x+2, y+h+20), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 2)
        
        cv2.imshow('Live',frame)   
        
    if cv2.waitKey(1)==27:
        break

cam.release()
cv2.destroyAllWindows()