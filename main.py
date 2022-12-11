from keras.models import load_model
import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# model = load_model('model-003.model')
model = load_model('moodel.model')

face_clf = cv2.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')
labels_dict={0:'mask',1:'no mask'}
color_dict={0:(0,255,0),1:(0,0,255)}

def convart(x):
    if x>=0.5:
        return 1
    else:
        return 0


def video():
    cap = cv2.VideoCapture(0)

    

    while True:
        success , img = cap.read()
        img = cv2.resize(img,(512,512))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_clf.detectMultiScale(gray,1.3,5) 
        
            # print(faces)

        

        for (x,y,w,h) in faces:
            
            face_img=gray[y:y+w,x:x+w]
            resized=cv2.resize(face_img,(100,100))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,100,100,1)) #reshape to 4D
            result=model.predict(reshaped)
            # label=np.argmax(result,axis=1)[0]

            label = convart(result[0])

            cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2) #for bounding box
            cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1) #for closed or filled rectangle on top of bounding box
            cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        cv2.imshow('mask_video',img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def image():
    img = cv2.imread('mask.jpg')
    img = cv2.resize(img,(512,512))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = face_clf.detectMultiScale(gray,1.1, 9) 
    print(faces)

    for (x,y,w,h) in faces:

        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1)) #reshape to 4D

        # result=model.predict(reshaped)
        result2 = model.predict(reshaped)
        
        # label=np.argmax(result,axis=1)[0]
        label2 = convart(result2[0])
        # print(result,label,result2,label2)

        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label2],2) #for bounding box
        cv2.rectangle(img,(x,y-30),(x+w,y),color_dict[label2],-1) #for closed or filled rectangle on top of bounding box
        cv2.putText(img, labels_dict[label2], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    cv2.imshow('mask',img)
    cv2.waitKey(0)

# image()
video()
# Importing OpenCV package

