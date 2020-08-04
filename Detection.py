import cv2
import numpy as np
from keras.models import load_model

model = load_model("model-005.model")
results = {0:'without mask', 1:'mask'}
GR_dict = {0:(0,0,255), 1:(0,255,0)}
rect_size = 4
cap = cv2.VideoCapture(0) 

# Loading Classifier file from the cv2 library
haarcascade = cv2.CascadeClassifier('C:\\Users\\Smit\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

# Check if Classifier loaded correctly else exit
if( haarcascade.empty() ):
    print("Cannot load!")
    exit

while True:
    (rval, im) = cap.read()
    im = cv2.flip(im, 1, 1) 
    
    rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
    faces = haarcascade.detectMultiScale(rerect_size)
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f] 
        # Process image for faster and accurate predicton
        face_img     = im[y:y+h, x:x+w]
        rerect_sized = cv2.resize(face_img, (150, 150))
        normalized   = rerect_sized/255.0
        reshaped     = np.reshape(normalized, (1, 150, 150, 3))
        reshaped     = np.vstack([reshaped])
        result       = model.predict(reshaped)
        label        = np.argmax(result,axis = 1)[0]
        
        # Rectangles aroung the detected face
        cv2.rectangle(im, (x, y),(x+w, y+h), GR_dict[label], 2)
        cv2.rectangle(im, (x,y-40), (x+w,y), GR_dict[label], -1)
        
        # Writing Text
        cv2.putText(im, results[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    
    cv2.imshow('LIVE',   im)
    key = cv2.waitKey(10)
    
    # Escape Key Closes the opened window
    if key == 27: 
        break
cap.release()
cv2.destroyAllWindows()