import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from os import listdir
from os.path import isfile, join


model = tf.keras.models.load_model("vgg_face_recognition.model")
vgg_face_descriptor = tf.keras.Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

enc_path = "encodings/"
user_encodings_dir = [f for f in listdir(enc_path) if isfile(join(enc_path,f))]
names = []
user_encodings = []

for file_name in user_encodings_dir:
	user_encodings.append(np.load(join(enc_path,file_name)))
	names.append(file_name.split("_")[0])

required_size=(224, 224)
face_cascade_name = "haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier()
face_cascade.load(face_cascade_name)


def encoder_return(face_reg):
    face = cv2.resize(face_reg, required_size)
    face = face.reshape(1,224,224,3)
    face = face/255.0
    face_representation = vgg_face_descriptor.predict(face)[0,:]
    return face_representation

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def runtime_compare(live_enc):
	min_choice = float("inf")
	for i,encs in enumerate(user_encodings):
		similarity = findCosineDistance(live_enc, encs)
		if similarity < min_choice:
			min_choice = similarity
			id = names[i]
			confidence = "  {0}%".format(100 - round(similarity * 100))

	if min_choice <= epsilon:
		return id, confidence
	else:
		id = "unknown"
		confidence = "  {0}%".format(100 - round(similarity * 100))
		return id, confidence


epsilon = 0.40

font = cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(0)
try:
    while(True):
        ret, img = cam.read()
        frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)
        faces = face_cascade.detectMultiScale(frame_gray)
    
        for x1,y1,w,h in faces:
            x2, y2 = x1 + w, y1 + h
            live_enc = encoder_return(img[y1:y2,x1:x2])

            id, confidence= runtime_compare(live_enc)              
            img = cv2.rectangle(img,(x1,y1), (x2, y2), (255,0,0), 2)
            cv2.putText(
                        img, 
                        str(id), 
                        (x1+5,y1-5), 
                        font, 
                        1, 
                        (255,255,255), 
                        2
                       )
            cv2.putText(
                        img, 
                        str(confidence), 
                        (x1+5,y1+h-5), 
                        font, 
                        1, 
                        (255,255,0), 
                        1
                       ) 
        
        cv2.imshow('camera',img) 
        if cv2.waitKey(20) & 0xFF == ord('q'): # Press 'q' for exiting video
        	break

except Exception as e:             
    # Do a bit of cleanup
    print(e)

finally:
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

