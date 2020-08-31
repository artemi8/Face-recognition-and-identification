import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2

start_time = time.time()

model = tf.keras.models.load_model("vgg_face_recognition.model")
vgg_face_descriptor = tf.keras.Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

required_size=(224, 224)
face_cascade_name = "haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier()
face_cascade.load(face_cascade_name)
enc_acumulator = np.zeros((2622,1))

def encoder_return(face_reg):
	face = cv2.resize(face_reg, required_size)
	face = face.reshape(1,224,224,3)
	face = face/255.0
	face_representation = vgg_face_descriptor.predict(face)[0,:]
	return face_representation

face_id = input('\n enter user id end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
count = 0
face_frames = []

try:
	cam = cv2.VideoCapture(0)
	while(True):
		ret, img = cam.read()
		frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		frame_gray = cv2.equalizeHist(frame_gray)
		faces = face_cascade.detectMultiScale(frame_gray)
		for x1,y1,w,h in faces:
			x2, y2 = x1 + w, y1 + h
			count += 1
			face_frames.append(img[y1:y2,x1:x2])
			img = cv2.rectangle(img,(x1,y1), (x2, y2), (255,0,0), 2)
		cv2.imshow('camera',img)
		k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
		if k == 27:
			break
		 # Take 30 face sample and stop video
		if count >= 40:
			break
except Exception as e:
	print(e)
# Do a bit of cleanup
finally:
	print("\n [INFO] Exiting Program and cleanup stuff")
	cam.release()
	cv2.destroyAllWindows()

for i,single_frame in enumerate(face_frames):
	if i == 0:
		enc_acumulator = encoder_return(single_frame).reshape(2622,1)
	else:
		temp_enc = encoder_return(single_frame).reshape(2622,1)
		enc_acumulator = np.concatenate((enc_acumulator, temp_enc), axis = 1)

averaged_encodings = np.average(enc_acumulator, axis = 1)
np.save("encodings/{}_avg_enc.npy".format(face_id), averaged_encodings)

print("Total Runtime: {}".format(time.time() - start_time))


