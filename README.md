# Face-recognition-and-identification

This is a practice project of mine to get working understanding of CNNs and it's capabilities in face identification by using transfer learning.

Model used here is VGG-Face model which is created with the [weights](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/ "VGG Face Descriptor") contributed by the authors of the paper [Deep Face Recognition](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf "Paper") 

## Files
### avg_face_enc_gen.py
This program promts for a name to save the encodings and takes in the camera feed and identifies the face using [Haar Cascade Classifiers](https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html "Face detection using Haar cascades") and VGG-Face model is loaded and the program captures 40 pictures of your face and each face is separately encoded with the model and averaged with all fourty encodings of the face.

### face_recognition.py
This simply works like a camera with feature of identifying the people in the camera if they have already saved their encodings using the previous program.

#### Note
Make sure you download the haarcascade_frontalface_alt.xml file along with the model from the this [link](https://drive.google.com/drive/folders/1-n-_uJISARomEVbSLCacItyl1oImzCTq?usp=sharing)
