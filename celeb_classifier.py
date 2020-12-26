import numpy as np 
import cv2 as cv
import sys

# Define celebrity names (classification labels)
people = [
"anne_hathaway",
"dwayne_johnson",
"kate_beckinsale",	
"madonna",			
"sofia_vergara",
"arnold_schwarzenegger",
"elton_john",		
"keanu_reeves",		
"mindy_kaling",		
"will_smith",
"ben_afflek",		
"jerry_seinfeld",		
"lauren_cohan",		
"simon_pegg"]

# import facial recognition model
haar_cascade = cv.CascadeClassifier('./haarcascade_face.xml')

# import celeb recognition model
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('model_specs.yaml')

# read in inputted picture and then turn into grayscale
img = cv.imread(sys.argv[1])
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# scan for face in inputted picture
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

# iterate over the face
for(x,y,w,h) in faces_rect:

	# zoom in on only the face
	faces_roi = gray[y:y+h, x:x+w]

	label, confidence = face_recognizer.predict(faces_roi)

	print(f'--> Predicting individual is -- {label} --, with a confidence of {confidence}')

	cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,0, 255), thickness = 2)

	cv.rectangle(img, (x,y), (x+w, y+h), (0,0,255), thickness = 1)

cv.imshow('Detected Face', img)

cv.waitKey(0)
