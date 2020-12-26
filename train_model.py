import cv2 as cv
import os
import numpy as np 


## Training Function ##
def create_train():

	# loop through files of people
	for person in people:
		path = os.path.join(DIR, person)
		label = people.index(person)

		# loop through pictures located in the files of each person
		for img in os.listdir(path):

			img_path = os.path.join(path, img)
			img_array = cv.imread(img_path)
			gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

			faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1)

			# obtain and store only the face of the person out of the entire picture
			for(x,y,w,h) in faces_rect:
				faces_roi = gray[y:y+h, x:x+w]
				features.append(faces_roi)
				labels.append(label)


## Script Beginning ##

# labels of the training files
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

# directory of training files
DIR = r'./training_dataset'

# import facial recognition model
haar_cascade = cv.CascadeClassifier('./haarcascade_face.xml')

features = [] # used for storage of faces
labels = []   # used for storage of the labels

# call training function
create_train()
print('-> training finished!')

# convert to numpy
features = np.array(features, dtype = 'object')
labels = np.array(labels)

# Train our own facial recognition model
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

# save the model specs for transfer learning
face_recognizer.save('model_specs.yaml')

