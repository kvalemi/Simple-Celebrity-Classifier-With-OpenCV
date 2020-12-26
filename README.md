## Project Description

In this project, I built a very simple celebrity classifier with OpenCV. I obtained a dataset of images which includes a single celrbtiy from the following list:

- anne_hathaway
- dwayne_johnson
- kate_beckinsale	
- madonna			
- sofia_vergara
- arnold_schwarzenegger
- elton_john		
- keanu_reeves		
- mindy_kaling		
- will_smith
- ben_afflek		
- jerry_seinfeld		
- lauren_cohan		
- simon_pegg

I used a pre-built model called haar cascades to obtain the celebrities faces from the training dataset, and then used a built-in OpenCV algorithm called LBPHFaceRecognizer_create() for training the facial recognition model. Note, this is NOT a very good classifier given that it is something I hacked together in about an hour. Despite it's shortcomings it actually works the majority of the time. See below for some examples of correct classifications, ran on the validation dataset:

![](/Classification%20Examples/Ben%20Afflek.png)
![](/Classification%20Examples/Elton%20John.png)
![](/Classification%20Examples/Jerry%20Sienfeild.png)
![](/Classification%20Examples/Madona.png)
![](/Classification%20Examples/Mindy%20Kaling.png)

## How to Build the Project

1) Unzip the dataset `data.zip`.

2) Train the classifier by executing the following command: `python3 train_model.py`.
(Note this step isn't neccessary if you are cloning this repo given that I included the XML file that contains the specifications of the trained model)

3) Run the classifier on the validation examples by inputting the relative path of a picture. See below for an example:

`python3 celeb_classifier.py "./validation_dataset/arnold_schwarzenegger/800px-Arnold_Schwarzenegger_(33730957348).jpg"`

## Credits

- Haar Cascades Model XML Obtained From: https://github.com/opencv/opencv/tree/master/data/haarcascades

- Dataset Obtained From: https://www.kaggle.com/danupnelson/14-celebrity-faces-dataset?select=14-celebrity-faces-dataset.zip
