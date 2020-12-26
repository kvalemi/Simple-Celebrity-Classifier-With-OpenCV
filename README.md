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

I used a pre-built model called haar cascades to obtain the celebrities faces from the training dataset, and then used an OpenCV built-in algorithm called LBPHFaceRecognizer_create() for training the facial recognition model. This is NOT a very good classifier given that it is something I hacked together in about an hour. Despite it's shortcomings it actually works the majority of the time. See below for some examples of correct classifications, ran on the validation dataset:

![](/Classification%20Examples/Ben%20Afflek.png)
![](/Classification%20Examples/Elton%20John.png)
![](/Classification%20Examples/Jerry%20Sienfeild.png)
![](/Classification%20Examples/Madona.png)
![](/Classification%20Examples/Mindy%20Kaling.png)

## How to Build the Project


