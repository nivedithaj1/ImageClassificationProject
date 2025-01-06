# ImageClassificationProject
Product Quality checker for E –Grocery using Image Classification

#Objective
To develop an automated system for quality evaluation and grading of e-grocery items (fruits & vegetables) Using onboard resources such as camera or from collected images.

#Data Preparation
Data  collected for the grocery items (fruits and vegetables). Due to covid situation images were collected from online source.  
The collected data is then Labelled into good, average and bad.
Then the data is cleaned (low resolution and pixelated images are removed )
Now the data is split into train and test and only the trained data is Augmented.
The augmented data is then fed to the model for evaluation.

#Model Building
The model building was done using Teachable Machine, TensorFlow Lite Model Maker and other pretrained models.
The models were saved and then uploaded into server for prediction and the performance was tested 
Finally we have concluded TensorFlow Lite Model Maker as it gives better performance than the teachable machine and other pre-trained models.

#Future Scope
The project helps in smooth purchase and cost efficient in the field of e-commerce.
Increases the client customer relationship 
The predicted/classified data is stored and used for retraining of the model, which increases the efficiency and performance of the model.
The model can easily be developed into a mobile application by using other servers like AWS ,Microsoft Azure.
This project helps the farmer in achieving the good price for his items.

#Challenges
Data collection -: The collection ,labelling and cleaning data was a important task and took more time 
Model size –: The model had to be lite in order to support a mobile application
Selecting the model which can give better performance with low size took lot of research and work.
ICT equipment [Needed software’s to be installed.]
For creating this kind of algorithm, we need GPU machine, but they are costly so we used google-colab on our local machines.


<img width="813" alt="image" src="https://github.com/user-attachments/assets/0542d46b-ba2d-4871-a682-38831e9842c8" />

