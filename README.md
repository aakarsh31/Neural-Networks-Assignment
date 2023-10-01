# Neural-Networks-Assignment
A simple assignment for my Neural Networks class involving training two different CNNs for the MNIST and CIFAR-10 datasets

The following Assignments require us to Create two CNN (Convolutional Neural Network) Architectures for the following Datasets.

•	MNIST – A database of handwritten digits with labels from 0 to 9 (10 classes) each with a size 28x28.

![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/b6ff077b-87bd-4866-9b39-2f82276869a0)


•	CIFAR-10 – A computer-vision-based dataset with 10 different objects (car, airplane, cat, etc,) with each image of size 32x32.

![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/0563ae7d-6115-4973-a8e0-3e7d5e723f02)

Both these datasets demand a multiclass neural network that can correctly identify among the 10 classes.


Let us begin deconstructing our neural Networks.


MNIST
Data collection
1.	For this dataset we first extract the images from the tf.keras.datasets.
2.	First, we split the dataset into training and validation sets in the ratio 6:1.
3.	Followed by reshaping the images and scaling the pixel values as float values between 0 and 1.
4.	Then we do one hot encoding for our labels for both training and Validation.
We can visualize the first few images as follows.


![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/92e1ec1c-e81c-4e83-8902-b4dc5a90f0b8)




 





CNN ARCHITECTURE


![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/2274b3db-70a8-4dc8-afbc-d87751407f8a)

Our CNN Architecture is as follows- 
•	Here we have 32 3x3 filters followed by BatchNormalization, 2x2 MaxPooling and a Dropout Layer. (Conv. Is described by yellow colour).
•	Then we repeat the above layer again which is then followed by another layer of 64 3x3 filters twice with the above regularization and Dropout methods. (BatchNormalization is Red; Max pooling is Green and Dropout is Blue).
•	We then flatten our CNN and then create a 256-node dense Layer.
•	This corresponds to our final layer which is a Dense Layer containing 10 nodes where each node corresponds to one of the classes that we want to identify.
•	Note that we need to make sure that the inputs are of size 28x28.
•	We also set padding to the same to keep the dimensions the same after convolution.




ADDITIONAL PARAMETERS
•	In the architecture itself we have used BatchNormalization and Dropouts consistently after each convolutional layer to tackle overfitting.
•	We set Dropout values to 0.1 as we found it gave us the best results when it came to the trade-off between accuracy and generalizing the model.
•	We use the RMSprop as compared to Adam as we found this helps the model converge better and also the fact that it dynamically adjusts its learning rate.
•	We took the batch sizes as 64 and 32 for training and validation sets respectively since the model trains relatively quickly, hence we could afford smaller sizes to get a better accuracy.
•	We use “categorical_crossentropy” as our loss function as this is standard when dealing with multi-class problems.
•	Our model converges around 30 epochs with validation accuracy being consistently close to our Training accuracy and slightly less, so we stopped with our training at said number of epochs.
•	So, we have concluded that our model has trained well without overfitting to our data with around 99.7% training accuracy and 99.21% Validation Accuracy.










Each model layer is defined using model.summary()

![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/67d7b61f-3337-44a6-bd4d-bc2e0b38e2a0)


 

TESTING
 
•	As you can see in the above code cell, we use the following code to upload a given image and ask our model to classify it.
•	We have taken handwritten images of 9,3 and 4 and observed that our model had correctly classified the images thus confirming that our model works as intended.

A few examples of testing are as follows-

![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/2b96b775-0ad1-46f5-a94a-f5d98a7fa3c6)

![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/1c8b8294-dc5c-4480-b9c8-44dd1d4780bc)

![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/979d7665-8eff-4da6-8f2c-8cd109e159d9)

![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/69ec83e1-160b-4e73-98aa-d01d93230079)

![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/797390ff-cf60-4a89-8cad-7aa0198f9d63)
  
![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/b1c79fb4-cf83-4cc8-a0e7-3a0b0975ec7e)




 

 
 
 
 









CIFAR-10
Data Collection
1.	Compared to MNIST, this dataset is relatively more complex.
2.	These are coloured images of different objects each pertaining to a class; hence a need has come for a more complex architecture.
3.	The data is divided into 50,000 training and 10,000 validation images (5:1).
4.	This is where we also introduce Data Augmentation as another means to combat overfitting and train our model on more data.
5.	Like MNIST, we load our data from Keras and perform data scaling and one hot encoding for the images and labels respectively.

We can visualize the first few images as follows:

 

![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/c3d8e718-38f3-4694-8a4f-c5e99b91d91c)














CNN ARCHITECTURE

![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/583c3cc6-b405-494d-8519-ca95672d857b)

 
•	Here, our architecture is much more complex than the previous one where we apply Double convolution of 32 3x3, 64 3x3, and 128 3x3 filters respectively. (Conv. Is described in yellow color).
•	Then we apply a 256 5X5 filters right before flattening.
•	After each double convolution we apply BatchNormalization followed by Max pooling of 2x2 and then applying a dropout layer.
•	We then Flatten into shape 1152 followed by a Dense layer of 256 Nodes. (BatchNormalization is Red; Max pooling is Green and Dropout is Blue).
•	The final layer is again a Dense layer of 10 nodes each pertaining to the 10 classes we want to classify.
•	Here again we keep the padding as same to preserve the size dimensions after convolution is applied.


ADDITIONAL PARAMETERS
•	In the architecture itself we have used BatchNormalization and Dropouts consistently after each convolutional layer to tackle overfitting.
•	Here we use ReLU as our activation function as it works very well for CNNs.
•	We set Dropout values to 0.2 as we found it gave us the best results when it came to the trade-off between accuracy and generalizing the model.
•	We use the Adam Optimizer as compared to RMS as we found this helps the model converge better and we made use of the fact that Adam uses an Adaptive Learning Rate to help with Our Performance.
•	We took the batch sizes as 32 and 32 for training and validation sets respectively since we wanted our model to learn accurately as we trained it for a lot of epochs and the training time for each epoch did not take much time.
•	We use “categorical_crossentropy” as our loss function as this is standard when dealing with multi-class problems.
•	Our model converges around 200 epochs with validation accuracy being consistently close to our Training accuracy and slightly less, so we stopped with our training at said number of epochs from here which we get similar values of training accuracy.
•	It is here we also introduced data augmentation to allow the model to learn slower, but more generalized and better.
•	So, we have concluded that our model has trained well without overfitting to our data with around 95.34% training accuracy and 87.54% Validation Accuracy.






DATA AUGMENTATION


![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/d66c05d9-cfd2-4dc4-9c33-58394dcd5965)

1.	This turned out to be a crucial part of developing this model as it helped us generalize the data as we were initially struggling with the problem of overfitting.
2.	We considered the fact that in terms of vision, an object when looked at from a different perspective would still be considered the same object (AN image of a car zoomed in slightly or flipped horizontally would still be considered a car)
3.	This is where data augmentation comes in to allow us to make a more generalized dataset.
4.	In our CNN, using the ImageDataGenerator, we modified the images into batches of augmented images as follows-
	Rotated the image by around 10 degrees.
	Changed the width and height of the image by a factor of 0.1.
	Slanted (0.1) the image by a factor of 0.1.
	Zoomed on the image by about 10%.
	Fill_mode here is set to “nearest” as it’s used to fill in any new pixels generated after a transformation.
	We also flip the image horizontally but not vertically.

5.	Let us consider an image and visualize its augmented versions.
 
6.	We decided to just apply augmentation slightly as we wanted a more general network just enough for our network to learn quickly and efficiently enough. 



The model layer is defined by model.summary()


![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/80981e61-cf40-440c-bd06-c0c1e604c3a2)






TESTING
We use the following code to verify our model for a given input-

![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/d2c3f417-d078-4390-833c-c1f41fb3aae4)



 





We test our code with the following inputs-
1.

![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/f699a4b9-49cf-4ef4-9201-0eb23c6e64d5)
![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/7c1e5e2c-4a7c-4821-aacd-6b456f45f255)



 
2.
![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/25bd0dbf-d50e-426b-b2ca-99ab536b8e9c)
![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/113edcef-4a28-4ff7-8caa-12f12a82a611)


 
3.
![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/bb72c319-54a6-4127-990e-1e1366598715)
![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/33b25fc0-6ce6-4718-bb38-e1b74d95db92)


