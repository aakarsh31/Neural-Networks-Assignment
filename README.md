# Neural-Networks-Assignment
A simple assignment for my Neural Networks class involving training two different CNNs for the MNIST and CIFAR-10 datasets

The following Assignments require us to Create two CNN (Convolutional Neural Network) Architectures for the following Datasets.

•	MNIST – A database of handwritten digits with labels from 0 to 9 (10 classes) each with a size 28x28.

  ![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/1a521baf-ec61-4d50-a219-0fa1f6f0d27f)


•	CIFAR-10 – A computer-vision-based dataset with 10 different objects (car, airplane, cat, etc,) with each image of size 32x32.

 ![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/43394bdf-517e-4118-8de3-9bcb9552d50c)

Both these datasets demand a multiclass neural network that can correctly identify among the 10 classes.


Let us begin deconstructing our neural Networks.


MNIST
Data collection
1.	For this dataset we first extract the images from the tf.keras.datasets.
2.	First, we split the dataset into training and validation sets in the ratio 6:1.
3.	Followed by reshaping the images and scaling the pixel values as float values between 0 and 1.
4.	Then we do one hot encoding for our labels for both training and Validation.
We can visualize the first few images as follows.


![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/550ee9d3-1b75-4f91-a5f1-1d8be0cc8988)




 





CNN ARCHITECTURE


![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/1bbf313e-568d-4f2a-86a9-72d04bb40d7e)

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

![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/6c5de5ad-f496-4b1f-a7ef-767f7cb784bc)


 

TESTING
 
•	As you can see in the above code cell, we use the following code to upload a given image and ask our model to classify it.
•	We have taken handwritten images of 9,3 and 4 and observed that our model had correctly classified the images thus confirming that our model works as intended.

A few examples of testing are as follows-

  ![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/d10c931d-21c2-47ed-b184-9463f13e65f9)

  ![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/b5fe5b59-567f-4ffb-90c2-44869222904d)

  ![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/d4de7220-fbef-45c2-9467-acc496258e70)

  ![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/ec5d2da3-b109-4726-ace8-025d1a0f1dc8)

  ![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/dce8500e-b38d-47e0-a793-c5e783c1266d)
  
  ![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/182f214d-6ebd-419c-afd7-0bd11ffb8a4a)




 

 
 
 
 









CIFAR-10
Data Collection
1.	Compared to MNIST, this dataset is relatively more complex.
2.	These are coloured images of different objects each pertaining to a class; hence a need has come for a more complex architecture.
3.	The data is divided into 50,000 training and 10,000 validation images (5:1).
4.	This is where we also introduce Data Augmentation as another means to combat overfitting and train our model on more data.
5.	Like MNIST, we load our data from Keras and perform data scaling and one hot encoding for the images and labels respectively.

We can visualize the first few images as follows:

 

![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/971cb74b-bafb-438b-9e11-b036af1e953f)














CNN ARCHITECTURE

![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/92fcc2a0-253f-44d1-8c2e-9341044f11e4)

 
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


![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/2f9d91e8-1aac-48f0-ae11-eba09c353b40)

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


 ![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/23f9c849-0250-49cd-9845-1245311bd5de)






TESTING
We use the following code to verify our model for a given input-

![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/22615fb6-fed4-4725-9e7b-b2c5db7e674c)



 





We test our code with the following inputs-
1.

 ![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/e54d10ce-f3d1-4b1b-98b4-2be6b6271a6c)
![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/d0572c8f-272b-4d43-a8be-fc6a872d998b)



 
2.
 ![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/e64c831e-62de-4d6f-a4ac-b0baeb319efb)
![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/68c45331-3da6-4641-bb3d-661c7fc1f1fa)


 
3.
 ![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/8de8ab3a-d89c-4c1c-8cf7-f3447c89ab18)
![image](https://github.com/aakarsh31/Neural-Networks-Assignment/assets/89195418/bb03dfe9-6dbb-46e8-be4c-808fda243a60)


