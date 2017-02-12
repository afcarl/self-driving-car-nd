# Data Processing

Data quality is very important for this project more than the importance of amount of the data. I chose Udacity's original track 1 data because when I manully record driving data I could occassionally drive off the track. The technique used are as follow:
1. Remove zero angle row from the original driving log to make the distribution normal to prevent bias towards learning only driving straight line. Excessive zero labeled angle is a problem because in track one there are alot of sharp turn after the bridge. The car initially always failed the first turn and drive off the road, and even if it did passed it failed the second one immediatly. By equalizing the sample distribution, 4000 zero sample are removed, and surprisingly the model learns better in this case and keep the car always on track. I can say this is the most important part of the project.

2. Left and right camera images are randomly chosen and offset by 0.27 angle.  
3. Some common image augmentation idea are borrowed from others when reading online post, this includes RGB to HSV color sapce transformation to learn different road color and lighting condition, but as it turns out these technique are not essential for the success of driving on track one.
4. angles jittering is also a technique to generalised steering behaviour. it randomly offset the angle for a given image by a small amount, which makes steering smoother.
5. images are randomly flipped so the model can learn a unseen but a similar condition.
6. images are resize to 32x32 to reduce training time.

# The Architecture of the Model

The network architecture design compact and small for my samll. The reason is image is resize and cropped to 32x32 so it is not neccessary have deep and huge network to remember the details in the image. Typical convolutional neuralnetwork is used, which is summarised as follow:
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 32, 32, 3)     0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 32, 32, 32)    896         lambda_1[0][0]                   
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 32, 32, 32)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 16, 16, 32)    0           elu_1[0][0]                      
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 16, 16, 16)    4624        maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 16, 16, 16)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 8, 8, 16)      0           elu_2[0][0]                      
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 8, 8, 8)       1160        maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 8, 8, 8)       0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 4, 4, 8)       0           elu_3[0][0]                      
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 128)           0           maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 128)           0           flatten_1[0][0]                  
____________________________________________________________________________________________________
Dense0 (Dense)                   (None, 512)           66048       dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 512)           0           Dense0[0][0]                     
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 512)           0           dropout_2[0][0]                  
____________________________________________________________________________________________________
Dense1 (Dense)                   (None, 128)           65664       elu_4[0][0]                      
____________________________________________________________________________________________________
elu_5 (ELU)                      (None, 128)           0           Dense1[0][0]                     
____________________________________________________________________________________________________
Out (Dense)                      (None, 1)             129         elu_5[0][0]                      
====================================================================================================
Total params: 138521
____________________________________________________________________________________________________

Conv -> Activation -> Pool pattern is repeated three times to extract necessary features (two lines actually), and using two dense layers before the output. I also tried the model with only two convolution layers 32x3x3 and 16x3x3, which also works well with samples_per_epoch set to 20000. I added 8x8 layer later on and increase the sample size to 30000 and find out it slightly reduce the final loss and seems to drive more smoothly.
Subsampling is only made in pooling layers. As activation function, ELU is used to overcome "dying ReLU" problem. Using dropout in just after the first dense layer (with more weights!) was enough to prevent overfitting.

# Training
The training and validation dataset are split with 0.1 ratio. There are 3632 training samples, batch size is set to 100 and samples_per_epoch is 30000, epoch is set to 4. The reason for larger samples_per_epoch is that even though I only have 3632 original training data, since data are generated on the fly randomly by generator so that I can train the model with more unique data generated per epoch.  
Initially I did not use generator correctly and preload tons of manipulated image data on the memory, it took a very long time before it starts training, which is frustrating. By using fit_generator function in Keras, it became easy. The batches of 100 images are generated from the original Udacity data after removing 4000 zero samples with data augmentation techniques, and it is repeated 30 times per epoch. Training for 4 epochs was enough for a smooth driving. What I found the best driving result is obtained not when the loss is minimum, which is a little bit wired, but I guess in my case smallest loss means the model is overfitting rather than generalizing for the untested data. After 4 epochs, The final loss: loss: 0.0292 - val_loss: 0.0016 which enables me to drive along the track with speed 2 for hours without problem.
