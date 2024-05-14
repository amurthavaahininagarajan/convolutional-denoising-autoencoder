# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
Using autoencoder, we are trying to remove the noise added in the encoder part and tent to get the output which should be same as the input with minimal loss. The dataset which is used is mnist dataset.
![image](https://github.com/amurthavaahininagarajan/convolutional-denoising-autoencoder/assets/118679102/6a35cd5a-0aa2-43bc-8d3b-fbc356e1c1ae)


## Convolution Autoencoder Network Model
![image](https://github.com/amurthavaahininagarajan/convolutional-denoising-autoencoder/assets/118679102/78444f91-49f4-4400-a8fe-ed0c4327dc3b)


## DESIGN STEPS

### STEP 1:
Import the necessary libraries and dataset.
### STEP 2:
Load the dataset and scale the values for easier computation.

### STEP 3:
Add noise to the images randomly for both the train and test sets.

 ### Step 4:
Build the Neural Model using Convolutional, Pooling and Up Sampling layers. Make sure the input shape and output shape of the model are identical.

### Step 5:
Pass test data for validating manually.

### Step 6:
Plot the predictions for visualization.

## PROGRAM
```
DEVELOPED BY: AMURTHA VAAHINI.KN
REG.NO:212222240008
```
```
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()


x_train.shape


x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape) 
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

input_img = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32,(3,3),activation='relu',padding='same')(input_img)
x = layers.MaxPooling2D((2,2),padding='same')(x)
x = layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

x = layers.Conv2D(32,(3,3),activation='relu',padding='same')(encoded)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)
x = layers.UpSampling2D((2,2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)

autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train_noisy, x_train_scaled,epochs=2,batch_size=128,shuffle=True,validation_data=(x_test_noisy, x_test_scaled))

decoded_imgs = autoencoder.predict(x_test_noisy)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    

    # Display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![Screenshot 2024-05-14 214757](https://github.com/amurthavaahininagarajan/convolutional-denoising-autoencoder/assets/118679102/d956a29a-1b2d-417a-9b63-2adeaf31b570)

### MODEL SUMMARY
![Screenshot 2024-05-14 214715](https://github.com/amurthavaahininagarajan/convolutional-denoising-autoencoder/assets/118679102/053dddf2-a30a-4d68-a472-b292e5ed342f)

### Original vs Noisy Vs Reconstructed Image
![Screenshot 2024-05-14 214810](https://github.com/amurthavaahininagarajan/convolutional-denoising-autoencoder/assets/118679102/2bb6abbc-830e-4c3f-b1b5-cc427796b875)


## RESULT
Thus we have successfully developed a convolutional autoencoder for image denoising application.

