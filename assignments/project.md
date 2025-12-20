[<img width=900 src="../img/title.png?raw=yes">](../README.md)   
[Syllabus](../README.md) |
[Slides and Assignments](README.md) |
[Project](project.md) |
[Lecturer](http://zhe-yu.github.io) 

Your framework assignment is to implement a CNN and train it using CIFAR10 data in [TensorFlow](https://www.tensorflow.org/install/pip).

### Project Part 1

For first steps and to get an introduction on how to easily import the CIFAR10 data, please visit this tutorial:

[CIFAR-10 Image Classification in TensorFlow](https://www.geeksforgeeks.org/deep-learning/cifar-10-image-classification-in-tensorflow/)

After going through the tutorial, you will need to utilize what you learned to complete the CNN_model class in [project/tf_cifar10.py](project/tf_cifar10.py). Note that your input would be 3x32x32 (three channels of 32x32 pixels):

part 1 (note each of these feature maps will be 32x32 until the max pool):

- 2D convolutional layer, padding 1, kernel size 3, with 9 output feature maps
- ReLU on the outputs
- batch normalization
- 2D convolutional layer, padding 1, kernel size 3, with 9 output feature maps
- ReLU on the outputs
- batch normalization
- 2D max pool with a pool size of 2 and a stride of 2

part 2 (note each of these feature maps will be 16x16 until the max pool):

- 2D convolutional layer, padding 1, kernel size 3, with 18 output feature maps
- ReLU on the outputs
- batch normalization
- 2D convolutional layer, padding 1, kernel size 3, with 18 output feature maps
- ReLU on the outputs
- batch normalization
- 2D max pool with a pool size of 2 and a stride of 2


part 3 (note each of these feature maps will be 8x8 until the max pool):

- 2D convolutional layer, padding 1, kernel size 3, with 36 output feature maps
- ReLU on the outputs
- batch normalization
- 2D convolutional layer, padding 1, kernel size 3, with 36 output feature maps
- ReLU on the outputs
- batch normalization
- 2D max pool with a pool size of 2 and a stride of 2


part 4 (dense layers, after the flatten these will be 1xY where Y is the number of feature maps):

- Flatten (this should result in 576 single values as the input layer would be 36x4x4 after the max pool)
- Linear with 100 outputs
- ReLU
- Dropout (with probability 50%)
- Linear with 10 outputs (this is your final output layer)
- Softmax

The model should look like this:
```
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 32, 32, 3)]       0

 conv2d (Conv2D)             (None, 32, 32, 9)         252

 batch_normalization (Batch  (None, 32, 32, 9)         36
 Normalization)

 conv2d_1 (Conv2D)           (None, 32, 32, 9)         738

 batch_normalization_1 (Bat  (None, 32, 32, 9)         36
 chNormalization)

 max_pooling2d (MaxPooling2  (None, 16, 16, 9)         0
 D)

 conv2d_2 (Conv2D)           (None, 16, 16, 18)        1476

 batch_normalization_2 (Bat  (None, 16, 16, 18)        72
 chNormalization)

 conv2d_3 (Conv2D)           (None, 16, 16, 18)        2934

 batch_normalization_3 (Bat  (None, 16, 16, 18)        72
 chNormalization)

 max_pooling2d_1 (MaxPoolin  (None, 8, 8, 18)          0
 g2D)

 conv2d_4 (Conv2D)           (None, 8, 8, 36)          5868

 batch_normalization_4 (Bat  (None, 8, 8, 36)          144
 chNormalization)

 conv2d_5 (Conv2D)           (None, 8, 8, 36)          11700

 batch_normalization_5 (Bat  (None, 8, 8, 36)          144
 chNormalization)

 max_pooling2d_2 (MaxPoolin  (None, 4, 4, 36)          0
 g2D)

 flatten (Flatten)           (None, 576)               0

 dense (Dense)               (None, 100)               57700

 dropout (Dropout)           (None, 100)               0

 dense_1 (Dense)             (None, 10)                1010

=================================================================
Total params: 82182 (321.02 KB)
Trainable params: 81930 (320.04 KB)
Non-trainable params: 252 (1008.00 Byte)
_________________________________________________________________

```

The following will help with adding in particular layers:

[https://www.tensorflow.org/api_docs/python/tf/keras/layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers)

Train this model with optimizer='adam', loss='sparse_categorical_crossentropy', batch_size=128 for 50 epochs. 
 - Save a screenshot of the output as _project/screenshot.png_.
 - Save the model weights to _project/checkpoint/weights.keras_. Rename it as _weights_P1.keras_ to keep it for examination.
 - Make sure you can test the saved model using the [project/test.py](project/test.py) file.
 - Also make sure the training and validation accuracies are recorded in _project/training.png_. Rename it as _training_P1.png_ to keep it for examination.

### Project Part 2

You can try whatever you want to train a better model (changing the hyperparameters, changing the model architectural, etc.). The goal is to get as high accuracy on the test set as possible. You can use generative AI to assist you in this process but eventually, you will need to make sure you train the model with [project/tf_cifar10.py](project/tf_cifar10.py) without data leakage (cannot use any information from the test set to train/tune the model) and that [project/test.py](project/test.py) can be used to evaluate the accuracy on the test set of your saved model.

### Grading Rubric 20 points in total

- 10 points if
  + _project/screenshot.png_ looks right.
  + _training_P1.png_ looks right.
  + accuracy with _weights_P1.keras_ looks right.
- 5 points if
  + A higher accuracy than the Part 1 model is achieved by the Part 2 model by testing with [project/test.py](project/test.py).
  + Part 2 model runs without error.
- 5 points for
  + How high your test accuracy is when compared to the other students in class.
