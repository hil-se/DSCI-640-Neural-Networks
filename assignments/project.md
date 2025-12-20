[<img width=900 src="../img/title.png?raw=yes">](../README.md)   
[Syllabus](../README.md) |
[Slides and Assignments](README.md) |
[Project](project.md) |
[Lecturer](http://zhe-yu.github.io) 

Your framework assignment is to implement a CNN and train it using CIFAR10 data in [TensorFlow](https://www.tensorflow.org/install/pip).

For first steps and to get an introduction on how to easily import the CIFAR10 data, please visit this tutorial:

[CIFAR-10 Image Classification in TensorFlow](https://www.geeksforgeeks.org/deep-learning/cifar-10-image-classification-in-tensorflow/)

After going through the tutorial and being able to train on the CIFAR-10 data using their simple linear model, you will need to update the model to be the following, noting that your input tensor is going to be 3x32x32 (three channels of 32x32 pixels):

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


The following will help with adding in particular layers:

[https://www.tensorflow.org/api_docs/python/tf/keras/layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers)

Train this model and test the model.  You can try tweaking the number of epochs and learning rate(s), or swapping out SGD for other optimizers to see how high you can get the accuracy.


Put this code and your output file(s) in a ./framework/ directory in your gitlab repository, and push these to your repo by the deadline.
