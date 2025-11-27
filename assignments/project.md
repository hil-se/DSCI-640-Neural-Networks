[<img width=900 src="../img/title.png?raw=yes">](../README.md)   
[Syllabus](../README.md) |
[Slides and Assignments](README.md) |
[Project](project.md) |
[Lecturer](http://zhe-yu.github.io) 

Your framework assignment is to implement a CNN and train it using CIFAR10 data in PyTorch.

For first steps and to get an introduction on how to easily import the CIFAR10 data, please visit this tutorial:

Deep Learning in PyTorch with CIFAR-10 dataset

After going through the tutorial and being able to train on the CIFAR-10 data using their simple linear model, you will need to update the model to be the following, noting that your input tensor is going to be 3x32x32 (three channels of 32x32 pixels):

part 1 (note each of these feature maps will be 32x32 until the max pool):

1. 2D convolutional layer, padding 1, kernel size 3, with 9 output feature maps

2. ReLU on the outputs

3. 2D convolutional layer, padding 1, kernel size 3, with 9 output feature maps

4. ReLU on the outputs

5. 2D max pool with a pool size of 2 and a stride of 2




part 2 (note each of these feature maps will be 16x16 until the max pool):

6. 2D convolutional layer, padding 1, kernel size 3, with 18 output feature maps

7. ReLU on the outputs

8. 2D convolutional layer, padding 1, kernel size 3, with 18 output feature maps

9. ReLU on the outputs

10. 2D max pool with a pool size of 2 and a stride of 2




part 3 (note each of these feature maps will be 8x8 until the max pool):

11. 2D convolutional layer, padding 1, kernel size 3, with 36 output feature maps

12. ReLU on the outputs

13. 2D convolutional layer, padding 1, kernel size 3, with 36 output feature maps

14. ReLU on the outputs

15. 2D max pool with a pool size of 2 and a stride of 2




part 4 (dense layers, after the flatten these will be 1xY where Y is the number of feature maps):

16. Flatten (this should result in 576 single values as the input layer would be 36x4x4 after the max pool)

17. Linear with 100 outputs

18. ReLU

19. Dropout (with probability 50%)

19. Linear with 10 outputs (this is your final output layer)




The following will help with adding in particular layers:

https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html

https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html

https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html




Your model should have 81678 parameters (for validation). You can use the following function to count the number of parameters in your model:

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


Train this model and report the output in an output.txt file.  You can try tweaking the number of epochs and learning rate(s), or swapping out SGD for other optimizers to see how high you can get the accuracy.




BONUS (5%): After each convolution and ReLU layer, add a batch normalization layer. Train this model and report the output in a bonus_output.txt file.




Put this code and your output file(s) in a ./framework/ directory in your gitlab repository, and push these to your repo by the deadline.
