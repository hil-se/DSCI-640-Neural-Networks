from collections import Counter
from tf_cifar10 import CNN_model
import tensorflow as tf

def accuracy(labels, preds):
    # Calculate accuracy
    return float(Counter(labels == preds)[True])/len(labels)

cifar10 = tf.keras.datasets.cifar10

# Distribute it to train and test set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Reduce pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# flatten the label values
y_train, y_test = y_train.flatten(), y_test.flatten()

model = CNN_model()
model.load()
preds = model.predict(x_test)
acc_test = accuracy(y_test, preds)
print("Accuracy on test set: %.2f" %acc_test)
