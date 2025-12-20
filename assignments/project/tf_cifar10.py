import tensorflow as tf

# Display the version
print(tf.__version__)

from collections import Counter
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model



class CNN_model:
    def __init__(self, input_shape = (32, 32, 3)):

        # TODO: implement the model architectural below

        # model description
        self.model.summary()

        # Compile
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    def fit(self, x_train, y_train, val_split = 0.2):
        # Fit
        self.r = self.model.fit(x_train, y_train, batch_size=128, validation_split = val_split, epochs=50)

    def plot(self):
        # Plot accuracy per iteration
        plt.plot(self.r.history['accuracy'], label='acc', color='red')
        plt.plot(self.r.history['val_accuracy'], label='val_acc', color='green')
        plt.legend()
        plt.savefig('training.png')

    def predict(self, x_test):
        # Predict on test data
        return tf.argmax(self.model(x_test),1).numpy()

    def save(self):
        self.model.save_weights('checkpoint/weights.keras')

    def load(self):
        self.model.load_weights('checkpoint/weights.keras')

def accuracy(labels, preds):
    # Calculate accuracy
    return float(Counter(labels == preds)[True])/len(labels)

if __name__ == "__main__":
    # Load in the data
    cifar10 = tf.keras.datasets.cifar10

    # Distribute it to train and test set
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # Reduce pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # flatten the label values
    y_train, y_test = y_train.flatten(), y_test.flatten()

    # number of classes
    K = len(set(y_train))

    # calculate total number of classes
    # for output layer
    print("number of classes:", K)

    model = CNN_model()
    model.fit(x_train, y_train)
    model.plot()
    model.save()
    preds = model.predict(x_test)
    acc_test = accuracy(y_test, preds)
    print("Accuracy on test set: %.2f" %acc_test)