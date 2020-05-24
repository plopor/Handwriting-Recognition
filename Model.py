import tensorflow as tf
import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=tf.nn.softmax))
    return model

if __name__ == "__main__":
    xTrain, yTrain = loadlocal_mnist(images_path='C:/Users/bowenall/PycharmProjects/POC/train-images.idx3-ubyte',
                                     labels_path='C:/Users/bowenall/PycharmProjects/POC/train-labels.idx1-ubyte')
    xTest, yTest = loadlocal_mnist(images_path='C:/Users/bowenall/PycharmProjects/POC/t10k-images.idx3-ubyte',
                                     labels_path='C:/Users/bowenall/PycharmProjects/POC/t10k-labels.idx1-ubyte')
    # (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.mnist.load_data()
    xTrain = xTrain.reshape(60000, 28, 28)
    # print('Dimensions: %s x %s' % (xTrain.shape[0], xTrain.shape[1]))
    # print(yTrain[0])
    # print(xTrain[0].shape)
    # print(xTrain[0])
    # plt.imshow(xTrain[0], cmap='Greys')
    # plt.show()

    testGenerator = ImageDataGenerator().flow_from_directory('Img',
                                                             target_size=(28, 28),
                                                             batch_size=3410,
                                                             color_mode='grayscale',
                                                             class_mode='sparse')

    xAdd, yAdd = testGenerator.next()
    print(xAdd.shape)
    print(yAdd.shape)
    # print(xAdd[0])
    print(yAdd[0])
    print(testGenerator.class_indices)
    xAdd = xAdd.reshape(xAdd.shape[0], 28, 28)
    xAdd = 255 - xAdd
    print(xAdd[0])
    plt.imshow(xAdd[0], cmap='Greys')
    plt.show()

    xTrain = xTrain.reshape(xTrain.shape[0], 28, 28, 1)
    xTest = xTest.reshape(xTest.shape[0], 28, 28, 1)

    xTrain = xTrain.astype('float32')
    xTest = xTest.astype('float32')

    xTrain /= 255
    xTest /= 255

    #print(xTrain.shape)
    #print(xTrain[0])

    #inputShape = (28, 28, 1)
    #convolutionModel = create_model(inputShape)
    #convolutionModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #convolutionModel.fit(x=xTrain, y=yTrain, epochs=10)
    #convolutionModel.evaluate(xTest, yTest)
