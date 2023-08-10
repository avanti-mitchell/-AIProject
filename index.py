# import the nescessary libraries
import tensorflow as tf
import matplotlib.pyplot as plt

# load the data and split the data to training set and test set

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# scale down the value of the image pixels from 0-255 to 0-1

train_images = train_images / 255.0
test_images = test_images / 255.0

#visualize the data

print(train_images.shape)
print(test_images.shape)
print(train_labels)

plt.imshow(train_images[0], cmap='gray')
plt.show

"""
 (60000, 28, 28)
(10000, 28, 28)
[5 0 4 ... 5 6 8]
<function matplotlib.pyplot.show(close=None, block=None)>
"""

# define the model
my_model = tf.keras.models.Sequential()
my_model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
my_model.add(tf.keras.layers.Dense(128, activation='relu'))
my_model.add(tf.keras.layers.Dense(10, activation='softmax'))

# compile the model

my_model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model
my_model.fit(train_images, train_labels, epochs=3)

"""
Epoch 1/3
1875/1875 [==============================] - 7s 3ms/step - loss: 2.3898 - accuracy: 0.8497
Epoch 2/3
1875/1875 [==============================] - 8s 4ms/step - loss: 0.3824 - accuracy: 0.9103
Epoch 3/3
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2780 - accuracy: 0.9294
<keras.callbacks.History at 0x789bc3a85360>
"""

# check the model fro accuracy on the test data
val_loss, val_acc = my_model.evaluate(test_images, test_labels)
print("test accuracy: "), val_acc

"""
313/313 [==============================] - 1s 2ms/step - loss: 0.2639 - accuracy: 0.9383
test accuracy: 
(None, 0.9383000135421753)
"""

# save the model for later use
my_model.save('my_mnist_model')

# load the model from file system
my_new_model = tf.keras.models.load_model('my_mnist_model')