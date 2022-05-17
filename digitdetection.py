import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')
model.fit(x_train,y_train,epochs=3)

model.save('detection.model')
model = tf.keras.models.load_model('detection.model')

x = 0
y = 9

while x < y:
    prediction = model.predict(x_test)
    predict = np.argmax(prediction[x])
    plt.title(f"This is probably: {predict}")
    plt.imshow(x_test[x], cmap=plt.cm.binary)
    plt.show()
    x += 1
