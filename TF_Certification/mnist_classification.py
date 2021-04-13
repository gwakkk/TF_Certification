import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint


mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_valid, y_valid) = mnist.load_data()
x_train = x_train / 255.0
x_valid = x_valid / 255.0

model = Sequential([
                    Flatten(input_shape=(28,28)),
                    Dense(1024,activation='relu'),
                    Dense(512,activation='relu'),
                    Dense(256,activation='relu'),
                    Dense(10,activation='softmax'),
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])

checkpoint_path="my_check1.ckpt"
checkpoint=ModelCheckpoint(filepath=checkpoint_path,
                           monitor='val_loss',
                           verbose=1,
                           save_best_only=True,
                           save_weights_only=True
                           )
history=model.fit(x_train,y_train,
              validation_data=(x_valid,y_valid),
              epochs=20,
              callbacks=[checkpoint]
               )
model.load_weights(checkpoint_path)


