
import tensorflow_datasets as tfds
import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

dataset_name = 'horses_or_humans'
train_dataset = tfds.load(name=dataset_name, split='train[:80%]')
valid_dataset = tfds.load(name=dataset_name, split='train[80%:]')


def preprocess(data):
    x = data['image']
    y = data['label']
    x = x / 255
    x = tf.image.resize(x, size=(300, 300))

    return x, y


train_data = train_dataset.map(preprocess).batch(32)
valid_data = valid_dataset.map(preprocess).batch(32)

model = Sequential([

    Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    MaxPooling2D(2, 2),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(32, activation='relu'),


    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
checkpoint_path = "mycheckpoint.ckpt"
checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_loss',
                             )

model.fit(train_data, validation_data=(valid_data),
          epochs=20,
          callbacks=[checkpoint]
          )

model.load_weights(checkpoint_path)
