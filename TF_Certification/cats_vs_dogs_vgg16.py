import tensorflow_datasets as tfds
import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16

dataset_name = 'cats_vs_dogs'
train_dataset = tfds.load(name=dataset_name, split='train[:80%]', shuffle_files=True)
valid_dataset = tfds.load(name=dataset_name, split='train[80%:]', shuffle_files=True)


def preprocess(data):
    x = data['image']
    y = data['label']
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.image.resize(x, size=(224, 224))

    return x, y



train_data = train_dataset.map(preprocess).batch(32)
valid_data = valid_dataset.map(preprocess).batch(32)
transfer_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
transfer_model.trainable = False

model = Sequential([
    transfer_model,
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

checkpoint_path = "mycheckpt.ckpt"
checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_loss'
                             )
model.fit(train_data,
          validation_data=(valid_data),
          epochs=10,
          callbacks=[checkpoint],
          )

model.load_weights(checkpoint_path)

