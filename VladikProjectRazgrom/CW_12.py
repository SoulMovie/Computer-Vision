import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.python.layers.normalization import normalization

from CW_11 import history

train_ds = tf.keras.preprocessing.image_dataset_from_directory('data/train', image_size=(128, 128), batch_size=30, label_mode='categorical')

test_ds = tf.keras.preprocessing.image_dataset_from_directory('data/test', image_size=(128, 128), batch_size=30, label_mode='categorical')

normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, validation_data=test_ds, epochs=50)

test_loss, test_acc = model.evaluate(test_ds)
print(f'Pravdivost: {test_acc}')

class_name = ["cars", "cats", "dogs"]

img = image.load_img("images/img.png", target_size=(128, 128))
x = image.img_to_array(img)

image_array = image.img_to_array(img)
image_array = image_array / 255.0
image_array = np.expand_dims(image_array, axis=0)
predictions = model.predict(image_array)
predict_index = np.argmax(predictions[0])


print(f'Imovirnost po klassam: {predictions[0]}')
print(f'Imovirnost viznachilla: {class_name[predict_index]}')