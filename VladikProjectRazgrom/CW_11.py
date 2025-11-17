import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

#vidkrivaem tablitsu
df = pd.read_csv('data/figures.csv')
print(df.head())

#rozshifrovuem
encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label_enc'])

#chitaem model
X = df[["area", "perimeter", "corners"]] #oznaki
y = df["label_enc"] #mitki

#stvoruyem model
model = keras.Sequential([
    layers.Dense(8, activation = "relu", input_shape = (3,)),
    layers.Dense(8, activation = "relu"),
    layers.Dense(8, activation = "softmax"),
])

#compilaciya modeli
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])


history = model.fit(X, y, epochs = 300, verbose = 0)


plt.plot(history.history['loss'], label = "loss")
plt.plot(history.history['accuracy'], label = "accuracy")
plt.xlabel('epoha')
plt.ylabel('znachennya')
plt.title('process navchannya')
plt.legend()
plt.show()

test = np.array([[25, 20, 0]])
pred = model.predict(test)

print(f'imovirnist: {pred}')
print(f'model skazala: {encoder.inverse_transform([np.argmax(pred)])}')