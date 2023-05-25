# Sentiment Analisys
## Introducción
El objetivo del proyecto consiste en analizar comentarios de peliculas para saber si esta es buena, mala o neutral.
## Dataset
El dataset fue obtenido en de la libreria tensorflow_datasets
```import tensorflow_datasets as tfds

datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
train_size = info.splits["train"].num_examples
```
## Modelo

El modelo se tiene una precisión de 0.9716, esto en su época número 5.
```from tensorflow import keras

embed_size = 128
model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size,
                            input_shape=[None]),
    keras.layers.GRU(128, return_sequences=True), 
    keras.layers.GRU(128),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(train_set, epochs=5)
```

```
Epoch 1/5
782/782 [==============================] - 1709s 2s/step - loss: 0.4976 - accuracy: 0.7557
Epoch 2/5
782/782 [==============================] - 1689s 2s/step - loss: 0.2521 - accuracy: 0.8989
Epoch 3/5
782/782 [==============================] - 1705s 2s/step - loss: 0.1526 - accuracy: 0.9460
Epoch 4/5
782/782 [==============================] - 1680s 2s/step - loss: 0.1022 - accuracy: 0.9650
Epoch 5/5
782/782 [==============================] - 1692s 2s/step - loss: 0.0778 - accuracy: 0.9716


```
## Conclusiones y predicciones
El modelo puede predecir de una manera precisa, para confirmarlo haremos la prueba con las 3 etiquetas posibles.
```import numpy as np

predict_x = model.predict(table.lookup(tf.constant([b"This movie is bullshit".split()]))) 

if predict_x[0] >= 0.7:
  print('Good')
elif predict_x[0] >= 0.4:
  print('Neutral')
else:
  print('Bad')
```

```
1/1 [==============================] - 0s 26ms/step
Bad

```
```import numpy as np

predict_x = model.predict(table.lookup(tf.constant([b"THIS MOVIE IS AWSOME".split()]))) 

if predict_x[0] >= 0.7:
  print('Good')
elif predict_x[0] >= 0.4:
  print('Neutral')
else:
  print('Bad')
```

```
1/1 [==============================] - 0s 26ms/step
Good


```
```import numpy as np

predict_x = model.predict(table.lookup(tf.constant([b"The movie is fun".split()]))) 

if predict_x[0] >= 0.7:
  print('Good')
elif predict_x[0] >= 0.4:
  print('Neutral')
else:
  print('Bad')
```

```
1/1 [==============================] - 0s 25ms/step
Neutral

```
