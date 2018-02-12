---
title: "Keras"
categories:
  - Research
tags:
  - deep learning
  - tools
  - keras
header:
  teaser: /assets/images/keras/envir.png
  overlay_image: /assets/images/keras/envir.png
  overlay_filter: 0.4
sidebar:
  nav: "dl"
author_profile: false
---

**Keras** is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.
This article is about summary and tips on Keras.

{% include toc title="Table of Contents" icon="file-text" %}

# Simple Example
The core data structure of Keras is a **model**, a way to organize layers.
The simplest type of model is the `Sequential` model, a linear stack of layers.
For more complex architectures, you should use the *Keras functional API*, which allows to build arbitrary graphs of layers.

```python
from keras.models import Sequential, Dense

# Construct `Sequential` model
model = Sequential()

# Stacking layers by `.add()`
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

# Configure its learning process with `.compile()`
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
              
# Iterate on the training data in batches
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Alternatively, you can feed batches to your model manually:
model.train_on_batch(x_batch, y_batch)

# Evaluate performance:
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

# Generate predictions on new data:
classes = model.predict(x_test, batch_size=128)
```

# References
- Web: Keras Documentation [[Link](https://keras.io/)]