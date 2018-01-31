---
title: "Deep Learning Tools"
categories:
  - Research
tags:
  - deep learning
  - tools
  - tensorflow
  - keras
  - theano
header:
  teaser: /assets/images/deep learning tools/envir.png
  overlay_image: /assets/images/deep learning tools/envir.png
  overlay_filter: 0.4
sidebar:
  nav: "dl"
author_profile: false
---

Summary and tips on deep learning tools such as TensorFlow, Keras and Theano.

{% include toc title="Table of Contents" icon="file-text" %}

# TensorFlow

## The Programming Stack
TensorFlow provides a programming stack consisting of multiple API layers

![Environment]({{ site.url }}{{ site.baseurl }}/assets/images/deep learning tools/envir.png){: .align-center}

They recommend writing TensorFlow programs with the following APIs:

- **Estimator**: provides methods to train the model, to judge the model's accuracy, and to generate predictions.
  - "Estimator" is any class derived from `tf.estimator.Estimator`.
  - TensorFlow provides a collection of pre-made Estimators (for example, `LinearRegressor`).

- **Datasets**: has methods to load and manipulate data, and feed it into your model. The datasets API meshes well with the Estimators API.
  - An **input function** is a function that returns the following two-element tuple:
    - "Features": A Python dictionary in which each key is the name of a feature, each value is an array containing all of that features values.
    - "Label": An array containing the values of the label for every example.
  - Simple demonstration of the format of the input function:
  
```python
def input_evaluation_set():
    features = { 'SepalLength': np.array([6.4, 5.0]),
                 'SepalWidth':  np.array([2.8, 2.3]),
                 'PetalLength': np.array([5.6, 3.3]),
                 'PetalWidth':  np.array([2.2, 1.0])}
    labels = np.array([2, 1])
    return features, labels
```

- Dataset API consists of the following classes:
![Dataset]({{ site.url }}{{ site.baseurl }}/assets/images/deep learning tools/dataset.png){: .align-center}
  - Dataset: Base class containing methods to create and transform datasets. Also allows you to initialize a dataset from data in memory, or from a Python generator.
  - TextLineDataset: Reads lines from text files.
  - TFRecordDataset: Reads records from TFRecord files.
  - FixedLengthRecordDataset: Reads fixed size records from binary files.
  - Iterator: Provides a way to access one data set element at a time.

- **Input pipeline example**
```python
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Build the Iterator, and return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()
```

### Reading Data
- There are three other methods of getting data into a TensorFlow program:
  - Feeding: Python code provides the data when running each step.
  - Reading from files: an input pipeline reads the data from files at the beginning of a TensorFlow graph.
  - Preloaded data: a constant or variable in the TensorFlow graph holds all the data (for small data sets).

- **Tensor**: `tf.Tensor` object represents the symbolic result of a TensorFlow operation. 
A tensor itself does not hold or store values in memory, but provides only an interface for retrieving the value referenced by the tensor.
- **Variable**: think of it as a normal variable which we use in programming languages. We initialize variables, we can modify it later as well.
  - Variables can be described as persistent, mutable handles to in-memory buffers storing tensors. As such, variables are characterized by a certain shape and a fixed type.
- **Placeholder**: It doesn't require initial value. Placeholder simply allocates block of memory for future use by using `feed_dict` and it has an unconstrained shape.

## Instantiate an Estimator
TensorFlow provides several pre-made classifier Estimators,
- `tf.estimator.DNNClassifier` — for deep models that perform multi-class classification.
- `tf.estimator.DNNLinearCombinedClassifier` — for wide-n-deep models.
- `tf.estimator.LinearClassifier` — for classifiers based on linear models.

```python
# Build 2 hidden layer DNN with 10, 10 units respectively.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    n_classes=3)
```

## Train the Model
Train the model by calling the Estimator's `train` method as follows:
```python
# Train the Model.
classifier.train(
    input_fn=lambda:iris_data.train_input_fn(train_x, train_y, args.batch_size),
    steps=args.train_steps)
```

## Evaluate the Trained Model
Evaluates the accuracy of the trained model on the test data:
```python
# Evaluate the model.
eval_result = classifier.evaluate(
    input_fn=lambda:iris_data.eval_input_fn(test_x, test_y, args.batch_size))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
```

# Keras
**Keras** is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.

## Simple Example
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

# Theano

# References
- Web: TensorFlow Tutorial [[Link](https://www.tensorflow.org/get_started/premade_estimators)]
- Web: Keras Documentation [[Link](https://keras.io/)]