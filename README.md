# CNN

Applications of Convolutional Neural Networks (CNNs) to solve image classification problems.

- cats vs. dogs is a model to classify images of cats and dogs. Code is based on [Jeremy Howard's course](http://course.fast.ai/index.html) and [VGG16 architecture](https://arxiv.org/abs/1409.1556). It is written on Keras and runs on top of TensorFlow. Link to Kaggle competition is [here](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition).
- mnist solves the classic [handwritten digit](http://yann.lecun.com/exdb/mnist/) classification problem. Model is implemented on TensorFlow and is based on code from [tonanhngo](https://github.com/tonanhngo/handon-tensorflow). Architecture is composed of two convolutional layers with max-pooling, a dropout layer to reduce probability of overfitting and a fully connected layer.
