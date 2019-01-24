# A simple sample for "Neural Networks and Deep Learning"

* Notice these codes are written before taking Machine Learning couse and mastering Numpy and so I might edit them ASAP.

This is a sample implementaion of the "Neural Networks and Deep Learning".
The Neural-networks-and-deep-learning package is the very basic implementation and the
Neural_Network_Scikit package is the Neural Networks and Deep Learning implementation using scikit_learn and
scipy libraries.


## Getting Started

To work with the implementaion of Neural Newtwork using Scikit_learn , you have to
consider num_samples as the number of rows and num_features as the number of columns of a
matrix that will be the input matrix with the shape (num_samples, num_features).
To Simplify reading the matrices' shapes, let num_samples be N and num_featrues be p.
then matrices that will be used in MLPClassifier will have theses shapes:
* X -> N * p
* w -> p * first_hidden_layer_nurons_number
* y -> N * 1
* b -> N * 1

where w and b are weights and biases matrices that you can see in the simple implementation version.
aslo the point is that hidden_layer_sizes used in the number of hidden layers excluding the input and
output layers. So if we need 2 hidden layers we will use a tupple (x, u) with x as the number of nurons
in first layer and so on.

## Prerequisites

* numpy
* scikit-learn
* scipy



## Acknowledgments

Special Thanks to:

* [3blue1brown](http://www.3blue1brown.com)
* [http://neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com)

