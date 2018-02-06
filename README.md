# Sushi Artificial Neural Network
This is a library for a simple neural network, developed as project for the course of Machine Learning in University of Pisa.

## Getting Started
I will briefly introduce you to my neural network. It could solve all the basic tasks a neural network should solve and I try to make it simple to configure and not to difficult to understand. I try to develop a full customizable network, so a lot of parameters could be tuned as you prefer.
I developed and tested it only on Linux environment.
I have used [nlohmann::json](https://github.com/nlohmann/json) library for parsing json file.

### Docs
The documentation can be found [here](http://giulioaur.com/sann). It has been generated thanks to [Doxygen](http://www.stack.nl/~dimitri/doxygen/).

### Network
To create a neural network we have to instantiate a new [Network](http://giulioaur.com/sann/classsann_1_1Network.html) object, passing three parameters:
* The topology : the shape of the layers.
* The activation function: the activation function of each layer. For the input layer the activation function is always a linear function and it cannot be modified.
* The weights initialization: a function for the initialization of the weights. m is the number of the units in the current layer and n the weights for each neuron. The matrix to return must be m x n. 

```c++
/* Creates a net with three layers: an input layer of 5 units, a middle layer of 2 units and an output layer
with 1 unit. Here we use two basic functions as activation functions, but a new one could be created as
instance of the class Func. Then we initialize the weights from gaussian distribution. */
Network net{{5, 2, 1}, {Func::ReLU, Func::sigmoid}, [](const size_t m, const size_t n){
            weightsMatrix weights(m);

            for(size_t i = 0; i < m; ++i)
                weights[i] = math::Randomizer::randomGaussianVector<double>(0, 1./sqrt(n-1), n);

            return weights;
        }};
```

Then to train the net we need three (or more) parameters:
* The dataset : the data on which train.
* The Estimator : an object that interacts with the train (read the class documentation for more info).
* The hyperparameters : the hyperparameters with which train.

```c++
/* Train the net on 1000 epochs, with mini batch size of 100 (1 = online, numOfPatterns = stochastic),
with learning rate = 0.1, momentum = 0.9 and L2 reg = 0.001. */
net.train(myDataset, myEstimator, {1000, 100, 0.1, 0.9, 0.001});
```

The net will train using a gradient descent algorithm and if no error function is passed through the method setErrorFunction(), it will try to minimize the mean square error.
Then compute the output passing the input.

```c++
vector<double> res = net.compute(inputs);
```

### Use Configuration file
I have also developed a method to parse the net from a json formatted configuration file. The method is _parse_net()_ from the _"examples/parse.hpp"_ files, while the configuration file is in _"files/config/config.json"_ file. An example on how to use it could be found commented on both _"examples/monk.cpp"_ and _"example/cup.cpp"_ files. I think that the configuration files is self explained, and the parse function is really easy to understand.

## Running the example
### Prerequisites
The only prerequisite is to have boost library installed (or, at least, the boost::filesystem). As you know it makes the file system management very easy in C++, so I totally need it.

### Compilation
If you think brave enough, you can compile it by hand including all the files inside _"src"_ folder and its sub-folders (you have also to manually set the three variables inside "constants.h") plus one of the main file inside _"examples"_ folder, otherwise you can just use the CMake file. So create a directory name _"build"_, enter in it and type:

```
cmake -D DATA_SET=@Desidered_dataset@ ..
```

where @desidered\_dataset@ could be monk1, monk2, monk3 or cup (I will explain it later). 

### Execution
Once compiled one of the four exemples, you can just run it typing

```
./@Desidered_dataset@ [n]
```
The n parameter is only needed if the cup dataset has been compiled, it represents the number of time the model selection has to be executed.

### Configuration Files
The parameters on which execute the model selection could be tuned on the relative _"\_validation.json"_ files inside the folder _"files/config"_.

### Plot Result
To plot the result I have done a choise that could seem weird, but I have not found a good and simple C++ library to make some plot, so I have used Python. The scripts to plot the data are in the _"scripts"_ folder, and must be executed on that folder (for relative path reason, I'm a bit lazy and do not know Python enough). To plot the result of the train phase:

```
#python trainErrors.py trainingDataSeterrorFilename [testDataSetErrorFilename] [epochsToPlot]
python trainErrors.py trainErrors testErrors 250
```

To see the result of the validation phase, instead:

```
#python validation.py [numberOfWindows] [epochsToPlot]
python trainErrors.py 2 250
```

This last scripts contain a bug: it always opens one more windows with no plot.

### Data Set
#### Monk
It's a dataset for a classification task. It is composed by 3 sub-dataset, the first two without any kind of noise, the last one with some noisy data. You can find them [here](https://archive.ics.uci.edu/ml/datasets/MONK's+Problems).
#### Cup 
It is a dataset provided by our teacher for a regression task.
