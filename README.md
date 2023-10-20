# Neural Networks V Module
A V module to use multi-layer neural networks. Includes backpropagation but you can train them the way you want. If you have any questions or suggestions I'd be happy to read them !

An example can be found in example.v (It covers nearly the whole module but if you have any question, dont mind asking me). And a better example is coming soon! 

If you run the file it will train a small neural network with backpropagation to make it learn a XOR logic gate. The dataset can be found in training_data.toml (contains fields for training data, associated training outputs, test data, associated test outputs).

The number of layer in the neural network is ajustable with the nb_neurons field by using it like that : `[number of inputs, number of neurons of the first hidden layer, insert as many hidden layers as you want, number of outputs]`
