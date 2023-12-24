import neural_networks as nn
import os

/*
 If you get a lot of errors you probably need to run :

 v install vsl

 and then run :

 v run .
*/

/*
The minibatches are not very efficient on this too little dataset that
does not require generalisation over the whole data.
It's here to show how to use it.
*/

fn main() {
	mut model := nn.NeuralNetwork.new(0)

	if os.input('Do you want to load a saved model ? [y/n]') != 'y' {
		println('Creating a new model')
		model.add_layer(nn.Dense.new(2, 3, 0.7, 0.65))
		model.add_layer(nn.Activation.new(.leaky_relu))
		model.add_layer(nn.Dense.new(3, 1, 0.6, 0.65))
		model.add_layer(nn.Activation.new(.leaky_relu))
	} else {
		println('Loading the saved model')
		model.load_model('saveXOR')
	}

	training_parameters := nn.MinibatchesBackpropTrainingParams{
		learning_rate: 1.0
		momentum: 0.5
		batch_size: 2
		nb_epochs: 30
		print_interval: 1
		print_batch_interval: 2
		cost_function: .mse // mean squared error
		training: nn.Dataset {
			inputs: [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
			expected_outputs: [[0.0], [1.0], [1.0], [0.0]]
		}
		test: nn.Dataset {
			inputs: [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
			expected_outputs: [[0.0], [1.0], [1.0], [0.0]]
		}
		test_params: nn.TestParams{
			print_start: 0
			print_end: 3
			training_interval: 30
			training_batch_interval: 2
		}
	}
	model.train(training_parameters)

	model.save_model('saveXOR')
}