import neural_networks as nn

// remove the input / output if not needed
// all the features of the other module

fn main() {
	mut model := nn.NeuralNetwork.new(0)
	model.add_layer(nn.Dense.new(2, 3, 0.6, 0.6))
	model.add_layer(nn.Activation.new(3, .leaky_relu))
	model.add_layer(nn.Dense.new(3, 1, 0.6, 0.6))
	model.add_layer(nn.Activation.new(1, .leaky_relu))

	training_parameters := nn.TrainingParams {
		learning_rate: 0.37
		nb_epochs: 300
		print_interval: 1
		cost_function: .mse // mean squared error
		training_inputs: [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
		expected_training_outputs: [[0.0], [1.0], [1.0], [0.0]]
	}
	
	model.train(training_parameters)
}
