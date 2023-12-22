import neural_networks as nn
import os
/*
 If you get a lot of errors you certainly need to run : 

 v install vsl 

 and then run :

 v run .

*/
// TODO:
// remove the input / output if not needed in the layer
// add all the features of the other module
// Later :
// Save the costs for momentum ?

fn main() {
	mut model := nn.NeuralNetwork.new(0)

	if os.input("Do you want to load a saved model ? [y/n]")  != "y" {
		println("Creating a new model")
		model.add_layer(nn.Dense.new(2, 3, 0.7, 0.65))
		model.add_layer(nn.Activation.new(.leaky_relu))
		model.add_layer(nn.Dense.new(3, 1, 0.6, 0.65))
		model.add_layer(nn.Activation.new(.leaky_relu))
	} else {
		println("Loading the saved model")
		model.load_model("saveXOR")
	}

	training_parameters := nn.TrainingParams {
		learning_rate: 0.37
		momentum: 0.5
		nb_epochs: 300
		print_interval: 50
		cost_function: .mse // mean squared error
		training_inputs: [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
		expected_training_outputs: [[0.0], [1.0], [1.0], [0.0]]
	}
	model.train(training_parameters)

	model.save_model("saveXOR")
}
