module preceptron

import os
import time

/*
Backpropagation implementation
*/

/*
Backprop training loop
Input	: number of epochs that will run
*/
pub fn (mut nn NeuralNetwork) train_backprop(nb_epochs u64) {
	mut need_to_save := false
	mut cost_to_save := 0.0
	mut weights_to_save := [][][]Weight{}
	mut layers_to_save := [][]Neuron{}

	mut timestamp := time.now()

	for epoch in 0 .. nb_epochs {
		if epoch > 0 {
			nn.apply_delta()
		}
		nn.global_cost = 0.0 // reset the cost before the training of this epoch
		for i in 0 .. nn.training_inputs.len {
			nn.neurons_costs_reset()
			nn.backprop(i)
		}
		if nn.print_epoch > 0 {
			if epoch % u64(nn.print_epoch) == 0 {
				println('\nEpoch: ${epoch} Global Cost: ${nn.global_cost} Time Elapsed: ${(time.now() - timestamp)}')
				timestamp = time.now()
			}
		}
		if nn.best_cost / nn.global_cost > 1.0 {
			need_to_save = true
			cost_to_save = nn.global_cost
			weights_to_save = nn.weights_list.clone()
			layers_to_save = nn.layers_list.clone()
			nn.best_cost = nn.global_cost
		}
	}
	if nn.print_epoch > 0 {
		println('____________________________________________________________\nFinal Results: \nCost: ${nn.global_cost}')
	}
	if need_to_save && nn.save_path != '' {
		println(' Saving the progress !')
		file := 'cost=${cost_to_save}\nweights=${get_weights(weights_to_save)}\nbiases=${get_biases(layers_to_save)}'
		os.write_file(nn.save_path + nn.nb_neurons.str() + '.nntoml', file) or { panic(err) }
	}
}

/*
Calculates the costs of each wieghts and biases
*/
//[direct_array_access]
fn (mut nn NeuralNetwork) backprop(index int) {
	nn.fprop_value(nn.training_inputs[index])

	// Cost for the print
	for i, neuron in nn.layers_list[nn.nb_neurons.len - 1] { // for each output
		tmp := neuron.output - nn.expected_training_outputs[index][i]
		nn.global_cost += tmp * tmp
	}

	// Start of the backprop
	// Deriv nactiv of all neurons to do it only one time
	for i, mut layer in nn.layers_list {
		if i > 0 {
			for mut neuron in layer {
				neuron.nactiv = nn.deriv_activ_funcs[i-1](neuron.nactiv)
			}
		}
	}

	// deltaC/deltaA(last)
	for i, mut neuron in nn.layers_list[nn.nb_neurons.len - 1] { // for each output
		neuron.cost = 2.0 * (neuron.output - nn.expected_training_outputs[index][i])
	}

	// deltaC/deltaW(last)
	for j, mut weight_list in nn.weights_list[nn.nb_neurons.len - 2] { // j is the nb of the input neuron
		for k, mut weight in weight_list { // k is the nb of the output neuron
			weight.cost += nn.layers_list[nn.nb_neurons.len - 2][j].output * nn.layers_list[nn.nb_neurons.len - 1][k].nactiv * nn.layers_list[nn.nb_neurons.len - 1][k].cost
		}
	}

	// deltaC/deltaB(last)
	for mut neuron in nn.layers_list[nn.nb_neurons.len - 1] {
		neuron.b_cost += neuron.nactiv * neuron.cost
	}

	// deltaC/deltaA(i)
	for i := nn.nb_neurons.len - 2; i > 0; i-- { // for each hidden layer but starting at the end
		for j in 0 .. nn.nb_neurons[i] { // for each neuron of the layer
			for k in 0 .. nn.nb_neurons[i + 1] { // for each neuron of the next layer
				nn.layers_list[i][j].cost += nn.weights_list[i][j][k].weight * nn.layers_list[i + 1][k].nactiv * nn.layers_list[
					i + 1][k].cost
			}
		}
	}

	for i in 1 .. nn.nb_neurons.len - 1 { // for each hidden layer (output already done and nothing to do on input)
		// Weights
		for j, mut weight_list in nn.weights_list[i - 1] { // j = nb input neuron
			for k, mut weight in weight_list { // k = nb output neuron
				weight.cost += nn.layers_list[i - 1][j].output * nn.layers_list[i][k].nactiv * nn.layers_list[i][k].cost
			}
		}

		// Biases
		for mut neuron in nn.layers_list[i] { // for each neuron of each layer
			neuron.b_cost += neuron.nactiv * neuron.cost
		}
	}
}

/*
Apply the modifications based on the cost calculated in the backprop
*/
fn (mut nn NeuralNetwork) apply_delta() {
	// Weights
	for mut layer in nn.weights_list {
		for mut weight_list in layer {
			for mut weight in weight_list {
				weight.weight -= weight.cost * nn.learning_rate
				weight.cost = 0.0
			}
		}
	}

	// Biases
	for mut layer in nn.layers_list[1..] { // for each layer excluding the input layer
		for mut neuron in layer {
			neuron.bias -= neuron.b_cost * nn.learning_rate
			neuron.b_cost = 0.0
		}
	}
}

/*
Reset the costs that aren't reset in the backprop
*/
fn (mut nn NeuralNetwork) neurons_costs_reset() {
	for mut layer in nn.layers_list[1..] {
		for mut neuron in layer {
			neuron.cost = 0.0
		}
	}
}
