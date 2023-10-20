module preceptron

// import os
import time
import rand as rd

/*
Backpropagation implementation
*/

/*
Backprop training loop
Input	: number of epochs that will run
*/
pub fn (mut nn NeuralNetwork) train_backprop(nb_epochs u64) {
	mut timestamp := time.now()
	mut print_cost := 0.0
	mut accuracy := 0.0
	mut print_accu := 0.0
	println('\n\nActual time: ${timestamp}')
	for epoch in 1 .. nb_epochs + 1 {
		accuracy = 0.0
		if epoch > 0 {
			nn.apply_delta()
		}
		nn.global_cost = 0.0 // reset the cost before the training of this epoch
		for i in 0 .. nn.training_inputs.len {
			nn.neurons_costs_reset()
			nn.backprop(i)
			if nn.classifier {
				accuracy += if nn.test_value_classifier(i, false) { 1 } else { 0 }
			}
		}
		nn.global_cost /= nn.training_inputs.len
		print_cost += nn.global_cost
		accuracy /= nn.training_inputs.len
		print_accu += accuracy
		if nn.print_epoch > 0 {
			if epoch % u64(nn.print_epoch) == 0 {
				if nn.classifier {
					println('\nEpoch: ${epoch} - Mean Cost: ${print_cost / nn.print_epoch} - Last Cost: ${nn.global_cost} - Accuracy: ${(print_accu / nn.print_epoch * 100):.2}% - Time Elapsed: ${(time.now() - timestamp)}')
				} else {
					println('\nEpoch: ${epoch} - Mean Cost: ${print_cost / nn.print_epoch} - Last Cost: ${nn.global_cost} - Time Elapsed: ${(time.now() - timestamp)}')
				}
				timestamp = time.now()
				print_cost = 0
				print_accu = 0
			}
		}
	}
	println('\nActual time: ${timestamp}\n')
}

pub fn (mut nn NeuralNetwork) train_backprop_minibatches(nb_epochs u64, batch_size int) {
	mut timestamp := time.now()
	mut print_cost := 0.0
	mut accuracy := 0.0
	mut print_accu := 0.0
	println('\n\nActual time: ${timestamp}')
	for epoch in 1 .. nb_epochs + 1 {
		accuracy = 0.0
		if epoch > 1 {
			nn.apply_delta()
		}
		nn.global_cost = 0.0 // reset the cost before the training of this epoch
		nn.mini_batch_start = rd.int_in_range(0, nn.training_inputs.len - batch_size) or {
			panic(err)
		}
		nn.mini_batch_end = nn.mini_batch_start + batch_size
		for i in nn.mini_batch_start .. nn.mini_batch_end {
			nn.neurons_costs_reset()
			nn.backprop(i)
			if nn.classifier {
				accuracy += if nn.test_value_classifier(i, false) { 1 } else { 0 }
			}
		}
		nn.global_cost /= batch_size
		print_cost += nn.global_cost
		accuracy /= batch_size
		print_accu += accuracy
		if nn.print_epoch > 0 {
			if epoch % u64(nn.print_epoch) == 0 {
				if nn.classifier {
					println('\nEpoch: ${epoch} - Global Cost: ${print_cost / nn.print_epoch} - Accuracy: ${(print_accu / nn.print_epoch * 100):.2}% - Time Elapsed: ${(time.now() - timestamp)}')
				} else {
					println('\nEpoch: ${epoch} - Global Cost: ${print_cost / nn.print_epoch} - Time Elapsed: ${(time.now() - timestamp)}')
				}
				nn.test_unseen_data()
				timestamp = time.now()
				print_cost = 0
				print_accu = 0
			}
		}
	}
	println('\nActual time: ${timestamp}\n')
}

/*
Calculates the costs of each wieghts and biases
*/
[direct_array_access]
pub fn (mut nn NeuralNetwork) backprop(index int) {
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
				neuron.nactiv = nn.deriv_activ_funcs[i - 1](neuron.nactiv)
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
pub fn (mut nn NeuralNetwork) apply_delta() {
	// Weights
	for mut layer in nn.weights_list {
		for mut weight_list in layer {
			for mut weight in weight_list {
				weight.weight -= weight.cost / nn.training_inputs.len * nn.learning_rate
				weight.cost = 0.0
			}
		}
	}

	// Biases
	for mut layer in nn.layers_list[1..] { // for each layer excluding the input layer
		for mut neuron in layer {
			neuron.bias -= neuron.b_cost / nn.training_inputs.len * nn.learning_rate
			neuron.b_cost = 0.0
		}
	}
}

/*
Reset the costs that aren't reset in the backprop
*/
pub fn (mut nn NeuralNetwork) neurons_costs_reset() {
	for mut layer in nn.layers_list[1..] {
		for mut neuron in layer {
			neuron.cost = 0.0
		}
	}
}
