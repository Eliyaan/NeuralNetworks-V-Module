module neural_networks

import time

pub fn (mut nn NeuralNetwork) threaded_train_bp_minibatches(nb_epochs u64, batch_size int, nb_threads int) {
	mut timestamp := time.now()
	mut print_cost := 0.0
	mut print_accu := 0.0
	println('\n\nActual time: ${timestamp}')
	for epoch in 1 .. nb_epochs + 1 {
		for batch in 1 .. int(nn.training_inputs.len/batch_size) + 1 { // be careful for the size of the batches to not lose some data over the division rounding
			if epoch+batch > 2 {
				nn.apply_delta()
			}
			nn.global_accuracy = 0.0
			nn.global_cost = 0.0 // reset the cost before the training of this epoch
			channel := chan int{}
			for nb in 0 .. nb_threads {
				spawn nn.threaded_backprop_work(nb, batch, batch_size, nb_threads, channel)
			}
			for _ in 0 .. nb_threads {
				<-channel
			}
			nn.global_cost /= batch_size
			print_cost += nn.global_cost
			nn.global_accuracy /= batch_size
			print_accu += nn.global_accuracy
			if batch%nn.print_batch == 0 && nn.print_epoch > 0 {
				if epoch % u64(nn.print_epoch) == 0 {
					if nn.classifier {
						println('\nEpoch: ${epoch}/${nb_epochs} - Batch: ${batch}/${int(nn.training_inputs.len/batch_size)} - Global Cost: ${print_cost / (nn.print_epoch*nn.print_batch)} - Accuracy: ${(print_accu / (nn.print_epoch*nn.print_batch)* 100):.2}% - Time Elapsed: ${(time.now() - timestamp)}')
					} else {
						println('\nEpoch: ${epoch}/${nb_epochs} - Batch: ${batch}/${int(nn.training_inputs.len/batch_size)} - Global Cost: ${print_cost / (nn.print_epoch*nn.print_batch)} - Time Elapsed: ${(time.now() - timestamp)}')
					}
					timestamp = time.now()
					print_cost = 0
					print_accu = 0
					
				}
			}
			if batch%nn.test_batch == 0 {
				nn.test_unseen_data()
				if nn.test_cost < nn.save_cost || nn.test_accuracy > nn.save_accuracy {
					println('\n${nn.test_cost} < ${nn.save_cost} or ${nn.test_accuracy} > ${nn.save_accuracy}\nThreshold reached -> Saving')
					nn.save('nn_save-e${epoch}-')
				}
			}
		}
		
	}
	println('\nActual time: ${timestamp}\n')
}

fn (mut nn NeuralNetwork) threaded_backprop_work(thread_nb int, batch_nb int, batch_size int, nb_threads int, channel chan int) {
	for i in 0 .. int(batch_size/nb_threads) { // be careful for the nb of threads you choose to not lose some data
		elem := thread_nb*int(batch_size/nb_threads) + i + (batch_nb-1)*batch_size
		if elem < nn.training_inputs.len {
			nn.neurons_costs_reset()
			nn.threaded_backprop(elem)
			if nn.classifier {
				nn.global_accuracy += if nn.test_value_classifier(elem, false) { 1 } else { 0 }
			}
		}
	}
	channel <- 727
}

/*
Calculates the costs of each wieghts and biases
*/
@[direct_array_access]
pub fn (mut nn NeuralNetwork) threaded_backprop(index int) {
	nn.fprop(nn.training_inputs[index])

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