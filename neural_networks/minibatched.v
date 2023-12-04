module neural_networks

import time

pub fn (mut nn NeuralNetwork) train_backprop_minibatches(nb_epochs u64, batch_size int) {
	mut timestamp := time.now()
	mut print_cost := 0.0
	mut print_accu := 0.0
	println('\n\nActual time: ${timestamp}')
	for epoch in 1 .. nb_epochs + 1 {
		for batch in 1 .. int(nn.training_inputs.len/batch_size) + 1{ // be careful for the size of the batches to not lose some data over the division rounding
			if epoch+batch > 2 {
				nn.apply_delta_minibatches(batch_size)
			}
			nn.global_accuracy = 0.0
			nn.global_cost = 0.0 // reset the cost before the training of this epoch
			for i in 0 .. batch_size {
				elem := i + (batch-1)*batch_size
				if elem < nn.training_inputs.len {
					nn.neurons_costs_reset()
					nn.backprop(elem)
					if nn.classifier {
						nn.global_accuracy += if nn.test_value_classifier(elem, false) { 1 } else { 0 }
					}
				}
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

/*
Apply the modifications based on the cost calculated in the backprop
*/
pub fn (mut nn NeuralNetwork) apply_delta_minibatches(batch_size int) {
	// Weights
	for mut layer in nn.weights_list {
		for mut weight_list in layer {
			for mut weight in weight_list {
				
				change := (weight.cost / batch_size + weight.last_change*nn.momentum) * nn.learning_rate
				weight.weight -= change
				weight.last_change = change
				weight.cost = 0.0
			}
		}
	}

	// Biases
	for mut layer in nn.layers_list[1..] { // for each layer excluding the input layer
		for mut neuron in layer {
			change := (neuron.b_cost / nn.training_inputs.len + neuron.last_change*nn.momentum) * nn.learning_rate
			neuron.bias -= change
			neuron.last_change = change
			neuron.b_cost = 0.0
		}
	}
}