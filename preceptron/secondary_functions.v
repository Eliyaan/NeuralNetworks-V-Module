module preceptron

import rand as rd

/*
Other useful functions
*/

/*
To initialise the nn with random weights and biases
*/
[direct_array_access; inline]
fn (mut nn NeuralNetwork) set_rd_wb_values() {
	// Weights
	for mut layer in nn.weights_list {
		for mut weights_list in layer {
			for mut weight in weights_list {
				weight.weight = rd.f64_in_range(-1, 1) or { panic(err) }
			}
		}
	}

	// Biases
	for mut layer in nn.layers_list {
		for mut neuron in layer {
			neuron.bias = rd.f64_in_range(-1, 1) or { panic(err) }
		}
	}
}

pub fn (mut nn NeuralNetwork) softmax() []Neuron {
	mut sum := 0.0
	for neuron in nn.layers_list[nn.nb_neurones.len - 1] {
		sum += neuron.output
	}
	for mut neuron in nn.layers_list[nn.nb_neurones.len - 1] {
		neuron.output /= sum
	}
	return nn.layers_list[nn.nb_neurones.len - 1]
}
