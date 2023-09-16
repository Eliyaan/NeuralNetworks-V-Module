module preceptron

import rand as rd

// To initialise the nn with random weights and biases
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

[direct_array_access; inline]
fn (mut nn NeuralNetwork) randomise_i_exp_o() { // To shuffle the dataset I think
	mut base_inputs := nn.training_inputs.clone()
	range := base_inputs.len
	mut base_expd_o := nn.excpd_training_outputs.clone()
	nn.training_inputs.clear()
	nn.excpd_training_outputs.clear()
	for _ in 0 .. range {
		i := rd.int_in_range(0, base_inputs.len) or { panic(err) }
		nn.training_inputs << base_inputs[i]
		base_inputs.delete(i)
		nn.excpd_training_outputs << base_expd_o[i]
		base_expd_o.delete(i)
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
