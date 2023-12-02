module preceptron

import rand as rd
import os

/*
Other useful functions
*/

/*
To initialise the nn with random weights and biases
*/

pub fn (mut nn NeuralNetwork) test_value_classifier(i int, test bool) bool {
	mut highest := 0
	mut true_highest := 0
	if test {
		for nb, neuron in nn.layers_list[nn.nb_neurons.len - 1] {
			if neuron.output > nn.layers_list[nn.nb_neurons.len - 1][highest].output {
				highest = nb
			}
		}
		for nb, value in nn.expected_test_outputs[i] {
			if value > nn.expected_test_outputs[i][true_highest] {
				true_highest = nb
			}
		}
	} else {
		for nb, neuron in nn.layers_list[nn.nb_neurons.len - 1] {
			if neuron.output > nn.layers_list[nn.nb_neurons.len - 1][highest].output {
				highest = nb
			}
		}
		for nb, value in nn.expected_training_outputs[i] {
			if value > nn.expected_training_outputs[i][true_highest] {
				true_highest = nb
			}
		}
	}
	return highest == true_highest
}

@[direct_array_access; inline]
pub fn (mut nn NeuralNetwork) set_rd_wb_values() {
	// Weights
	for mut layer in nn.weights_list {
		for mut weights_list in layer {
			for mut weight in weights_list {
				weight.weight = rd.f64_in_range(-nn.w_random_interval, nn.w_random_interval) or {
					panic(err)
				}
			}
		}
	}

	// Biases
	for mut layer in nn.layers_list {
		for mut neuron in layer {
			neuron.bias = rd.f64_in_range(-nn.b_random_interval, nn.b_random_interval) or {
				panic(err)
			}
		}
	}
}

fn abs(value f64) f64 {
	return if value >= 0 { value } else { -value }
}

pub fn (mut nn NeuralNetwork) softmax() []Neuron {
	mut sum := 0.0
	mut min := 0.0
	for neuron in nn.layers_list[nn.nb_neurons.len - 1] {
		sum += abs(neuron.output)
		if neuron.output < min {
			min = neuron.output
		}
	}
	for mut neuron in nn.layers_list[nn.nb_neurons.len - 1] {
		neuron.output += abs(min)
		neuron.output /= sum
	}
	return nn.layers_list[nn.nb_neurons.len - 1]
}

pub fn (mut nn NeuralNetwork) save(save_path string) {
	if nn.classifier {
		file := 'arch=${nn.nb_neurons}\ntest_cost=${nn.test_cost}\ntest_accuracy=${nn.test_accuracy}\nweights=${get_weights(nn.weights_list)}\nbiases=${get_biases(nn.layers_list)}'
		os.write_file(save_path + nn.nb_neurons.str() + '.nntoml', file) or { panic(err) }
		println('\nSaved the neural network weights and biases as ${save_path +
			nn.nb_neurons.str() + '.nntoml'} !')
	} else {
		file := 'arch=${nn.nb_neurons}\ntest_cost=${nn.test_cost}\nweights=${get_weights(nn.weights_list)}\nbiases=${get_biases(nn.layers_list)}'
		os.write_file(save_path + nn.nb_neurons.str() + '.nntoml', file) or { panic(err) }
		println('\nSaved the neural network weights and biases as ${save_path +
			nn.nb_neurons.str() + '.nntoml'} !')
	}
}

/*
Function for the input noise
*/
fn (nn NeuralNetwork) noise() f64 { // negative input noise -> ranges from -input_noise to input_noise
	if nn.input_noise > 0 {
		if rd.int_in_range(0, nn.input_noise_chance) or { 50 } == 0 {
			return rd.f64_in_range(0, nn.input_noise) or { 0 }
		}
	} else {
		if rd.int_in_range(0, nn.input_noise_chance) or { 50 } == 0 {
			return rd.f64_in_range(-nn.input_noise, nn.input_noise) or { 0 }
		}
	}
	return 0
}
