module preceptron

import toml
import os
import rand as rd

/*
Main functions and structs for the neural networks
*/

pub struct Neuron {
mut:
	bias   f64
	b_cost f64
	nactiv f64
	output f64
	cost   f64
}

pub struct Weight {
mut:
	weight f64
	cost   f64
}

pub struct NeuralNetwork {
pub mut:
	learning_rate f64
	nb_neurons    []int
	activ_funcs   []fn (f64) f64

	print_epoch int

	deriv_activ_funcs []fn (f64) f64

	w_random_interval f64 = 0.005
	b_random_interval f64 = 0.005
	// [layer_nbr][input_neuron_nbr][output_neuron_nbr]
	weights_list [][][]Weight
	// [layer_nbr][neuron_nbr]
	layers_list [][]Neuron
	// For backprop:
	global_cost               f64
	test_cost                 f64 = 10000000
	training_inputs           [][]f64
	expected_training_outputs [][]f64
	test_inputs               [][]f64
	expected_test_outputs     [][]f64
	best_cost                 f64 = 100000000000
	mini_batch_start          int = -1
	mini_batch_end            int = -1
	classifier                bool

	seed []u32 = [u32(0), 0]
}

/*
Initialise the neural network
Input	: name -> name of the file to load
*/
pub fn (mut nn NeuralNetwork) init(load_path string) {
	rd.seed(nn.seed)
	if load_path != '' {
		file := toml.parse_file(load_path) or { panic(err) }
		nn.best_cost = file.value('cost').f64()
		base_weights_list := file.value('weights').array()
		base_layers_list := file.value('biases').array()
		mut base_layers_list_good := [][]Neuron{}
		mut base_weights_listgood := [][][]Weight{}
		for a, layer in base_weights_list {
			base_weights_listgood << [][]Weight{}
			for b, weight_list in layer.array() {
				base_weights_listgood[a] << []Weight{}
				for weight in weight_list.array() {
					base_weights_listgood[a][b] << Weight{weight.f64(), 0}
				}
			}
		}
		for a, layer in base_layers_list {
			base_layers_list_good << []Neuron{}
			for bias in layer.array() {
				base_layers_list_good[a] << Neuron{bias.f64(), 0, 0, 0, 0}
			}
		}
		nn.layers_list = base_layers_list_good
		nn.weights_list = base_weights_listgood
	} else { // If it's a new nn
		nn.weights_list = [][][]Weight{len: nn.nb_neurons.len - 1}

		for i, mut layer in nn.weights_list {
			for _ in 0 .. nn.nb_neurons[i] {
				layer << []Weight{len: nn.nb_neurons[i + 1]}
			}
		}

		nn.layers_list = [][]Neuron{}
		for nb in nn.nb_neurons {
			nn.layers_list << []Neuron{len: nb}
		}

		nn.set_rd_wb_values()
	}
}

/*
To load the data from a toml file
*/
pub fn (mut nn NeuralNetwork) load_dataset(name string) {
	file := toml.parse_file(name) or { panic(err) }
	base_t_i_list := file.value('training_inputs').array()
	base_e_t_o_list := file.value('expected_training_outputs').array()
	base_test_i_list := file.value('test_inputs').array()
	base_e_test_o_list := file.value('expected_test_outputs').array()

	nn.training_inputs = [][]f64{}
	nn.expected_training_outputs = [][]f64{}
	for i, t_i in base_t_i_list {
		nn.training_inputs << []f64{}
		for value in t_i.array() {
			nn.training_inputs[i] << value.f64()
		}
	}
	for i, e_t_o in base_e_t_o_list {
		nn.expected_training_outputs << []f64{}
		for value in e_t_o.array() {
			nn.expected_training_outputs[i] << value.f64()
		}
	}
	assert nn.training_inputs[0].len == nn.nb_neurons[0]
	assert nn.expected_training_outputs[0].len == nn.nb_neurons[nn.nb_neurons.len - 1]
	assert nn.training_inputs.len == nn.expected_training_outputs.len

	nn.test_inputs = [][]f64{}
	nn.expected_test_outputs = [][]f64{}
	for i, test_i in base_test_i_list {
		nn.test_inputs << []f64{}
		for value in test_i.array() {
			nn.test_inputs[i] << value.f64()
		}
	}
	for i, e_test_o in base_e_test_o_list {
		nn.expected_test_outputs << []f64{}
		for value in e_test_o.array() {
			nn.expected_test_outputs[i] << value.f64()
		}
	}
	assert nn.test_inputs[0].len == nn.nb_neurons[0]
	assert nn.expected_test_outputs[0].len == nn.nb_neurons[nn.nb_neurons.len - 1]
	assert nn.test_inputs.len == nn.expected_test_outputs.len
}

/*
Load data from the mnist dataset
0 < nb_training <= 60000
0 < nb_tests <= 10000
*/
[direct_array_access]
pub fn (mut nn NeuralNetwork) load_mnist(nb_training int, nb_tests int) {
	println('Loading mnist...')
	test_labels := os.open('mnist\\t10k-labels-idx1-ubyte') or { panic(err) }
	test_images := os.open('mnist\\t10k-images-idx3-ubyte') or { panic(err) }
	nn.test_inputs = [][]f64{}
	nn.expected_test_outputs = [][]f64{}
	for i in 0 .. nb_tests {
		nn.test_inputs << [
			scale_and_process_img(test_images.read_bytes_at(784, i * 784 + 16).map(f64(it)),
				28, 28 + rd.int_in_range(-7, 8) or { panic(err) }, 28 + rd.int_in_range(-7, 8) or {
				panic(err)
			}, true),
		]
		nn.expected_test_outputs << [
			match test_labels.read_bytes_at(1, i + 8)[0] {
				0 { [f64(1), 0, 0, 0, 0, 0, 0, 0, 0, 0] }
				1 { [f64(0), 1, 0, 0, 0, 0, 0, 0, 0, 0] }
				2 { [f64(0), 0, 1, 0, 0, 0, 0, 0, 0, 0] }
				3 { [f64(0), 0, 0, 1, 0, 0, 0, 0, 0, 0] }
				4 { [f64(0), 0, 0, 0, 1, 0, 0, 0, 0, 0] }
				5 { [f64(0), 0, 0, 0, 0, 1, 0, 0, 0, 0] }
				6 { [f64(0), 0, 0, 0, 0, 0, 1, 0, 0, 0] }
				7 { [f64(0), 0, 0, 0, 0, 0, 0, 1, 0, 0] }
				8 { [f64(0), 0, 0, 0, 0, 0, 0, 0, 1, 0] }
				9 { [f64(0), 0, 0, 0, 0, 0, 0, 0, 0, 1] }
				else { panic('Match test outputs') }
			},
		]
	}
	train_labels := os.open('mnist\\train-labels-idx1-ubyte') or { panic(err) }
	train_images := os.open('mnist\\train-images-idx3-ubyte') or { panic(err) }
	nn.training_inputs = [][]f64{}
	nn.expected_training_outputs = [][]f64{}
	for i in 0 .. nb_training {
		nn.training_inputs << [
			scale_and_process_img(train_images.read_bytes_at(784, i * 784 + 16).map(f64(it)),
				28, 28 + rd.int_in_range(-7, 8) or { panic(err) }, 28 + rd.int_in_range(-7, 8) or {
				panic(err)
			}, true),
		]
		nn.expected_training_outputs << [
			match train_labels.read_bytes_at(1, i + 8)[0] {
				0 { [f64(1), 0, 0, 0, 0, 0, 0, 0, 0, 0] }
				1 { [f64(0), 1, 0, 0, 0, 0, 0, 0, 0, 0] }
				2 { [f64(0), 0, 1, 0, 0, 0, 0, 0, 0, 0] }
				3 { [f64(0), 0, 0, 1, 0, 0, 0, 0, 0, 0] }
				4 { [f64(0), 0, 0, 0, 1, 0, 0, 0, 0, 0] }
				5 { [f64(0), 0, 0, 0, 0, 1, 0, 0, 0, 0] }
				6 { [f64(0), 0, 0, 0, 0, 0, 1, 0, 0, 0] }
				7 { [f64(0), 0, 0, 0, 0, 0, 0, 1, 0, 0] }
				8 { [f64(0), 0, 0, 0, 0, 0, 0, 0, 1, 0] }
				9 { [f64(0), 0, 0, 0, 0, 0, 0, 0, 0, 1] }
				else { panic('bu') }
			},
		]
	}
	println('Finished loading mnist!')
}

/*
For doing a simple forward propagation
Input	: array of values (1 value by input neuron)
Output	: array of the outputs
*/
[direct_array_access]
pub fn (mut nn NeuralNetwork) fprop_value(inputs []f64) []f64 {
	for i, input in inputs {
		nn.layers_list[0][i].output = input
	}
	for i, mut layer in nn.layers_list { // For each layer
		if i > 0 { // ignore the input layer
			for j, mut o_neuron in layer { // For each neuron in the output layer
				o_neuron.nactiv = 0
				for k, i_neuron in nn.layers_list[i - 1] { // For each neuron in the input layer
					o_neuron.nactiv += nn.weights_list[i - 1][k][j].weight * i_neuron.output
				}
				o_neuron.nactiv += o_neuron.bias
				o_neuron.output = nn.activ_funcs[i - 1](o_neuron.nactiv)
			}
		}
	}
	return get_outputs(nn.layers_list[nn.nb_neurons.len - 1])
}

pub fn (mut nn NeuralNetwork) test_unseen_data() f64 {
	nn.test_cost = 0
	mut accuracy := 0.0
	for index, inputs in nn.test_inputs {
		for i, output in nn.fprop_value(inputs) { // for each output
			tmp := output - nn.expected_test_outputs[index][i]
			nn.test_cost += tmp * tmp
		}
		if nn.classifier {
			accuracy += if nn.test_value_classifier(index, true) { 1 } else { 0 }
		}
	}
	nn.test_cost /= nn.test_inputs.len
	if nn.classifier {
		println('\nTest cost: ${nn.test_cost} - Accuracy: ${(accuracy / nn.test_inputs.len * 100):.3}%')
	} else {
		println('\nTest cost: ${nn.test_cost}')
	}

	return nn.test_cost
}

/*
Input	: Neuron array
Output	: The outputs of the neuron array
*/
pub fn get_outputs(neurons []Neuron) []f64 {
	return []f64{len: neurons.len, init: neurons[index].output}
}

/*
Input	: Neural network neuron array
Output	: The biases of the neurons
*/
pub fn get_biases(neurons [][]Neuron) [][]f64 {
	mut biases := [][]f64{}
	for layer in neurons {
		biases << []f64{len: layer.len, init: layer[index].bias}
	}
	return biases
}

/*
Input	: Neural network weights array
Output	: The weights values of the weights
*/
pub fn get_weights(weights_objs [][][]Weight) [][][]f64 {
	mut weights := [][][]f64{len: weights_objs.len}
	for l, mut layer in weights {
		for j in 0 .. weights_objs[l].len {
			layer << []f64{len: weights_objs[l][j].len, init: weights_objs[l][j][index].weight}
		}
	}
	return weights
}
