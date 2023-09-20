module preceptron

import toml

/*
Main functions and structs for the neural networks
*/

// TODO	:
// minibatches ?

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
	learning_rate f64
	nb_neurons   []int
	activ_funcs    []fn (f64) f64

	print_epoch int

	save_path string
	load_path string

	deriv_activ_funcs []fn (f64) f64
mut:
	// [layer_nbr][input_neuron_nbr][output_neuron_nbr]
	weights_list [][][]Weight
	// [layer_nbr][neuron_nbr]
	layers_list [][]Neuron
	// For backprop:
	global_cost            f64
	training_inputs        [][]f64
	expected_training_outputs [][]f64
	best_cost              f64 = 100000000000
}

/*
Initialise the neural network
Input	: name -> name of the file to load
*/
pub fn (mut nn NeuralNetwork) init() {
	if nn.load_path != '' {
		file := toml.parse_file(nn.load_path) or { panic(err) }
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
//[direct_array_access]
pub fn (mut nn NeuralNetwork) load_dataset(name string) {
	file := toml.parse_file(name) or {panic(err)}
	base_t_i_list := file.value("training_inputs").array()
	base_e_t_o_list := file.value("expected_training_outputs").array()
	
	nn.training_inputs = [][]f64{}
	nn.expected_training_outputs = [][]f64{}
	for i, t_i in base_t_i_list{
		nn.training_inputs << []f64{}
		for value in t_i.array(){
			nn.training_inputs[i] << value.f64()
		}
	}
	for i, e_t_o in base_e_t_o_list{
		nn.expected_training_outputs << []f64{}
		for value in e_t_o.array(){
			nn.expected_training_outputs[i] << value.f64()
		}
	}
}

/*
For doing a simple forward propagation
Input	: array of values (1 value by input neuron)
Output	: array of the outputs
*/
//[direct_array_access]
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
				o_neuron.output = nn.activ_funcs[i-1](o_neuron.nactiv)
			}
		}
	}
	return get_outputs(nn.layers_list[nn.nb_neurons.len - 1])
}

/*
Input	: Neuron array
Output	: The outputs of the neuron array
*/
//[direct_array_access; inline]
pub fn get_outputs(neurons []Neuron) []f64 {
	return []f64{len: neurons.len, init: neurons[index].output}
}

/*
Input	: Neural network neuron array
Output	: The biases of the neurons
*/
//[direct_array_access; inline]
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
//[direct_array_access; inline]
pub fn get_weights(weights_objs [][][]Weight) [][][]f64 {
	mut weights := [][][]f64{len: weights_objs.len}
	for l, mut layer in weights {
		for j in 0 .. weights_objs[l].len {
			layer << []f64{len: weights_objs[l][j].len, init: weights_objs[l][j][index].weight}
		}
	}
	return weights
}