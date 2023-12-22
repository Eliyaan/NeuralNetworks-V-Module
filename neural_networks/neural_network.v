module neural_networks

import la
import os
import rand

pub struct NeuralNetwork {
pub mut :
	layers []Layer
}

pub struct TrainingParams {
	learning_rate f64
	momentum f64
	nb_epochs int // An epoch is when the nn has seen the entire dataset
	print_interval int
	cost_function CostFunctions
	training_inputs [][]f64
	expected_training_outputs [][]f64
}

pub fn NeuralNetwork.new(seed u32) NeuralNetwork {
	rand.seed([seed, 0])
	return NeuralNetwork{}
}

pub fn (mut nn NeuralNetwork) add_layer(layer Layer) {
	nn.layers << layer
}

pub fn (mut nn NeuralNetwork) train(t_p TrainingParams) { // TODO: input an interface so allow for multi threading / mini batches
	cost_fn, cost_prime := get_cost_function(t_p.cost_function)
	for epoch in 0..t_p.nb_epochs {
		mut error := 0.0
		for i, input in t_p.training_inputs {
			output := nn.forward_propagation(input)
			error += cost_fn(t_p.expected_training_outputs[i], output)
			nn.backpropagation(t_p.expected_training_outputs[i], output, cost_prime)			
		}
		if (epoch+1) % t_p.print_interval == 0 || epoch == 0 {
			println("Epoch ${epoch+1}/$t_p.nb_epochs\t-\tCost : $error")
		}
		nn.apply_gradient_descent(t_p.training_inputs.len, t_p.learning_rate, t_p.momentum)
	}
}

pub fn (mut nn NeuralNetwork) forward_propagation(input []f64) []f64 {
	mut next_layer_input := input.clone()
	for mut layer in nn.layers {
		next_layer_input = layer.forward(next_layer_input)
	}
	return next_layer_input
}

pub fn (mut nn NeuralNetwork) backpropagation(expected_output []f64, output []f64, cost_prime fn([]f64, []f64) []f64) {
	mut gradient := cost_prime(expected_output, output)
	for j := nn.layers.len-1; j >= 0; j -= 1 {
		gradient = nn.layers[j].backward(gradient)
	}
}

pub fn (mut nn NeuralNetwork) apply_gradient_descent(nb_elems_seen int, lr f64, momentum f64) {
	for mut layer in nn.layers {
		layer.apply_grad(nb_elems_seen, lr, momentum)
		layer.reset()
	}
}

pub fn (mut nn NeuralNetwork) save_model() {
	mut file := os.create('save') or {panic(err)}
	file.write_raw(i64(nn.layers.len)) or {panic(err)}
	for layer in nn.layers {
		l_type := layer_type(layer)
		file.write_raw(l_type) or {panic(err)}
		match layer {
			Dense {
				file.write_raw(layer.input_size) or {panic(err)}
				file.write_raw(layer.output_size) or {panic(err)}
				for elem in layer.weights.data {
					file.write_raw(elem) or {panic(err)}
				}
				for elem in layer.bias {
					file.write_raw(elem) or {panic(err)}
				}
			}
			Activation {
				file.write_raw(layer.activ_type) or {panic(err)}
			}
			else {}
		}
	}
	file.close()
}

pub fn (mut nn NeuralNetwork) load_model() {
	mut load := os.open('save') or {panic(err)}
	nb_layers := load.read_raw[i64]() or {panic(err)}
	for _ in 0..nb_layers {
		ltype := load.read_raw[LayerType]() or {panic(err)}
		mut layer_base := layer_from_type(ltype)
		
		match mut layer_base {
			Dense {
				layer_base.input_size = load.read_raw[i64]() or {panic(err)}
				layer_base.output_size = load.read_raw[i64]() or {panic(err)}
				matrix_size := int(layer_base.input_size*layer_base.output_size)
				layer_base.weights = la.Matrix.raw(int(layer_base.output_size), int(layer_base.input_size), []f64{len:matrix_size, init:index-index+load.read_raw[f64]() or {panic(err)}})
				layer_base.weights_gradient = la.Matrix.new[f64](int(layer_base.output_size), int(layer_base.input_size))
				layer_base.old_weights_gradient = la.Matrix.new[f64](int(layer_base.output_size), int(layer_base.input_size))
				layer_base.bias = []f64{len:int(layer_base.output_size), init:index-index+load.read_raw[f64]() or {panic(err)}}
				layer_base.bias_gradient = []f64{len:int(layer_base.output_size)}
				layer_base.old_bias_gradient = []f64{len:int(layer_base.output_size)}
			}
			Activation {
				layer_base = Activation.new(load.read_raw[ActivationFunctions]() or {panic(err)})				
			}
			else {}
		}
		nn.add_layer(layer_base)
	}
	println(nn)
}