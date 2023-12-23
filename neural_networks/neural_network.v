module neural_networks

import la
import os
import rand

pub struct NeuralNetwork {
pub mut:
	layers []Layer
}

pub fn NeuralNetwork.new(seed u32) NeuralNetwork {
	rand.seed([seed, 0])
	return NeuralNetwork{}
}

pub fn (mut nn NeuralNetwork) add_layer(layer Layer) {
	nn.layers << layer
}

pub fn (mut nn NeuralNetwork) train(t_m TrainingMode) {
	println("\nTraining for $t_m.nb_epochs epochs:")
	match t_m {
		BackpropTrainingParams { nn.train_backprop(t_m) }
		MinibatchesBackpropTrainingParams { nn.train_minibatches_backprop(t_m) }
		else { exit_err('The training mode is not implemented') }
	}
}

pub fn (mut nn NeuralNetwork) forward_propagation(input []f64) []f64 {
	mut next_layer_input := input.clone()
	for mut layer in nn.layers {
		next_layer_input = layer.forward(next_layer_input)
	}
	return next_layer_input
}

pub fn (mut nn NeuralNetwork) backpropagation(expected_output []f64, output []f64, cost_prime fn ([]f64, []f64) []f64) {
	mut gradient := cost_prime(expected_output, output)
	for j := nn.layers.len - 1; j >= 0; j -= 1 {
		gradient = nn.layers[j].backward(gradient)
	}
}

pub fn (mut nn NeuralNetwork) apply_gradient_descent(nb_elems_seen int, lr f64, momentum f64) {
	for mut layer in nn.layers {
		layer.apply_grad(nb_elems_seen, lr, momentum)
		layer.reset()
	}
}

pub fn (mut nn NeuralNetwork) save_model(save_name string) {
	mut file := os.create(save_name) or { panic(err) }
	file.write_raw(i64(nn.layers.len)) or { panic(err) }
	for layer in nn.layers {
		l_type := layer_type(layer)
		file.write_raw(l_type) or { panic(err) }
		match layer {
			Dense {
				file.write_raw(layer.input_size) or { panic(err) }
				file.write_raw(layer.output_size) or { panic(err) }
				for elem in layer.weights.data {
					file.write_raw(elem) or { panic(err) }
				}
				for elem in layer.bias {
					file.write_raw(elem) or { panic(err) }
				}
			}
			Activation {
				file.write_raw(layer.activ_type) or { panic(err) }
			}
			else {}
		}
	}
	file.close()
}

@[noreturn]
fn exit_err(message string) {
	println(message)
	exit(1)
}

pub fn (mut nn NeuralNetwork) load_model(save_name string) {
	mut load := os.open(save_name) or { exit_err("The file doesn't exist") }
	nb_layers := load.read_raw[i64]() or { panic(err) }
	for _ in 0 .. nb_layers {
		ltype := load.read_raw[LayerType]() or { panic(err) }
		mut layer_base := layer_from_type(ltype)

		match mut layer_base {
			Dense {
				layer_base.input_size = load.read_raw[i64]() or { panic(err) }
				layer_base.output_size = load.read_raw[i64]() or { panic(err) }
				matrix_size := int(layer_base.input_size * layer_base.output_size)
				layer_base.weights = la.Matrix.raw(int(layer_base.output_size), int(layer_base.input_size),
					[]f64{len: matrix_size, init: index - index + load.read_raw[f64]() or {
					panic(err)
				}})
				layer_base.weights_gradient = la.Matrix.new[f64](int(layer_base.output_size),
					int(layer_base.input_size))
				layer_base.old_weights_gradient = la.Matrix.new[f64](int(layer_base.output_size),
					int(layer_base.input_size))
				layer_base.bias = []f64{len: int(layer_base.output_size), init: index - index + load.read_raw[f64]() or {
					panic(err)
				}}
				layer_base.bias_gradient = []f64{len: int(layer_base.output_size)}
				layer_base.old_bias_gradient = []f64{len: int(layer_base.output_size)}
			}
			Activation {
				layer_base = Activation.new(load.read_raw[ActivationFunctions]() or { panic(err) })
			}
			else {}
		}
		nn.add_layer(layer_base)
	}
}
