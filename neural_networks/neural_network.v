module neural_networks

import rand

pub struct NeuralNetwork {
pub mut :
	layers []Layer
}

pub struct TrainingParams {
	learning_rate f64
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
		nn.apply_gradient_descent(t_p.training_inputs.len, t_p.learning_rate)
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

pub fn (mut nn NeuralNetwork) apply_gradient_descent(nb_elems_seen int, lr f64) {
	for mut layer in nn.layers {
		layer.apply_grad(nb_elems_seen, lr)
		layer.reset()
	}
}