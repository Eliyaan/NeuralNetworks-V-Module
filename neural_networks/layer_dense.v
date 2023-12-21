module neural_networks

import la

pub struct Dense {
pub mut:
	input []f64
	output []f64
	weights &la.Matrix[f64]
	weights_gradient &la.Matrix[f64] // maybe not made in the right order
	bias []f64
	bias_gradient []f64
}

pub fn Dense.new(input_size int, output_size int, weights_range f64, biases_range f64) Dense {
	return Dense{[]f64{len:input_size}, []f64{len:output_size}, rand_matrix(output_size, input_size, weights_range), la.Matrix.new[f64](output_size, input_size), rand_array(output_size, biases_range), []f64{len:output_size}}
}

pub fn (mut d Dense) reset() {
	d.weights_gradient = la.Matrix.new[f64](d.output.len, d.input.len)
	d.bias_gradient = []f64{len:d.bias_gradient.len}
}

pub fn (mut d Dense) forward(input []f64) []f64 {
	d.input = input.clone()
	d.output = la.vector_add(1.0, la.matrix_vector_mul(1.0, d.weights, d.input), 1.0, d.bias)
	return d.output
}

pub fn (mut d Dense) backward(output_gradient []f64) []f64 {
	la.matrix_add(mut d.weights_gradient, 1.0, la.vector_vector_tr_mul(1.0, output_gradient, d.input), 1.0, d.weights_gradient)
	d.bias_gradient = la.vector_add(1.0, output_gradient, 1.0, d.bias_gradient)
	return la.matrix_tr_vector_mul(1.0, d.weights, output_gradient)
}

pub fn (mut d Dense) apply_grad(nb_elems_seen int, lr f64) {
	la.matrix_add(mut d.weights, lr/f64(nb_elems_seen), d.weights_gradient, 1.0, d.weights)
	d.bias = la.vector_add(lr/f64(nb_elems_seen), d.bias_gradient, 1.0, d.bias)
}