module neural_networks_acc

import vsl.la

/*
A Dense layer is also called a fully connected layer.
It connects all the neurones of the previous layer to all the neurones of the next layer.
*/

pub struct Dense {
pub mut:
	input_size           i64
	output_size          i64
	input                []f64
	output               []f64
	weights              &la.Matrix[f64]
	weights_gradient     &la.Matrix[f64]
	old_weights_gradient &la.Matrix[f64]
	bias                 []f64
	bias_gradient        []f64
	old_bias_gradient    []f64
}

pub fn Dense.new(input_size int, output_size int, weights_range f64, biases_range f64) Dense {
	return Dense{input_size, output_size, []f64{len: input_size}, []f64{len: output_size}, rand_matrix(output_size,
		input_size, weights_range), la.Matrix.new[f64](output_size, input_size), la.Matrix.new[f64](output_size,
		input_size), rand_array(output_size, biases_range), []f64{len: output_size}, []f64{len: output_size}}
}

pub fn (mut d Dense) reset() { // important, call after apply grad
	d.old_weights_gradient = d.weights_gradient.clone()
	d.weights_gradient = la.Matrix.new[f64](int(d.output_size), int(d.input_size))
	d.old_bias_gradient = d.bias_gradient.clone()
	d.bias_gradient = []f64{len: int(d.output_size)}
}

pub fn (mut d Dense) forward(input []f64) []f64 {
	d.input = input.clone()
	without_bias := la.matrix_vector_mul(1.0, d.weights, d.input)
	d.output = la.vector_add(1.0, without_bias, 1.0, d.bias)
	return d.output
}

pub fn (mut d Dense) backward(output_gradient []f64) []f64 {
	la.matrix_add(mut d.weights_gradient, 1.0, la.vector_vector_tr_mul(1.0, output_gradient,
		d.input), 1.0, d.weights_gradient)
	d.bias_gradient = la.vector_add(1.0, output_gradient, 1.0, d.bias_gradient)
	return la.matrix_tr_vector_mul(1.0, d.weights, output_gradient)
}

pub fn (mut d Dense) apply_grad(nb_elems_seen int, lr f64, momentum f64) {
	if momentum > 0 {
		la.matrix_add(mut d.weights_gradient, lr / f64(nb_elems_seen), d.weights_gradient,
			lr * momentum, d.old_weights_gradient)
		la.matrix_add(mut d.weights, 1.0, d.weights_gradient, 1.0, d.weights)
		d.bias = la.vector_add(lr / f64(nb_elems_seen), d.bias_gradient, 1.0, d.bias)
	} else {
		la.matrix_add(mut d.weights, lr / f64(nb_elems_seen), d.weights_gradient, 1.0,
			d.weights)
		d.bias = la.vector_add(lr / f64(nb_elems_seen), d.bias_gradient, 1.0, d.bias)
	}
}
