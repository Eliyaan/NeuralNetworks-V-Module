module neural_networks

pub interface Layer {
mut:
	input []f64
	output []f64
	forward(input []f64) []f64
	backward(output_gradient []f64) []f64
	apply_grad(nb_elems_seen int, lr f64)
	reset()
}