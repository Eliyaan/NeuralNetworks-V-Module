module neural_networks

pub struct Activation {
mut:
	input []f64
	output []f64
	activ_type ActivationFunctions
	activ fn(n f64) f64 @[required]
	activ_prime fn(n f64) f64 @[required]
}

pub fn Activation.new(activ_type ActivationFunctions) Activation {
	activ, activ_prime := get_activ_function(activ_type)
	return Activation{[]f64{}, []f64{}, activ_type, activ, activ_prime}
}

pub fn (mut a Activation) forward(input []f64) []f64 {
	a.input = input.clone()
	a.output = input.clone()
	vector_apply_func(mut a.output, a.activ)
	return a.output
}

pub fn (mut a Activation) backward(output_gradient []f64) []f64 {
	mut input_deriv := a.input.clone()
	vector_apply_func(mut input_deriv, a.activ_prime)
	output := vector_element_wise_mul(output_gradient, input_deriv)
	return output
}

pub fn (mut a Activation) apply_grad(nb_elems_seen int, lr f64, momentum f64) {
}

pub fn (mut a Activation) reset() {
}

pub fn vector_apply_func(mut a []f64, f fn(n f64) f64 ) {
	for mut elem in a {
		elem = f(elem)
	}
}