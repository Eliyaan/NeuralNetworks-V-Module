module neural_networks

import math

pub enum ActivationFunctions {
	tanh
	leaky_relu
}

fn get_activ_function(f ActivationFunctions) (fn (f64) f64, fn (f64) f64) {
	match f {
		.tanh { return tanh, tanh_prime }
		.leaky_relu { return lrelu, lrelu_prime }
	}
}

@[inline]
pub fn tanh(a f64) f64 {
	return math.tanh(a)
}

@[inline]
pub fn tanh_prime(a f64) f64 {
	tanha := math.tanh(a)
	return 1 - tanha * tanha
}

@[inline]
pub fn lrelu(value f64) f64 {
	return if value < 0 { value * 0.1 } else { value }
}

@[inline]
pub fn lrelu_prime(value f64) f64 {
	return if value < 0 { 0.1 } else { 1.0 }
}
