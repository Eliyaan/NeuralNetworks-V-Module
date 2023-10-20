module preceptron

import math as m

/*
Different activation functions and their derivatives
*/

[inline]
pub fn tanh(value f64) f64 {
	return (m.exp(value) - m.exp(-value)) / (m.exp(value) + m.exp(-value))
}

[inline]
pub fn dtanh(value f64) f64 {
	val := tanh(value)
	return 1 - val * val
}

[inline]
pub fn relu(value f64) f64 {
	return if value < 0 { 0 } else { value }
}

[inline]
pub fn drelu(value f64) f64 {
	return if value < 0 { 0.0 } else { 1.0 }
}

[inline]
pub fn leaky_relu(value f64) f64 {
	return if value < 0 { value * 0.1 } else { value }
}

[inline]
pub fn dleaky_relu(value f64) f64 {
	return if value < 0 { 0.1 } else { 1.0 }
}

[inline]
pub fn sigmoid(value f64) f64 {
	return 1 / (1 + m.exp(-value))
}

[inline]
pub fn dsig(value f64) f64 {
	sigx := sigmoid(value)
	return sigx * (1 - sigx)
}
