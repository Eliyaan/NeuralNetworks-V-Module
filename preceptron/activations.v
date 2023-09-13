module preceptron

import math as m


//Different activation functions and their derivatives
[inline]
fn tanh(value f64) f64{
	return (m.exp(value)-m.exp(-value)) / (m.exp(value)+m.exp(-value))
}

[inline]
fn dtanh(value f64) f64{
	val := tanh(value)
	return 1 - val*val
}

[inline]
fn relu(value f64) f64{
	return if value<0{0}else{value}
}

[inline]
fn drelu(value f64) f64{
	return if value<0{0.0}else{1.0}
}

[inline]
fn leaky_relu(value f64) f64{
	return if value<0{value*0.01}else{value}
}

[inline]
fn dleaky_relu(value f64) f64{
	return if value<0{0.01}else{1.0}
}

[inline]
fn sigmoid(value f64) f64{
	return 1 / (1 + m.exp(-value))
}

[inline]
fn dsig(value f64) f64{
	sigx := sigmoid(value)
	return sigx*(1 - sigx)
}