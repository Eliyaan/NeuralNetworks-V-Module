module neural_networks

import la

pub enum LayerType as i64 {
	dense
	activ
}

pub fn layer_type(l Layer) LayerType {
	match l {
		Dense { return .dense }
		Activation { return .activ }
		else { panic('strange') }
	}
}

pub fn layer_from_type(lt LayerType) Layer {
	match lt {
		.dense { return Dense{
				weights: la.Matrix.new[f64](0, 0)
				weights_gradient: la.Matrix.new[f64](0, 0)
				old_weights_gradient: la.Matrix.new[f64](0, 0)
			} }
		.activ { return Activation{
				activ: lrelu
				activ_prime: lrelu_prime
			} }
	}
	panic('Unknown LayerType value')
}

pub interface Layer {
mut:
	input  []f64
	output []f64
	forward(input []f64) []f64
	backward(output_gradient []f64) []f64
	apply_grad(nb_elems_seen int, lr f64, momentum f64)
	reset()
}
