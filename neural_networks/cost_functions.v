module neural_networks

import la

pub enum CostFunctions {
	mse
}

fn get_cost_function(f CostFunctions) (fn ([]f64, []f64) f64, fn ([]f64, []f64) []f64) {
	match f {
		.mse { return mse, mse_prime }
	}
}

pub fn mse(y_true []f64, y_pred []f64) f64 { // mean squared error
	not_squared_error := la.vector_add(1.0, y_true, -1.0, y_pred)
	mut mean := 0.0
	for elem in not_squared_error {
		mean += elem * elem
	}
	return mean / f64(not_squared_error.len)
}

pub fn mse_prime(y_true []f64, y_pred []f64) []f64 {
	return la.vector_add(1.0 * 2 / f64(y_true.len), y_true, -1.0 * 2 / f64(y_pred.len),
		y_pred)
}
