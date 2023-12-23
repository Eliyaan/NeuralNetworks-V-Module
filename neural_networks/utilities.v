module neural_networks

import la
import rand

fn rand_array(nb int, range f64) []f64 {
	mut a := []f64{}
	for _ in 0 .. nb {
		a << rand.f64_in_range(-range, range) or { panic(err) }
	}
	return a
}

fn rand_2darray(nb_lines int, nb_cols int, range f64) [][]f64 {
	mut a := [][]f64{len: nb_lines, init: []f64{}}
	for mut line in a {
		line << rand_array(nb_cols, range)
	}
	return a
}

fn rand_matrix(nb_lines int, nb_cols int, range f64) &la.Matrix[f64] {
	return la.Matrix.deep2(rand_2darray(nb_lines, nb_cols, range))
}

fn vector_element_wise_mul(u []f64, v []f64) []f64 {
	mut result := []f64{len: u.len}
	for i, elem in u {
		result[i] = elem * v[i]
	}
	return result
}
