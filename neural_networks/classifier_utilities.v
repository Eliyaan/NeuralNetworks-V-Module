module neural_networks

pub fn match_number_to_classifier_array(nb u8) []f64 {
	return []f64{len:10, init: if index == nb {1} else {0}}
}

pub fn match_classifier_array_to_number(a []f64) int {
	for i, elem in a {
		if elem == 1.0 {
			return i
		}
	}
	panic("No corresponding number")
}