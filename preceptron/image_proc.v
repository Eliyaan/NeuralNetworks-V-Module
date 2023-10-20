module preceptron

import rand as rd
import math as m

/*
Center an image in a different size image (with noise in the offset and the output or not)
*/
pub fn offset_diff_sized_image(a []f64, x_size int, y_size int, x_goal int, y_goal int, noise bool) []f64 {
	mut offset_x, mut offset_y := get_center_of_mass(a, x_size, y_size, x_goal, y_goal)
	if noise {
		offset_x += rd.int_in_range(-4, 5) or { panic(err) }
		offset_y += rd.int_in_range(-4, 5) or { panic(err) }
	}
	mut output := []f64{}
	for l in 0 .. y_goal {
		for c in 0 .. x_goal {
			if offset_x + c >= 0 && offset_x + c < x_size && offset_y + l >= 0
				&& offset_y + l < y_size && (offset_y + l) * x_size + offset_x + c < a.len {
				output << a[(offset_y + l) * x_size + offset_x + c]
			} else {
				output << 0.0
			}
			if noise {
				output[l * x_goal + c] += f64(if (rd.int_in_range(0, 28) or { 50 }) == 0 { rd.int_in_range(0, 256 - int(output[l * x_goal + c])) or {
						0} } else { 0 })
			}
		}
	}
	return output
}

/*
Scale an image to a certain nb of pixels and then transform it into a 28 x 28 image
*/
pub fn scale_and_process_img(a []f64, base int, goal_x int, goal_y int, noise bool) []f64 {
	mut new_a := []f64{len: goal_x * goal_y}
	for l in 0 .. goal_y {
		for c in 0 .. goal_x {
			mut val_l := f64(l * (base - 1))
			mut val_c := f64(c * (base - 1))
			if int(val_l) % (goal_y - 1) == 0 && int(val_c) % (goal_x - 1) == 0 {
				new_a[l * goal_x + c] = a[int(val_l) / (goal_y - 1) * base +
					int(val_c) / (goal_x - 1)]
			} else {
				mut l_int := true
				mut c_int := true
				if int(val_l) % (goal_y - 1) != 0 {
					l_int = false
				}
				if int(val_c) % (goal_x - 1) != 0 {
					c_int = false
				}
				val_l /= (goal_y - 1)
				val_c /= (goal_x - 1)
				if !(l_int || c_int) {
					new_a[l * goal_x + c] = (a[int(m.floor(val_l) * base +
						m.floor(val_c))] * (1 - (val_c - int(val_c))) * (1 - (val_l - int(val_l))) +
						a[int(m.floor(val_l) * base +
						m.ceil(val_c))] * (val_c - int(val_c)) * (1 - (val_l - int(val_l))) +
						a[int(m.ceil(val_l) * base +
						m.floor(val_c))] * (val_l - int(val_l)) * (1 - (val_c - int(val_c))) +
						a[int(m.ceil(val_l) * base +
						m.ceil(val_c))] * (val_l - int(val_l)) * (val_c - int(val_c)))
				} else if l_int {
					new_a[l * goal_x + c] = a[int(val_l * base +
						m.floor(val_c))] * (1 - (val_c - int(val_c))) + a[int(val_l * base +
						m.ceil(val_c))] * (val_c - int(val_c))
				} else {
					new_a[l * goal_x + c] = a[int(m.floor(val_l) * base +
						val_c)] * (1 - (val_l - int(val_l))) + a[int(m.ceil(val_l) * base +
						val_c)] * (val_l - int(val_l))
				}
			}
		}
	}
	return offset_diff_sized_image(new_a, goal_x, goal_y, 28, 28, noise)
}

/*
Get the center of mass of the image
*/
pub fn get_center_of_mass(a []f64, x_size int, y_size int, x_goal int, y_goal int) (int, int) {
	mut x := 0.0
	mut y := 0.0
	mut cpt := 0.0
	for l in 0 .. y_size {
		for c in 0 .. x_size {
			if a[l * x_size + c] != 0 {
				x += c * a[l * x_size + c]
				y += l * a[l * x_size + c]
				cpt += 1 * a[l * x_size + c]
			}
		}
	}
	if cpt != 0 {
		x /= cpt
		y /= cpt
	}
	return int(x - x_goal / 2), int(y - y_goal / 2) // offset (half image size)
}
