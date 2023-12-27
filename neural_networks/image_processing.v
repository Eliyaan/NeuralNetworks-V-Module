module neural_networks

// import math
import rand

@[direct_array_access]
pub fn rand_noise(a []f64, noise_probability int, noise_range int) []f64 {
	if noise_probability > 0 && noise_range > 0 {
		mut output := a.clone()
		for mut elem in output {
			if rand.int_in_range(0, noise_probability) or {1} == 0 {
				elem += rand.f64_in_range(0, f64(noise_range)-elem) or {0.0}
			}
		}
		return output
	} else {
		return a
	}
}

@[direct_array_access]
pub fn get_center_of_mass(a []f64, x_size int, y_size int) (int, int) {
	mut x := 0.0
	mut y := 0.0
	mut cpt := 0.0 // to divide everything by the total of values
	for l in 0 .. y_size {
		for c in 0 .. x_size {
			px_value := a[l * y_size + c]
			if px_value != 0 {
				x += c * px_value
				y += l * px_value
				cpt += 1 * px_value
			}
		}
	}
	if cpt != 0 {
		x /= cpt
		y /= cpt
	}
	return int(x - 28 / 2), int(y - 28 / 2) // offset (half goal/crop image size)
}

@[direct_array_access]
pub fn center_image(a []f64, x_size int, y_size int) []f64 {
	offset_x, offset_y := get_center_of_mass(a, x_size, y_size)
	mut output := []f64{cap: x_size * y_size}
	for l in 0 .. y_size {
		for c in 0 .. x_size {
			if in_range(offset_x + c, offset_y + l, 0, 0, x_size, y_size) {
				output << a[a_coords(offset_y + l, offset_x + c, x_size)]
			} else {
				output << 0.0
			}
		}
	}
	return output
}

@[direct_array_access]
pub fn scale_img(a []f64, scale_goal f64, x_size int, y_size int) []f64 {
	base_side_x := x_size
	base_side_y := y_size
	scaled_side_x := ceil(f64(base_side_x) * scale_goal)
	scaled_side_y := ceil(f64(base_side_y) * scale_goal)
	if scaled_side_y != base_side_y && scaled_side_x != base_side_x {
		mut new_a := []f64{len: scaled_side_y * scaled_side_x, cap: scaled_side_y * scaled_side_x}
		for l in 0 .. scaled_side_y {
			for c in 0 .. scaled_side_x {
				// Index in the new array of the current pixel
				new_i := l * scaled_side_y + c
				// needs division (for proportionality) but only if needed :
				mut val_l := f64(l * (base_side_y - 1))
				mut val_c := f64(c * (base_side_x - 1))

				// if the division is a integer (it corresponds to an exact pixel)
				l_is_int := int(val_l) % (scaled_side_y - 1) != 0
				c_is_int := int(val_c) % (scaled_side_x - 1) != 0
				// divide
				val_l /= (scaled_side_y - 1)
				val_c /= (scaled_side_x - 1)
				int_val_l := int(val_l)
				int_val_c := int(val_c)
				// Take the right pixel values
				if l_is_int && c_is_int {
					new_a[new_i] = a[int(val_l) * base_side_x + int_val_c]
				} else if !(l_is_int || c_is_int) {  // none of them
					new_a[new_i] = a[a_coords(int_val_l, int_val_c, base_side_x)] * float_gap(val_c) * float_gap(val_l) +
						a[a_coords(int_val_l, ceil(val_c), base_side_x)] * float_offset(val_c) * float_gap(val_l) +
						a[a_coords(ceil(val_l), int_val_c, base_side_x)] * float_offset(val_l) * float_gap(val_c) +
						a[a_coords(ceil(val_l), ceil(val_c), base_side_x)] * float_offset(val_l) * float_offset(val_c)
				} else if l_is_int {  // exact line (not useful for squares I think but there if needed)
					new_a[new_i] = a[a_coords(int_val_l, int_val_c, base_side_x)] * float_gap(val_c) +
						a[a_coords(int_val_l, ceil(val_c), base_side_x)] * float_offset(val_c)
				} else {  // exact collumn (not useful for squares I think but there if needed)
					new_a[new_i] = a[a_coords(int_val_l, int_val_c, base_side_x)] * float_gap(val_l) + 
						a[a_coords(ceil(val_l), int_val_c, base_side_x)] * float_offset(val_l)
				}
			}
		}
		return new_a // needs to be cropped
	} else {
		return a
	}
}

@[inline]
pub fn a_coords(y int, x int, size int) int {
	return y * size + x
}

@[inline]
pub fn in_range[T](x T, y T, x_start T, y_start T, x_end T, y_end T) bool {
	return x >= x_start && x < x_end && y >= y_start && y < y_end
}

@[inline]
fn ceil(nb f64) int {
	return -int(-nb)
}

@[inline]
fn float_offset(f f64) f64 {
	return f - int(f)
}

@[inline]
fn float_gap(f f64) f64 {
	return 1 - float_offset(f)
}