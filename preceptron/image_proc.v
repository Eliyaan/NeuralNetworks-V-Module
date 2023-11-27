module preceptron

import rand as rd
import math as m
import os


/*
Load data from the mnist dataset
0 < nb_training <= 60000
0 < nb_tests <= 10000
*/
[direct_array_access]
pub fn (mut nn NeuralNetwork) load_mnist(nb_training int, nb_tests int, noise_strength int, scale_range int, rotation_range int, offset_range int) {
	println('Loading mnist...')
	test_labels := os.open('mnist\\t10k-labels-idx1-ubyte') or { panic(err) }
	test_images := os.open('mnist\\t10k-images-idx3-ubyte') or { panic(err) }
	nn.test_inputs = [][]f64{}
	nn.expected_test_outputs = [][]f64{}
	for i in 0 .. nb_tests {
		nn.test_inputs << [
			process_img(test_images.read_bytes_at(784, i * 784 + 16).map(f64(it)), 28, 28 + rd.int_in_range(-scale_range, scale_range+1) or { panic(err) }, 28 + rd.int_in_range(-scale_range, scale_range+1) or {panic(err)}, noise_strength, nn.image_size_goal, rd.f64_in_range(-rotation_range, rotation_range) or {panic(err)}, offset_range)
		]
		nn.expected_test_outputs << [
			match test_labels.read_bytes_at(1, i + 8)[0] {
				0 { [f64(1), 0, 0, 0, 0, 0, 0, 0, 0, 0] }
				1 { [f64(0), 1, 0, 0, 0, 0, 0, 0, 0, 0] }
				2 { [f64(0), 0, 1, 0, 0, 0, 0, 0, 0, 0] }
				3 { [f64(0), 0, 0, 1, 0, 0, 0, 0, 0, 0] }
				4 { [f64(0), 0, 0, 0, 1, 0, 0, 0, 0, 0] }
				5 { [f64(0), 0, 0, 0, 0, 1, 0, 0, 0, 0] }
				6 { [f64(0), 0, 0, 0, 0, 0, 1, 0, 0, 0] }
				7 { [f64(0), 0, 0, 0, 0, 0, 0, 1, 0, 0] }
				8 { [f64(0), 0, 0, 0, 0, 0, 0, 0, 1, 0] }
				9 { [f64(0), 0, 0, 0, 0, 0, 0, 0, 0, 1] }
				else { panic('Match test outputs') }
			},
		]
	}
	train_labels := os.open('mnist\\train-labels-idx1-ubyte') or { panic(err) }
	train_images := os.open('mnist\\train-images-idx3-ubyte') or { panic(err) }
	nn.training_inputs = [][]f64{}
	nn.expected_training_outputs = [][]f64{}
	for i in 0 .. nb_training {
		nn.training_inputs << [
			process_img(train_images.read_bytes_at(784, i * 784 + 16).map(f64(it)),28, 28 + rd.int_in_range(-scale_range, scale_range+1) or { panic(err) }, 28 + rd.int_in_range(-scale_range, scale_range+1) or {panic(err)}, noise_strength, nn.image_size_goal, rd.f64_in_range(-rotation_range, rotation_range) or {panic(err)}, offset_range)
		]
		nn.expected_training_outputs << [
			match train_labels.read_bytes_at(1, i + 8)[0] {
				0 { [f64(1), 0, 0, 0, 0, 0, 0, 0, 0, 0] }
				1 { [f64(0), 1, 0, 0, 0, 0, 0, 0, 0, 0] }
				2 { [f64(0), 0, 1, 0, 0, 0, 0, 0, 0, 0] }
				3 { [f64(0), 0, 0, 1, 0, 0, 0, 0, 0, 0] }
				4 { [f64(0), 0, 0, 0, 1, 0, 0, 0, 0, 0] }
				5 { [f64(0), 0, 0, 0, 0, 1, 0, 0, 0, 0] }
				6 { [f64(0), 0, 0, 0, 0, 0, 1, 0, 0, 0] }
				7 { [f64(0), 0, 0, 0, 0, 0, 0, 1, 0, 0] }
				8 { [f64(0), 0, 0, 0, 0, 0, 0, 0, 1, 0] }
				9 { [f64(0), 0, 0, 0, 0, 0, 0, 0, 0, 1] }
				else { panic('bu') }
			},
		]
	}
	println('Finished loading mnist!')
}


[direct_array_access; inline]
fn max(a []f64) f64 {
    mut highest := 0
    for nb, val in a {
        if val > a[highest] {
            highest = nb
        }
    }
    return a[highest]
}

[direct_array_access; inline]
fn min(a []f64) f64 {
    mut highest := 0
    for nb, val in a {
        if val < a[highest] {
            highest = nb
        }
    }
    return a[highest]
}

[inline]
fn round_to_greater(nb f64) f64 {
    if nb >= 0 {
        return ceil_f(m.round_sig(nb, 5))
    }else {
        return f64(int(m.round_sig(nb, 5)))
    }
}

[inline]
fn floor_f(nb f64) f64 {
	return f64(int(nb))
}

[inline]
fn ceil_f(nb f64) f64 {
	return -floor_f(-nb)
}

/*
Center an image in a different size image (with noise in the offset and the output or not)
*/
[direct_array_access]
pub fn offset_diff_sized_image(a []f64, x_size int, y_size int, x_goal int, y_goal int, noise_strength int, offset_range int) []f64 {
	mut offset_x, mut offset_y := get_center_of_mass(a, x_size, y_size, x_goal, y_goal)
	if offset_range > 0 {
		offset_x += rd.int_in_range(-offset_range, offset_range+1) or { panic(err) }
		offset_y += rd.int_in_range(-offset_range, offset_range+1) or { panic(err) }
	}
	mut output := []f64{cap:y_goal*x_goal}
	for l in 0 .. y_goal {
		for c in 0 .. x_goal {
			if offset_x + c >= 0 && offset_x + c < x_size && offset_y + l >= 0
				&& offset_y + l < y_size && (offset_y + l) * x_size + offset_x + c < a.len {
				output << a[(offset_y + l) * x_size + offset_x + c]
			} else {
				output << 0.0
			}
			if noise_strength > 0 {
				output[l * x_goal + c] += f64(if (rd.int_in_range(0, 28) or { 50 }) == 0 { ((rd.int_in_range(0, m.max(0, noise_strength)) or {0})+int(output[l * x_goal + c]))%256 } else { 0 })
			}
		}
	}
	return output
}

[direct_array_access]
pub fn rotate(a []f64, alpha f64, im_size int) ([]f64, int) { 
	if alpha != 0  {
		angle := m.radians(alpha)

		full_x := (im_size-1)*m.cos(angle) - (im_size-1)*m.sin(angle)
		full_y := (im_size-1)*m.sin(angle) + (im_size-1)*m.cos(angle)
		only_x_x := (im_size-1)*m.cos(angle)// - 0*m.sin(angle)
		only_x_y := (im_size-1)*m.sin(angle)// + 0*m.cos(angle)
		only_y_x := -(im_size-1)*m.sin(angle)
		only_y_y := (im_size-1)*m.cos(angle)
		max_x := (max([full_x, only_x_x, only_y_x, 0]))
		min_x := (min([full_x, only_x_x, only_y_x, 0]))
		max_y := (max([full_y, only_x_y, only_y_y, 0]))
		min_y := (min([full_y, only_x_y, only_y_y, 0]))
		size_x := (max_x-min_x+1)
		size_y := (max_y-min_y+1)

		
		side := int(m.sqrt(round_to_greater(round_to_greater(size_x)*round_to_greater(size_y))))
		mut output := []f64{cap:side*side}
		mut twod_output := [][]f64{len:side, cap:side, init:[]f64{len:side, cap:side}} // need to opti

		for i, pixel in a {
			if pixel > 0 {
				x := f64(i%im_size)-(f64(im_size))/2.0+0.5
				y := f64(i/im_size)-(f64(im_size))/2.0+0.5
				xn := x*m.cos(angle) - y*m.sin(angle)
				yn := x*m.sin(angle) + y*m.cos(angle)
				
				array_coord_y := m.max(yn+(side)/2-0.5, 0)
				array_coord_x := m.max(xn+(side)/2-0.5, 0)
				twod_output[int(array_coord_y)][int(array_coord_x)] += pixel*(1-(array_coord_y - int(array_coord_y)))*(1-(array_coord_x - int(array_coord_x)))
				if twod_output[int(array_coord_y)][int(array_coord_x)] > 255 {
					twod_output[int(array_coord_y)][int(array_coord_x)] = 255
				}
				twod_output[int(array_coord_y)][int(ceil_f(array_coord_x))] += pixel*(1-(array_coord_y - int(array_coord_y)))*(array_coord_x - int(array_coord_x))
				if twod_output[int(array_coord_y)][int(ceil_f(array_coord_x))] > 255 {
					twod_output[int(array_coord_y)][int(ceil_f(array_coord_x))] = 255
				}
				twod_output[int(ceil_f(array_coord_y))][int(array_coord_x)] += pixel*(array_coord_y - int(array_coord_y))*(1-(array_coord_x - int(array_coord_x)))
				if twod_output[int(ceil_f(array_coord_y))][int(array_coord_x)] > 255 {
					twod_output[int(ceil_f(array_coord_y))][int(array_coord_x)] = 255
				}
				twod_output[int(ceil_f(array_coord_y))][int(ceil_f(array_coord_x))] += pixel*(array_coord_y - int(array_coord_y))*(array_coord_x - int(array_coord_x))
				if twod_output[int(ceil_f(array_coord_y))][int(ceil_f(array_coord_x))] > 255 {
					twod_output[int(ceil_f(array_coord_y))][int(ceil_f(array_coord_x))] = 255
				}
			}
		}
		for row in twod_output {
			output << row
		}
		return output, side
	}else {
		return a, im_size
	}
}

[direct_array_access]
pub fn process_img(a []f64, base int, goal_x int, goal_y int, noise_strength int, image_size_goal int, angle f64, offset_range int) []f64 {
	rotated_a, new_base := rotate(a, angle, base)
	return scale_and_process_img(offset_diff_sized_image(rotated_a, new_base, new_base, image_size_goal, image_size_goal, 0, 0), image_size_goal, goal_x, goal_y, noise_strength, image_size_goal, offset_range)
}

/*
Scale an image to a certain nb of pixels and then transform it into a 28 x 28 image
*/
[direct_array_access]
pub fn scale_and_process_img(a []f64, base int, goal_x int, goal_y int, noise_strength int, image_size_goal int, offset_range int) []f64 {
	mut new_a := []f64{len: goal_x * goal_y, cap:goal_x * goal_y}
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
					new_a[l * goal_x + c] = (a[int(int(val_l) * base +
						int(val_c))] * (1 - (val_c - int(val_c))) * (1 - (val_l - int(val_l))) +
						a[int(int(val_l) * base +
						ceil_f(val_c))] * (val_c - int(val_c)) * (1 - (val_l - int(val_l))) +
						a[int(ceil_f(val_l) * base +
						int(val_c))] * (val_l - int(val_l)) * (1 - (val_c - int(val_c))) +
						a[int(ceil_f(val_l) * base +
						ceil_f(val_c))] * (val_l - int(val_l)) * (val_c - int(val_c)))
				} else if l_int {
					new_a[l * goal_x + c] = a[int(val_l * base +
						int(val_c))] * (1 - (val_c - int(val_c))) + a[int(val_l * base +
						ceil_f(val_c))] * (val_c - int(val_c))
				} else {
					new_a[l * goal_x + c] = a[int(int(val_l) * base +
						val_c)] * (1 - (val_l - int(val_l))) + a[int(ceil_f(val_l) * base +
						val_c)] * (val_l - int(val_l))
				}
			}
		}
	}
	return offset_diff_sized_image(new_a, goal_x, goal_y, image_size_goal, image_size_goal, noise_strength, offset_range)
}

/*
Get the center of mass of the image
*/
[direct_array_access]
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
