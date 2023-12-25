import neural_networks as nn
import os
import math
import gg
import gx
import rand as rd
import rand.config as rdconfig

/*
/!\ the functions are for square images and/or for this specific project most of the time, but the core concepts are here
*/


/*
TODO:
Buttons with click color change, text change etc
to make the module transform ids into ints when creating function etc / in arguments
*/

const (
	// Mocha:
	surface0		= gx.Color{30, 30, 46, 255}
	lavender		= gx.Color{180, 190, 254, 255}
	blue			= gx.Color{137, 180, 250, 255}
	sapphire		= gx.Color{116, 199, 236, 255}
	sky				= gx.Color{137, 220, 235, 255}
	teal			= gx.Color{148, 227, 213, 255}
	green			= gx.Color{166, 214, 161, 255}
	yellow			= gx.Color{249, 226, 175, 255}
	peach			= gx.Color{250, 179, 135, 255}
	maroon			= gx.Color{235, 160, 172, 255}
	red				= gx.Color{243, 139, 168, 255}
	mauve			= gx.Color{203, 166, 247, 255}
	pink			= gx.Color{245, 194, 231, 255}
	flamingo		= gx.Color{242, 205, 205, 255}
	rosewater		= gx.Color{245, 224, 220, 255}
	mocha_text			= gx.Color{205, 214, 244, 255}
	
    win_width   	= 601
    win_height  	= 601
    bg_color    	= surface0
	px_size			= 4
	x_buttons_offset= 300
	buttons_shape	= ButtonShape{20, 20, 5, .top_right}
)

struct Id {
	id Ids
}

enum Ids {
	@none 	
	img_nb_text
	img_label
	final_scale_text
	noise_range_text
	noise_probability_text
	scale_range_text
	rota_range_text
}

struct App {
mut:
    gg    &gg.Context = unsafe { nil }
	clickables []Clickable
	elements []Element
	base_dataset nn.Dataset
	dataset nn.Dataset
	actual_image int
	final_scale f64 = 0.75
	noise_probability int = 15
	noise_range int = 255
	scale_range f64 = 0.1
	rota_range int = 30
	final_nb_pixels int
	augment_asked bool = true
}

fn main() {
    mut app := &App{}
    app.gg = gg.new_context(
        width: win_width
        height: win_height
        create_window: true
        window_title: 'Mnist Data Augmentation Visualiser'
        user_data: app
        bg_color: bg_color
        frame_fn: on_frame
        event_fn: on_event
        sample_count: 4
		ui_mode: true
    )
	app.base_dataset = load_mnist_training(100)
	app.dataset = app.base_dataset.clone()

	plus_text := Text{Id{}, 0, 0, "+", gx.TextCfg{color:surface0, size:20, align:.center, vertical_align:.middle}}
	minus_text := Text{Id{}, 0, 0, "-", gx.TextCfg{color:surface0, size:20, align:.center, vertical_align:.middle}}
	reload_text := Text{Id{}, 0, 0, "~", gx.TextCfg{color:surface0, size:20, align:.center, vertical_align:.middle}}
	button_description_cfg := gx.TextCfg{color:mocha_text, size:20, align:.right, vertical_align:.top}

	app.elements << Text{Id{.img_label}, 14*px_size, 28*px_size, match_classifier_array_to_number(app.dataset.expected_outputs[app.actual_image]).str(), gx.TextCfg{color:mocha_text, size:20, align:.center, vertical_align:.top}}

	app.clickables << Button{Id{}, x_buttons_offset+50, 10, buttons_shape, minus_text, red, prev_img}
	app.clickables << Button{Id{}, x_buttons_offset+75, 10, buttons_shape, plus_text, green, next_img}
	app.elements << Text{Id{.img_nb_text}, x_buttons_offset+45, 10, "Image n°${app.actual_image}", button_description_cfg}

	app.clickables << Button{Id{}, x_buttons_offset+105, 10, buttons_shape, reload_text, flamingo, ask_augment}

	app.clickables << Button{Id{}, x_buttons_offset+50, 35, buttons_shape, minus_text, red, sub_final_scale}
	app.clickables << Button{Id{}, x_buttons_offset+75, 35, buttons_shape, plus_text, green, add_final_scale}
	app.elements << Text{Id{.final_scale_text}, x_buttons_offset+45, 35, "Final scale: ${app.final_scale}", button_description_cfg}

	app.clickables << Button{Id{}, x_buttons_offset+50, 60, buttons_shape, minus_text, red, sub_noise_range}
	app.clickables << Button{Id{}, x_buttons_offset+75, 60, buttons_shape, plus_text, green, add_noise_range}
	app.elements << Text{Id{.noise_range_text}, x_buttons_offset+45, 60, "Noise range: ${app.noise_range}", button_description_cfg}

	app.clickables << Button{Id{}, x_buttons_offset+50, 85, buttons_shape, minus_text, red, sub_noise_probability}
	app.clickables << Button{Id{}, x_buttons_offset+75, 85, buttons_shape, plus_text, green, add_noise_probability}
	app.elements << Text{Id{.noise_probability_text}, x_buttons_offset+45, 85, "Noise probability: ${app.noise_probability}", button_description_cfg}

	app.clickables << Button{Id{}, x_buttons_offset+50, 110, buttons_shape, minus_text, red, sub_scale_range}
	app.clickables << Button{Id{}, x_buttons_offset+75, 110, buttons_shape, plus_text, green, add_scale_range}
	app.elements << Text{Id{.scale_range_text}, x_buttons_offset+45, 110, "Scale range: ${app.scale_range}", button_description_cfg}

	app.clickables << Button{Id{}, x_buttons_offset+50, 135, buttons_shape, minus_text, red, sub_rota_range}
	app.clickables << Button{Id{}, x_buttons_offset+75, 135, buttons_shape, plus_text, green, add_rota_range}
	app.elements << Text{Id{.rota_range_text}, x_buttons_offset+45, 135, "Rotation range: ${app.rota_range}", button_description_cfg}

	app.augment_images()
    app.gg.run()
}

fn on_frame(mut app App) {
    //Draw
    app.gg.begin()
	if app.augment_asked {
		app.augment(app.actual_image)
		app.augment_asked = false
		mut img_label_text := app.get_element_with_id(Id{.img_label}) or {panic(err)}
		img_label_text.y = app.final_nb_pixels*px_size
		img_label_text.x = img_label_text.y/2
	}
	app.render_image()
	app.render_clickables()
	app.render_elements()
    app.gg.end()
}

fn on_event(e &gg.Event, mut app App){
    match e.typ {
        .key_down {
            match e.key_code {
                .escape {app.gg.quit()}
                else {}
            }
        }
        .mouse_up {
            match e.mouse_button{
                .left{
					app.check(int(e.mouse_x), int(e.mouse_y))
				}
                else{}
        	}
		}
        else {}
    }
}

fn (mut app App) get_clickables_with_id(id Id) []&Clickable{
	mut result := []&Clickable{}
	for obj in app.clickables {
		unsafe {
			if obj.id.id == id.id {
				result << &obj
			}
		}
	}
	return result
}

fn (mut app App) get_clickable_with_id(id Id) !&Clickable{
	for obj in app.clickables {
		unsafe {
			if obj.id.id == id.id {
				return &obj
			}
		}
	}
	return error("No object matching")
}

fn (mut app App) get_elements_with_id(id Id) []&Element{
	mut result := []&Element{}
	for obj in app.elements {
		unsafe {
			if obj.id.id == id.id {
				result << &obj
			}
		}
	}
	return result
}

fn (mut app App) get_element_with_id(id Id) !&Element{
	for obj in app.elements {
		unsafe {
			if obj.id.id == id.id {
				return &obj
			}
		}
	}
	return error("No object matching")
}


interface Shape {
mut:
	width f32
	height f32
	relative_pos Pos
}

interface Clickable {
mut:
	id Id
	x f32
	y f32
	shape Shape
	click_func fn (mut app App)
	render(mut app App)
}

interface Element {
mut:
	id Id
	x f32
	y f32
	render(mut app App, x_offset f32, y_offset f32)
}

struct ButtonShape {
mut:
	width f32
	height f32
	rounded int
	relative_pos Pos
}

@[heap]
struct Button {
mut:
	id Id
	x f32
	y f32
	shape Shape
	text Text
	color gx.Color
	click_func fn (mut app App) @[required]
}

@[heap]
struct Text {
mut:
	id Id
	x f32
	y f32
	text string
	cfg gx.TextCfg
}

fn (t Text) render(mut app App, x_offset f32, y_offset f32) {
	if t.text != "" {
		app.gg.draw_text(int(t.x + x_offset), int(t.y + y_offset), t.text, t.cfg)
	}
}

fn (mut app App) augment_images() {
	for i, _ in app.dataset.inputs {
		app.augment(i)
	}
}

fn (mut app App) augment(i int) {
	app.dataset.inputs[i] = rotate(app.base_dataset.inputs[i], rd.f64_in_range(-app.rota_range, app.rota_range) or {0})
	app.dataset.inputs[i] = scale_img(app.dataset.inputs[i], rd.f64_in_range(1-app.scale_range, 1+app.scale_range) or {0})
	app.dataset.inputs[i] = rand_noise(app.dataset.inputs[i], app.noise_probability, app.noise_range)
	app.dataset.inputs[i] = center_image(app.dataset.inputs[i])
	app.dataset.inputs[i] = crop(app.dataset.inputs[i])
	app.dataset.inputs[i] = scale_img(app.dataset.inputs[i], app.final_scale)
	app.final_nb_pixels = int(math.sqrt(app.dataset.inputs[i].len))
}

fn (mut app App) check(mouse_x int, mouse_y int) { // need to do the relative pos
	for obj in app.clickables {
		mut x_coo := obj.x
		mut y_coo := obj.y
		match obj.shape.relative_pos {
			.center {
				x_coo -= obj.shape.width/2
				y_coo -= obj.shape.height/2
			}
			.top_right {}
			.top_left {
				x_coo -= obj.shape.width
			}
			.bottom_right {
				y_coo -= obj.shape.height
			}
			.bottom_left {
				y_coo -= obj.shape.height
				x_coo -= obj.shape.width
			}
		}
		if in_range(f32(mouse_x), f32(mouse_y), x_coo, y_coo, x_coo + obj.shape.width, y_coo + obj.shape.height) {
			obj.click_func(mut app)
		}
	}
}

fn (b Button) render(mut app App) {
	mut x_coo := b.x
	mut y_coo := b.y
	mut text_x_offset := b.x
	mut text_y_offset := b.y
	match b.shape.relative_pos {
		.center {
			x_coo -= b.shape.width/2
			y_coo -= b.shape.height/2
		}
		.top_right {
			text_x_offset += b.shape.width/2
			text_y_offset += b.shape.height/2
		}
		.top_left {
			x_coo -= b.shape.width
			text_x_offset -= b.shape.width/2
			text_y_offset -= b.shape.height/2
		}
		.bottom_right {
			y_coo -= b.shape.height
			text_x_offset += b.shape.width/2
			text_y_offset -= b.shape.height/2
		}
		.bottom_left {
			y_coo -= b.shape.height
			x_coo -= b.shape.width
			text_x_offset -= b.shape.width/2
			text_y_offset -= b.shape.height/2
		}
	}
	match b.shape {
		ButtonShape {app.gg.draw_rounded_rect_filled(x_coo, y_coo, b.shape.width, b.shape.height, b.shape.rounded, b.color)}
		else {app.gg.draw_rect_filled(x_coo, y_coo, b.shape.width, b.shape.height, b.color)}
	}
	b.text.render(mut app, text_x_offset, text_y_offset)
}

fn (mut app App) change_text(id Id, text string) {
	mut text_obj := app.get_element_with_id(id) or {panic(err)}
	if mut text_obj is Text {
		text_obj.text = text
	}
}

fn next_img(mut app App) {
	if app.actual_image == app.dataset.inputs.len - 1 {
		app.actual_image = 0
	}else{
		app.actual_image += 1
	}
	app.change_text(Id{.img_nb_text}, "Image n°${app.actual_image}")
	app.change_text(Id{.img_label}, match_classifier_array_to_number(app.dataset.expected_outputs[app.actual_image]).str())

	ask_augment(mut app)
}

fn prev_img(mut app App) {
	if app.actual_image == 0 {
		app.actual_image = app.dataset.inputs.len - 1
	}else{
		app.actual_image -= 1
	}
	app.change_text(Id{.img_nb_text}, "Image n°${app.actual_image}")
	app.change_text(Id{.img_label}, match_classifier_array_to_number(app.dataset.expected_outputs[app.actual_image]).str())
	ask_augment(mut app)
}

fn ask_augment(mut app App) {
	app.augment_asked = true
}

fn add_final_scale(mut app App) {
	app.final_scale = math.round_sig(app.final_scale+0.05, 2)
	app.change_text(Id{.final_scale_text}, "Final scale: ${app.final_scale}")
	ask_augment(mut app)
}

fn sub_final_scale(mut app App) {
	app.final_scale = math.round_sig(app.final_scale-0.05, 2)
	app.change_text(Id{.final_scale_text}, "Final scale: ${app.final_scale}")
	ask_augment(mut app)
}

fn add_noise_range(mut app App) {
	app.noise_range += 1
	app.change_text(Id{.noise_range_text}, "Noise range: ${app.noise_range}")
	ask_augment(mut app)
}

fn sub_noise_range(mut app App) {
	app.noise_range -= 1
	app.change_text(Id{.noise_range_text}, "Noise range: ${app.noise_range}")
	ask_augment(mut app)
}

fn add_noise_probability(mut app App) {
	app.noise_probability += 1
	app.change_text(Id{.noise_probability_text}, "Noise probability: ${app.noise_probability}")
	ask_augment(mut app)
}

fn sub_noise_probability(mut app App) {
	app.noise_probability -= 1
	app.change_text(Id{.noise_probability_text}, "Noise probability: ${app.noise_probability}")
	ask_augment(mut app)
}

fn add_scale_range(mut app App) {
	app.scale_range = math.round_sig(app.scale_range+0.01, 2)
	app.change_text(Id{.scale_range_text}, "Scale range: ${app.scale_range}")
	ask_augment(mut app)
}

fn sub_scale_range(mut app App) {
	app.scale_range = math.round_sig(app.scale_range-0.01, 2)
	app.change_text(Id{.scale_range_text}, "Scale range: ${app.scale_range}")
	ask_augment(mut app)
}

fn add_rota_range(mut app App) {
	app.rota_range += 1
	app.change_text(Id{.rota_range_text}, "Rotation range: ${app.rota_range}")
	ask_augment(mut app)
}

fn sub_rota_range(mut app App) {
	app.rota_range -= 1
	app.change_text(Id{.rota_range_text}, "Rotation range: ${app.rota_range}")
	ask_augment(mut app)
}

enum Pos {
	center
	top_right
	top_left
	bottom_right
	bottom_left
}

fn (mut app App) render_clickables() {
	for mut obj in app.clickables {
		obj.render(mut app)
	}
}

fn (mut app App) render_elements() {
	for mut elem in app.elements {
		elem.render(mut app, 0, 0)
	}
}

fn (mut app App) render_image() {
	img_size := int(math.sqrt(app.dataset.inputs[app.actual_image].len))
	for y in 0..img_size {
		for x in 0..img_size {
			px := u8(app.dataset.inputs[app.actual_image][y*img_size+x])
			app.gg.draw_rect_filled(f32(x*px_size), f32(y*px_size), px_size, px_size, gx.Color{px,px,px,255})
		}
	}
}

@[direct_array_access]
fn load_mnist_training(nb_training int) nn.Dataset {
	println('Loading training mnist...')
	train_labels := os.open('mnist\\train-labels-idx1-ubyte') or { panic(err) }
	train_images := os.open('mnist\\train-images-idx3-ubyte') or { panic(err) }
	mut dataset := nn.Dataset{[][]f64{}, [][]f64{}}
	mut order_array := []u64{len:nb_training, init:u64(index)}
	rd.shuffle(mut order_array, rdconfig.ShuffleConfigStruct{}) or {panic(err)}
	for i in order_array {
		dataset.inputs << [train_images.read_bytes_at(784, i * 784 + 16).map(f64(it))]
		dataset.expected_outputs << [
			match_number_to_classifier_array(train_labels.read_bytes_at(1, i + 8)[0])
		]
	}
	println('Finished loading training mnist!')
	return dataset
}

@[direct_array_access]
fn load_mnist_test(nb_tests int) nn.Dataset {
	println('Loading test mnist...')
	test_labels := os.open('mnist\\t10k-labels-idx1-ubyte') or { panic(err) }
	test_images := os.open('mnist\\t10k-images-idx3-ubyte') or { panic(err) }
	mut dataset := nn.Dataset{[][]f64{}, [][]f64{}}
	for i in 0 .. nb_tests {
		dataset.inputs << [test_images.read_bytes_at(784, i * 784 + 16).map(f64(it))]
		dataset.expected_outputs << [
			match_number_to_classifier_array(test_labels.read_bytes_at(1, i + 8)[0])
		]
	}
	println('Finished loading test mnist!')
	return dataset
}

fn match_number_to_classifier_array(nb u8) []f64 {
	return []f64{len:10, init: if index == nb {1} else {0}}
}

fn match_classifier_array_to_number(a []f64) int {
	for i, elem in a {
		if elem == 1.0 {
			return i
		}
	}
	panic("No corresponding number")
}

@[direct_array_access]
fn get_center_of_mass(a []f64) (int, int) {
	mut x := 0.0
	mut y := 0.0
	mut cpt := 0.0 // to divide everything by the total of values
	im_size := int(math.sqrt(a.len))
	for l in 0 .. im_size {
		for c in 0 .. im_size {
			px_value := a[l * im_size + c]
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
pub fn center_image(a []f64) []f64 {
	offset_x, offset_y := get_center_of_mass(a)
	base_im_size := ceil(math.sqrt(a.len))
	mut output := []f64{cap: base_im_size * base_im_size}
	for l in 0 .. base_im_size {
		for c in 0 .. base_im_size {
			if in_range(offset_x + c, offset_y + l, 0, 0, base_im_size, base_im_size) {
				output << a[a_coords(offset_y + l, offset_x + c, base_im_size)]
			} else {
				output << 0.0
			}
		}
	}
	return output
}

@[direct_array_access]
fn crop(a []f64) []f64 {
	base_im_size := ceil(math.sqrt(a.len))
	mut output := []f64{cap: 28 * 28}
	for l in 0 .. 28 {
		for c in 0 .. 28 {
			if in_range(c, l, 0, 0, base_im_size, base_im_size) {
				output << a[a_coords(l, c, base_im_size)]
			} else {
				output << 0.0
			}
		}
	}
	return output
}

@[direct_array_access]
pub fn rand_noise(a []f64, noise_probability int, noise_range int) []f64 {
	if noise_probability > 0 && noise_range > 0 {
		mut output := a.clone()
		for mut elem in output {
			if rd.int_in_range(0, noise_probability) or {1} == 0 {
				elem += rd.f64_in_range(0, f64(noise_range)-elem) or {0.0}
			}
		}
		return output
	} else {
		return a
	}
}

@[direct_array_access]
pub fn scale_img(a []f64, scale_goal f64) []f64 {
	base_side := int(math.sqrt(a.len))
	scaled_side := ceil(f64(base_side) * scale_goal)
	if scaled_side != base_side {
		mut new_a := []f64{len: scaled_side * scaled_side, cap: scaled_side * scaled_side}
		for l in 0 .. scaled_side {
			for c in 0 .. scaled_side {
				// Index in the new array of the current pixel
				new_i := l * scaled_side + c
				// needs division (for proportionality) but only if needed :
				mut val_l := f64(l * (base_side - 1))
				mut val_c := f64(c * (base_side - 1))

				// if the division is a integer (it corresponds to an exact pixel)
				l_is_int := int(val_l) % (scaled_side - 1) != 0
				c_is_int := int(val_c) % (scaled_side - 1) != 0
				// divide
				val_l /= (scaled_side - 1)
				val_c /= (scaled_side - 1)
				int_val_l := int(val_l)
				int_val_c := int(val_c)
				// Take the right pixel values
				if l_is_int && c_is_int {
					new_a[new_i] = a[int(val_l) * base_side + int_val_c]
				} else if !(l_is_int || c_is_int) {  // none of them
					new_a[new_i] = a[a_coords(int_val_l, int_val_c, base_side)] * float_gap(val_c) * float_gap(val_l) +
						a[a_coords(int_val_l, ceil(val_c), base_side)] * float_offset(val_c) * float_gap(val_l) +
						a[a_coords(ceil(val_l), int_val_c, base_side)] * float_offset(val_l) * float_gap(val_c) +
						a[a_coords(ceil(val_l), ceil(val_c), base_side)] * float_offset(val_l) * float_offset(val_c)
				} else if l_is_int {  // exact line (not useful for squares I think but there if needed)
					new_a[new_i] = a[a_coords(int_val_l, int_val_c, base_side)] * float_gap(val_c) +
						a[a_coords(int_val_l, ceil(val_c), base_side)] * float_offset(val_c)
				} else {  // exact collumn (not useful for squares I think but there if needed)
					new_a[new_i] = a[a_coords(int_val_l, int_val_c, base_side)] * float_gap(val_l) + 
						a[a_coords(ceil(val_l), int_val_c, base_side)] * float_offset(val_l)
				}
			}
		}
		return new_a // needs to be cropped
	} else {
		return a
	}
}

@[inline]
fn a_coords(y int, x int, size int) int {
	return y * size + x
}

// the decimal part
@[inline]
fn float_offset(f f64) f64 {
	return f - int(f)
}

@[inline]
fn float_gap(f f64) f64 {
	return 1 - float_offset(f)
}

@[direct_array_access]
pub fn rotate(a []f64, alpha f64) []f64 {
	if alpha != 0 {
		angle := math.radians(alpha)

		full_x := (28 - 1) * math.cos(angle) - (28 - 1) * math.sin(angle)
		full_y := (28 - 1) * math.sin(angle) + (28 - 1) * math.cos(angle)
		only_x_x := (28 - 1) * math.cos(angle) // - 0*math.sin(angle)
		only_x_y := (28 - 1) * math.sin(angle) // + 0*math.cos(angle)
		only_y_x := -(28 - 1) * math.sin(angle)
		only_y_y := (28 - 1) * math.cos(angle)
		max_x := (max([full_x, only_x_x, only_y_x, 0]))
		min_x := (min([full_x, only_x_x, only_y_x, 0]))
		max_y := (max([full_y, only_x_y, only_y_y, 0]))
		min_y := (min([full_y, only_x_y, only_y_y, 0]))
		size_x := f64(max_x - min_x + 1)
		size_y := f64(max_y - min_y + 1)

		side := int(math.sqrt(round_to_greater(round_to_greater(size_x) * round_to_greater(size_y))))
		mut output := []f64{cap: side * side}
		mut twod_output := [][]f64{len: side, cap: side, init: []f64{len: side, cap: side}} // need to opti
		for i, pixel in a {
			if pixel > 0 {
				x := f64(i % 28) - (f64(28) / 2.0) + 0.5
				y := f64(i / 28) - (f64(28) / 2.0) + 0.5
				xn := x * math.cos(angle) - y * math.sin(angle)
				yn := x * math.sin(angle) + y * math.cos(angle)

				array_coord_y := math.max(yn + side / 2 - 0.5, 0)
				array_coord_x := math.max(xn + side / 2 - 0.5, 0)
				twod_output[int(array_coord_y)][int(array_coord_x)] += pixel * float_gap(array_coord_y) * float_gap(array_coord_x)
				if twod_output[int(array_coord_y)][int(array_coord_x)] > 255 {
					twod_output[int(array_coord_y)][int(array_coord_x)] = 255
				}
				twod_output[int(array_coord_y)][int(ceil(array_coord_x))] += pixel * float_gap(array_coord_y ) * float_offset(array_coord_x)
				if twod_output[int(array_coord_y)][int(ceil(array_coord_x))] > 255 {
					twod_output[int(array_coord_y)][int(ceil(array_coord_x))] = 255
				}
				twod_output[int(ceil(array_coord_y))][int(array_coord_x)] += pixel * float_offset(array_coord_y) * float_gap(array_coord_x)
				if twod_output[int(ceil(array_coord_y))][int(array_coord_x)] > 255 {
					twod_output[int(ceil(array_coord_y))][int(array_coord_x)] = 255
				}
				twod_output[int(ceil(array_coord_y))][int(ceil(array_coord_x))] += pixel * float_offset(array_coord_y) * float_offset(array_coord_x)
				if twod_output[int(ceil(array_coord_y))][int(ceil(array_coord_x))] > 255 {
					twod_output[int(ceil(array_coord_y))][int(ceil(array_coord_x))] = 255
				}
			}
		}
		for row in twod_output {
			output << row
		}
		return output
	} else {
		return a
	}
}

@[direct_array_access; inline]
fn max(a []f64) f64 {
	mut highest := 0
	for nb, val in a {
		if val > a[highest] {
			highest = nb
		}
	}
	return a[highest]
}

@[direct_array_access; inline]
fn min(a []f64) f64 {
	mut highest := 0
	for nb, val in a {
		if val < a[highest] {
			highest = nb
		}
	}
	return a[highest]
}

@[inline]
fn round_to_greater(nb f64) f64 {
	if nb >= 0 {
		return f64(ceil(math.round_sig(nb, 5)))
	} else {
		return f64(int(math.round_sig(nb, 5)))
	}
}

@[inline]
fn ceil(nb f64) int {
	return -int(-nb)
}


fn in_range[T](x T, y T, x_start T, y_start T, x_end T, y_end T) bool {
	return x >= x_start && x < x_end && y >= y_start && y < y_end
}