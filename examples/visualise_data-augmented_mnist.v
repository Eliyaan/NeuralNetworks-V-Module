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
*/

const (
    win_width   	= 601
    win_height  	= 601
    bg_color    	= gx.white
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
}

struct App {
mut:
    gg    &gg.Context = unsafe { nil }
	clickables []Clickable
	elements []Element
	base_dataset nn.Dataset
	dataset nn.Dataset
	actual_image int
	offset_range int = 4
	noise_probability int = 28
	noise_range int = 255
	augment_asked bool
}

fn main() {
    mut app := &App{}
    app.gg = gg.new_context(
        width: win_width
        height: win_height
        create_window: true
        window_title: '- Application -'
        user_data: app
        bg_color: bg_color
        frame_fn: on_frame
        event_fn: on_event
        sample_count: 4
		ui_mode: true
    )
	app.base_dataset = load_mnist_training(100)
	app.dataset = app.base_dataset.clone()

	plus_text := Text{Id{}, 0, 0, "+", gx.TextCfg{color:gx.black, size:20, align:.center, vertical_align:.middle}}
	minus_text := Text{Id{}, 0, 0, "-", gx.TextCfg{color:gx.black, size:20, align:.center, vertical_align:.middle}}
	reload_text := Text{Id{}, 0, 0, "~", gx.TextCfg{color:gx.black, size:20, align:.center, vertical_align:.middle}}

	app.elements << Text{Id{.img_label}, 14*px_size, 28*px_size, match_classifier_array_to_number(app.dataset.expected_outputs[app.actual_image]).str(), gx.TextCfg{color:gx.dark_green, size:20, align:.center, vertical_align:.top}}

	app.clickables << Button{Id{}, x_buttons_offset+50, 10, buttons_shape, minus_text, gx.red, prev_img}
	app.clickables << Button{Id{}, x_buttons_offset+75, 10, buttons_shape, plus_text, gx.green, next_img}
	app.elements << Text{Id{.img_nb_text}, x_buttons_offset+45, 10, "Image n°${app.actual_image}", gx.TextCfg{color:gx.black, size:20, align:.right, vertical_align:.top}}

	app.clickables << Button{Id{}, x_buttons_offset+105, 10, buttons_shape, reload_text, gx.gray, ask_augment}

	app.clickables << Button{Id{}, x_buttons_offset+50, 35, buttons_shape, minus_text, gx.red, sub_offset}
	app.clickables << Button{Id{}, x_buttons_offset+75, 35, buttons_shape, plus_text, gx.green, add_offset}

	app.clickables << Button{Id{}, x_buttons_offset+50, 60, buttons_shape, minus_text, gx.red, sub_noise_range}
	app.clickables << Button{Id{}, x_buttons_offset+75, 60, buttons_shape, plus_text, gx.green, add_noise_range}

	app.clickables << Button{Id{}, x_buttons_offset+50, 85, buttons_shape, minus_text, gx.red, sub_noise_probability}
	app.clickables << Button{Id{}, x_buttons_offset+75, 85, buttons_shape, plus_text, gx.green, add_noise_probability}

	app.augment_images()
    app.gg.run()
}

fn on_frame(mut app App) {
    //Draw
    app.gg.begin()
	if app.augment_asked {
		app.augment(app.actual_image)
		app.augment_asked = false
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
	width f32
	height f32
	relative_pos Pos
}

interface Clickable {
	id Id
	x f32
	y f32
	shape Shape
	click_func fn (mut app App)
	render(mut app App)
}

interface Element {
	id Id
	x f32
	y f32
	render(mut app App, x_offset f32, y_offset f32)
}

struct ButtonShape {
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
	app.dataset.inputs[i] = rand_offset_image(app.base_dataset.inputs[i], app.offset_range)
	app.dataset.inputs[i] = rand_noise(app.dataset.inputs[i], app.noise_probability, app.noise_range)
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
	b.text.render(mut app, text_x_offset, text_y_offset) // need to add the relative pos
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

fn add_offset(mut app App) {
	app.offset_range += 1
	ask_augment(mut app)
}

fn sub_offset(mut app App) {
	app.offset_range -= 1
	ask_augment(mut app)
}

fn add_noise_range(mut app App) {
	app.noise_range += 1
	ask_augment(mut app)
}

fn sub_noise_range(mut app App) {
	app.noise_range -= 1
	ask_augment(mut app)
}

fn add_noise_probability(mut app App) {
	app.noise_probability += 1
	ask_augment(mut app)
}

fn sub_noise_probability(mut app App) {
	app.noise_probability -= 1
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
	for obj in app.clickables {
		obj.render(mut app)
	}
}

fn (mut app App) render_elements() {
	for elem in app.elements {
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
	return int(x - im_size / 2), int(y - im_size / 2) // offset (half image size)
}

@[direct_array_access]
pub fn rand_offset_image(a []f64, offset_range int) []f64 {
	mut offset_x, mut offset_y := get_center_of_mass(a)
	im_size := int(math.sqrt(a.len))
	if offset_range > 0 {
		offset_x += rd.int_in_range(-offset_range, offset_range + 1) or { panic(err) }
		offset_y += rd.int_in_range(-offset_range, offset_range + 1) or { panic(err) }
	}
	mut output := []f64{cap: im_size * im_size}
	for l in 0 .. im_size {
		for c in 0 .. im_size {
			if in_range(offset_x + c, offset_y + l, 0, 0, im_size, im_size) {
				output << a[(offset_y + l) * im_size + offset_x + c]
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

fn in_range[T](x T, y T, x_start T, y_start T, x_end T, y_end T) bool {
	return x >= x_start && x < x_end && y >= y_start && y < y_end
}