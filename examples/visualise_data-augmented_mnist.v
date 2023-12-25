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
Text struct to render nice text (and on buttons)
Buttons with click color change, text change etc
*/

const (
    win_width    = 601
    win_height   = 601
    bg_color     = gx.white
	px_size		 = 4
)


struct App {
mut:
    gg    &gg.Context = unsafe { nil }
	objects []Object
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
	app.objects << Button{150, 10, "-", 20, 20, .top_right, gx.red, 5, prev_img}
	app.objects << Button{175, 10, "+", 20, 20, .top_right, gx.green, 5, next_img}
	app.objects << Button{205, 10, "~", 20, 20, .top_right, gx.gray, 5, ask_augment}
	app.objects << Button{150, 35, "-", 20, 20, .top_right, gx.red, 5, sub_offset}
	app.objects << Button{175, 35, "+", 20, 20, .top_right, gx.green, 5, add_offset}
	app.objects << Button{150, 60, "-", 20, 20, .top_right, gx.red, 5, sub_noise_range}
	app.objects << Button{175, 60, "+", 20, 20, .top_right, gx.green, 5, add_noise_range}
	app.objects << Button{150, 85, "-", 20, 20, .top_right, gx.red, 5, sub_noise_probability}
	app.objects << Button{175, 85, "+", 20, 20, .top_right, gx.green, 5, add_noise_probability}
	app.augment_images()
    app.gg.run()
}

fn on_frame(mut app App) {
    //Draw
    app.gg.begin()
	app.render_image()
	app.render_objects()
	if app.augment_asked {
		app.augment(app.actual_image)
		app.augment_asked = false
	}
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
        }}
        else {}
    }
}

interface Object {
	x f32
	y f32
	width f32
	height f32
	click_func fn (mut app App)
	render(mut app App)
}

struct Button { // refaire pour meilleur texte
	x f32
	y f32
	text string
	width f32
	height f32
	relative_pos Pos
	color gx.Color
	rounded int
	click_func fn (mut app App) @[required]
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
	for obj in app.objects {
		if in_range(f32(mouse_x), f32(mouse_y), obj.x, obj.y, obj.x + obj.width, obj.y + obj.height) {
			obj.click_func(mut app)
		}
	}
}

fn (b Button) render(mut app App) { // need to do the relative pos
	app.gg.draw_rounded_rect_filled(b.x, b.y, b.width, b.height, b.rounded, b.color)
	app.gg.draw_text(int(b.x), int(b.y), b.text, gx.TextCfg{})
}

fn next_img(mut app App) {
	if app.actual_image == app.dataset.inputs.len - 1 {
		app.actual_image = 0
	}else{
		app.actual_image += 1
	}
	ask_augment(mut app)
}

fn prev_img(mut app App) {
	if app.actual_image == 0 {
		app.actual_image = app.dataset.inputs.len - 1
	}else{
		app.actual_image -= 1
	}
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

fn (mut app App) render_objects() {
	for obj in app.objects {
		obj.render(mut app)
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
	app.gg.draw_text(0, int(img_size*px_size), match_classifier_array_to_number(app.dataset.expected_outputs[app.actual_image]).str(), gx.TextCfg{})
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