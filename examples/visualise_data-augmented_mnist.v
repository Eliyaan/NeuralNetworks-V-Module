import neural_networks as nn
import ggui
import os
import math
import gg
import gx
import rand as rd
import rand.config as rdconfig


const (
    win_width   	= 601
    win_height  	= 601
	theme			= ggui.CatppuchinMocha{}
    bg_color    	= theme.base
	px_size			= 4
	box_offset_x	= 300
	box_offset_y	= 10
	buttons_shape	= ggui.RoundedShape{20, 20, 5, .top_left}
)

enum Id {
	@none 	
	img_nb_text
	img_label
	final_scale_text
	noise_range_text
	noise_probability_text
	scale_range_text
	rota_range_text
}

fn id(id Id) int {
	return int(id)
}

@[heap]
struct App {
mut:
    gg    &gg.Context = unsafe { nil }
	gui		&ggui.Gui = unsafe { nil }
	clickables []ggui.Clickable
	elements []ggui.Element
	base_dataset nn.Dataset
	dataset nn.Dataset
	actual_image int
	final_scale f64 = 1
	noise_probability int = 15
	noise_range int = 255
	scale_range f64 = 0.2
	rota_range int = 45
	final_nb_pixels int
	augment_asked bool = true
}

fn main() {
    mut app := &App{}
	app.gui = &ggui.Gui(app)
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

	plus_text := ggui.Text{0, 0, 0, "+", gx.TextCfg{color:theme.base, size:20, align:.center, vertical_align:.middle}}
	minus_text := ggui.Text{0, 0, 0, "-", gx.TextCfg{color:theme.base, size:20, align:.center, vertical_align:.middle}}
	reload_text := ggui.Text{0, 0, 0, "~", gx.TextCfg{color:theme.base, size:20, align:.center, vertical_align:.middle}}
	button_description_cfg := gx.TextCfg{color:theme.text, size:20, align:.right, vertical_align:.top}

	app.clickables << ggui.Button{0, box_offset_x+105, box_offset_y+5, buttons_shape, reload_text, theme.flamingo, ask_augment}

	app.clickables << ggui.Button{0, box_offset_x+50, box_offset_y+5, buttons_shape, minus_text, theme.red, prev_img}
	app.clickables << ggui.Button{0, box_offset_x+75, box_offset_y+5, buttons_shape, plus_text, theme.green, next_img}
	
	app.clickables << ggui.Button{0, box_offset_x+50, box_offset_y+30, buttons_shape, minus_text, theme.red, sub_final_scale}
	app.clickables << ggui.Button{0, box_offset_x+75, box_offset_y+30, buttons_shape, plus_text, theme.green, add_final_scale}

	app.clickables << ggui.Button{0, box_offset_x+50, box_offset_y+55, buttons_shape, minus_text, theme.red, sub_noise_range}
	app.clickables << ggui.Button{0, box_offset_x+75, box_offset_y+55, buttons_shape, plus_text, theme.green, add_noise_range}

	app.clickables << ggui.Button{0, box_offset_x+50, box_offset_y+80, buttons_shape, minus_text, theme.red, sub_noise_probability}
	app.clickables << ggui.Button{0, box_offset_x+75, box_offset_y+80, buttons_shape, plus_text, theme.green, add_noise_probability}

	app.clickables << ggui.Button{0, box_offset_x+50, box_offset_y+105, buttons_shape, minus_text, theme.red, sub_scale_range}
	app.clickables << ggui.Button{0, box_offset_x+75, box_offset_y+105, buttons_shape, plus_text, theme.green, add_scale_range}

	app.clickables << ggui.Button{0, box_offset_x+50, box_offset_y+130, buttons_shape, minus_text, theme.red, sub_rota_range}
	app.clickables << ggui.Button{0, box_offset_x+75, box_offset_y+130, buttons_shape, plus_text, theme.green, add_rota_range}

	app.elements << ggui.Text{id(.img_label), 14*px_size, 28*px_size, nn.match_classifier_array_to_number(app.dataset.expected_outputs[app.actual_image]).str(), gx.TextCfg{color:theme.text, size:20, align:.center, vertical_align:.top}}

	app.elements << ggui.Rect{x:150, y:10, shape:ggui.RoundedShape{280, 160, 5, .top_left}, color:theme.mantle}

	app.elements << ggui.Text{id(.img_nb_text), box_offset_x+45, box_offset_y+5, "Image n°${app.actual_image}", button_description_cfg}

	app.elements << ggui.Text{id(.final_scale_text), box_offset_x+45, box_offset_y+30, "Final scale: ${app.final_scale}", button_description_cfg}

	app.elements << ggui.Text{id(.noise_range_text), box_offset_x+45, box_offset_y+55, "Noise range: ${app.noise_range}", button_description_cfg}

	app.elements << ggui.Text{id(.noise_probability_text), box_offset_x+45, box_offset_y+80, "Noise probability: ${app.noise_probability}", button_description_cfg}

	app.elements << ggui.Text{id(.scale_range_text), box_offset_x+45, box_offset_y+105, "Scale range: ${app.scale_range}", button_description_cfg}

	app.elements << ggui.Text{id(.rota_range_text), box_offset_x+45, box_offset_y+130, "Rotation range: ${app.rota_range}", button_description_cfg}

	app.augment_images()
    app.gg.run()
}

fn on_frame(mut app App) {
	if app.augment_asked {
		app.augment(app.actual_image)
		app.augment_asked = false
		mut img_label_text := app.gui.get_element_with_id(id(.img_label)) or {panic(err)}
		img_label_text.y = app.final_nb_pixels*px_size
		img_label_text.x = img_label_text.y/2
	}
    //Draw
    app.gg.begin()
	app.gui.render()
	app.render_image()
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
					app.gui.check_clicks(int(e.mouse_x), int(e.mouse_y))
				}
                else{}
        	}
		}
        else {}
    }
}

fn (mut app App) augment_images() {
	println("Start data augmentation")
	for i, _ in app.dataset.inputs {
		app.augment(i)
	}
	println("Data augmentation finished!")
}

fn (mut app App) augment(i int) {
	app.dataset.inputs[i] = app.base_dataset.inputs[i].clone()
	app.dataset.inputs[i] = nn.rotate(app.dataset.inputs[i], rd.f64_in_range(-app.rota_range, app.rota_range) or {0}, 28, 28)
	mut image_side_size := nn.ceil(math.sqrt(app.dataset.inputs[i].len))
	app.dataset.inputs[i] = nn.scale_img(app.dataset.inputs[i], rd.f64_in_range(1-app.scale_range, 1+app.scale_range) or {0}, image_side_size, image_side_size)
	image_side_size = nn.ceil(math.sqrt(app.dataset.inputs[i].len))
	app.dataset.inputs[i] = nn.center_image(app.dataset.inputs[i], image_side_size, image_side_size)
	app.dataset.inputs[i] = nn.rand_noise(app.dataset.inputs[i], app.noise_probability, app.noise_range)
	app.dataset.inputs[i] = nn.crop(app.dataset.inputs[i], image_side_size, image_side_size, 28, 28)
	image_side_size = nn.ceil(math.sqrt(app.dataset.inputs[i].len))
	app.dataset.inputs[i] = nn.scale_img(app.dataset.inputs[i], app.final_scale, image_side_size, image_side_size)
	app.final_nb_pixels = int(math.sqrt(app.dataset.inputs[i].len))
}

fn next_img(mut app ggui.Gui) {
	if mut app is App {
		if app.actual_image == app.dataset.inputs.len - 1 {
			app.actual_image = 0
		}else{
			app.actual_image += 1
		}
		app.gui.change_text(id(.img_nb_text), "Image n°${app.actual_image}")
		app.gui.change_text(id(.img_label), nn.match_classifier_array_to_number(app.dataset.expected_outputs[app.actual_image]).str())

		ask_augment(mut app)
	}
}

fn prev_img(mut app ggui.Gui) {
	if mut app is App {
		if app.actual_image == 0 {
			app.actual_image = app.dataset.inputs.len - 1
		}else{
			app.actual_image -= 1
		}
		app.gui.change_text(id(.img_nb_text), "Image n°${app.actual_image}")
		app.gui.change_text(id(.img_label), nn.match_classifier_array_to_number(app.dataset.expected_outputs[app.actual_image]).str())
		ask_augment(mut app)
	}
}

fn ask_augment(mut app ggui.Gui) {
	if mut app is App {
		app.augment_asked = true
	}
}

fn add_final_scale(mut app ggui.Gui) {
	if mut app is App {
		app.final_scale = math.round_sig(app.final_scale+0.05, 2)
		app.gui.change_text(id(.final_scale_text), "Final scale: ${app.final_scale}")
		ask_augment(mut app)
	}
}

fn sub_final_scale(mut app ggui.Gui) {
	if mut app is App {
		app.final_scale = math.round_sig(app.final_scale-0.05, 2)
		app.gui.change_text(id(.final_scale_text), "Final scale: ${app.final_scale}")
		ask_augment(mut app)
	}
}

fn add_noise_range(mut app ggui.Gui) {
	if mut app is App {
		app.noise_range += 1
		app.gui.change_text(id(.noise_range_text), "Noise range: ${app.noise_range}")
		ask_augment(mut app)
	}
}

fn sub_noise_range(mut app ggui.Gui) {
	if mut app is App {
		app.noise_range -= 1
		app.gui.change_text(id(.noise_range_text), "Noise range: ${app.noise_range}")
		ask_augment(mut app)
	}
}

fn add_noise_probability(mut app ggui.Gui) {
	if mut app is App {
		app.noise_probability += 1
		app.gui.change_text(id(.noise_probability_text), "Noise probability: ${app.noise_probability}")
		ask_augment(mut app)
	}
}

fn sub_noise_probability(mut app ggui.Gui) {
	if mut app is App {
		app.noise_probability -= 1
		app.gui.change_text(id(.noise_probability_text), "Noise probability: ${app.noise_probability}")
		ask_augment(mut app)
	}
}

fn add_scale_range(mut app ggui.Gui) {
	if mut app is App {
		app.scale_range = math.round_sig(app.scale_range+0.01, 2)
		app.gui.change_text(id(.scale_range_text), "Scale range: ${app.scale_range}")
		ask_augment(mut app)
	}
}

fn sub_scale_range(mut app ggui.Gui) {
	if mut app is App {
		app.scale_range = math.round_sig(app.scale_range-0.01, 2)
		app.gui.change_text(id(.scale_range_text), "Scale range: ${app.scale_range}")
		ask_augment(mut app)
	}
}

fn add_rota_range(mut app ggui.Gui) {
	if mut app is App {
		app.rota_range += 1
		app.gui.change_text(id(.rota_range_text), "Rotation range: ${app.rota_range}")
		ask_augment(mut app)
	}
}

fn sub_rota_range(mut app ggui.Gui) {
	if mut app is App {
		app.rota_range -= 1
		app.gui.change_text(id(.rota_range_text), "Rotation range: ${app.rota_range}")
		ask_augment(mut app)
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
	train_labels := os.open('mnist/train-labels-idx1-ubyte') or { panic(err) }
	train_images := os.open('mnist/train-images-idx3-ubyte') or { panic(err) }
	mut dataset := nn.Dataset{[][]f64{}, [][]f64{}}
	mut order_array := []u64{len:nb_training, init:u64(index)}
	rd.shuffle(mut order_array, rdconfig.ShuffleConfigStruct{}) or {panic(err)}
	for i in order_array {
		dataset.inputs << [train_images.read_bytes_at(784, i * 784 + 16).map(f64(it))]
		dataset.expected_outputs << [
			nn.match_number_to_classifier_array(train_labels.read_bytes_at(1, i + 8)[0])
		]
	}
	println('Finished loading training mnist!')
	return dataset
}

@[direct_array_access]
fn load_mnist_test(nb_tests int) nn.Dataset {
	println('Loading test mnist...')
	test_labels := os.open('mnist/t10k-labels-idx1-ubyte') or { panic(err) }
	test_images := os.open('mnist/t10k-images-idx3-ubyte') or { panic(err) }
	mut dataset := nn.Dataset{[][]f64{}, [][]f64{}}
	for i in 0 .. nb_tests {
		dataset.inputs << [test_images.read_bytes_at(784, i * 784 + 16).map(f64(it))]
		dataset.expected_outputs << [
			nn.match_number_to_classifier_array(test_labels.read_bytes_at(1, i + 8)[0])
		]
	}
	println('Finished loading test mnist!')
	return dataset
}