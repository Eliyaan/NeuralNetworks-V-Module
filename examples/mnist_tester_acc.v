import neural_networks_acc as nn
import gg
import gx
import ggui
import math

const path			= 'saveMNIST-0014-93'
const theme			= ggui.CatppuchinLatte{}
const buttons_shape	= ggui.RoundedShape{20, 20, 5, .center}
const px_size		= 5

enum Id {
	@none 
	prediction
	text_0	
	text_1
	text_2
	text_3
	text_4
	text_5
	text_6
	text_7
	text_8
	text_9
}

fn id(id Id) int {
	return int(id)
}


struct App {
mut:
    gg    &gg.Context = unsafe { nil }
	gui		&ggui.Gui = unsafe { nil }
	clickables []ggui.Clickable
	elements []ggui.Element
	model nn.NeuralNetwork
	image []f64 = []f64{len:28*28}
	mouse_held bool
}


fn main() {
    mut app := &App{}
	app.gui = &ggui.Gui(app)
    app.gg = gg.new_context(
        width: 300
        height: 280
        create_window: true
        window_title: 'Mnist Tester'
        user_data: app
        bg_color: theme.base
        frame_fn: on_frame
        event_fn: on_event
        sample_count: 6
    )
	app.model.load_model(path)

	erase_text := ggui.Text{0, 0, 0, "~", gx.TextCfg{color:theme.base, size:20, align:.center, vertical_align:.middle}}
	text_cfg := gx.TextCfg{color:theme.text, size:20, align:.left, vertical_align:.top}

	app.clickables << ggui.Button{0, 280, 30, buttons_shape, erase_text, theme.flamingo, erase}
	
	app.elements << ggui.Rect{x:150, y:10, shape:ggui.RoundedShape{160, 265, 8, .top_left}, color:theme.mantle}

	app.elements << ggui.Text{id(.text_0), 125+45, 10+10, "0 : 00.00%", text_cfg}
	app.elements << ggui.Text{id(.text_1), 125+45, 10+35, "1 : 00.00%", text_cfg}
	app.elements << ggui.Text{id(.text_2), 125+45, 10+60, "2 : 00.00%", text_cfg}
	app.elements << ggui.Text{id(.text_3), 125+45, 10+85, "3 : 00.00%", text_cfg}
	app.elements << ggui.Text{id(.text_4), 125+45, 10+110, "4 : 00.00%", text_cfg}
	app.elements << ggui.Text{id(.text_5), 125+45, 10+135, "5 : 00.00%", text_cfg}
	app.elements << ggui.Text{id(.text_6), 125+45, 10+160, "6 : 00.00%", text_cfg}
	app.elements << ggui.Text{id(.text_7), 125+45, 10+185, "7 : 00.00%", text_cfg}
	app.elements << ggui.Text{id(.text_8), 125+45, 10+210, "8 : 00.00%", text_cfg}
	app.elements << ggui.Text{id(.text_9), 125+45, 10+235, "9 : 00.00%", text_cfg}

	app.elements << ggui.Text{id(.prediction), 14*px_size, 28*px_size, "-", gx.TextCfg{color:theme.text, size:40, align:.center, vertical_align:.top}}
    
    app.gg.run()
}

fn on_frame(mut app App) {
	centered_image := nn.center_image(app.image, 28, 28)
	output := softmax(app.model.forward_propagation(centered_image))

	
	app.gui.change_text(id(.text_0), "0 : ${output[0]*100:.2}%")
	app.gui.change_text(id(.text_1), "1 : ${output[1]*100:.2}%")
	app.gui.change_text(id(.text_2), "2 : ${output[2]*100:.2}%")
	app.gui.change_text(id(.text_3), "3 : ${output[3]*100:.2}%")
	app.gui.change_text(id(.text_4), "4 : ${output[4]*100:.2}%")
	app.gui.change_text(id(.text_5), "5 : ${output[5]*100:.2}%")
	app.gui.change_text(id(.text_6), "6 : ${output[6]*100:.2}%")
	app.gui.change_text(id(.text_7), "7 : ${output[7]*100:.2}%")
	app.gui.change_text(id(.text_8), "8 : ${output[8]*100:.2}%")
	app.gui.change_text(id(.text_9), "9 : ${output[9]*100:.2}%")
	guess := nn.match_output_array_to_number(output)
	app.gui.change_text(id(.prediction), "$guess")



    app.gg.begin()
	app.render_image()
	app.gui.render()
    app.gg.end()
}

fn (mut app App) render_image() {
	img_size := 28
	for y in 0..img_size {
		for x in 0..img_size {
			px := u8(app.image[y*img_size+x])
			app.gg.draw_rect_filled(f32(x*px_size), f32(y*px_size), px_size, px_size, gg.Color{px,px,px,255})
		}
	}
}

fn softmax(a []f64) []f64 {
	e := []f64{len:a.len, init:math.exp(a[index]*10)}
	mut total := 0.0
	for elem in e {
		total += elem
	}
	output := []f64{len:e.len, init:e[index]/total}
	return output
}

fn erase(mut app ggui.Gui) {
	if mut app is App {
		app.image = []f64{len:28*28}
	}
}

fn on_event(e &gg.Event, mut app App){
    match e.typ {
        .key_down {
            match e.key_code {
                .escape {app.gg.quit()}
				.backspace {erase(mut app)}
                else {}
            }
		}
        .mouse_down {app.mouse_held = true}
        .mouse_up {
            match e.mouse_button{
                .left{app.gui.check_clicks(e.mouse_x, e.mouse_y)}
                else{}
        	}
			app.mouse_held = false
		}
        else {}
    }
    app.check_buttons(e.mouse_x, e.mouse_y)
}

fn (mut app App) check_buttons(mouse_x f64, mouse_y f64){
    if app.mouse_held{
        if mouse_x < (px_size*28) && mouse_y < (px_size*28) && mouse_x >= 0 && mouse_y >= 0{
            index := int(mouse_y/px_size)*28 + int(mouse_x/px_size)
            for l in -1..2{
                for c in -1..2{
                    color := 255 - (120*c*c) - (120*l*l)
                    if index%28+c < 28 && index%28+c >= 0 && index + c + l*28 < 28*28 && index + c + l*28 >= 0 {
                        if (app.image[index+c+l*28] + color) > 255{
                            app.image[index+c+l*28] = 255
                        }else{
                            app.image[index+c+l*28] =  app.image[index+c+l*28] + color
                        }
                    }
                }
            }
        }
    }
}