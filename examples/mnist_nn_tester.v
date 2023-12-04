module main
import gg
import gx
import neural_networks as p

const (
    bg_color     = gx.white

    image_size = 28
    pixel_size = 9

    win_width    = (image_size+1)*pixel_size+90
    win_height   = (image_size+1)*pixel_size+25

    text_cfg = gx.TextCfg{color: gx.black, size: 13, align: .left, vertical_align: .middle, mono:true, family:"agency fb"}
    good_text_cfg = gx.TextCfg{color: gx.dark_green, size: 16, align: .left, vertical_align: .middle, mono:true, family:"agency fb"}
    prediction_text_cfg = gx.TextCfg{color: gx.dark_green, size: 30, align: .left, vertical_align: .top, bold:true, family:"agency fb"}
)


struct App {
mut:
    gg    &gg.Context = unsafe { nil }
    
    pixel_values []f32
    mouse_held bool
    

    nn p.NeuralNetwork = p.NeuralNetwork{
		activ_funcs: [p.leaky_relu,p.leaky_relu,p.leaky_relu,p.leaky_relu,p.leaky_relu]
	}
}


fn main() {
    mut app := &App{}
    app.gg = gg.new_context(
        width: win_width
        height: win_height
        create_window: true
        window_title: "- Reflection of MNIST -"
        user_data: app
        bg_color: bg_color
        frame_fn: on_frame
        event_fn: on_event
        sample_count: 2
        ui_mode: true
    )
    app.pixel_values = []f32{len:image_size*image_size}
    app.nn.init("nn_save-e4-[784, 400, 250, 10].nntoml")
    //app.nn.load_mnist(10, 0)
    //rotate(app.nn.training_inputs[0], -m.pi_2, 28)
    //lancement du programme/de la fenêtre
    app.gg.run()
}




fn on_frame(mut app App) {
    app.gg.begin()
    for l in 0..image_size{
        for c in 0..image_size{
            pixel_grey_scale := u8(app.pixel_values[l*image_size+c])
            app.gg.draw_square_filled(c*pixel_size, l*pixel_size, pixel_size, gx.Color{pixel_grey_scale,pixel_grey_scale,pixel_grey_scale,255})
        }
    }
    //pixels := p.process_img(app.nn.training_inputs[3], 28, 28, 28, false, 0, app.nn.image_size_goal, 45)

    pixels := p.process_img(app.pixel_values, image_size, 24, 24, 0, app.nn.image_size_goal, 0, 0)
    /*
    for l in 0..28{
        for c in 0..28{
            pixel_grey_scale := u8(pixels[l*28+c])
            //pixel_grey_scale := u8(app.nn.training_inputs[3][l*28+c])
            app.gg.draw_square_filled(c*pixel_size+(image_size+1)*pixel_size, l*pixel_size, pixel_size, gx.Color{pixel_grey_scale,pixel_grey_scale,pixel_grey_scale,255})
        }
    }*/
    
    /*
    //test displays
    for l in 0..28{
        for c in 0..28{
            pixel_grey_scale := u8(pixels[l*28+c])
            app.gg.draw_square_filled(c*pixel_size, l*pixel_size+(40+1)*pixel_size, pixel_size, gx.Color{pixel_grey_scale,pixel_grey_scale,pixel_grey_scale,255})
        }
    }
    */
    /*
    for l in 0..38{
        for c in 0..20{
            app.gg.draw_square_filled(c*pixel_size+120, l*pixel_size+(80+1)*pixel_size, pixel_size, gx.Color{u8(app.tmp_a[l*20+c]),u8(app.tmp_a[l*20+c]),u8(app.tmp_a[l*20+c]),255})
        }
    }
    */

    app.nn.fprop(pixels)
    mut guesses_array := p.get_outputs(app.nn.softmax())
    mut highest := 0
    for nb, guess in guesses_array{
        if guess > guesses_array[highest]{
            highest = nb
        }
    }
    for i, guess in guesses_array{
        text := "${guess*100:.0}"
        app.gg.draw_text((image_size+1)*pixel_size+20, i*20+10, "${text:02}% -> $i", if i == highest {good_text_cfg} else {text_cfg})
    }
    app.gg.draw_text(0, 28*pixel_size, "Prediction : ${highest}", prediction_text_cfg)
    app.gg.end()
}

fn on_event(e &gg.Event, mut app App){
    match e.typ {
        .key_down {
            match e.key_code {
                .escape {app.gg.quit()}
                .backspace{
                    app.pixel_values = []f32{len:image_size*image_size}
                }
                else {}
            }
        }
        .mouse_up {
            match e.mouse_button{
                .left{app.mouse_held = false}
                else{}
        }}
        .mouse_down {
            match e.mouse_button{
                .left{app.mouse_held = true}
                else{}
        }}
        else {}
    }
    app.check_buttons(e.mouse_x, e.mouse_y)
}

fn (mut app App) check_buttons(mouse_x f64, mouse_y f64){
    if app.mouse_held{
        if mouse_x < (pixel_size*image_size) && mouse_y < (pixel_size*image_size) && mouse_x >= 0 && mouse_y >= 0{
            index := int(mouse_y/pixel_size)*image_size + int(mouse_x/pixel_size)
            for l in -1..2{
                for c in -1..2{
                    color := 255 - (120*c*c) - (120*l*l)
                    if index%image_size+c < image_size && index%image_size+c >= 0 && index + c + l*image_size < image_size*image_size && index + c + l*image_size >= 0 {
                        if (app.pixel_values[index+c+l*image_size] + color) > 255{
                            app.pixel_values[index+c+l*image_size] = 255
                        }else{
                            app.pixel_values[index+c+l*image_size] =  app.pixel_values[index+c+l*image_size] + color
                        }
                    }
                }
            }
        }
    }
}

