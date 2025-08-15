import ggui
import gg


const (
    win_width   	= 601
    win_height  	= 601
    theme           = ggui.CatppuchinMocha{}
	buttons_shape	= ggui.RoundedShape{20, 20, 5, .top_left}
)

enum Id {
	@none 	
	img_nb_text
    click_nb_text
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
	actual_image int
    actual_click int
}

fn main() {
    mut app := &App{}
	app.gui = &ggui.Gui(app)
    app.gg = gg.new_context(
        width: win_width
        height: win_height
        create_window: true
        window_title: 'ggui example'
        bg_color: theme.base
        user_data: app
        frame_fn: on_frame
        event_fn: on_event
        sample_count: 4
		ui_mode: true
    )
	plus_text := ggui.Text{0, 0, 0, "+", gg.TextCfg{color:theme.base, size:20, align:.center, vertical_align:.middle}}
	minus_text := ggui.Text{0, 0, 0, "-", gg.TextCfg{color:theme.base, size:20, align:.center, vertical_align:.middle}}
	text_cfg := gg.TextCfg{color:theme.text, size:20, align:.right, vertical_align:.top}

	app.clickables << ggui.Button{0, 300+50, 10, buttons_shape, minus_text, theme.red, prev_img}
	app.clickables << ggui.Button{0, 300+75, 10, buttons_shape, plus_text, theme.green, next_img}
	app.elements << ggui.Text{id(.img_nb_text), 300+45, 10, "Image n°${app.actual_image}", text_cfg}

    app.elements << ggui.Rect{x:300, y:40, shape:ggui.RoundedShape{160, 30, 5, .top}, color:theme.mantle}

    app.clickables << ggui.Button{0, 320, 45, buttons_shape, minus_text, theme.red, prev_click}
    app.clickables << ggui.Button{0, 345, 45, buttons_shape, plus_text, theme.green, next_click}
    app.elements << ggui.Text{id(.click_nb_text), 315, 45, "Click n°${app.actual_click}", text_cfg}
    
    app.gg.run()
}

fn on_frame(mut app App) {
    //Draw
    app.gg.begin()
	app.gui.render()
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
					app.gui.check_clicks(e.mouse_x, e.mouse_y)
				}
                else{}
        	}
		}
        else {}
    }
}

fn next_img(mut app ggui.Gui) {
	if mut app is App {
		app.actual_image += 1
		app.gui.change_text(id(.img_nb_text), "Image n°${app.actual_image}")
	}
}

fn prev_img(mut app ggui.Gui) {
	if mut app is App {
		app.actual_image -= 1
		app.gui.change_text(id(.img_nb_text), "Image n°${app.actual_image}")
	}
}

fn next_click(mut app ggui.Gui) {
	if mut app is App {
		app.actual_click += 1
        app.gui.change_text(id(.click_nb_text), "Click n°${app.actual_click}")
	}
}

fn prev_click(mut app ggui.Gui) {
	if mut app is App {
		app.actual_click -= 1
		app.gui.change_text(id(.click_nb_text), "Click n°${app.actual_click}")
	}
}
