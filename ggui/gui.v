module ggui
import gg
import gx

pub interface Gui {
mut:
	gg &gg.Context
	id int
	clickables []Clickable
	elements []Element
	parts []Gui
	x f32
	y f32
	shape Area
	color gx.Color
}

pub struct Part{
pub mut:
	gg &gg.Context
	gui		&Gui = unsafe { nil }
	id int
	clickables []Clickable
	elements []Element
	parts []Gui
	x f32
	y f32
	shape Area
	color gx.Color
}

pub fn (mut g Gui) get_clickables_with_id(id int) []&Clickable{
	mut result := []&Clickable{}
	for obj in g.clickables {
		unsafe {
			if obj.id == id {
				result << &obj
			}
		}
	}
	return result
}

pub fn (mut g Gui) get_clickable_with_id(id int) !&Clickable{
	for obj in g.clickables {
		unsafe {
			if obj.id == id {
				return &obj
			}
		}
	}
	return error("No object matching")
}

pub fn (mut g Gui) get_elements_with_id(id int) []&Element{
	mut result := []&Element{}
	for obj in g.elements {
		unsafe {
			if obj.id == id {
				result << &obj
			}
		}
	}
	return result
}

pub fn (mut g Gui) get_element_with_id(id int) !&Element{
	for obj in g.elements {
		unsafe {
			if obj.id == id {
				return &obj
			}
		}
	}
	return error("No object matching")
}

pub fn (mut g Gui) get_part_with_id(id int) !&Gui{
	for part in g.parts {
		unsafe {
			if part.id == id {
				return &part
			}
		}
	}
	return error("No object matching")
}

pub fn (mut g Gui) render() {
	g.shape.render(mut g, g.x, g.y, g.color)
	g.render_parts()
	g.render_clickables()
	g.render_elements()
}

pub fn (mut g Gui) render_parts() {
	for mut part in g.parts {
		part.render()
	}
}

pub fn (mut g Gui) render_clickables() {
	for mut obj in g.clickables {
		obj.render(mut g, g.x, g.y)
	}
}

pub fn (mut g Gui) render_elements() {
	for mut elem in g.elements {
		elem.render(mut g, g.x, g.y)
	}
}

pub fn (mut g Gui) check_clicks(mouse_x f32, mouse_y f32, mut base_app Gui) {
	click_x := mouse_x - g.x 
	click_y := mouse_y - g.y 
	println("$click_x $click_y")
	for obj in g.clickables {
		x_rel, y_rel := obj.shape.offset()
		obj_x := obj.x + x_rel
		obj_y := obj.y + y_rel
		
		if in_range(click_x, click_y, obj_x, obj_y, obj_x + obj.shape.width, obj_y + obj.shape.height) {
			obj.click_func(mut base_app, mut g)
		}
	}
	for mut part in g.parts {
		part.check_clicks(click_x, click_y, mut base_app)
	}
}

pub fn (mut g Gui) change_text(id int, text string) {
	mut text_obj := g.get_element_with_id(id) or {panic(err)}
	if mut text_obj is Text {
		text_obj.text = text
	}
}