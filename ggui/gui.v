module ggui
import gg

pub interface Gui {
mut:
	gg &gg.Context
	clickables []Clickable
	elements []Element
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

pub fn (mut g Gui) render() {
	g.render_elements()
	g.render_clickables()
}

pub fn (mut g Gui) render_clickables() {
	for mut obj in g.clickables {
		obj.render(mut g, 0, 0)
	}
}

pub fn (mut g Gui) render_elements() {
	for mut elem in g.elements {
		elem.render(mut g, 0, 0)
	}
}

pub fn (mut g Gui) check_clicks(mouse_x f32, mouse_y f32) {
	click_x := mouse_x
	click_y := mouse_y
	for obj in g.clickables {
		x_rel, y_rel := obj.shape.offset()
		obj_x := obj.x + x_rel
		obj_y := obj.y + y_rel
		
		if in_range(click_x, click_y, obj_x, obj_y, obj_x + obj.shape.width, obj_y + obj.shape.height) {
			obj.click_func(mut g)
		}
	}
}

pub fn (mut g Gui) change_text(id int, text string) {
	mut text_obj := g.get_element_with_id(id) or {panic(err)}
	if mut text_obj is Text {
		text_obj.text = text
	}
}