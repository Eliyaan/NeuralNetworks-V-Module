module ggui
import gx

pub struct Rect {
mut:
	id int
	x f32
	y f32
	shape Area
	color gx.Color
}

fn (r Rect) render(mut g Gui, x_offset f32, y_offset f32) {
	r.shape.render(mut g, r.x, r.y, r.color)
}