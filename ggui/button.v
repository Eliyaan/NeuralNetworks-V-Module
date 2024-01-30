module ggui
import gx

@[heap]
pub struct Button {
pub mut:
	id int
	x f32
	y f32
	shape Area
	text Text
	color gx.Color
	click_func fn (mut g Gui) @[required]
}

pub fn (b Button) render(mut g Gui, x_offset f32, y_offset f32) {
	x_coo := b.x + x_offset
	y_coo := b.y + y_offset
	mut text_x_offset := x_coo
	mut text_y_offset := y_coo
	match b.shape.relative_pos {
		.center {}
		.left {text_x_offset += b.shape.width/2}
		.right {text_x_offset -= b.shape.width/2}
		.top {text_y_offset += b.shape.height/2}
		.bottom {text_y_offset -= b.shape.height/2}
		.top_left {
			text_x_offset += b.shape.width/2
			text_y_offset += b.shape.height/2
		}
		.top_right {
			text_x_offset -= b.shape.width/2
			text_y_offset += b.shape.height/2
		}
		.bottom_left {
			text_x_offset += b.shape.width/2
			text_y_offset -= b.shape.height/2
		}
		.bottom_right {
			text_x_offset -= b.shape.width/2
			text_y_offset -= b.shape.height/2
		}
	}
	b.shape.render(mut g, x_coo, y_coo, b.color)
	b.text.render(mut g, text_x_offset, text_y_offset)
}