module ggui
import gx

@[heap]
pub struct Text {
pub mut:
	id int
	x f32
	y f32
	text string
	cfg gx.TextCfg
}

pub fn (t Text) render(mut g Gui, x_offset f32, y_offset f32) {
	if t.text != "" {
		g.gg.draw_text(int(t.x + x_offset), int(t.y + y_offset), t.text, t.cfg)
	}
}