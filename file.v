module main
import preceptron as p

fn main(){
	mut neunet := p.NeuralNetwork{		
								learning_rate: 0.3
								nb_neurones: [2, 4, 3, 2] 
								activ_func: p.tanh
							}
	neunet.init()

	println(neunet.fprop_value([0.0, 0.0])) // possible way to use the nn (without necessarly using a dataset)
	println(neunet.fprop_value([1.0, 1.0]))
}