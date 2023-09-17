module main

import preceptron as p

fn main() {
	mut neunet := p.NeuralNetwork{
		learning_rate: 0.3
		nb_neurons: [2, 3, 1]
		activ_func: p.sigmoid
		deriv_activ_func: p.dsig
		print_epoch: 500
		save_path: 'nn_save'
		// load_path: 'nn_save[2, 3, 1].nntoml'
		training_inputs: [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
		expected_training_outputs: [[0.0], [1.0], [1.0], [0.0]]
	}
	
	neunet.init()

	print(neunet.fprop_value([0.0, 0.0]))
	print(neunet.fprop_value([0.0, 1.0]))
	print(neunet.fprop_value([1.0, 0.0]))
	print(neunet.fprop_value([1.0, 1.0]))

	neunet.train_backprop(2000)

	print(neunet.fprop_value([0.0, 0.0]))
	print(neunet.fprop_value([0.0, 1.0]))
	print(neunet.fprop_value([1.0, 0.0]))
	print(neunet.fprop_value([1.0, 1.0]))
}
