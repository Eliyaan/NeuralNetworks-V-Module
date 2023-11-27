module main

import preceptron as p
import os

fn main() {
	mut neunet := p.NeuralNetwork{
		learning_rate: 0.37
		nb_neurons: [2, 3, 1]
		activ_funcs: [p.leaky_relu, p.leaky_relu]
		deriv_activ_funcs: [p.dleaky_relu, p.dleaky_relu]
		w_random_interval: 0.6
		b_random_interval: 0.6
		print_epoch: 50
	}

	neunet.init(os.input('Load or new random NN ? [name of the file/Enter] ? > ')) // leave "" if no NN to load else put the name of the file you want to load
	neunet.load_dataset('XOR_data.toml')

	neunet.train_backprop(300)
	neunet.test_unseen_data()
	neunet.save('XOR-nn_save')

	println('Actual | Expected output')
	println('${neunet.fprop(neunet.test_inputs[0])}  | ${neunet.expected_test_outputs[0]}')
	println('${neunet.fprop(neunet.test_inputs[1])}  | ${neunet.expected_test_outputs[1]}')
	println('${neunet.fprop(neunet.test_inputs[2])}  | ${neunet.expected_test_outputs[2]}')
	println('${neunet.fprop(neunet.test_inputs[3])}  | ${neunet.expected_test_outputs[3]}')
}
