module main

/*
A small and fast example, very great to manipulate and test to get used to the hyperparametters.
*/

import neural_networks as n
import os

fn main() {
	mut neunet := n.NeuralNetwork{
		learning_rate: 0.37
		momentum: 0.5
		nb_neurons: [2, 3, 1]
		activ_funcs: [n.leaky_relu, n.leaky_relu]
		deriv_activ_funcs: [n.dleaky_relu, n.dleaky_relu]
		w_random_interval: 0.6
		b_random_interval: 0.6
		print_epoch: 50
		test_batch: 100
	}

	neunet.init(os.input('Load or new random NN ? [name of the file/Enter] ? > ')) // leave "" if no NN to load else put the name of the file you want to load
	neunet.load_dataset('XOR_data.toml')

	neunet.threaded_train_bp_minibatches(300, 2, 2)
	neunet.test_unseen_data()
	neunet.save('XOR-nn_save')

	println('Actual | Expected output')
	println('${neunet.fprop(neunet.test_inputs[0])}  | ${neunet.expected_test_outputs[0]}')
	println('${neunet.fprop(neunet.test_inputs[1])}  | ${neunet.expected_test_outputs[1]}')
	println('${neunet.fprop(neunet.test_inputs[2])}  | ${neunet.expected_test_outputs[2]}')
	println('${neunet.fprop(neunet.test_inputs[3])}  | ${neunet.expected_test_outputs[3]}')
}
