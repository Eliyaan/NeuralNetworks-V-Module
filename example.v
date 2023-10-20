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

	neunet.init(os.input('Enter the name of the save you want to load ([enter] to create a random nn) ? > ')) // leave "" if no NN to load else put the name of the file you want to load
	neunet.load_dataset('training_data.toml')

	neunet.train_backprop(300)
	neunet.test_unseen_data()
	neunet.save('nn_save')

	println(neunet.fprop_value(neunet.test_inputs[0]))
	println(neunet.expected_test_outputs[0])
	println(neunet.fprop_value(neunet.test_inputs[1]))
	println(neunet.expected_test_outputs[1])
	println(neunet.fprop_value(neunet.test_inputs[2]))
	println(neunet.expected_test_outputs[2])
}
