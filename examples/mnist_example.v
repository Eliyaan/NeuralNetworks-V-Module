module main

import neural_networks as n

/* TODO: 
- adjust the data augmentation to better fit the need
- dropout
- noise input
- -prod -prof prof.txt optimisation
- function to visualise on which is there a fail in the test (to know if it's a dataset problem that causes overfitting)
*/			

fn main() {
	mut neunet := n.NeuralNetwork{
		learning_rate: 1.0
		momentum: 0.9
		nb_neurons: [784, 400, 250, 10]
		activ_funcs: [n.leaky_relu,n.leaky_relu,n.leaky_relu,n.leaky_relu]
		deriv_activ_funcs: [n.dleaky_relu,n.dleaky_relu,n.dleaky_relu,n.dleaky_relu]
		w_random_interval: 0.01
		b_random_interval: 0.005
		
		print_epoch: 1
		print_batch: 400
		test_batch: 2400
		classifier: true
		save_accuracy: 90
		save_cost: 0.20
		input_noise: 128
		input_noise_chance: 28
	}
	
	neunet.init("")

	neunet.load_mnist(60000, 10000, 6, 1, 256, 7, 45, 4)
	
	neunet.train_backprop_minibatches(5, 50)
	neunet.save('mnist_save')
	neunet.train_backprop_minibatches(5, 50)
	neunet.save('mnist_save')
	neunet.train_backprop_minibatches(5, 50) //  not useful for now
	neunet.test_unseen_data()
	println('Actual | Expected output')
	println('${neunet.fprop(neunet.test_inputs[0])}  | ${neunet.expected_test_outputs[0]}')
	println('${neunet.fprop(neunet.test_inputs[1])}  | ${neunet.expected_test_outputs[1]}')
	println('${neunet.fprop(neunet.test_inputs[2])}  | ${neunet.expected_test_outputs[2]}')
	println('${neunet.fprop(neunet.test_inputs[3])}  | ${neunet.expected_test_outputs[3]}')
	neunet.save('mnist_save')
}
