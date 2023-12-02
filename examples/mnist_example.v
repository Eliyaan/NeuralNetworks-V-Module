module main

import preceptron as p

// TODO: function to visualise on which is there a fail in the test (to know if it's a dataset problem that causes overfitting)

fn main() {
	mut neunet := p.NeuralNetwork{
		learning_rate: 1.0
		nb_neurons: [784, 350, 200, 10]
		activ_funcs: [p.leaky_relu,p.leaky_relu,p.leaky_relu,p.leaky_relu]
		deriv_activ_funcs: [p.dleaky_relu,p.dleaky_relu,p.dleaky_relu,p.dleaky_relu]
		w_random_interval: 0.01
		b_random_interval: 0.005
		
		print_epoch: 200
		classifier: true
		save_accuracy: 82
		save_cost: 0.30
		input_noise: 128
		input_noise_chance: 28
	}
	
	neunet.init("")

	neunet.load_mnist(60000, 10000, 256, 7, 45, 4)
	
	neunet.train_backprop_minibatches(5000, 50)
	neunet.train_backprop_minibatches(5000, 100)
	neunet.train_backprop_minibatches(5000, 200)
	neunet.test_unseen_data()
	println('Actual | Expected output')
	println('${neunet.fprop(neunet.test_inputs[0])}  | ${neunet.expected_test_outputs[0]}')
	println('${neunet.fprop(neunet.test_inputs[1])}  | ${neunet.expected_test_outputs[1]}')
	println('${neunet.fprop(neunet.test_inputs[2])}  | ${neunet.expected_test_outputs[2]}')
	println('${neunet.fprop(neunet.test_inputs[3])}  | ${neunet.expected_test_outputs[3]}')
	neunet.save('mnist_save')
}
