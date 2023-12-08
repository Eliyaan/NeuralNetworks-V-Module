module main

import neural_networks as n

/* 
TODO: 
- error if the dataset is not there
- multithreading
- adjust the data augmentation to better fit the need
- dropout -> latter not nedded rn
- noise input -> latter not needed rn
- taux de divergence des w&b
- function to visualise on which is there a fail in the test (to know if it's a dataset problem that causes overfitting)
- make a struct for the mnist loading
*/			

// This file is long to train, so compiling with -prod is advised

fn main() {
	mut neunet := n.NeuralNetwork{
		learning_rate: 0.01
		momentum: 0.9
		nb_neurons: [784, 500, 400, 300, 10]
		activ_funcs: [n.leaky_relu,n.leaky_relu,n.leaky_relu,n.leaky_relu]
		deriv_activ_funcs: [n.dleaky_relu,n.dleaky_relu,n.dleaky_relu,n.dleaky_relu]
		w_random_interval: 0.01
		b_random_interval: 0.005
		seed: [u32(0), 0]
		
		print_epoch: 1
		print_batch: 150
		test_batch: 1200
		classifier: true
		save_accuracy: 95
		save_cost: 0.10
		input_noise: 128
		input_noise_chance: 28
	}
	
	neunet.init("")

	for _ in 0..30 {
		neunet.load_mnist(60000, 10000, 1, 1, 256, 7, 45, 4)  // regenerating the data-augmented dataset to not have overfitting
		neunet.train_backprop_minibatches(1, 50)
		neunet.save('mnist_save')
	}
	neunet.test_unseen_data()
	println('Actual | Expected output')
	println('${neunet.fprop(neunet.test_inputs[0])}  | ${neunet.expected_test_outputs[0]}')
	println('${neunet.fprop(neunet.test_inputs[1])}  | ${neunet.expected_test_outputs[1]}')
	println('${neunet.fprop(neunet.test_inputs[2])}  | ${neunet.expected_test_outputs[2]}')
	println('${neunet.fprop(neunet.test_inputs[3])}  | ${neunet.expected_test_outputs[3]}')
	neunet.save('mnist_save')
}
