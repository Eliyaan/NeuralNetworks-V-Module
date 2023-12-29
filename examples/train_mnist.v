import neural_networks as nn
import math
import os
import rand
import rand.config as rdconfig
/*
 If you get a lot of errors you probably need to run :

 v install vsl

 and then run :

 v run .
*/

fn main() {
	mut model := nn.NeuralNetwork.new(0)

	println('Creating a new model')
	model.add_layer(nn.Dense.new(784, 400, 0.01, 0.01))
	model.add_layer(nn.Activation.new(.leaky_relu))
	model.add_layer(nn.Dense.new(400, 300, 0.01, 0.01))
	model.add_layer(nn.Activation.new(.leaky_relu))
	model.add_layer(nn.Dense.new(300, 200, 0.01, 0.01))
	model.add_layer(nn.Activation.new(.leaky_relu))
	model.add_layer(nn.Dense.new(200, 10, 0.01, 0.01))
	model.add_layer(nn.Activation.new(.leaky_relu))

		// model.load_model('')

	for i in 0..10 {
		println("Epoch n°$i")
		training_parameters := nn.MinibatchesBackpropTrainingParams{
			learning_rate: 0.02
			momentum: 0.9
			batch_size: 150
			classifier: true
			nb_epochs: 1
			print_interval: 1
			print_batch_interval: 10
			cost_function: .mse // mean squared error
			training: load_mnist_training(60000)
			test: load_mnist_test(10000)
			test_params: nn.TestParams{
				print_start: 0
				print_end: 3
				training_interval: 1
				training_batch_interval: 200
			}
		}

		model.train(training_parameters)
	}

	model.save_model('saveMNIST-${model.cost}-${model.accuracy}')
}

@[direct_array_access]
fn load_mnist_training(nb_training int) nn.Dataset {
	println('Loading training mnist...')
	train_labels := os.open('mnist\\train-labels-idx1-ubyte') or { panic(err) }
	train_images := os.open('mnist\\train-images-idx3-ubyte') or { panic(err) }
	mut dataset := nn.Dataset{[][]f64{}, [][]f64{}}
	mut order_array := []u64{len:nb_training, init:u64(index)}
	rand.shuffle(mut order_array, rdconfig.ShuffleConfigStruct{}) or {panic(err)}
	for i in order_array {
		dataset.inputs << [train_images.read_bytes_at(784, i * 784 + 16).map(f64(it))]
		dataset.expected_outputs << [
			nn.match_number_to_classifier_array(train_labels.read_bytes_at(1, i + 8)[0])
		]
	}
	augment_images(mut dataset)
	println('Finished loading training mnist!')
	return dataset
}

@[direct_array_access]
fn load_mnist_test(nb_tests int) nn.Dataset {
	println('Loading test mnist...')
	test_labels := os.open('mnist\\t10k-labels-idx1-ubyte') or { panic(err) }
	test_images := os.open('mnist\\t10k-images-idx3-ubyte') or { panic(err) }
	mut dataset := nn.Dataset{[][]f64{}, [][]f64{}}
	for i in 0 .. nb_tests {
		dataset.inputs << [test_images.read_bytes_at(784, i * 784 + 16).map(f64(it))]
		dataset.expected_outputs << [
			nn.match_number_to_classifier_array(test_labels.read_bytes_at(1, i + 8)[0])
		]
	}
	augment_images(mut dataset)
	println('Finished loading test mnist!')
	return dataset
}

fn augment_images(mut d nn.Dataset) {
	for mut input in d.inputs {
		augment(mut input)
	}
}

fn augment(mut input []f64) {
	input = nn.rotate(input, rand.f64_in_range(-45, 45) or {0}, 28, 28)
	mut image_side_size := nn.ceil(math.sqrt(input.len))
	input = nn.scale_img(input, rand.f64_in_range(1-0.2, 1+0.2) or {0}, image_side_size, image_side_size)
	image_side_size = nn.ceil(math.sqrt(input.len))
	input = nn.rand_noise(input, 15, 255)
	input = nn.center_image(input, image_side_size, image_side_size)
	input = nn.crop(input, image_side_size, image_side_size, 28, 28)
}