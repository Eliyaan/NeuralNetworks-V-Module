module neural_networks

pub struct MinibatchesBackpropTrainingParams {
	learning_rate             f64
	momentum                  f64
	batch_size                int = 1
	nb_epochs                 int
	classifier				  bool
	print_interval            int
	print_batch_interval      int
	cost_function             CostFunctions
	training        Dataset
	test 			Dataset
	test_params 	TestParams
}

fn (mut nn NeuralNetwork) train_minibatches_backprop(t_p MinibatchesBackpropTrainingParams) {
	cost_fn, cost_prime := get_cost_function(t_p.cost_function)
	nb_batches :=  t_p.training.inputs.len / t_p.batch_size
	for epoch in 0 .. t_p.nb_epochs {
		mut epoch_error := 0.0
		mut epoch_accuracy := 0.0
		print_epoch := t_p.print_interval != 0 && ((epoch + 1) % t_p.print_interval == 0 || epoch == 0)
		test_epoch := t_p.test_params.training_interval != 0 && ((epoch + 1) % t_p.test_params.training_interval == 0 || epoch == 0)
		for batch in 0 .. nb_batches { // be careful for the size of the batches to not lose some data over the division rounding
			mut error := 0.0
			mut accuracy := 0.0
			print_batch := t_p.print_batch_interval != 0 && ((batch + 1) % t_p.print_batch_interval == 0 || batch == 0)
			test_batch := t_p.test_params.training_batch_interval != 0 && ((batch + 1) % t_p.test_params.training_batch_interval == 0)
			for i, input in t_p.training.inputs[batch * t_p.batch_size..(batch + 1) * t_p.batch_size] {
				output := nn.forward_propagation(input)
				error += cost_fn(t_p.training.expected_outputs[i], output)
				nn.backpropagation(t_p.training.expected_outputs[i], output, cost_prime)
				if t_p.classifier {
					if match_output_array_to_number(output) == match_classifier_array_to_number(t_p.training.expected_outputs[i]) {
						accuracy += 1
					}
				}
			}
			epoch_error += error
			epoch_accuracy += accuracy
			error /= t_p.batch_size
			accuracy /= f64(t_p.batch_size)/100.0
			nn.apply_gradient_descent(t_p.batch_size, t_p.learning_rate, t_p.momentum)
			if print_batch && print_epoch {
				if t_p.classifier {
					println('  batch ${batch + 1}/${nb_batches}\t-\tCost : ${error}\t-\tAccuracy : ${accuracy:.2}%')
				} else {
					println('  batch ${batch + 1}/${nb_batches}\t-\tCost : ${error}')
				}
			}
			if test_epoch && test_batch {
				nn.test(t_p)
			}
		}
		epoch_error /= (nb_batches*t_p.batch_size)
		epoch_accuracy /= (nb_batches*t_p.batch_size)/100
		if print_epoch {
			if t_p.classifier {
				println('Epoch ${epoch + 1}/${t_p.nb_epochs}\t-\tCost : ${epoch_error}\t-\tAccuracy : ${epoch_accuracy:.2}%')
			} else {
				println('Epoch ${epoch + 1}/${t_p.nb_epochs}\t-\tCost : ${epoch_error}')
			}
		}
		nn.cost = epoch_error
		nn.accuracy = epoch_accuracy
	}
}
