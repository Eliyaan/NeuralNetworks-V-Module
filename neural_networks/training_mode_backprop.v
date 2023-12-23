module neural_networks

// An epoch is when the nn has seen the entire dataset
pub struct BackpropTrainingParams {
	learning_rate             f64
	momentum                  f64
	nb_epochs                 int
	print_interval            int
	cost_function             CostFunctions
	training        Dataset
	test 			Dataset
	test_params 	TestParams
}

fn (mut nn NeuralNetwork) train_backprop(t_p BackpropTrainingParams) {
	cost_fn, cost_prime := get_cost_function(t_p.cost_function)
	for epoch in 0 .. t_p.nb_epochs {
		mut error := 0.0
		print_epoch := t_p.print_interval != 0 && ((epoch + 1) % t_p.print_interval == 0 || epoch == 0)
		test_epoch := t_p.test_params.training_interval != 0 && ((epoch + 1) % t_p.test_params.training_interval == 0 || epoch == 0)
		for i, input in t_p.training.inputs {
			output := nn.forward_propagation(input)
			error += cost_fn(t_p.training.expected_outputs[i], output)
			nn.backpropagation(t_p.training.expected_outputs[i], output, cost_prime)
		}
		if print_epoch {
			println('Epoch ${epoch + 1}/${t_p.nb_epochs}\t-\tCost : ${error/t_p.training.inputs.len}')
		}
		nn.apply_gradient_descent(t_p.training.inputs.len, t_p.learning_rate, t_p.momentum)
		if test_epoch {
			nn.test(t_p)
		}
	}
}
