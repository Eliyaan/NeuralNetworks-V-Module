module neural_networks

// An epoch is when the nn has seen the entire dataset
interface TrainingMode {
	learning_rate   f64
	nb_epochs       int
	classifier		bool
	cost_function   CostFunctions
	training        Dataset
	test 			Dataset
	test_params 	TestParams
}

pub struct Dataset {
pub mut:
	inputs  [][]f64
	expected_outputs [][]f64
}

pub fn (dataset Dataset) clone() Dataset {
	return Dataset{dataset.inputs.clone(), dataset.expected_outputs.clone()}
}

// [ start -> end ]
// test_interval in epochs
pub struct TestParams { 
pub:
	print_start int
	print_end 	int
	training_interval 	int
	training_batch_interval 	int
}

pub fn (mut nn NeuralNetwork) test(t_m TrainingMode) {
	println("\nTest Dataset:")
	cost_fn, _ := get_cost_function(t_m.cost_function)
	mut cost := 0.0
	mut accuracy := 0.0
	for i, input in t_m.test.inputs {
		output := nn.forward_propagation(input)
		cost += cost_fn(t_m.test.expected_outputs[i], output)
		if t_m.classifier {
			if match_output_array_to_number(output) == match_classifier_array_to_number(t_m.test.expected_outputs[i]) {
				accuracy += 1
			}
		}
		if i >= t_m.test_params.print_start && i <= t_m.test_params.print_end { // if there is an interval to print
			println("$i -> $output / ${t_m.test.expected_outputs[i]}")
		}
	}
	accuracy /= f64(t_m.test.inputs.len)/100.0
	cost /= t_m.training.inputs.len
	if t_m.classifier {
		println('Test Cost : ${cost} - Test Accuracy : ${accuracy:.2}%\n')
	}else{
		println('Test Cost : ${cost}\n')
	}
}
