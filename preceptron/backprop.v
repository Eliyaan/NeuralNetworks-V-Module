module preceptron

import os

[direct_array_access]
pub fn (mut nn NeuralNetwork) train_backprop(nb_epochs u64){ // exemple of a training loop
	// vars for the save 
	mut need_to_save := false
	mut cost_to_save := 0.0
	mut weights_to_save := [][][]Weight{}
	mut layers_to_save := [][]Neuron{}
	//Training
	for epoch in 0..nb_epochs{
		if epoch > 0{ // there to not change the last tested nn at the end of the training
			nn.apply_delta()
		}
		if nn.shuffle_dataset{
			nn.randomise_i_exp_o()
		}
		nn.global_cost = 0.0  // reset the cost before the training of this epoch
		for i in 0..nn.training_inputs.len{ // mesure the time ?
			nn.neurons_costs_reset()
			nn.backprop(i)
		}
		if nn.print_epoch > 0 && nn.print{
			if epoch%u64(nn.print_epoch) == 0{
				println('\nEpoch: $epoch Global Cost: ${nn.global_cost}')
			}
		}
		if nn.best_cost/nn.global_cost > 1.0{
			need_to_save = true
			cost_to_save = nn.global_cost
			weights_to_save = nn.weights_list.clone()  // KASSé
			layers_to_save = nn.layers_list.clone()  // KASSé
			nn.best_cost = nn.global_cost
		}
	}
	if nn.print{
		println('____________________________________________________________\nFinal Results: \nCost: ${nn.global_cost}')
	}
	if need_to_save && nn.save_path != ""{
		println(" Saving the progress !")
		file := "cost=${cost_to_save}\nweights=${weights_to_save}\nlayers=${layers_to_save}"
		os.write_file(nn.save_path, file) or {panic(err)}
	}
}

// A possible way to train the network if you use a dataset
[inline]
[direct_array_access]
fn (mut nn NeuralNetwork) backprop(index int){
	nn.fprop_value(nn.training_inputs[index])
	
	// cost for the eyes
	for i, neuron in nn.layers_list[nn.nb_neurones.len-1]{ // pour chaque output
		tmp := neuron.output - nn.excpd_training_outputs[index][i]
		nn.global_cost += tmp*tmp  // Cost of each output
	}

	//Start backprop
	//Deriv nactiv of all neurons
	for mut layer in nn.layers_list{
		for mut neuron in layer{
			neuron.nactiv = nn.deriv_activ_func(neuron.nactiv)
		}
	}
	// deltaC/deltaA(last)
	for i, mut neuron in nn.layers_list[nn.nb_neurones.len-1]{ // pour chaque output
		neuron.cost = 2.0*(neuron.output - nn.excpd_training_outputs[index][i])
	}
	// deltaC/deltaW(last)
	for j, mut weight_list in nn.weights_list[nn.nb_neurones.len-2]{  // j is the nb of the input neuron
		for k, mut weight in weight_list{ // k is the nb of the output neuron
			weight.cost += nn.layers_list[nn.nb_neurones.len-2][j].output * nn.layers_list[nn.nb_neurones.len-1][k].nactiv * nn.layers_list[nn.nb_neurones.len-1][k].cost
		}
	}
	// deltaC/deltaB(last)
	for mut neuron in nn.layers_list[nn.nb_neurones.len-1]{
		neuron.b_cost += neuron.nactiv*neuron.cost
	}

	// deltaC/deltaA(i)
	for i := nn.nb_neurones.len-2; i>0; i--{ // for each hidden layer
		for j in 0..nn.nb_neurones[i]{ // for each neuron of the layer
			for k in 0..nn.nb_neurones[i+1]{ // for each neuron of the next layer
				nn.layers_list[i][j].cost += nn.weights_list[i][j][k].weight * nn.layers_list[i+1][k].nactiv * nn.layers_list[i+1][k].cost
			}
		}
	}

	for i in 1..nn.nb_neurones.len-1{ // for each hidden layer (output already done and nothing on input)
		//Weights
		for j, mut weight_list in nn.weights_list[i-1]{ // j = nb input neuron
			for k, mut weight in weight_list{  // k = nb output neuron
				weight.cost += nn.layers_list[i-1][j].output * nn.layers_list[i][k].nactiv * nn.layers_list[i][k].cost
			}
		}
		//Biases
		for mut neuron in nn.layers_list[i]{
			neuron.b_cost += neuron.nactiv*neuron.cost
		}
	}
}

[inline]
[direct_array_access]
fn (mut nn NeuralNetwork) apply_delta(){ // Apply the modifications calculated with the costs of the backprop
	//Weights
	for mut layer in nn.weights_list{
		for mut weight_list in layer{
			for mut weight in weight_list{
				weight.weight -= weight.cost * nn.learning_rate
				weight.cost = 0.0
			}
		}
	}
	//Biases
	for mut layer in nn.layers_list[1..]{ // cause we dont care about the bias of the input neuron that doesn't need to exist
		for mut neuron in layer{
			neuron.bias -= neuron.b_cost * nn.learning_rate
			neuron.b_cost = 0.0
		}
	}
}

[inline]
[direct_array_access]
fn (mut nn NeuralNetwork) neurons_costs_reset(){ // reset the layers that changed
	for mut layer in nn.layers_list[1..]{
		for mut neuron in layer{
			neuron.cost = 0.0
		}		 
	}	
}