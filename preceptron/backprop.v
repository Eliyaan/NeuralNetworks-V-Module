module preceptron

import os

[direct_array_access]
pub fn (mut nn NeuralNetwork) train_backprop(nb_epochs u64){ // exemple of a training loop
	// vars for the save 
	mut need_to_save := false
	mut cost_to_save := 0.0
	mut weights_to_save := [][][][]f64{}
	mut layers_to_save := [][][]f64{}
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
			weights_to_save = nn.weights_list.clone()
			layers_to_save = nn.layers_list.clone()
			nn.best_cost = nn.global_cost
		}
	}
	if nn.print{
		println('____________________________________________________________\nFinal Results: \nCost: ${nn.global_cost}')
	}
	if need_to_save{
		println(" Saving the progress !")
		file := "cost=${cost_to_save}\nweights=${weights_to_save}\nlayers=${layers_to_save}"
		os.write_file(nn.save_path, file) or {panic(err)}
	}
}

// A possible way to train the network if you use a dataset
[inline]
[direct_array_access]
fn (mut nn NeuralNetwork) backprop(index int){
	//Deriv nactiv all neurons 
	nn.fprop_value(nn.training_inputs[index])
	// cost for the eyes
	for i in 0..nn.nb_neurones[nn.nb_neurones.len-1]{ // pour chaque output
		tmp := nn.layers_list[nn.nb_neurones.len-1][3][i] - unsafe{nn.excpd_training_outputs[index][i]}
		nn.global_cost += tmp*tmp  // Cost of each output
	}

	//Start backprop
	for mut hidden_lay in nn.layers_list{ //dsig all nactiv
		for mut elem in hidden_lay[2]{
			elem = nn.deriv_activ_func(*elem)
		}
	}
	// deltaC/deltaA(last)
	for i in 0..nn.nb_neurones[nn.nb_neurones.len-1]{ // pour chaque output
		nn.layers_list[nn.nb_neurones.len-1][4][i] = 2(nn.layers_list[nn.nb_neurones.len-1][3][i] - unsafe{nn.excpd_training_outputs[index][i]})
	}
	// deltaC/deltaW(last)
	for k, mut inputlist in nn.weights_list[nn.nb_neurones.len-2][1]{  // k is the nb of the input neuron
		for j, mut weight_cost in inputlist{ // j is the nb of the output neuron
			weight_cost += nn.layers_list[nn.nb_neurones.len-2][3][k]*nn.layers_list[nn.nb_neurones.len-1][2][j]*nn.layers_list[nn.nb_neurones.len-1][4][j]
		}
	}
	// deltaC/deltaB(last)
	for l, mut bias_cost in nn.layers_list[nn.nb_neurones.len-1][1]{
		bias_cost += nn.layers_list[nn.nb_neurones.len-1][2][l]*nn.layers_list[nn.nb_neurones.len-1][4][l]
	}

	// deltaC/deltaA(i)
	for i := nn.nb_neurones.len-2; i>0; i--{ // for each layer
		for j in 0..nn.nb_neurones[i]{ // pour chaque neurone de la couche (pour tous les faire)
			for k in 0..nn.nb_neurones[i+1]{ // pour chaque neurones de la couche d'après (on en a besoin pour le calcul)
				nn.layers_list[i][4][j] += nn.weights_list[i][0][j][k]*nn.layers_list[i+1][2][j]*nn.layers_list[i+1][4][j]
			}
		}
	}

	for i in 1..nn.nb_neurones.len-2{ 
		//Weights
		for j, mut inputlist in nn.weights_list[i-1][1]{
			for k, mut weight_cost in inputlist{
				weight_cost += nn.layers_list[i-1][3][j]*nn.layers_list[i][2][k]*nn.layers_list[i][4][k]
			}
		}
		//Biases
		for l, mut bias_cost in nn.layers_list[i][1]{
			bias_cost += nn.layers_list[i][2][l]*nn.layers_list[i][4][l]
		}
	}
}

[inline]
[direct_array_access]
fn (mut nn NeuralNetwork) apply_delta(){ // Apply the modifications calculated with the costs of the backprop
	//Weights
	for mut hidd_lay in nn.weights_list{
		for i, mut w_list in hidd_lay[0]{
			for j, mut weight in w_list{
				weight -= hidd_lay[1][i][j] * nn.learning_rate
			}
		}
	}
	//Biases
	for j in 1..nn.layers_list.len{ // cause we dont care about the bias of the input neuron that doesn't need to exist
		for i, mut bias in nn.layers_list[j][0]{
			bias -= nn.layers_list[j][1][i] * nn.learning_rate
		}
	}
}

[inline]
[direct_array_access]
fn (mut nn NeuralNetwork) neurons_costs_reset(){ // reset the layers that changed
	for i in 1..nn.layers_list.len{
		 nn.layers_list[i][4] = []f64{len:nn.nb_neurones[i]}
	}
}