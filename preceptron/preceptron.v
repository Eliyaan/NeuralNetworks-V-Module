module preceptron
import rand as rd
import math as m
import os
import toml

// make that the save includes the size of the network, could be handy ?


pub struct NeuralNetwork{
	learning_rate f64
	nb_neurones []int // contains the nbr of neurones for each layer
	activ_func fn(f64) f64 = sigmoid

	//if you use a dataset
	deriv_activ_func fn(f64) f64 = dsig // no dtanh yet

	shuffle_dataset bool
	print_epoch int
	
	save_path string
	load_path string 
	print bool

mut:
	weights_list [][][][]f64 // [layer_nbr][weights/weights_cost (0or1)][start_neuron_nbr][result_neuron_nbr] it stores the weights and their associated costs
	layers_list [][][]f64  // [layer_nbr][bias/bcost/nactiv...*(see at the end of the comment)][neuron_nbr] (contains all the hidden layers and the output sothe 0 is the first hidden layer) * bias, bias cost, non-activated_output (nactiv), output (activ), cost
	
	// the followings are if you are training it with a dataset most probably, you wont need it for exemple for a genetic algorithm
	global_cost f64
	training_inputs [][]f64 // store the inputs that will be used for the training
	excpd_training_outputs [][]f64  // store the wanted neuron activations ex : [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]] for 2 inputs, the first one should fire the second neuron and no other ones and the second input should fire the third one and the fourth (normally can suport 0.3 or floats like that, so not limited to either 0 or 1)

	best_cost f64 = 100000000000
}

// For doing a simple forward propagation 
[inline]
[direct_array_access]
pub fn (mut nn NeuralNetwork) fprop_value(input []f64) []f64{ // Return the result of the nn for this input, use this if you dont have a dataset
	nn.layers_list[0][3] = input.clone()
	for i, mut hidd_lay in nn.layers_list{ // For each layer (hidden and output)
		if i > 0{ // ignore the input layer
			for j, mut nactiv in hidd_lay[2]{ // For each neuron in the concerned layer
				nactiv = 0
				for k, elem in nn.layers_list[i-1][3]{  // For each input (each neuron of the last layer)
					nactiv += nn.weights_list[i-1][0][k][j] * elem // add the multiplication of the input (of the last layer) and the concerned weight to the non-activated output
				}
				
				nactiv += hidd_lay[0][j]  // Ajout du bias
				hidd_lay[3][j] = nn.activ_func(*nactiv)  //activation function
			}
		}
	}
	
	return nn.layers_list[nn.nb_neurones.len-2][3]	
}

//To initialise the nn with random weights and biases
[inline]
[direct_array_access]
fn (mut nn NeuralNetwork) set_rd_wb_values(){
	//Weights
	for mut hw_wc_list in nn.weights_list{
		for mut weights_list in hw_wc_list[0]{
			for mut weight in weights_list{
				weight = rd.f64_in_range(-1, 1)or{panic(err)}
			}
		}	
	}

	//Biases 
	for mut neuron in nn.layers_list{
		for mut bias in neuron[0]{
			bias = rd.f64_in_range(-1, 1)or{panic(err)}
		}
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
fn (mut nn NeuralNetwork) randomise_i_exp_o(){ // To shuffle the dataset I think
	mut base_inputs := nn.training_inputs.clone()
	range := base_inputs.len
	mut base_expd_o := nn.excpd_training_outputs.clone()
	nn.training_inputs.clear()
	nn.excpd_training_outputs.clear()
	for _ in 0..range{
		i := rd.int_in_range(0, base_inputs.len) or {panic(err)}
		nn.training_inputs << base_inputs[i]
		base_inputs.delete(i)
		nn.excpd_training_outputs << base_expd_o[i]
		base_expd_o.delete(i)
	}
}

[inline]
[direct_array_access]
fn (mut nn NeuralNetwork) reset(){ // reset the layers that changed
	for i in 1..nn.layers_list.len{
		 nn.layers_list[i][4] = []f64{len:nn.nb_neurones[i]}
	}
}

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
			nn.reset()
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

pub fn (mut nn NeuralNetwork) softmax() []f64{
	mut sum := 0.0
	for value in nn.layers_list[nn.nb_neurones.len-1][3]{
		sum += value
	}
	for mut value in nn.layers_list[nn.nb_neurones.len-1][3]{
		value /= sum
	}
    return nn.layers_list[nn.nb_neurones.len-1][3]
}

//Different activation functions and their derivatives
[inline]
fn tanh(value f64) f64{
	return (m.exp(value)-m.exp(-value)) / (m.exp(value)+m.exp(-value))
}

[inline]
fn dtanh(value f64) f64{
	val := tanh(value)
	return 1 - val*val
}

[inline]
fn relu(value f64) f64{
	return if value<0{0}else{value}
}

[inline]
fn drelu(value f64) f64{
	return if value<0{0.0}else{1.0}
}

[inline]
fn leaky_relu(value f64) f64{
	return if value<0{value*0.01}else{value}
}

[inline]
fn dleaky_relu(value f64) f64{
	return if value<0{0.01}else{1.0}
}

[inline]
fn sigmoid(value f64) f64{
	return 1 / (1 + m.exp(-value))
}

[inline]
fn dsig(value f64) f64{
	sigx := sigmoid(value)
	return sigx*(1 - sigx)
}

[direct_array_access]
pub fn (mut nn NeuralNetwork) init(){ // To initialise the neural network
	if nn.load_path != ""{ 
		file := toml.parse_file(nn.load_path) or {panic(err)}
		nn.best_cost = file.value("cost").f64()
		base_weights_list := file.value("weights").array()
		base_layers_list := file.value("layers").array()
		mut base_layers_list_good := [][][]f64{}
		mut base_weights_listgood := [][][][]f64{}
		for a, layer in base_weights_list{
			base_weights_listgood << [][][]f64{}
			for b, diff_types_lists in layer.array(){
				base_weights_listgood[a] << [][]f64{}
				for c, weights_lists in diff_types_lists.array(){
					base_weights_listgood[a][b] << []f64{}
					for weight in weights_lists.array(){
						base_weights_listgood[a][b][c] << weight.f64()
					}
				}
			}
		}
		for a, layer in base_layers_list{
			base_layers_list_good << [][]f64{}
			for b, diff_types_lists in layer.array(){
				base_layers_list_good[a] << []f64{}
				for value in diff_types_lists.array(){
					base_layers_list_good[a][b] << value.f64()
				}
			}
		}
		nn.layers_list = base_layers_list_good
		nn.weights_list = base_weights_listgood
	}else{ // If it's a new nn
		nn.weights_list = [][][][]f64{len:nn.nb_neurones.len-1, init:[][][]f64{len:2, init:[][]f64{len:nn.nb_neurones[index], init:[]f64{len:nn.nb_neurones[index+1]}}}}
		nn.layers_list = [][][]f64{len:nn.nb_neurones.len, init:[][]f64{len:5, init:[]f64{len:nn.nb_neurones[index]}}}

		nn.set_rd_wb_values()
	}
}