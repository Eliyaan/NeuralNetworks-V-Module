module preceptron
import rand as rd
import math as m
import os
import toml


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
	glob_output [][]f64
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
				for k, elem in nn.layers_list[i-1][3]{  // For each input (each neuron of the last layer)
					nactiv += nn.weights_list[i-1][0][j][k] * elem // add the multiplication of the input (of the last layer) and the concerned weight to the non-activated output
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


// Used to get the cost and things for backprop
[inline]
[direct_array_access]
fn (mut nn NeuralNetwork) forward_prop(index int){ 
	fprop_value(nn.training_inputs[i])

	// Used if you have a dataset if not you can use fprop_value()
	excpd_training_outputs := &nn.excpd_training_outputs[index] // take the right expected output
	for i in 0..nn.nb_neurones[nn.nb_neurones.len-1]{ // pour chaque output
		tmp := nn.layers_list[nn.nb_neurones.len-1][3][i] - unsafe{excpd_training_outputs[i]}
		nn.layers_list[nn.nb_neurones.len-2][4][i] += (tmp*tmp)/2.0
	}	
	for cost in nn.layers_list[nn.nb_neurones.len-2][4]{
		nn.global_cost += cost
	}	
}


// A possible way to train the network if you use a dataset
[inline]
[direct_array_access]
// pas fonctionnel
fn (mut nn NeuralNetwork) backprop(index int){
	//Deriv nactiv all neurons 
	for mut hidden_lay in nn.layers_list{
		for mut elem in hidden_lay[2]{
			elem = nn.deriv_activ_func(*elem)
		}
	}
	for i := nn.nb_neurones.len-2; i>0; i--{ 
		hidd_lay := nn.layers_list[i]
		if i == nn.nb_neurones.len-2{ // for the output
			//Costs of the neurons (I think)
			for l, mut outputlist in nn.weights_list[i][1]{
				for j, mut weight_cost in outputlist{
					weight_cost += nn.layers_list[i][3][j]*hidd_lay[2][l]*(hidd_lay[3][l]-nn.excpd_training_outputs[index][l]) 
				}
			}
			//Biases
			for l, mut bias_cost in hidd_lay[1]{
				bias_cost += hidd_lay[2][l]*(hidd_lay[3][l]-nn.excpd_training_outputs[index][l]) 
			}
		}else if i == 0{ // for the first layer
			//Costs of the neurons (I think)
			if nn.nb_neurones.len > 1{ // if there is more than 1 hidden layer (idk why I didn't use the nb of hidden layers)
				for l, mut hidden_cost in hidd_lay[4]{
					for j in 0..nn.nb_neurones[1]{//a changer pour I+1 pour le reste + faire un condi si au moins 1
						hidden_cost += nn.weights_list[1][0][j][l]*nn.layers_list[1][2][j]*nn.layers_list[1][4][j]  // idk why it is 1 on this line and nn.nb_neurones.len-2 in the else one (may be an mistake)
					}
				}
			}else{
				for l, mut hidden_cost in hidd_lay[4]{
					for j in 0..nn.nb_neurones[nn.nb_neurones.len-1]{//a changer pour I+1 pour le reste + faire un condi si au moins 1
						hidden_cost += nn.weights_list[1][0][j][l]*nn.layers_list[1][2][j]*(nn.layers_list[nn.nb_neurones.len-2][3][j]-nn.excpd_training_outputs[index][j])
					}
				}
			}
			//Weights
			for l, mut outputlist in nn.weights_list[i][1]{
				for j, mut weight_cost in outputlist{
					weight_cost += nn.training_inputs[index][j]*hidd_lay[2][l]*hidd_lay[4][l] 
				}
			}
			//Biases
			for l, mut bias_cost in hidd_lay[1]{
				bias_cost += hidd_lay[2][l]*hidd_lay[4][l]
			}					
		}else if i == nn.nb_neurones.len-2-1{ // the hidden layber before the output layer
			//Costs of the neurons (I think)
			for l, mut hidden_cost in hidd_lay[4]{
				for j in 0..nn.layers_list[nn.nb_neurones.len-2][0].len{//Len of last layer
					hidden_cost += nn.weights_list[nn.nb_neurones.len-2][0][j][l]*nn.layers_list[nn.nb_neurones.len-2][2][j]*(nn.layers_list[nn.nb_neurones.len-2][3][j]-nn.excpd_training_outputs[index][j])
				}
			}
			//Weights
			for l, mut outputlist in nn.weights_list[i][1]{
				for j, mut weight_cost in outputlist{
					weight_cost += nn.layers_list[i-1][3][j]*hidd_lay[2][l]*hidd_lay[4][l]
				}
			}
			//Biases
			for l, mut bias_cost in hidd_lay[1]{
				bias_cost += hidd_lay[2][l]*hidd_lay[4][l]
			}
		}else{ // every other hidden layer
			//Costs of the neurons (I think)
			for l, mut hidden_cost in hidd_lay[4]{
				for j in 0..nn.nb_neurones[i+1]{
					hidden_cost += nn.weights_list[i+1][0][j][l]*nn.layers_list[i+1][2][j]*nn.layers_list[i+1][4][j]
				}
			}
			//Weights
			for k, mut outputlist in nn.weights_list[i][1]{
				for j, mut weight_cost in outputlist{
					weight_cost += nn.layers_list[i-1][3][j]*hidd_lay[2][k]*hidd_lay[4][k]
				}
			}
			//Biases
			for l, mut bias_cost in hidd_lay[1]{
				bias_cost += hidd_lay[2][l]*hidd_lay[4][l]
			}
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
	for mut hidd_lay in nn.layers_list{
		for i, mut bias in hidd_lay[0]{
			bias -= hidd_lay[1][i] * nn.learning_rate
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

[direct_array_access]
pub fn (mut nn NeuralNetwork) init(){ // To initialise the neural network
	if nn.load_path != ""{ // If there was a saved nn jsp si ca marche
		file := toml.parse_file(nn.load_path) or {panic(err)}
		nn.best_cost = file.value("cost").f64()
		base_weights_list := file.value("weights").array()
		base_layers_list := file.value("layers").array()
		mut base_layers_list_good := [][][]f64{}
		mut base_weights_listgood := [][][][]f64{}
		for a, layer in base_weights_list{
			base_weights_listgood << [][][]f64{}
			for b, flist in layer.array(){
				base_weights_listgood[a] << [][]f64{}
				for c, maybeline in flist.array(){
					base_weights_listgood[a][b] << []f64{}
					for maybecoll in maybeline.array(){
						base_weights_listgood[a][b][c] << maybecoll.f64()
					}
				}
			}
		}
		for a, layer in base_layers_list{
			base_layers_list_good << [][]f64{}
			for b, flist in layer.array(){
				base_layers_list_good[a] << []f64{}
				for value in flist.array(){
					base_layers_list_good[a][b] << value.f64()
				}
			}
		}
		nn.layers_list = base_layers_list_good
		nn.weights_list = base_weights_listgood
		nn.glob_output = [][]f64{len:nn.excpd_training_outputs.len, init:[]f64{len:nn.nb_neurones[0]}}
	}else{ // If it's a new nn
		nn.weights_list = [][][][]f64{len:nn.nb_neurones.len-1, init:[][][]f64{len:2, init:[][]f64{len:nn.nb_neurones[index+1], init:[]f64{len:nn.nb_neurones[index]}}}}
		nn.layers_list = [][][]f64{len:nn.nb_neurones.len, init:[][]f64{len:5, init:[]f64{len:nn.nb_neurones[index]}}}
		//nn.glob_output = [][]f64{len:nn.excpd_training_outputs.len, init:[]f64{len:nn.nb_neurones[0]}}

		nn.set_rd_wb_values()
	}
}

pub fn (mut nn NeuralNetwork) softmax() []f64{ // softmax function if needed
	mut sum := 0.0
	for value in nn.layers_list[nn.nb_neurones.len-2][3]{
		sum += value
	}
	for mut value in nn.layers_list[nn.nb_neurones.len-2][3]{
		value /= sum
	}
    return nn.layers_list[nn.nb_neurones.len-2][3]
}



[direct_array_access]
pub fn (mut nn NeuralNetwork) train_backprop(nb_epochs u64){ // exemple of a training loop
	mut need_to_save := false
	mut cost_to_save := 0.0
	mut weights_to_save := [][][][]f64{}
	mut layers_to_save := [][][]f64{}
	for epoch in 0..nb_epochs{
		if epoch != 0{
			nn.apply_delta()
		}
		if nn.shuffle_dataset{
			nn.randomise_i_exp_o()
		}
		for mut hidd_lay in nn.weights_list{
			for mut costs_list in hidd_lay[1]{
				costs_list = []f64{len:costs_list.len}
			}
		}
		for mut hidd_lay in nn.layers_list{
			hidd_lay[1] = []f64{len:hidd_lay[1].len}
			hidd_lay[4] = []f64{len:hidd_lay[4].len}
		}
		nn.global_cost = 0.0
		for i, _ in nn.training_inputs{
			nn.reset()
			nn.forward_prop(i)
			nn.backprop(i)
			nn.glob_output[i] = nn.layers_list[nn.nb_neurones.len-2][3]
		}
		if nn.print_epoch > 0 && nn.print{
			if epoch%u64(nn.print_epoch) == 0{
				println('\nEpoch: $epoch Global Cost: ${nn.global_cost} \nOutputs: $nn.glob_output \nExpected Outputs: $nn.excpd_training_outputs')
			}
		}
		if nn.best_cost/nn.global_cost > 1.0{
			need_to_save = true
			cost_to_save = nn.global_cost
			weights_to_save = nn.weights_list.clone()
			layers_to_save = nn.layers_list.clone()
			nn.best_cost = nn.global_cost
		}else{
			println(nn.best_cost)
			println(nn.global_cost)
		}
	}
	if nn.print{
		println('____________________________________________________________\nFinal Results: \nCost: ${nn.global_cost} \nOutputs: $nn.glob_output \nExpected Outputs: $nn.excpd_training_outputs')
	}
	if need_to_save{
		println(" Saving the progress !")
		file := "cost=${cost_to_save}\nweights=${weights_to_save}\nlayers=${layers_to_save}"
		os.write_file(nn.save_path, file) or {panic(err)}
	}
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

[inline]
[direct_array_access]
//Jsp si ca marche
fn (mut nn NeuralNetwork) reset(){ // reset the layers that changed
	for i, mut hidden_lay in nn.layers_list{
		hidden_lay[3] = []f64{len:nn.nb_neurones[i]}
		hidden_lay[2] = []f64{len:nn.nb_neurones[i]}
		if i < nn.nb_neurones.len-2{ // idk why
			hidden_lay[4] = []f64{len:nn.nb_neurones[i]}
		}
	}
}