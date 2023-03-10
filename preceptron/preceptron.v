module preceptron
import rand as rd
import math as m
import os
import toml


pub struct NeuralNet{
	//Consts
	learning_rate f64
	nb_inputs int 
	nb_hidden_layer int
	nb_hidden_neurones []int 
	nb_outputs int
	activ_func fn(f64) f64
	deriv_activ_func fn(f64) f64

	shuffle_dataset bool
	print_epoch int
	
	save_path string
	load_path string 
	print bool

mut:
	weights_list [][][][]f64
	layers_list [][][]f64  // bias, bias cost, nactiv, output(activ), cost
	glob_output [][]f64
	global_cost f64
	
	inputs [][]f64 
	excpd_outputs [][]f64  // first : prob for 0 ; sec : prob for 1

	best_cost f64 = 100000000000
}

fn (mut nn NeuralNet) set_rd_wb_values(){
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

// fn (mut nn NeuralNet) softmax(){
// 	mut sum := 0.0
// 	for value in nn.output{
// 		sum += value
// 	}
// 	for mut value in nn.output{
// 		value /= sum
// 	}
// }

fn (mut nn NeuralNet) forward_prop(index int){
	inputs := nn.inputs[index]
	excpd_outputs := nn.excpd_outputs[index]
	for i, mut hidd_lay in nn.layers_list{
		for j, mut nactiv in hidd_lay[2]{
			if i == 0{
				for k, elem in inputs{  // Pour chaque input
					nactiv += nn.weights_list[i][0][j][k] * elem //Le bon weight fois le bon input
				}
			}else{
				for k, elem in nn.layers_list[i-1][3]{  // Pour chaque input
					nactiv += nn.weights_list[i][0][j][k] * elem //Le bon weight fois le bon input
				}
			}
			
			nactiv += hidd_lay[0][j]  // Ajout du bias
			hidd_lay[3][j] = nn.activ_func(*nactiv)  //activation function
		}
	}

	for i in 0..nn.nb_outputs{
		nn.layers_list[nn.nb_hidden_layer][4][i] += m.pow(nn.layers_list[nn.nb_hidden_layer][3][i] - excpd_outputs[i], 2)/2
	}	
	for cost in nn.layers_list[nn.nb_hidden_layer][4]{
		nn.global_cost += cost
	}
}

fn (mut nn NeuralNet) reset(){
	for i, mut hidden_lay in nn.layers_list{
		if i == nn.nb_hidden_layer{
			hidden_lay[3] = []f64{len:nn.nb_outputs}
			hidden_lay[2] = []f64{len:nn.nb_outputs}
			if i < nn.nb_hidden_layer{
				hidden_lay[4] = []f64{len:nn.nb_outputs}
			}
		}else{
			hidden_lay[3] = []f64{len:nn.nb_hidden_neurones[i]}
			hidden_lay[2] = []f64{len:nn.nb_hidden_neurones[i]}
			if i < nn.nb_hidden_layer{
				hidden_lay[4] = []f64{len:nn.nb_hidden_neurones[i]}
			}
		}
	}
}

fn (mut nn NeuralNet) backprop(index int){
	//Dsig nactiv all neurons
	for mut hidden_lay in nn.layers_list{
		for mut elem in hidden_lay[2]{
			elem = nn.deriv_activ_func(*elem)
		}
	}
	//Reverif toute la backprop pour s'assurer qu'elle soit dans le bon sens avec les bons trucs car c pas ca x)
	for i := nn.nb_hidden_layer; i>=0; i--{ 
		hidd_lay := nn.layers_list[i]
		if i == nn.nb_hidden_layer{ //Correct
			//Weights
			for l, mut outputlist in nn.weights_list[i][1]{
				for j, mut weight_cost in outputlist{
					weight_cost += nn.layers_list[i-1][3][j]*hidd_lay[2][l]*(hidd_lay[3][l]-nn.excpd_outputs[index][l]) 
				}
			}
			for l, mut bias_cost in hidd_lay[1]{
				bias_cost += hidd_lay[2][l]*(hidd_lay[3][l]-nn.excpd_outputs[index][l]) 
			}
		}else if i == 0{
			if nn.nb_hidden_neurones.len > 1{
				for l, mut hidden_cost in hidd_lay[4]{
					for j in 0..nn.nb_hidden_neurones[1]{//a changer pour I+1 pour le reste + faire un condi si au moins 1
						hidden_cost += nn.weights_list[1][0][j][l]*nn.layers_list[1][2][j]*nn.layers_list[1][4][j]
					}
				}
			}else{
				for l, mut hidden_cost in hidd_lay[4]{
					for j in 0..nn.nb_outputs{//a changer pour I+1 pour le reste + faire un condi si au moins 1
						hidden_cost += nn.weights_list[1][0][j][l]*nn.layers_list[1][2][j]*(nn.layers_list[nn.nb_hidden_layer][3][j]-nn.excpd_outputs[index][j])
					}
				}
			}
			//Weights
			for l, mut outputlist in nn.weights_list[i][1]{
				for j, mut weight_cost in outputlist{
					weight_cost += nn.inputs[index][j]*hidd_lay[2][l]*hidd_lay[4][l] 
				}
			}
			for l, mut bias_cost in hidd_lay[1]{
				bias_cost += hidd_lay[2][l]*hidd_lay[4][l]
			}					
		}else if i == nn.nb_hidden_layer-1{
			for l, mut hidden_cost in hidd_lay[4]{
				for j in 0..nn.layers_list[nn.nb_hidden_layer][0].len{//Len of last layer
					hidden_cost += nn.weights_list[nn.nb_hidden_layer][0][j][l]*nn.layers_list[nn.nb_hidden_layer][2][j]*(nn.layers_list[nn.nb_hidden_layer][3][j]-nn.excpd_outputs[index][j])
				}
			}
			//Weights
			for l, mut outputlist in nn.weights_list[i][1]{
				for j, mut weight_cost in outputlist{
					weight_cost += nn.layers_list[i-1][3][j]*hidd_lay[2][l]*hidd_lay[4][l]
				}
			}
			for l, mut bias_cost in hidd_lay[1]{
				bias_cost += hidd_lay[2][l]*hidd_lay[4][l]
			}
		}else{
			for l, mut hidden_cost in hidd_lay[4]{
				for j in 0..nn.nb_hidden_neurones[i+1]{
					hidden_cost += nn.weights_list[i+1][0][j][l]*nn.layers_list[i+1][2][j]*nn.layers_list[i+1][4][j]
				}
			}
			//Weights
			for k, mut outputlist in nn.weights_list[i][1]{
				for j, mut weight_cost in outputlist{
					weight_cost += nn.layers_list[i-1][3][j]*hidd_lay[2][k]*hidd_lay[4][k]
				}
			}
			for l, mut bias_cost in hidd_lay[1]{
				bias_cost += hidd_lay[2][l]*hidd_lay[4][l]
			}
		}
	}
}

fn (mut nn NeuralNet) apply_delta(){
	//Output Weights
	for mut hidd_lay in nn.weights_list{
		for i, mut w_list in hidd_lay[0]{
			for j, mut weight in w_list{
				weight -= hidd_lay[1][i][j] * nn.learning_rate
			}
		}
	}

	for mut hidd_lay in nn.layers_list{
		for i, mut bias in hidd_lay[0]{
			bias -= hidd_lay[1][i] * nn.learning_rate
		}
	}
}

fn (mut nn NeuralNet) randomise_i_exp_o(){
	mut base_inputs := nn.inputs.clone()
	range := base_inputs.len
	mut base_expd_o := nn.excpd_outputs.clone()
	nn.inputs.clear()
	nn.excpd_outputs.clear()
	for _ in 0..range{
		i := rd.int_in_range(0, base_inputs.len) or {panic(err)}
		nn.inputs << base_inputs[i]
		base_inputs.delete(i)
		nn.excpd_outputs << base_expd_o[i]
		base_expd_o.delete(i)
	}
}

pub fn (mut nn NeuralNet) init(){
	if nn.load_path != ""{
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
		nn.glob_output = [][]f64{len:nn.excpd_outputs.len, init:[]f64{len:nn.nb_inputs}}
	}else{
		nn.weights_list = [][][][]f64{len:nn.nb_hidden_layer+1, init:[][][]f64{len:2, init:[][]f64{}}}
		nn.layers_list = [][][]f64{len:nn.nb_hidden_layer+1, init:[][]f64{len:5, init:[]f64{}}}
		nn.glob_output = [][]f64{len:nn.excpd_outputs.len, init:[]f64{len:nn.nb_inputs}}
		for i in 0..nn.nb_hidden_layer+1{
			if i == 0{
				nn.weights_list[i][0] = [][]f64{len:nn.nb_hidden_neurones[0], init:[]f64{len:nn.nb_inputs}}
				nn.weights_list[i][1] = [][]f64{len:nn.nb_hidden_neurones[0], init:[]f64{len:nn.nb_inputs}}
			}
			else if i == nn.nb_hidden_layer{
				nn.weights_list[i][0] = [][]f64{len:nn.nb_outputs, init:[]f64{len:nn.nb_hidden_neurones[i-1]}}
				nn.weights_list[i][1] = [][]f64{len:nn.nb_outputs, init:[]f64{len:nn.nb_hidden_neurones[i-1]}}
			}else{
				nn.weights_list[i][0] = [][]f64{len:nn.nb_hidden_neurones[i], init:[]f64{len:nn.nb_hidden_neurones[i-1]}}
				nn.weights_list[i][1] = [][]f64{len:nn.nb_hidden_neurones[i], init:[]f64{len:nn.nb_hidden_neurones[i-1]}}
			}
		}
		for i in 0..nn.nb_hidden_layer+1{
			if i == nn.nb_hidden_layer{
				nn.layers_list[i][0] = []f64{len:nn.nb_outputs}
				nn.layers_list[i][1] = []f64{len:nn.nb_outputs}
				nn.layers_list[i][2] = []f64{len:nn.nb_outputs}
				nn.layers_list[i][3] = []f64{len:nn.nb_outputs}
				nn.layers_list[i][4] = []f64{len:nn.nb_outputs}
			}else{
				nn.layers_list[i][0] = []f64{len:nn.nb_hidden_neurones[i]}
				nn.layers_list[i][1] = []f64{len:nn.nb_hidden_neurones[i]}
				nn.layers_list[i][2] = []f64{len:nn.nb_hidden_neurones[i]}
				nn.layers_list[i][3] = []f64{len:nn.nb_hidden_neurones[i]}
				nn.layers_list[i][4] = []f64{len:nn.nb_hidden_neurones[i]}
			}
		}
		nn.set_rd_wb_values()
		cost := (toml.parse_file(nn.save_path) or {panic(err)}).value("cost").f64()
		if cost != 0.0{
			nn.best_cost = cost
		}
	}
	
}

fn (mut nn NeuralNet) test_fprop(inputs []f64){
	for i, mut hidd_lay in nn.layers_list{
		for j, mut nactiv in hidd_lay[2]{
			if i == 0{
				for k, elem in inputs{  // Pour chaque input
					nactiv += nn.weights_list[i][0][j][k] * elem //Le bon weight fois le bon input
				}
			}else{
				for k, elem in nn.layers_list[i-1][3]{  // Pour chaque input
					nactiv += nn.weights_list[i][0][j][k] * elem //Le bon weight fois le bon input
				}
			}
			
			nactiv += hidd_lay[0][j]  // Ajout du bias
			hidd_lay[3][j] = nn.activ_func(*nactiv)  //activation function
		}
	}
	println("Tested Inputs: ${inputs}\nTested Output: ${nn.layers_list[nn.nb_hidden_layer][3]} ")	
}

pub fn (mut nn NeuralNet) test_value(value []f64){
	nn.reset()
	nn.test_fprop(value)
}

pub fn (mut nn NeuralNet) train(nb_epochs u64){
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
		for i, _ in nn.inputs{
			nn.reset()
			nn.forward_prop(i)
			nn.backprop(i)
			nn.glob_output[i] = nn.layers_list[nn.nb_hidden_layer][3]
		}
		if nn.print_epoch > 0 && nn.print{
			if epoch%u64(nn.print_epoch) == 0{
				println('\nEpoch: $epoch Global Cost: ${nn.global_cost} \nOutputs: $nn.glob_output \nExpected Outputs: $nn.excpd_outputs')
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
		println('____________________________________________________________\nFinal Results: \nCost: ${nn.global_cost} \nOutputs: $nn.glob_output \nExpected Outputs: $nn.excpd_outputs')
	}
	if need_to_save{
		file := "cost=${cost_to_save}\nweights=${weights_to_save}\nlayers=${layers_to_save}"
		os.write_file(nn.save_path, file) or {panic(err)}
	}
}

