//module preceptron
import rand as rd
import math as m

const(
	nb_i = 2
	nb_hl = 1
	hd_neu = [3]
	nb_o = 2
	nb_epochs = 10000
)


struct NeuralNet{
	//Consts
	learning_rate f64
	inputs [][]f64 
	excpd_outputs [][]f64  // first : prob for 0 ; sec : prob for 1
	nb_inputs int 
	nb_hidden_layer int
	nb_hidden_neurones []int 
	nb_outputs int

mut:
	weights_list [][][][]f64
	layers_list [][][]f64  // bias, bias cost, nactiv, output(activ), cost
	glob_output [][]f64

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

// fn relu(value f64) f64{
// 	if value < 0{
// 		return 0
// 	}else{
// 		return value
// 	}
// }

fn sigmoid(value f64) f64{
	return 1 / (1 + m.exp(-value))
}

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
			hidd_lay[3][j] = sigmoid(nactiv)  //activation function
		}
	}

	for i in 0..nn.nb_outputs{
		nn.layers_list[nn.nb_hidden_layer][4][i] += m.pow(nn.layers_list[nn.nb_hidden_layer][3][i] - excpd_outputs[i], 2)/2  //this is just to have the result I think
	}
}

fn (mut nn NeuralNet) reset(){
	for i, mut hidden_lay in nn.layers_list{
		if i == nn.nb_hidden_layer{
			hidden_lay[3] = []f64{len:nn.nb_outputs}
			if i < nn.nb_hidden_layer{
				hidden_lay[4] = []f64{len:nn.nb_outputs}
			}
		}else{
			hidden_lay[3] = []f64{len:nn.nb_hidden_neurones[i]}
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
			elem = dsig(elem)
		}
	}
	//Reverif toute la backprop pour s'assurer qu'elle soit dans le bon sens avec les bons trucs car c pas ca x)
	for i := nn.nb_hidden_layer; i>=0; i--{ 
		hidd_lay := nn.layers_list[i]
		if i == nn.nb_hidden_layer{ 
			//Weights
			for l, mut outputlist in nn.weights_list[i][1]{
				for j, mut weight_cost in outputlist{
					weight_cost += nn.layers_list[i-1][3][j]*hidd_lay[2][l]*(hidd_lay[3][l]-nn.excpd_outputs[index][l])  // It's normal that we dont apply the dsig bcz already applied 
				}
			}
			for l, mut bias_cost in hidd_lay[1]{
				bias_cost += hidd_lay[2][l]*(hidd_lay[3][l]-nn.excpd_outputs[index][l])   // It's normal that we dont apply the dsig bcz already applied 
			}
		}else if i == 0{
			if nn.nb_hidden_neurones.len > 1{
				for l, mut hidden_cost in hidd_lay[4]{
					for j in 0..nn.nb_hidden_neurones[1]{//a changer pour I+1 pour le reste + faire un condi si au moins 1
						hidden_cost += nn.weights_list[1][0][j][l]*nn.layers_list[1][2][j]*nn.layers_list[1][4][j]  // It's normal that we dont apply the dsig bcz already applied 
					}
				}
			}else{
				for l, mut hidden_cost in hidd_lay[4]{
					for j in 0..nn.nb_outputs{//a changer pour I+1 pour le reste + faire un condi si au moins 1
						hidden_cost += nn.weights_list[1][0][j][l]*nn.layers_list[1][2][j]*nn.layers_list[1][4][j]  // It's normal that we dont apply the dsig bcz already applied 
					}
				}
			}
			//Weights
			for l, mut outputlist in nn.weights_list[i][1]{
				for j, mut weight_cost in outputlist{
					weight_cost += nn.inputs[index][j]*hidd_lay[2][l]*hidd_lay[4][l]  // It's normal that we dont apply the dsig bcz already applied 
				}
			}
			for l, mut bias_cost in hidd_lay[1]{
				bias_cost += hidd_lay[2][l]*hidd_lay[4][l]   // It's normal that we dont apply the dsig bcz already applied 
			}
			
			
		}else if i == nn.nb_hidden_layer-1{
			println("antépénulflem")
			for l, mut hidden_cost in hidd_lay[4]{
				for j in 0..nn.nb_hidden_neurones[nn.nb_hidden_layer]{
					hidden_cost += nn.weights_list[nn.nb_hidden_layer][0][j][l]*nn.layers_list[nn.nb_hidden_layer][2][j]*(nn.layers_list[nn.nb_hidden_layer][3][j]-nn.excpd_outputs[index][j])  // It's normal that we dont apply the dsig bcz already applied 
				}
			}
			println("typ")
			//Weights
			for l, mut outputlist in nn.weights_list[i][1]{
				for j, mut weight_cost in outputlist{
					println(hidd_lay[4][l])
					weight_cost += nn.layers_list[i-1][3][j]*hidd_lay[2][l]*hidd_lay[4][l]  // It's normal that we dont apply the dsig bcz already applied 
				}
			}
			println("k")
			for l, mut bias_cost in hidd_lay[1]{
				bias_cost += hidd_lay[2][l]*(hidd_lay[3][l]-nn.excpd_outputs[index][l])   // It's normal that we dont apply the dsig bcz already applied 
			}			
		}else{
			//Weights
			for k, mut outputlist in nn.weights_list[i][1]{
				for j, mut weight_cost in outputlist{
					weight_cost += nn.layers_list[i-1][3][j]*hidd_lay[2][k]*hidd_lay[4][k]  // It's normal that we dont apply the dsig bcz already applied 
				}
			}
			for l, mut bias_cost in hidd_lay[1]{
				bias_cost += hidd_lay[2][l]*hidd_lay[4][l]   // It's normal that we dont apply the dsig bcz already applied 
			}
			for l, mut hidden_cost in hidd_lay[4]{
				for j in 0..nn.nb_hidden_neurones[1]{
					hidden_cost += nn.weights_list[i+1][0][j][l]*nn.layers_list[i+1][2][j]*nn.layers_list[i+1][4][j]  // It's normal that we dont apply the dsig bcz already applied 
				}
			}
		}
		
		
	}
}

fn (mut nn NeuralNet) apply_delta(){
	//Output Weights
	for l, mut hidd_lay in nn.weights_list{
		for i, mut w_list in hidd_lay[0]{
			for j, mut weight in w_list{
				weight -= hidd_lay[1][i][j] * nn.learning_rate
			}
		}
	}

	for h, mut hidd_lay in nn.layers_list{
		for i, mut bias in hidd_lay[0]{
			bias -= hidd_lay[1][i] * nn.learning_rate
		}
	}
}

//not done yet
fn main(){
	mut neunet := NeuralNet{		
								learning_rate: 0.3
								inputs: [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
								excpd_outputs: [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
								nb_inputs: nb_i 
								nb_hidden_layer: nb_hl
								nb_hidden_neurones: hd_neu 
								nb_outputs: nb_o

								weights_list: [][][][]f64{len:nb_hl+1, init:[][][]f64{len:2, init:[][]f64{}}}
								layers_list: [][][]f64{len:nb_hl+1, init:[][]f64{len:5, init:[]f64{}}}
								glob_output: [][]f64{len:4, init:[]f64{len:nb_i}}
							}
	for i in 0..nb_hl+1{
		if i == 0{
			neunet.weights_list[i][0] = [][]f64{len:hd_neu[0], init:[]f64{len:nb_i}}
			neunet.weights_list[i][1] = [][]f64{len:hd_neu[0], init:[]f64{len:nb_i}}
		}
		else if i == nb_hl{
			neunet.weights_list[i][0] = [][]f64{len:nb_o, init:[]f64{len:hd_neu[i-1]}}
			neunet.weights_list[i][1] = [][]f64{len:nb_o, init:[]f64{len:hd_neu[i-1]}}
		}else{
			neunet.weights_list[i][0] = [][]f64{len:hd_neu[i], init:[]f64{len:hd_neu[i-1]}}
			neunet.weights_list[i][1] = [][]f64{len:hd_neu[i], init:[]f64{len:hd_neu[i-1]}}
		}
	}
	for i in 0..nb_hl+1{
		if i == nb_hl{
			neunet.layers_list[i][0] = []f64{len:nb_o}
			neunet.layers_list[i][1] = []f64{len:nb_o}
			neunet.layers_list[i][2] = []f64{len:nb_o}
			neunet.layers_list[i][3] = []f64{len:nb_o}
			neunet.layers_list[i][4] = []f64{len:nb_o}
		}else{
			neunet.layers_list[i][0] = []f64{len:hd_neu[i]}
			neunet.layers_list[i][1] = []f64{len:hd_neu[i]}
			neunet.layers_list[i][2] = []f64{len:hd_neu[i]}
			neunet.layers_list[i][3] = []f64{len:hd_neu[i]}
			neunet.layers_list[i][4] = []f64{len:hd_neu[i]}
		}
	}
	neunet.set_rd_wb_values()
	for epoch in 0..nb_epochs{
		for mut hidd_lay in neunet.weights_list{
			for mut costs_list in hidd_lay[1]{
				costs_list = []f64{len:costs_list.len}
			}
		}
		for mut hidd_lay in neunet.layers_list{
			hidd_lay[1] = []f64{len:hidd_lay[1].len}
			hidd_lay[4] = []f64{len:hidd_lay[4].len}
		}
		for i, _ in neunet.inputs{
			neunet.reset()
			neunet.forward_prop(i)
			neunet.backprop(i)
			neunet.glob_output[i] = neunet.layers_list[nb_hl][3]
		}
		if epoch%10000 == 0{
			println('\nEpoch: $epoch Cost: ${neunet.layers_list[nb_hl][4]} \nOutputs: $neunet.glob_output \nExpected Outputs: $neunet.excpd_outputs')
		}
		neunet.apply_delta()
	}
	println('____________________________________________________________\nFinal Results: \nCost: ${neunet.layers_list[nb_hl][4]} \nOutputs: $neunet.glob_output \nExpected Outputs: $neunet.excpd_outputs')
}