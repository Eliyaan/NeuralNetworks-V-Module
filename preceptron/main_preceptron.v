module preceptron
import toml

// make that the save includes the size of the network, could be handy ?
// minibatches (use of the shuffle)


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
	print bool = true

mut:
	weights_list [][][][]f64 // [layer_nbr][weights/weights_cost (0or1)][start_neuron_nbr][result_neuron_nbr] it stores the weights and their associated costs
	layers_list [][][]f64  // [layer_nbr][bias/bcost/nactiv...*(see at the end of the comment)][neuron_nbr] (contains all the hidden layers and the output sothe 0 is the first hidden layer) * bias, bias cost, non-activated_output (nactiv), output (activ), cost
	
	// the followings are if you are training it with a dataset most probably, you wont need it for exemple for a genetic algorithm
	global_cost f64
	training_inputs [][]f64 // store the inputs that will be used for the training
	excpd_training_outputs [][]f64  // store the wanted neuron activations ex : [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]] for 2 inputs, the first one should fire the second neuron and no other ones and the second input should fire the third one and the fourth (normally can suport 0.3 or floats like that, so not limited to either 0 or 1)

	best_cost f64 = 100000000000
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
		nn.weights_list = [][][][]f64{len:nn.nb_neurones.len-1, init:[][][]f64{len:2, init:[][]f64{}}}
		for i, mut layer in nn.weights_list{ //for each layer
			for mut type_list in layer{  // for each data type (weight/associated cost)
				for j in 0..nn.nb_neurones[i]{  // for each input neuron of the weights
					type_list << []f64{len:nn.nb_neurones[i+1]}  // for each output neuron
				}
			}
		}

		nn.layers_list = [][][]f64{len:nn.nb_neurones.len, init:[][]f64{}}
		for i, mut layer in nn.layers_list{  // for each layer
			for _ in 0..5{  // for each type (bias, bias cost, non-activated_output (nactiv), output (activ), cost)
				layer << []f64{len:nn.nb_neurones[i]} // for each neuron
			}
		}
	
		nn.set_rd_wb_values() // init the random weights and biases
	}
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