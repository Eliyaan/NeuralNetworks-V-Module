module preceptron
import toml

// make that the save includes the size of the network, could be handy ?
// minibatches (use of the shuffle)

pub struct Neuron{
mut:
	bias f64
	b_cost f64
	nactiv f64
	output f64
	cost f64
}

pub struct Weight{
mut:
	weight f64
	cost f64
}

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
	weights_list [][][]Weight // [layer_nbr][input_neuron_nbr][output_neuron_nbr] it stores the all weights and their associated costs
	layers_list [][]Neuron  // [layer_nbr][neuron_nbr] contains all the layers
	
	// the followings are if you are training it with a dataset most probably, you wont need it for exemple for a genetic algorithm
	global_cost f64
	training_inputs [][]f64 // store the inputs that will be used for the training
	excpd_training_outputs [][]f64  // store the wanted neuron activations ex : [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]] for 2 inputs, the first one should fire the second neuron and no other ones and the second input should fire the third one and the fourth (normally can suport 0.3 or floats like that, so not limited to either 0 or 1)

	best_cost f64 = 100000000000
}

[direct_array_access]
pub fn (mut nn NeuralNetwork) init(){ // To initialise the neural network
	if nn.load_path != ""{  // KASSé
	/*
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
						base_weights_listgood[a][b][c] << Weight{weight.f64()}
					}
				}
			}
		}
		for a, layer in base_layers_list{
			base_layers_list_good << [][]f64{}
			for b, diff_types_lists in layer.array(){
				base_layers_list_good[a] << []f64{}
				for value in diff_types_lists.array(){
					base_layers_list_good[a][b] << Neuron{value.f64()}
				}
			}
		}
		nn.layers_list = base_layers_list_good
		nn.weights_list = base_weights_listgood
		*/
	}else{ // If it's a new nn
		nn.weights_list = [][][]Weight{len:nn.nb_neurones.len-1}
		
		for i, mut layer in nn.weights_list{
			for _ in 0..nn.nb_neurones[i]{
				layer << []Weight{len:nn.nb_neurones[i+1]}
			}
		}

		nn.layers_list = [][]Neuron{}
		for nb in nn.nb_neurones{
			nn.layers_list << []Neuron{len:nb}
		}

		nn.set_rd_wb_values()
	}
}

// For doing a simple forward propagation 
[inline]
[direct_array_access]
pub fn (mut nn NeuralNetwork) fprop_value(inputs []f64) []Neuron{ // Return the result of the nn for this input, use this if you dont have a dataset
	for i, input in inputs{
		nn.layers_list[0][i].output = input
	}
	for i, mut hidd_lay in nn.layers_list{ // For each layer (hidden and output)
		if i > 0{ // ignore the input layer
			for j, mut o_neuron in hidd_lay{ // For each neuron in the concerned layer
				o_neuron.nactiv = 0
				for k, i_neuron in nn.layers_list[i-1]{  // For each input (each neuron of the last layer)
					o_neuron.nactiv += nn.weights_list[i-1][k][j].weight * i_neuron.output // add the multiplication of the input (of the last layer) and the concerned weight to the non-activated output
				}
				o_neuron.nactiv += o_neuron.bias  // Ajout du bias
				hidd_lay[j].output = nn.activ_func(o_neuron.nactiv)  //activation function
			}
		}
	}
	
	return nn.layers_list[nn.nb_neurones.len-1]
}

[inline]
[direct_array_access]
pub fn get_neuron_array_output(neurons []Neuron) []f64{
	return []f64{len:neurons.len, init:neurons[index].output}
}