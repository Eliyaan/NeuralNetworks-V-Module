module preceptron

import rand as rd

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