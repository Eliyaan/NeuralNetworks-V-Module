module main
import preceptron as p

fn main(){
	nb_epochs := 10000
	mut neunet := p.NeuralNetwork{		
								learning_rate: 1
								inputs: [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] // exemple of inputs for 2 input neurons
								excpd_outputs: [[0.0], [0.0], [0.0], [1.0]]  // exemple of output for 1 output neuron
								nb_inputs: 2 
								nb_hidden_layer: 1
								nb_hidden_neurones: [3] 
								nb_outputs: 1
								shuffle_dataset: false
								print_epoch: 10000
								activ_func: p.sigmoid
								deriv_activ_func: p.dsig
								save_path: "saved_nn.toml"
								load_path: ""
							}
	neunet.init()
	neunet.train_backprop(u64(nb_epochs)) // possible way to train if you have a dataset

	println(neunet.fprop_value([0.0, 0.0])) // possible way to use the nn (without necessarly using a dataset)
	println(neunet.fprop_value([1.0, 1.0]))
}