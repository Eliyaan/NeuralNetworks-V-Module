module main
import preceptron as p


/*
Save le meilleur nn, load un nn
pouvoir test 1 seul truc
*/



fn main(){
	nb_epochs := 1000
	mut neunet := p.NeuralNet{		
								learning_rate: 0.3
								inputs: [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
								excpd_outputs: [[0.0], [0.0], [0.0], [1.0]]
								nb_inputs: 2 
								nb_hidden_layer: 1
								nb_hidden_neurones: [3] 
								nb_outputs: 1
								shuffle_dataset: true
								print_epoch: 10000
								activ_func: p.sigmoid
								deriv_activ_func: p.dsig
								save_path: "preceptron_module/preceptron_v_module/saved_nn.toml"
								load_path: "preceptron_module/preceptron_v_module/saved_nn.toml"
							}
	neunet.init()
	neunet.train(u64(nb_epochs))
}