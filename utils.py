import math
import os 

import numpy as np 
import tensorflow as tf

import evaluation
from tensorflow import keras
import time


def save_mymodel(save_path, save_name, need_save_model):
	cur_path = save_path + '/' + save_name
	need_save_model.save_weights(cur_path )
	print("Already saved the model's weights in file" + cur_path)

def divide_groups(concated_data, input_t):
    i0 = tf.cast((tf.where(input_t < 1)[:, 0]), tf.int32)
    i1 = tf.cast((tf.where(input_t > 0)[:, 0]), tf.int32)
    group_T = tf.gather(concated_data, i1)
    group_C = tf.gather(concated_data, i0)
    return tf.constant(group_T), tf.constant(group_C), i0, i1

def split_train_val_test(data, train_ratio, val_ratio, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    train_set_size = int(len(data) * train_ratio)
    val_set_size = int(len(data) * val_ratio)
    train_indices = shuffled_indices[:train_set_size]
    val_indices = shuffled_indices[train_set_size:train_set_size+val_set_size]
    test_indices = shuffled_indices[train_set_size+val_set_size:]
    return train_indices, val_indices, test_indices

def save_results(save_result, save_name):
	np.save(save_name, save_result) 
	print("saved all results")

def normalize_adj_numpy(mx):
    rowsum = np.sum(mx, axis=-1)  
    msq_rowsum = np.power(rowsum, -0.5)  
    msq_rowsum[np.isinf(msq_rowsum)] = 0.  
    msq_D = np.diag(msq_rowsum)
    L = msq_D @ mx @ msq_D
    return L

def config_pare_DITE(iterations, lr_rate, lr_weigh_decay, flag_early_stop, use_batch,
    rep_alpha, out_dropout, GNN_dropout, rep_dropout, inp_dropout, rep_hidden_layer, rep_hidden_shape,
    GNN_hidden_layer, GNN_hidden_shape, out_T_layer, out_C_layer, out_hidden_shape, activation,
    phi_shape, att_hidden_shape, GRU_hidden_shape, reg_lambda, window_size, flag_matt):
        cur_activation = activation
        config = {}
        config["iterations"] = iterations
        config["lr_rate"] = lr_rate
        config["lr_weigh_decay"] = lr_weigh_decay
        config["flag_early_stop"] = flag_early_stop
        config['use_batch'] = use_batch
        config['rep_alpha'] = rep_alpha
        config['out_dropout'] = out_dropout
        config['GNN_dropout'] = GNN_dropout
        config['rep_dropout'] = rep_dropout
        config['inp_dropout'] = inp_dropout
        config['reg_lambda'] = reg_lambda
        config['phi_shape'] = phi_shape
        config['rep_hidden_layer'] = rep_hidden_layer
        config['rep_hidden_shape'] = rep_hidden_shape
        config['GNN_hidden_layer'] = GNN_hidden_layer
        config['GNN_hidden_shape'] = GNN_hidden_shape
        config['GRU_hidden_shape'] = GRU_hidden_shape
        config['out_T_layer'] = out_T_layer
        config['out_C_layer'] = out_C_layer
        config['out_hidden_shape'] = out_hidden_shape
        config['att_hidden_shape'] = att_hidden_shape
        config["window_size"] = window_size
        config["flag_matt"] = flag_matt
        return config,cur_activation

def load_data_dy(data_name):
    data = []
    if data_name == 'flickr_5':
        xs = np.load("data_dy/flk_sic/flk_x_5time.npy")
        adjs= np.load("data_dy/flk_sic/flk_A_5time.npy")
        Ts = np.load("data_dy/flk_sic/flk_t_5time.npy")
        all_yfs = np.load("data_dy/flk_sic/flk_yf_5time.npy")
        all_y1s = np.load("data_dy/flk_sic/flk_y1_5time.npy")
        all_y0s = np.load("data_dy/flk_sic/flk_y0_5time.npy")
    elif data_name == 'flickr_new_5':
        xs = np.load("data_dy/flk_sdc/flk_x_5time.npy")
        adjs= np.load("data_dy/flk_sdc/flk_A_5time.npy")
        Ts = np.load("data_dy/flk_sdc/flk_t_5time.npy")
        all_yfs = np.load("data_dy/flk_sdc/flk_yf_5time.npy")
        all_y1s = np.load("data_dy/flk_sdc/flk_y1_5time.npy")
        all_y0s = np.load("data_dy/flk_sdc/flk_y0_5time.npy")
    elif data_name == 'blog_new_5':
        xs = np.load("data_dy/flk_sdc/blog_x_5time.npy")
        adjs= np.load("data_dy/flk_sdc/blog_A_5time.npy")
        Ts = np.load("data_dy/flk_sdc/blog_t_5time.npy")
        all_yfs = np.load("data_dy/flk_sdc/blog_yf_5time.npy")
        all_y1s = np.load("data_dy/flk_sdc/blog_y1_5time.npy")
        all_y0s = np.load("data_dy/flk_sdc/blog_y0_5time.npy")
    elif data_name == 'peer_new_10':
        xs = np.load("data_dy/flk_sdc/peer_x_10time.npy")
        adjs= np.load("data_dy/flk_sdc/peer_A_10time.npy")
        Ts = np.load("data_dy/flk_sdc/peer_t_10time.npy")
        all_yfs = np.load("data_dy/flk_sdc/peer_yf_10time.npy")
        all_y1s = np.load("data_dy/flk_sdc/peer_y1_10time.npy")
        all_y0s = np.load("data_dy/flk_sdc/peer_y0_10time.npy")
    elif data_name == 'blog':
        xs = np.load("data_dy/flk_sic/blog_x_5time.npy")
        adjs= np.load("data_dy/flk_sic/blog_A_5time.npy")
        Ts = np.load("data_dy/flk_sic/blog_t_5time.npy")
        all_yfs = np.load("data_dy/flk_sic/blog_yf_5time.npy")
        all_y1s = np.load("data_dy/flk_sic/blog_y1_5time.npy")
        all_y0s = np.load("data_dy/flk_sic/blog_y0_5time.npy")
    elif data_name == 'peer':
        xs = np.load("data_dy/flk_sic/peer_x_10time.npy")
        adjs= np.load("data_dy/flk_sic/peer_A_10time.npy")
        Ts = np.load("data_dy/flk_sic/peer_t_10time.npy")
        all_yfs = np.load("data_dy/flk_sic/peer_yf_10time.npy")
        all_y1s = np.load("data_dy/flk_sic/peer_y1_10time.npy")
        all_y0s = np.load("data_dy/flk_sic/peer_y0_10time.npy")
    data.append(xs)
    data.append(adjs)
    data.append(Ts)
    data.append(all_yfs)
    data.append(all_y1s)
    data.append(all_y0s)
    return data

def data_preparation_dy(data_name, data):
    xs = data[0]
    adjs= data[1]
    Ts = data[2]
    all_yfs = data[3]
    all_y1s = data[4]
    all_y0s = data[5]
    all_y1s = list(all_y1s)
    all_y0s = list(all_y0s)
    cur_ites_true = []
    cur_all_inputs = []

    for i in range(len(all_y1s)): all_y1s[i] = all_y1s[i].reshape(len(all_y1s[i]), 1)
    for i in range(len(all_y0s)): all_y0s[i] = all_y0s[i].reshape(len(all_y0s[i]), 1)
    for i in range(len(all_y0s)): cur_ites_true.append(all_y1s[i] - all_y0s[i])
    for i in range(len(xs)):
        temp = np.concatenate([xs[i], Ts[i]], axis=1)
        cur_all_inputs.append(temp)
    return cur_all_inputs, cur_ites_true 

def train_DITE(Model_name, cur_all_inputs, data, config, val_idx, train_idx, activation=tf.nn.relu):
	losslist=[]
	cur_init_A = data[1]
	all_yfs = data[3]
	from tensorflow.keras import mixed_precision
	policy = mixed_precision.Policy('float32')
	mixed_precision.set_global_policy(policy)
	cur_model = Model_name(config, activation=activation, init_adjs=cur_init_A) 
	count = 0
	losslist_CV = []
	sum_loss = 0
	sum_val_loss = 0
	losslist = []
	start_time = time.time()
	for i in range(config['iterations']):
		print("iter", i)
		loss = cur_model.val_y(tf.cast(cur_all_inputs,tf.float32),tf.cast(all_yfs,tf.float32),train_idx, train_all_mask=train_idx)
		total_loss = cur_model.network_learn(tf.cast(cur_all_inputs,tf.float32),tf.cast(all_yfs,tf.float32),train_idx)
		val_loss = cur_model.val_y(tf.cast(cur_all_inputs,tf.float32),  tf.cast(all_yfs,tf.float32),val_idx, train_all_mask=train_idx)
		sum_loss += loss
		sum_val_loss += val_loss
		if (i+1) % 20 == 0:
			if len(losslist_CV) > 0 and sum_val_loss / 20 >= losslist_CV[-1]:
				count += 1
			else:
				count = 0
			if config['flag_early_stop']:
				if i > 400 and count >= 1:
					break
			losslist.append(sum_loss / 20)
			losslist_CV.append(sum_val_loss / 20)
			sum_loss = 0
			sum_val_loss = 0
	return cur_model
   
def implement_DITE(config, data_name, Model_name, f_activation, start_i, end_i):
    data = load_data_dy(data_name)
    cur_all_inputs, cur_ites_true = data_preparation_dy(data_name,data)
    train_idx, val_idx, test_idx = split_train_val_test(data[0][0], 0.7, 0.15, 0.15)

    for i in range(start_i, end_i):
        cur_model = train_DITE(Model_name, cur_all_inputs, data, config, val_idx, train_idx, activation=f_activation)
        cur_save_model_name = data_name + "_" + str(Model_name)[8:-2] + "_split_" + str(i)
        cur_save_path = './dy_models/Model_' + data_name + "_" + str(Model_name)[8:-2] + "_split_" + str(i)
        os.makedirs(cur_save_path, exist_ok=True)
        save_mymodel(cur_save_path, cur_save_model_name, cur_model)
        cur_test_results = []
        mse_y, pehe, err_ate = evaluation.evalate_DITE(
            cur_model, 
            data[2],
            cur_all_inputs,
            test_idx,
            data[3],
            cur_ites_true, 
            mask_idx=train_idx
        )
        cur_test_results.append(pehe)
        cur_test_results.append(err_ate)
        cur_test_results_name = './dy_results/test_results_'+ data_name + '_' + str(Model_name)[8:-2] + "_split_" + str(i)
        save_results(cur_test_results, cur_test_results_name)

