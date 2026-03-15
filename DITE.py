import math
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

import utils
import layers


class DITE(keras.Model):
    def __init__(self, config, activation, init_adjs, weight_neg="att"):
        super(DITE, self).__init__()
        print("Initialization ...")
        self.adjs_self_loop = []
        for i in range(len(init_adjs)):
            if weight_neg == "att":
                adj_self_loop = init_adjs[i] + np.eye(init_adjs[i].shape[0], dtype=init_adjs[i].dtype)
                adj_np = adj_self_loop * 1.0  
                indices = np.array(np.nonzero(adj_self_loop)).T  # 形状 (num_nonzero, 2)
                values = adj_np[indices[:, 0], indices[:, 1]]
                adj_sparse = tf.sparse.SparseTensor(
                    indices=indices,
                    values=values,
                    dense_shape=adj_np.shape
                )
                adj_sparse = tf.sparse.reorder(adj_sparse)
                self.adjs_self_loop.append(adj_sparse * 1.0)
            else:
                adj_self_loop = init_adjs[i] + np.eye(init_adjs[i].shape[0], dtype=init_adjs[i].dtype)
                adj_np = utils.normalize_adj_numpy(adj_self_loop) 
                coo = np.array(np.nonzero(adj_np)).T  # [num_edges, 2]
                values = adj_np[coo[:, 0], coo[:, 1]]
                indices = coo.astype(np.int64)
                sparse_tensor = tf.sparse.SparseTensor(
                        indices=indices,
                        values=values,
                        dense_shape=mx.shape
                    )
                adj_sparse = tf.sparse.reorder(sparse_tensor)
                self.adjs_self_loop.append(adj_sparse * 1.0)
        print("adj number", len(self.adjs_self_loop))
        self.rep_weights = []
        self.gnn_weights = []
        self.rep_layers = []
        self.rep_layers_for_dis = []
        self.rep_layers_h_n = []
        self.gnn_layers = []
        self.f1_layers = []
        self.f0_layers = []
        self.activation = activation
        self.rep_alpha = config["rep_alpha"]
        self.reg_lambda = config["reg_lambda"] 
        self.out_dropout = config["out_dropout"]
        self.GNN_dropout = config["GNN_dropout"]
        self.rep_dropout = config["rep_dropout"]
        self.inp_drop = config["inp_dropout"]
        self.use_batch = config["use_batch"]
        self.window_size = config["window_size"]
        self.train_loss = None
        self.activation = activation
        self.weight_neg = weight_neg
        self.optimizer=keras.optimizers.Adam(lr=config['lr_rate'], decay=config['lr_weigh_decay'])

        for ly in range(config["rep_hidden_layer"]):
            h = layers.RepreLayer(config["rep_hidden_shape"][ly], activation=self.activation)
            self.rep_layers.append(h)
            h_for_d = layers.RepreLayer(config["rep_hidden_shape"][ly], activation=self.activation)
            self.rep_layers_for_dis.append(h_for_d)
            h_n = layers.MixSelfAttention(config["rep_hidden_shape"][ly], activation=self.activation)
            self.rep_layers_h_n.append(h_n)

        for ly in range(config["GNN_hidden_layer"]):
            g = layers.MixSelfAttention(d_model=config["GNN_hidden_shape"][ly], activation=self.activation)
            self.gnn_layers.append(g)

        for ly in range(config["out_T_layer"]):
            out_t = keras.layers.Dense(config["out_hidden_shape"][ly], activation=self.activation)
            self.f1_layers.append(out_t)

        for ly in range(config["out_C_layer"]):
            out_c = keras.layers.Dense(config["out_hidden_shape"][ly], activation=self.activation)
            self.f0_layers.append(out_c)
            
        self.f1_out = keras.layers.Dense(1)
        self.f0_out = keras.layers.Dense(1)
        
        self.encoder_t_net = []
        for ly in range(config["GNN_hidden_layer"]):
            encoder_t_ly = keras.layers.Dense(config["GNN_hidden_shape"][ly], activation=self.activation)
            self.encoder_t_net.append(encoder_t_ly)

        self.bc_encoder = []
        for ly in range(config["GNN_hidden_layer"]):
            encoder_t_ly = keras.layers.Dense(config["GNN_hidden_shape"][ly], activation=self.activation)
            self.bc_encoder.append(encoder_t_ly)

        self.encoder_t_g_net = []
        for ly in range(config["GNN_hidden_layer"]):
            encoder_t_g_ly = keras.layers.Dense(config["GNN_hidden_shape"][ly], activation=self.activation)
            self.encoder_t_g_net.append(encoder_t_g_ly)

        self.pre_t = keras.layers.Dense(1, activation=tf.nn.sigmoid)
        self.pre_t_g = keras.layers.Dense(1, activation=tf.nn.sigmoid)

        self.wformer_f1 = layers.WeightFormer(config["out_hidden_shape"][0])
        self.wformer_f0 = layers.WeightFormer(config["out_hidden_shape"][0])
        self.wformer_z_pre_t_net = layers.WeightFormer(config["GNN_hidden_shape"][0])
        self.wformer_pre_t = layers.WeightFormer(1)
        self.wformer_out1 = layers.WeightFormer(1)
        self.wformer_out0 = layers.WeightFormer(1)

    def call(self, inputtensors, test_idx, all_ys, training=False, train_all_mask=None):
        print("Call ...")
        pre_y1s = []
        pre_y0s = []

        features = tf.cast(inputtensors[0][:, :-1], tf.float32)
        input_t = tf.cast(tf.constant(inputtensors[0][:, -1], shape = [features.shape[0], 1]), tf.float32)
        yf = tf.cast(tf.constant(all_ys[0], shape = [features.shape[0], 1]), tf.float32)
        mask = tf.scatter_nd(
            indices=tf.expand_dims(train_all_mask, axis=1),
            updates=tf.ones_like(train_all_mask, dtype=tf.float32),
            shape=(len(features), )
        )
        mask = tf.reshape(mask, (len(features), 1))
        mask = tf.cast(mask, dtype=yf.dtype)
        cur_adj = self.adjs_self_loop[0]
        hidden = features * 1.0
        hidden_dis = features * 1.0
        his = tf.concat([features], axis=-1) # (N,1, d+2)
        his_zs = [] 
        his_zs.append(tf.expand_dims(his, axis=1))
        z = his * 1.0 # (N,1, d+2)
        for ly in range(len(self.rep_layers)):
            hidden_dis = self.rep_layers_for_dis[ly](hidden_dis)
            hidden = self.rep_layers[ly](hidden)
            z = tf.expand_dims(z, axis=1)
            z = self.rep_layers_h_n[ly](z, cur_adj)
            his_zs.append(tf.expand_dims(z, axis=1))

        rep = hidden * 1.0
        rep_dis = hidden_dis * 1.0
        weights_pre_t_net = []
        z_p_t = tf.concat([z], axis=-1)
        for ly in range(len(self.encoder_t_net)):
            z_p_t = self.encoder_t_net[ly](z_p_t)
            weights_pre_t_net.append(tf.expand_dims(self.encoder_t_net[ly].kernel, axis=1))
        his_zs_s = []
        his_zs_s.append(tf.expand_dims(tf.concat([rep_dis, z_p_t], axis=-1), axis=1))

        z_s = tf.concat([rep_dis, z_p_t], axis=-1) * 1.0
        for ly in range(len(self.gnn_layers)):
            z_s = tf.expand_dims(z_s * 1.0, axis=1)
            z_s = self.gnn_layers[ly](z_s, cur_adj)
            his_zs_s.append(tf.expand_dims(z_s * 1.0, axis=1))

        concat_rep_z = tf.concat([z], axis=-1)
        for ly in range(len(self.bc_encoder)):
            concat_rep_z = self.bc_encoder[ly](concat_rep_z)

        concated_data = tf.concat([rep, concat_rep_z, z_s], axis=-1)
        test_concated_data = tf.cast(tf.gather(concated_data, test_idx), tf.float32) 
        pre_f1_weights = []
        pre_f0_weights = []

        pre_y1_emb = test_concated_data * 1.0
        for ly in range(len(self.f1_layers)):
            pre_y1_emb = self.f1_layers[ly](pre_y1_emb)
            pre_f1_weights.append(tf.expand_dims(self.f1_layers[ly].kernel, axis=1))
        pre_y1 = self.f1_out(pre_y1_emb)
        pre_f1_weights.append(tf.expand_dims(self.f1_out.kernel, axis=1))
        pre_y1s.append(pre_y1 * 1.0)

        pre_y0_emb = test_concated_data * 1.0
        for ly in range(len(self.f0_layers)):
            pre_y0_emb = self.f0_layers[ly](pre_y0_emb)
            pre_f0_weights.append(tf.expand_dims(self.f0_layers[ly].kernel, axis=1))
        pre_y0 = self.f0_out(pre_y0_emb)
        pre_f0_weights.append(tf.expand_dims(self.f0_out.kernel, axis=1))
        pre_y0s.append(pre_y0)
        
        last_input_t = input_t * 1.0
        last_yf = yf * mask
        for time in range(1, len(inputtensors)):
            features = tf.cast(inputtensors[time][:, :-1], tf.float32)
            input_t = tf.cast(tf.constant(inputtensors[time][:, -1], shape=[features.shape[0], 1]), tf.float32)
            hidden = features * 1.0
            hidden_dis = features * 1.0

            yf = tf.cast(tf.constant(all_ys[time], shape = [features.shape[0], 1]), tf.float32)
            cur_adj = self.adjs_self_loop[time]

            z = tf.concat([features], axis=-1) # (N,time, d+2)
            for ly in range(len(self.rep_layers)):
                hidden_dis = self.rep_layers_for_dis[ly](hidden_dis)
                hidden = self.rep_layers[ly](hidden)
                z = tf.expand_dims(z, axis=1)
                z = tf.concat([his_zs[ly], z], axis=1)
                his_zs[ly] = z * 1.0
                if tf.shape(z)[1] > self.window_size:
                    z = z[:, time - self.window_size:time + 1,:]
                z = self.rep_layers_h_n[ly](z, cur_adj)

            rep_dis = hidden_dis * 1.0
            rep = hidden * 1.0
            z_p_t = tf.concat([z], axis=-1)
            for ly in range(len(self.encoder_t_net)):
                cur_weight = self.wformer_z_pre_t_net(weights_pre_t_net[ly])
                z_p_t = tf.matmul(z_p_t, cur_weight) + self.encoder_t_net[ly].bias
                weights_pre_t_net[ly] = tf.concat([weights_pre_t_net[ly], tf.expand_dims(cur_weight, axis=1)], axis=1)
            concat_rep_z = tf.concat([z], axis=-1)
            for ly in range(len(self.bc_encoder)):
                concat_rep_z = self.bc_encoder[ly](concat_rep_z)
            
            z_s = tf.concat([rep_dis, z_p_t], axis=-1) * 1.0
            for ly in range(len(self.gnn_layers)):
                z_s = tf.concat([tf.expand_dims(z_s, axis=1), his_zs_s[ly]], axis=1)
                his_zs_s[ly] = z_s * 1.0
                if tf.shape(z_s)[1] > self.window_size:
                    z_s = z_s[:, time - self.window_size:time + 1,:]
                z_s = self.gnn_layers[ly](z_s, cur_adj)

            concated_data = tf.concat([rep, concat_rep_z, z_s], axis=-1)
            test_concated_data = tf.cast(tf.gather(concated_data, test_idx), tf.float32) 

            pre_y1_emb = test_concated_data * 1.0
            for ly in range(len(self.f1_layers)):
                cur_weight = self.wformer_f1(pre_f1_weights[ly])
                pre_y1_emb = tf.matmul(pre_y1_emb, cur_weight) + self.f1_layers[ly].bias
                pre_f1_weights[ly] = tf.concat([pre_f1_weights[ly], tf.expand_dims(cur_weight, axis=1)], axis=1)
            cur_weight = self.wformer_out1(pre_f1_weights[-1])
            pre_y1 = tf.matmul(pre_y1_emb, cur_weight) + self.f1_out.bias
            pre_f1_weights[-1] = tf.concat([pre_f1_weights[-1], tf.expand_dims(cur_weight, axis=1)], axis=1)
            pre_y1s.append(pre_y1 * 1.0)

            pre_y0_emb = test_concated_data * 1.0
            for ly in range(len(self.f0_layers)):
                cur_weight = self.wformer_f0(pre_f0_weights[ly])
                pre_y0_emb = tf.matmul(pre_y0_emb, cur_weight) + self.f0_layers[ly].bias
                pre_f0_weights[ly] = tf.concat([pre_f0_weights[ly], tf.expand_dims(cur_weight, axis=1)], axis=1)
            cur_weight = self.wformer_out0(pre_f0_weights[-1])
            pre_y0 = tf.matmul(pre_y0_emb, cur_weight) + self.f0_out.bias
            pre_f0_weights[-1] = tf.concat([pre_f0_weights[-1], tf.expand_dims(cur_weight, axis=1)], axis=1)
            pre_y0s.append(pre_y0)
            last_input_t = input_t * 1.0
            last_yf = yf * mask

        return pre_y1s, pre_y0s

    def get_loss(self, inputtensors, all_ys, train_idx, training=True):
        if self.use_batch:
            I = random.sample(range(0, len(train_idx)), self.use_batch)
        pre_t_loss = 0.0
        pre_y_loss = 0.0
        bc_loss = 0.0
        bi_loss = 0.0
        pre_t_g_loss = 0.0

        features = tf.cast(inputtensors[0][:, :-1], tf.float32)
        input_t = tf.cast(tf.constant(inputtensors[0][:, -1], shape = [features.shape[0], 1]), tf.float32)
        yf = tf.cast(tf.constant(all_ys[0], shape = [features.shape[0], 1]), tf.float32)
        mask = tf.scatter_nd(
            indices=tf.expand_dims(train_idx, axis=1),
            updates=tf.ones_like(train_idx, dtype=tf.float32),
            shape=(len(features), )
        )
        mask = tf.reshape(mask, (len(features), 1))
        mask = tf.cast(mask, tf.float32)
        cur_adj = self.adjs_self_loop[0]
        
        hidden = features * 1.0
        hidden_dis = features * 1.0
        his = tf.concat([features], axis=-1) # (N,1, d+2)
        his_zs = [] 
        his_zs.append(tf.expand_dims(his, axis=1))
        z = his * 1.0 # (N,1, d+2)
        for ly in range(len(self.rep_layers)):
            hidden_dis = self.rep_layers_for_dis[ly](hidden_dis)
            hidden_dis = tf.nn.dropout(hidden_dis, self.rep_dropout)
            hidden = self.rep_layers[ly](hidden)
            hidden = tf.nn.dropout(hidden, self.rep_dropout)
            z = tf.expand_dims(z, axis=1)
            z = self.rep_layers_h_n[ly](z, cur_adj)
            z = tf.nn.dropout(z, self.rep_dropout)
            his_zs.append(tf.expand_dims(z, axis=1))
        
        rep = hidden * 1.0
        rep_dis = hidden_dis * 1.0
        weights_pre_t_net = []
        z_p_t = tf.concat([z], axis=-1)
        for ly in range(len(self.encoder_t_net)):
            z_p_t = self.encoder_t_net[ly](z_p_t)
            z_p_t = tf.nn.dropout(z_p_t, self.rep_dropout)
            weights_pre_t_net.append(tf.expand_dims(self.encoder_t_net[ly].kernel, axis=1))
        pred_t = self.pre_t(tf.concat([rep_dis, z_p_t], axis=-1))
        pred_t = tf.clip_by_value(pred_t, 1e-7, 1 - 1e-7)
        weights_pre_t_net.append(tf.expand_dims(self.pre_t.kernel, axis=1))

        concat_rep_z = tf.concat([z], axis=-1)
        for ly in range(len(self.bc_encoder)):
            concat_rep_z = self.bc_encoder[ly](concat_rep_z)
            concat_rep_z = tf.nn.dropout(concat_rep_z, self.rep_dropout)
        pre_bc_t = self.pre_t(tf.concat([rep, concat_rep_z], axis=-1))

        his_zs_s = []
        his_zs_s.append(tf.expand_dims(tf.concat([rep_dis, z_p_t], axis=-1), axis=1))
        his_t_g = []
        his_t_g.append(tf.expand_dims(input_t * 1.0, axis=1))
        t_g = input_t * 1.0

        z_s = tf.concat([rep_dis, z_p_t], axis=-1) * 1.0
        for ly in range(len(self.gnn_layers)):
            z_s = tf.expand_dims(z_s * 1.0, axis=1)
            t_g = tf.expand_dims(t_g * 1.0, axis=1)
            z_s = self.gnn_layers[ly](z_s, cur_adj, t_g)
            t_g = self.gnn_layers[ly].computed_t_g
            z_s = tf.nn.dropout(z_s, self.rep_dropout)
            his_zs_s.append(tf.expand_dims(z_s * 1.0, axis=1))
            his_t_g.append(tf.expand_dims(t_g * 1.0, axis=1))
        t_g_vals = tf.nn.sigmoid(self.gnn_layers[-1].computed_t_g)

        concat_x_z_t = tf.concat([z], axis=-1)
        for ly in range(len(self.encoder_t_g_net)):
            concat_x_z_t = self.encoder_t_g_net[ly](concat_x_z_t)
            concat_x_z_t = tf.nn.dropout(concat_x_z_t, self.rep_dropout)

        train_concat_x_z_t = tf.gather(concat_x_z_t, train_idx)
        train_concat_x_z_t = tf.gather(train_concat_x_z_t, I)
        
        concated_data = tf.concat([rep, concat_rep_z, z_s], axis=-1)
        train_concated_data = tf.gather(concated_data, train_idx)
        train_input_t = tf.gather(input_t, train_idx)
        train_y = tf.gather(yf, train_idx)
        train_z_s = tf.gather(z_s, train_idx)
        train_pre_t = tf.gather(pred_t, train_idx)
        train_pre_bc_t = tf.gather(pre_bc_t, train_idx)
        train_t_g_vals = tf.gather(t_g_vals, train_idx)
        train_concated_data = tf.gather(train_concated_data, I)
        train_input_t = tf.gather(train_input_t, I)
        train_y = tf.gather(train_y, I)
        train_pre_t = tf.gather(train_pre_t, I)
        train_pre_bc_t = tf.gather(train_pre_bc_t, I)
        train_z_s = tf.gather(train_z_s, I)
        train_t_g_vals = tf.gather(train_t_g_vals, I)
        train_z_p_t = tf.gather(z_p_t, train_idx)
        train_z_p_t = tf.gather(train_z_p_t, I)
        train_concat_rep_z = tf.gather(concat_rep_z, train_idx)
        train_concat_rep_z = tf.gather(train_concat_rep_z, I)
        train_rep_dis = tf.gather(rep_dis, train_idx)
        train_rep_dis = tf.gather(train_rep_dis, I)
        train_rep = tf.gather(rep, train_idx)
        train_rep = tf.gather(train_rep, I)

        ran_vals = tf.random.uniform(shape=(len(train_z_s), 1), minval=0, maxval=1)
        train_pre_t_g = self.pre_t_g(tf.concat([train_rep_dis, train_concat_x_z_t, train_input_t], axis=-1))
        train_pre_t_g_balance = self.pre_t_g(tf.concat([train_rep, train_concat_rep_z, train_input_t], axis=-1))

        pre_t_loss += -tf.reduce_sum(train_input_t * tf.math.log(train_pre_t) + (1 - train_input_t) * tf.math.log(1 - train_pre_t))
        pre_t_g_loss += tf.reduce_sum((train_pre_t_g - train_t_g_vals) ** 2)

        bc_loss += tf.reduce_sum((train_pre_bc_t - 0.5) ** 2)
        bi_loss += tf.reduce_sum((train_pre_t_g_balance - ran_vals) ** 2)

        pre_f1_weights = []
        pre_f0_weights = []
        group_t, group_c, i_0, i_1= utils.divide_groups(train_concated_data, train_input_t)
        pre_y1_emb = group_t * 1.0
        for ly in range(len(self.f1_layers)):
            pre_y1_emb = self.f1_layers[ly](pre_y1_emb)
            pre_y1_emb = tf.nn.dropout(pre_y1_emb, self.rep_dropout)
            pre_f1_weights.append(tf.expand_dims(self.f1_layers[ly].kernel, axis=1))
        pre_y1 = self.f1_out(pre_y1_emb)
        pre_f1_weights.append(tf.expand_dims(self.f1_out.kernel, axis=1))

        pre_y0_emb = group_c * 1.0
        for ly in range(len(self.f0_layers)):
            pre_y0_emb = self.f0_layers[ly](pre_y0_emb)
            pre_y0_emb = tf.nn.dropout(pre_y0_emb, self.rep_dropout)
            pre_f0_weights.append(tf.expand_dims(self.f0_layers[ly].kernel, axis=1))
        pre_y0 = self.f0_out(pre_y0_emb)
        pre_f0_weights.append(tf.expand_dims(self.f0_out.kernel, axis=1))

        y_pre = tf.dynamic_stitch([i_0, i_1], [pre_y0, pre_y1])
        pre_y_loss += tf.reduce_sum(tf.square(train_y - y_pre))

        last_input_t = input_t * 1.0
        last_yf = yf * mask
        for time in range(1, len(inputtensors)):
            features = tf.cast(inputtensors[time][:, :-1], tf.float32)
            input_t = tf.cast(tf.constant(inputtensors[time][:, -1], shape=[features.shape[0], 1]), tf.float32)
            hidden = features * 1.0
            hidden_dis = features * 1.0
            yf = tf.cast(tf.constant(all_ys[time], shape = [features.shape[0], 1]), tf.float32)
            cur_adj = self.adjs_self_loop[time]

            z = tf.concat([features], axis=-1) # (N,time, d+2)
            for ly in range(len(self.rep_layers)):
                hidden_dis = self.rep_layers_for_dis[ly](hidden_dis)
                hidden_dis = tf.nn.dropout(hidden_dis, self.rep_dropout)
                
                hidden = self.rep_layers[ly](hidden)
                hidden = tf.nn.dropout(hidden, self.rep_dropout)
                z = tf.expand_dims(z, axis=1)
                z = tf.concat([his_zs[ly], z], axis=1)
                his_zs[ly] = z * 1.0
                if tf.shape(z)[1] > self.window_size:
                    z = z[:, time - self.window_size:time + 1, :]
                z = self.rep_layers_h_n[ly](z, cur_adj)
                z = tf.nn.dropout(z, self.rep_dropout)

            rep = hidden * 1.0
            rep_dis = hidden_dis * 1.0
            z_p_t = tf.concat([z], axis=-1)
            for ly in range(len(self.encoder_t_net)):
                cur_weight = self.wformer_z_pre_t_net(weights_pre_t_net[ly])
                z_p_t = tf.matmul(z_p_t, cur_weight) + self.encoder_t_net[ly].bias
                z_p_t = tf.nn.dropout(z_p_t, self.rep_dropout)
                weights_pre_t_net[ly] = tf.concat([weights_pre_t_net[ly], tf.expand_dims(cur_weight, axis=1)], axis=1)
            cur_weight = self.wformer_pre_t(weights_pre_t_net[-1])
            pred_t = tf.matmul(tf.concat([hidden_dis, z_p_t], axis=-1), cur_weight) + self.pre_t.bias
            pred_t = tf.clip_by_value(pred_t, 1e-7, 1 - 1e-7)
            weights_pre_t_net[-1] = tf.concat([weights_pre_t_net[-1], tf.expand_dims(cur_weight, axis=1)], axis=1)

            concat_rep_z = tf.concat([z], axis=-1)
            for ly in range(len(self.bc_encoder)):
                concat_rep_z = self.bc_encoder[ly](concat_rep_z)
                concat_rep_z = tf.nn.dropout(concat_rep_z, self.rep_dropout)
            cur_weight = self.wformer_pre_t(weights_pre_t_net[-1])
            pre_bc_t = tf.matmul(tf.concat([rep, concat_rep_z], axis=-1), cur_weight) + self.pre_t.bias
            
            z_s = tf.concat([rep_dis, z_p_t], axis=-1) * 1.0
            t_g = input_t * 1.0
            for ly in range(len(self.gnn_layers)):
                z_s = tf.concat([tf.expand_dims(z_s, axis=1), his_zs_s[ly]], axis=1)
                t_g = tf.concat([tf.expand_dims(t_g, axis=1), his_t_g[ly]], axis=1)
                his_zs_s[ly] = z_s * 1.0
                his_t_g[ly] = t_g * 1.0
                if tf.shape(z_s)[1] > self.window_size:
                    z_s = z_s[:, time - self.window_size:time + 1,:]
                z_s = self.gnn_layers[ly](z_s, cur_adj, t_g)
                t_g = self.gnn_layers[ly].computed_t_g
                z_s = tf.nn.dropout(z_s, self.rep_dropout)

            t_g_vals = tf.nn.sigmoid(self.gnn_layers[-1].computed_t_g)
            concat_x_z_t = tf.concat([z], axis=-1)
            for ly in range(len(self.encoder_t_g_net)):
                concat_x_z_t = self.encoder_t_g_net[ly](concat_x_z_t)
                concat_x_z_t = tf.nn.dropout(concat_x_z_t, self.rep_dropout)
            train_concat_x_z_t = tf.gather(concat_x_z_t, train_idx)
            train_concat_x_z_t = tf.gather(train_concat_x_z_t, I)
            
            concated_data = tf.concat([rep, concat_rep_z, z_s], axis=-1)
            train_concated_data = tf.gather(concated_data, train_idx)
            train_input_t = tf.gather(input_t, train_idx)
            train_y = tf.gather(yf, train_idx)
            train_z_s = tf.gather(z_s, train_idx)
            train_pre_t = tf.gather(pred_t, train_idx)
            train_pre_bc_t = tf.gather(pre_bc_t, train_idx)
            train_t_g_vals = tf.gather(t_g_vals, train_idx)
            train_concated_data = tf.gather(train_concated_data, I)
            train_input_t = tf.gather(train_input_t, I)
            train_y = tf.gather(train_y, I)
            train_pre_t = tf.gather(train_pre_t, I)
            train_pre_bc_t = tf.gather(train_pre_bc_t, I)
            train_z_s = tf.gather(train_z_s, I)
            train_t_g_vals = tf.gather(train_t_g_vals, I)
            train_z_p_t = tf.gather(z_p_t, train_idx)
            train_z_p_t = tf.gather(train_z_p_t, I)
            train_concat_rep_z = tf.gather(concat_rep_z, train_idx)
            train_concat_rep_z = tf.gather(train_concat_rep_z, I)
            train_rep_dis = tf.gather(rep_dis, train_idx)
            train_rep_dis = tf.gather(train_rep_dis, I)
            train_rep = tf.gather(rep, train_idx)
            train_rep = tf.gather(train_rep, I)

            ran_vals = tf.random.uniform(shape=(len(train_t_g_vals), 1), minval=0, maxval=1)

            train_pre_t_g = self.pre_t_g(tf.concat([train_rep_dis, train_concat_x_z_t, train_input_t], axis=-1))
            train_pre_t_g_balance = self.pre_t_g(tf.concat([train_rep, train_concat_rep_z, train_input_t], axis=-1))

            pre_t_loss += -tf.reduce_sum(train_input_t * tf.math.log(train_pre_t) + (1 - train_input_t) * tf.math.log(1 - train_pre_t))
            pre_t_g_loss += tf.reduce_sum((train_pre_t_g - train_t_g_vals) ** 2)

            bc_loss += tf.reduce_sum((train_pre_bc_t - 0.5) ** 2)
            bi_loss += tf.reduce_sum((train_pre_t_g_balance - ran_vals) ** 2)

            group_t, group_c, i_0, i_1= utils.divide_groups(train_concated_data, train_input_t)

            pre_y1_emb = group_t * 1.0
            for ly in range(len(self.f1_layers)):
                cur_weight = self.wformer_f1(pre_f1_weights[ly])
                pre_y1_emb = tf.matmul(pre_y1_emb, cur_weight) + self.f1_layers[ly].bias
                pre_y1_emb = tf.nn.dropout(pre_y1_emb, self.rep_dropout)
                pre_f1_weights[ly] = tf.concat([pre_f1_weights[ly], tf.expand_dims(cur_weight, axis=1)], axis=1)
            cur_weight = self.wformer_out1(pre_f1_weights[-1])
            pre_y1 = tf.matmul(pre_y1_emb, cur_weight) + self.f1_out.bias
            pre_f1_weights[-1] = tf.concat([pre_f1_weights[-1], tf.expand_dims(cur_weight, axis=1)], axis=1)

            pre_y0_emb = group_c * 1.0
            for ly in range(len(self.f0_layers)):
                cur_weight = self.wformer_f0(pre_f0_weights[ly])
                pre_y0_emb = tf.matmul(pre_y0_emb, cur_weight) + self.f0_layers[ly].bias
                pre_y0_emb = tf.nn.dropout(pre_y0_emb, self.rep_dropout)
                pre_f0_weights[ly] = tf.concat([pre_f0_weights[ly], tf.expand_dims(cur_weight, axis=1)], axis=1)
            cur_weight = self.wformer_out0(pre_f0_weights[-1])
            pre_y0 = tf.matmul(pre_y0_emb, cur_weight) + self.f0_out.bias
            pre_f0_weights[-1] = tf.concat([pre_f0_weights[-1], tf.expand_dims(cur_weight, axis=1)], axis=1)

            y_pre = tf.dynamic_stitch([i_0, i_1], [pre_y0, pre_y1])
            pre_y_loss += tf.reduce_sum(tf.square(train_y - y_pre))
            last_input_t = input_t * 1.0
            last_yf = yf * mask

        mean_y_pre_loss = pre_y_loss / len(inputtensors) / self.use_batch
        mean_pre_t_loss = pre_t_loss / len(inputtensors) / self.use_batch
        mean_pre_t_g_loss = pre_t_g_loss / len(inputtensors) / self.use_batch
        mean_bc_loss = bc_loss / len(inputtensors) / self.use_batch
        mean_bi_loss = bi_loss / len(inputtensors) / self.use_batch
        total_loss = mean_y_pre_loss + self.reg_lambda * mean_pre_t_loss + self.reg_lambda * mean_pre_t_g_loss + self.rep_alpha * mean_bc_loss + self.rep_alpha * mean_bi_loss
        print("train total loss", total_loss)
        return total_loss

    def get_grad(self,inputtensor,y,train_idx):
        with tf.GradientTape() as tape:
            tape.watch(self.variables)
            L = self.get_loss(inputtensor,y,train_idx)
            self.train_loss = L
            g = tape.gradient(L, self.variables)
        return g
        
    def network_learn(self, inputtensor,y,train_idx):
        g = self.get_grad(inputtensor,y,train_idx)
        self.optimizer.apply_gradients(zip(g,self.variables))
        return self.train_loss
   
    def val_y(self,inputtensors,all_ys,train_idx,training=False, train_all_mask=None):
        pre_t_loss = 0.0
        pre_y_loss = 0.0
        bc_loss = 0.0
        bi_loss = 0.0
        
        features = tf.cast(inputtensors[0][:, :-1], tf.float32)
        input_t = tf.cast(tf.constant(inputtensors[0][:, -1], shape = [features.shape[0], 1]), tf.float32)
        yf = tf.cast(tf.constant(all_ys[0], shape = [features.shape[0], 1]), tf.float32)
        cur_adj = self.adjs_self_loop[0]
        mask = tf.scatter_nd(
            indices=tf.expand_dims(train_all_mask, axis=1),
            updates=tf.ones_like(train_all_mask, dtype=tf.float32),
            shape=(len(features), )
        )
        mask = tf.reshape(mask, (len(features), 1))

        mask = tf.cast(mask, dtype=yf.dtype)
        
        hidden = features * 1.0
        hidden_dis = features * 1.0

        his = tf.concat([features], axis=-1) # (N,1, d+2)
        his_zs = [] 
        his_zs.append(tf.expand_dims(his, axis=1))
        z = his * 1.0 # (N,1, d+2)
        for ly in range(len(self.rep_layers)):
            hidden = self.rep_layers[ly](hidden)
            hidden_dis = self.rep_layers_for_dis[ly](hidden_dis)
            z = tf.expand_dims(z, axis=1)
            z = self.rep_layers_h_n[ly](z, cur_adj)
            z = tf.nn.dropout(z, self.rep_dropout)
            his_zs.append(tf.expand_dims(z, axis=1))

        rep = hidden * 1.0
        rep_dis = hidden_dis * 1.0
        weights_pre_t_net = []
        z_p_t = tf.concat([z], axis=-1)
        for ly in range(len(self.encoder_t_net)):
            z_p_t = self.encoder_t_net[ly](z_p_t)
            weights_pre_t_net.append(tf.expand_dims(self.encoder_t_net[ly].kernel, axis=1))
        pred_t = self.pre_t(tf.concat([rep_dis, z_p_t], axis=1))
        weights_pre_t_net.append(tf.expand_dims(self.pre_t.kernel, axis=1))

        concat_rep_z = tf.concat([z], axis=-1)
        for ly in range(len(self.bc_encoder)):
            concat_rep_z = self.bc_encoder[ly](concat_rep_z)
        his_zs_s = []
        his_zs_s.append(tf.expand_dims(tf.concat([rep_dis, z_p_t], axis=-1), axis=1))
        z_s = tf.concat([rep_dis, z_p_t], axis=-1)
        for ly in range(len(self.gnn_layers)):
            z_s = tf.expand_dims(z_s * 1.0, axis=1)
            z_s = self.gnn_layers[ly](z_s, cur_adj)
            his_zs_s.append(tf.expand_dims(z_s * 1.0, axis=1))

        concated_data = tf.concat([rep, concat_rep_z, z_s], axis=-1)
        train_concated_data = tf.gather(concated_data, train_idx)
        train_input_t = tf.gather(input_t, train_idx)
        train_y = tf.gather(yf, train_idx)

        pre_f1_weights = []
        pre_f0_weights = []
        group_t, group_c, i_0, i_1= utils.divide_groups(train_concated_data, train_input_t)

        pre_y1_emb = group_t * 1.0
        for ly in range(len(self.f1_layers)):
            pre_y1_emb = self.f1_layers[ly](pre_y1_emb)
            #pre_y1_emb = tf.nn.dropout(pre_y1_emb, self.rep_dropout)
            pre_f1_weights.append(tf.expand_dims(self.f1_layers[ly].kernel, axis=1))
        pre_y1 = self.f1_out(pre_y1_emb)
        pre_f1_weights.append(tf.expand_dims(self.f1_out.kernel, axis=1))

        pre_y0_emb = group_c * 1.0
        for ly in range(len(self.f0_layers)):
            pre_y0_emb = self.f0_layers[ly](pre_y0_emb)
            pre_f0_weights.append(tf.expand_dims(self.f0_layers[ly].kernel, axis=1))
        pre_y0 = self.f0_out(pre_y0_emb)
        pre_f0_weights.append(tf.expand_dims(self.f0_out.kernel, axis=1))

        y_pre = tf.dynamic_stitch([i_0, i_1], [pre_y0, pre_y1])
        pre_y_loss += tf.reduce_sum(tf.square(train_y - y_pre))

        last_input_t = input_t * 1.0
        last_yf = yf * mask
        for time in range(1, len(inputtensors)):
            features = tf.cast(inputtensors[time][:, :-1], tf.float32)
            input_t = tf.cast(tf.constant(inputtensors[time][:, -1], shape=[features.shape[0], 1]), tf.float32)
            hidden = features * 1.0
            hidden_dis = features * 1.0
            yf = tf.cast(tf.constant(all_ys[time], shape = [features.shape[0], 1]), tf.float32)

            cur_adj = self.adjs_self_loop[time]
            z = tf.concat([features], axis=-1) # (N,time, d+2)
            for ly in range(len(self.rep_layers)):
                hidden_dis = self.rep_layers_for_dis[ly](hidden_dis)
                hidden = self.rep_layers[ly](hidden)
                z = tf.expand_dims(z, axis=1)
                z = tf.concat([his_zs[ly], z], axis=1)
                his_zs[ly] = z * 1.0
                if tf.shape(z)[1] > self.window_size:
                    z = z[:, time - self.window_size:time + 1,:]
                z = self.rep_layers_h_n[ly](z, cur_adj)

            rep = hidden * 1.0
            rep_dis = hidden_dis * 1.0
            z_p_t = tf.concat([z], axis=-1)
            for ly in range(len(self.encoder_t_net)):
                cur_weight = self.wformer_z_pre_t_net(weights_pre_t_net[ly])
                z_p_t = tf.matmul(z_p_t, cur_weight) + self.encoder_t_net[ly].bias
                weights_pre_t_net[ly] = tf.concat([weights_pre_t_net[ly], tf.expand_dims(cur_weight, axis=1)], axis=1)

            concat_rep_z = tf.concat([z], axis=-1)
            for ly in range(len(self.bc_encoder)):
                concat_rep_z = self.bc_encoder[ly](concat_rep_z)
                
            z_s = tf.concat([rep_dis, z_p_t], axis=-1)
            for ly in range(len(self.gnn_layers)):
                z_s = tf.concat([tf.expand_dims(z_s, axis=1), his_zs_s[ly]], axis=1)
                his_zs_s[ly] = z_s * 1.0
                if tf.shape(z_s)[1] > self.window_size:
                    z_s = z_s[:, time - self.window_size:time + 1,:]
                z_s = self.gnn_layers[ly](z_s, cur_adj)
            concated_data = tf.concat([rep, concat_rep_z, z_s], axis=-1)
            train_concated_data = tf.gather(concated_data, train_idx)
            train_input_t = tf.gather(input_t, train_idx)
            train_y = tf.gather(yf, train_idx)
            group_t, group_c, i_0, i_1= utils.divide_groups(train_concated_data, train_input_t)

            pre_y1_emb = group_t * 1.0
            for ly in range(len(self.f1_layers)):
                cur_weight = self.wformer_f1(pre_f1_weights[ly])
                pre_y1_emb = tf.matmul(pre_y1_emb, cur_weight) + self.f1_layers[ly].bias
                pre_f1_weights[ly] = tf.concat([pre_f1_weights[ly], tf.expand_dims(cur_weight, axis=1)], axis=1)
            cur_weight = self.wformer_out1(pre_f1_weights[-1])
            pre_y1 = tf.matmul(pre_y1_emb, cur_weight) + self.f1_out.bias
            pre_f1_weights[-1] = tf.concat([pre_f1_weights[-1], tf.expand_dims(cur_weight, axis=1)], axis=1)

            pre_y0_emb = group_c * 1.0
            for ly in range(len(self.f0_layers)):
                cur_weight = self.wformer_f0(pre_f0_weights[ly])
                pre_y0_emb = tf.matmul(pre_y0_emb, cur_weight) + self.f0_layers[ly].bias
                pre_f0_weights[ly] = tf.concat([pre_f0_weights[ly], tf.expand_dims(cur_weight, axis=1)], axis=1)

            cur_weight = self.wformer_out0(pre_f0_weights[-1])
            pre_y0 = tf.matmul(pre_y0_emb, cur_weight) + self.f0_out.bias
            pre_f0_weights[-1] = tf.concat([pre_f0_weights[-1], tf.expand_dims(cur_weight, axis=1)], axis=1)
            y_pre = tf.dynamic_stitch([i_0, i_1], [pre_y0, pre_y1])
            pre_y_loss += tf.reduce_sum(tf.square(train_y - y_pre))
            last_input_t = input_t * 1.0
            last_yf = yf * mask

        mean_y_pre_loss = pre_y_loss / len(inputtensors) / len(inputtensors[0])
        return mean_y_pre_loss
