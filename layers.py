import tensorflow as tf
from tensorflow import keras
import numpy as np


def m_tanh(x): 
    return 1-2/(tf.math.exp(2 * x)+1)


class WeightFormer(tf.keras.Model):
    def __init__(self, d_model, tem=1.0):
        super(WeightFormer, self).__init__()
        self.d_model = d_model  
        self.tem = tem
        self.Wq = keras.layers.Dense(d_model)
        self.Wk = keras.layers.Dense(d_model)
        self.Wv = keras.layers.Dense(d_model)
        self.FF = keras.layers.Dense(d_model)

    def call(self, inputs, training=False): 
        seq_len = tf.shape(inputs)[1]
        xs = inputs * 1.0
        Q_last = self.Wq(xs[:, -1, :])  # [batch_size, features]
        K = self.Wk(xs)
        V = self.Wv(xs) 
        attn_scores = tf.matmul(Q_last[:, None, :], K, transpose_b=True)  # [batch, 1, T]
        scaling_factor = tf.sqrt(tf.cast(self.d_model, tf.float32))
        attn_scores = attn_scores / scaling_factor  # 
        attn_weights = tf.nn.softmax(attn_scores / self.tem, axis=-1)  
        new_para = tf.reshape(tf.matmul(attn_weights, V), shape=[len(inputs), self.d_model]) # (W,1,d) -> (W,d)
        new_para = self.FF(new_para)
        output = new_para 
        return output


class MixSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, activation=tf.nn.relu, tem=1.0, flag_val_o_t=False, flag_sparse=True, weight_neg="att"):
        super().__init__()
        self.d_model = d_model  
        self.tem = tem
        self.flag_val_o_t = flag_val_o_t
        self.Wq = keras.layers.Dense(d_model) 
        self.Wk = keras.layers.Dense(d_model)  
        self.Wv = keras.layers.Dense(d_model)  
        self.act = activation
        self.FF = keras.layers.Dense(d_model, activation=activation)
        self.flag_sparse = flag_sparse
        self.weight_neg = weight_neg
        self.computed_t_g = None

    def call(self, inputs, adj, his_ts=None):
        seq_len = tf.shape(inputs)[1]
        his = inputs * 1.0
        Q_last = self.Wq(his[:, -1, :])  
        K = self.Wk(his)
        V = self.Wv(his)
        attn_scores = tf.matmul(Q_last[:, None, :], K, transpose_b=True)  # [batch, 1, T]
        scaling_factor = tf.sqrt(tf.cast(self.d_model, tf.float32))
        attn_scores = attn_scores / scaling_factor  
        attn_weights = tf.nn.softmax(attn_scores / self.tem, axis=-1)
        new_emb = tf.reshape(tf.matmul(attn_weights, V),shape=[len(his), self.d_model])  #[N, 1, d_model]
        if his_ts != None:
            new_emb_t_g = tf.reshape(tf.matmul(attn_weights, his_ts), shape=[len(his), 1])

        if self.flag_sparse:
            adj_sparse = adj
            row_idx = adj_sparse.indices[:, 0]
            col_idx = adj_sparse.indices[:, 1]
            q_i = tf.gather(Q_last, row_idx)  # [num_edges, d]
            k_j = tf.gather(new_emb, col_idx)  # [num_edges, d]
            attn_scores_edge = tf.reduce_sum(q_i * k_j, axis=-1) / tf.sqrt(tf.cast(self.d_model, tf.float32))
            attn_scores_edge = attn_scores_edge / self.tem
            max_scores = tf.math.unsorted_segment_max(attn_scores_edge, row_idx, tf.shape(adj)[0])
            attn_scores_edge -= tf.gather(max_scores, row_idx)
            exp_scores = tf.exp(attn_scores_edge)
            sum_exp_scores = tf.math.unsorted_segment_sum(exp_scores, row_idx, num_segments=tf.shape(adj)[0])
            sum_exp_scores_safe = sum_exp_scores + 1e-9
            softmax_scores = exp_scores / tf.gather(sum_exp_scores_safe, row_idx)
            attn_sparse = tf.sparse.SparseTensor(
                indices=adj_sparse.indices,
                values=softmax_scores,
                dense_shape=adj_sparse.dense_shape
            )
            output = tf.sparse.sparse_dense_matmul(attn_sparse, new_emb)
            if his_ts != None:
                self.computed_t_g = tf.sparse.sparse_dense_matmul(attn_sparse, new_emb_t_g)
        else:
            adj = tf.cast(adj, tf.float32)
            attn_scores_ne = tf.matmul(Q_last, new_emb, transpose_b=True)  # [B, N, N]
            scaling_factor = tf.sqrt(tf.cast(self.d_model, tf.float32))
            attn_scores_ne = attn_scores_ne / scaling_factor  
            attn_scores_ne = attn_scores_ne * adj - 1e9 * (1 - adj)
            attn_weights_ne = tf.nn.softmax(attn_scores_ne / self.tem, axis=-1) 
            output = tf.matmul(attn_weights_ne, new_emb)
            adj = np.array(adj)
            if his_ts != None:
                self.computed_t_g = tf.matmul(attn_weights_ne, new_emb_t_g)
        output = self.act(self.FF(output))
        return output


class RepreLayer(keras.layers.Layer):
    def __init__(self, num_outputs, activation=tf.nn.relu):
        """ representation layer
        input:
              num_outputs:  hidden shape
              activation: the activation function of representation layer
        Output:
              The representation of current layer
        """
        super(RepreLayer,self).__init__()
        self.num_outputs = num_outputs
        self.activation = activation

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel",
            shape=[int(input_shape[-1]),
            self.num_outputs],
            dtype=tf.float32,
            initializer=tf.keras.initializers.glorot_uniform()
        )
        self.bias = self.add_weight(
            "bias",
            shape=[self.num_outputs], 
            initializer=keras.initializers.Zeros()
        )

    def call(self, input, flag=False):
        output = tf.matmul(input, self.kernel) + self.bias
        output = self.activation(output)        
        self.result = output
        return output