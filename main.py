import math

import tensorflow as tf

import utils
import DITE


def main(dataname):
    configs, activation = utils.config_pare_DITE(
        iterations=2000,
        lr_rate=0.001,
        lr_weigh_decay=0.001,
        flag_early_stop=True,
        use_batch=1024,
        rep_alpha=0.1,
        out_dropout=0.1,
        GNN_dropout=0.1,
        rep_dropout=0.1,
        inp_dropout = 0.0,
        rep_hidden_layer=3,
        rep_hidden_shape=[100, 100, 100],
        GNN_hidden_layer=2,
        GNN_hidden_shape=[100, 100],
        out_T_layer=3,
        out_C_layer=3,
        out_hidden_shape=[100,100,100], 
        activation=tf.nn.relu,
        phi_shape=100,
        att_hidden_shape=100,
        GRU_hidden_shape=100,
        reg_lambda=0.1,
        window_size=10,
        flag_matt=False
    )
    utils.implement_DITE(
        config=configs,
        data_name=dataname,
        Model_name=DITE.DITE,
        f_activation=activation,
        start_i=0,
        end_i=10
    )


if __name__ == '__main__':
    main('flickr_new_5')    

