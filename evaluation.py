import numpy as np   
import tensorflow as tf 


def evalate_DITE(Model, Ts, cur_all_inputs, test_idx, true_y, true_ite, mask_idx):
    pre_y1s,pre_y0s = Model(cur_all_inputs, test_idx, true_y, False, train_all_mask=mask_idx)
    result_pehe = 0
    result_ate = 0
    for i in range(len(pre_y1s)):
        yf_pre = (Ts[i][test_idx] * pre_y1s[i]) + ((1 - Ts[i][test_idx]) * pre_y0s[i])
        ite_pre = pre_y1s[i] - pre_y0s[i]
        pehe = np.mean((ite_pre - true_ite[i][test_idx]) ** 2)
        ate_pre = np.mean(ite_pre)
        err_ate = np.abs(ate_pre - np.mean(true_ite[i][test_idx]))
        result_pehe += pehe
        result_ate += err_ate
    return result_pehe / len(pre_y1s), result_ate / len(pre_y1s)
