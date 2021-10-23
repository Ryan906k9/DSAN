import paddle
import paddle.nn as nn
import numpy as np

class LMMD_loss(nn.Layer):
    def __init__(self, class_num=31, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(LMMD_loss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.shape[ 0 ]) + int(target.shape[ 0 ])
        total = paddle.concat([ source, target ], axis=0)
        total0 = total.unsqueeze(0).expand(
            [int(total.shape[ 0 ]), int(total.shape[ 0 ]), int(total.shape[ 1 ])])
        total1 = total.unsqueeze(1).expand(
            [int(total.shape[ 0 ]), int(total.shape[ 0 ]), int(total.shape[ 1 ])])
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = paddle.sum(L2_distance) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [paddle.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def get_loss(self, source, target, s_label, t_label):
        batch_size = source.shape[0]
        weight_ss, weight_tt, weight_st = self.cal_weight(
            s_label, t_label, batch_size=batch_size, class_num=self.class_num)
        weight_ss = paddle.to_tensor(weight_ss,dtype='float32')
        weight_tt = paddle.to_tensor(weight_tt,dtype='float32')
        weight_st = paddle.to_tensor(weight_st,dtype='float32')

        kernels = self.guassian_kernel(source, target,
                                kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        
        # loss = paddle.Tensor([0]).cuda()
        loss = paddle.to_tensor([ 0 ],dtype='float32')
        
        if paddle.sum(paddle.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]
        

        loss += paddle.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
        
        return loss

    def convert_to_onehot(self, sca_label, class_num=31):
        return np.eye(class_num)[sca_label]

    def cal_weight(self, s_label, t_label, batch_size=32, class_num=31):
        batch_size = s_label.shape[ 0 ]
        s_sca_label = s_label.cpu().numpy()
        s_vec_label = self.convert_to_onehot(s_sca_label, class_num=self.class_num)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum
        # print(t_label)
        # t_sca_label = t_label.cpu().max(1)[1].numpy()
        t_sca_label = paddle.argmax(t_label, axis=1).numpy()
        t_vec_label = t_label.cpu().numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        index = list(set(s_sca_label) & set(t_sca_label))
        mask_arr = np.zeros((batch_size, class_num))
        mask_arr[:, index] = 1
        t_vec_label = t_vec_label * mask_arr
        s_vec_label = s_vec_label * mask_arr

        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)

        length = len(index)
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')